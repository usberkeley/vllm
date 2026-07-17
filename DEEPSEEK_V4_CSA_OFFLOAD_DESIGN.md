# DeepSeek V4 c4a CPU offload 与 selected-page GPU reload 设计

> **范围:** 本文验证 PD 分离部署下 DeepSeek V4 c4a 压缩 latent 在
> `fp8_ds_mla` layout 下的 P→D CPU DRAM offload 与 D 侧 selected-page GPU
> reload。接口按通用 sparse selected-page offload 设计,后续可通过 adapter
> 扩展到 GLM 5.2 DSA 等模型。
>
> **场景:** PD 分离(prefill/decode 分实例)。P 出首 token 后经 connector 把
> sealed c4a latent 送到 D 的本机 CPU DRAM 权威池;D 只在 HBM 保留 hot pool 和
> tail,decode 逐层按 top-k page-in。不支持纯聚合(prefill/decode 同实例)行为型
> offload。
>
> **实现边界:** 只面向 DeepSeek V4 sparse MLA decode 路径 + PD 分离。

---

## 1. 目标与结论

DeepSeek V4 的 CSA 层按 `compress_ratio` 分成三类:

| 层型 | 语义 | offload 结论 |
|---|---|---|
| `swaonly` | SWA 窗口 KV,每步必读 | 常驻 D HBM |
| `c4a` | 压缩池约 `S/4`,由 indexer top-k 选择 | **唯一 offload 候选** |
| `c128a` / HCA | 压缩池约 `S/128`,当前语义接近全量枚举 | 常驻 D HBM |

目标是让 D 实例在 decode 阶段只为 c4a 保留 hot pool 和 tail,c4a 的全量压缩池由
P 经 connector 下沉到 D 的本机 CPU DRAM 权威池,在同卡上容纳更长上下文或更高
并发。正确性要求是**逐 token logits 无损**;性能收益必须由页级观测数据证明。

核心结论:

1. c4a latent 封存后只读,CPU 权威副本由 P 经 connector 送到 D;D 侧 evict 是纯
   丢弃,不写回。
2. indexer key、SWA、C128A 必须常驻 D 的 HBM,不进入 offload 候选。
3. MLA kernel 只能从 HBM gather,所以 CPU miss 页必须先 load 到 D 的 GPU hot
   pool,再 patch 本层压缩 `block_table`。
4. offload 走自研 `SparsePageConnector`:拥有全部 V4 层,复用 NIXL 传输原语(DRAM
   注册 / `CopyBlocksOp` / notif),控制面自建。不复用 NIXL 的整层 rehydrate
   语义,不接 LMCache / Mooncake 控制面。

---

## 2. 关键事实与约束

### 2.1 关键数字

| 项 | DeepSeek V4 数值 / 说明 |
|---|---|
| c4a 压缩 latent 条目 | 584 B;仅指 `fp8_ds_mla` UE8M0 paged layout |
| indexer key 条目 | 132 B;当前 FP4/MXFP4 路径仍按 132 B 分配 |
| c4a 页 | 64 压缩条目 = 256 原始 token = 37,376 B |
| c4a 每层页数 | `ceil(S / 256)` |
| top-k 候选池 | c4a 的 indexer 在 `S/4` 压缩条目上选 top-k |

落地前还必须核对当前 backend 的真实 KV cache shape。当前代码中
`flashmla_sparse.py` 的注释写 DeepSeek V4 `fp8_ds_mla` 条目是 584B,但
`FlashMLASparseBackend.get_kv_cache_shape()` 在 `fp8_ds_mla` 下仍返回 656B。
这可能是 V3.2 旧路径、另一个 V4 backend 覆盖,或目标 commit 的待修正点。
实现 selected-page copy 前必须把最终使用的 c4a tensor shape、page size 和
alignment 写成断言,不能只依赖文档常量。

### 2.2 必须保持的正确性不变式

- **top-k 始终在全量逻辑池上计算。** indexer key 全驻留,offload 只影响被选中
  latent 的物理驻留位置。
- **MLA 前必须保证选中页在 HBM。** 对真实 top-k 命中的页直接 gather;对 miss 页
  先 CPU→GPU load,再等待完成。
- **`block_table` 必须按层 patch。** V4 metadata 的原始 `attn_metadata.block_table`
  没有 layer 维度,而 hot pool 是 per c4a layer 的,不能原地修改共享表。
- **尾页常驻。** D 上随 decode 延伸写入的尾部压缩页可变,不能使用 CPU 陈旧副本。
- **P 只发送、D 只按需驻留。** P 经 connector 把 sealed c4a latent 发送到 D 的 CPU
  权威池后,P 请求结束整体释放 KV;D 不为 c4a 全量分配 HBM,只分配 hot pool +
  tail(allocate-partial)。不允许用 NIXL 的整层 rehydrate 把 c4a 静默灌回 HBM。
- **关闭开关时零行为变化。** dense attention、SWA、C128A、现有 KV connector 与 CPU
  KV offload 路径不受影响。

### 2.3 `block_table` 映射

V4 decode 路径中,`compute_global_topk_indices_and_lens` 将 indexer 选出的压缩条目
`tok` 转成物理槽位:

```text
slot = block_table[req, tok // 64] * 64 + tok % 64
```

因此 c4a 的一个 offload page 对应本层压缩 `block_table` 的一列。`patch_block_table`
是 `LogicalPage -> PhysSlot` 的 1:1 标量或 scatter 写,没有 4 倍展开。

实现要求:

- 为每个 c4a 层准备固定地址的 layer-local compressed block table buffer。
- 每步从原始压缩表拷贝或初始化,只 patch 本层选中列。
- 不能 patch SWA 的 64-token `swa_metadata.block_table`。
- 落地前重新核对目标 commit 的 `attn_metadata.block_size`、`compress_ratio`、
  `storage_block_size` 和列语义。

不同 sparse MLA backend 的 hook 形态可能不同:

- 若 backend 接收 logical top-k 后内部调用
  `triton_convert_req_index_to_global_index`,则 patch 的输入应是
  layer-local compressed `block_table`。
- 若 backend 已经在 Python 侧生成 `topk_indices_physical`,则可以让
  coordinator 直接生成 patched physical top-k buffer,但必须保持
  `topk_indices_buffer` 原值不变,便于观测和回退。
- 不允许原地修改共享 `common_attn_metadata.block_table_tensor`,因为同一步内
  SWA、C128A、indexer 和其他 c4a 层可能继续读取它。

最低接口契约:

```text
input:
  req_id_per_token: int32[num_tokens]
  topk_indices: int32[num_tokens, index_topk]       # compressed logical token id
  source_block_table: int32[num_reqs, max_blocks]  # shared compressed table
  layer_name: str
  compress_ratio: 4
  storage_block_size: 64

output, 二选一:
  patched_block_table: int32[num_reqs, max_blocks]       # fixed address
  patched_topk_physical: int32[num_tokens, index_topk]    # fixed address
```

### 2.4 暂不支持范围

Phase 1 fail-closed,以下场景不启用行为型 offload:

- 纯聚合部署(prefill/decode 同实例)。本设计只支持 PD 分离。
- 非 DeepSeek V4、非 `fp8_ds_mla`、非 `compress_ratio == 4` c4a 层。
- DCP / decode context parallel。当前 indexer 对 `compress_ratio > 1` + DCP
  仍是 `NotImplementedError`,offload 不应绕过该限制。
- C128A、SWA、dense MLA、ROCm/XPU backend。
- MTP/native spec decode、多 token decode、DSpark non-causal decode,除非另有
  logits 无损验证。
- CPU pool 不足、hot pool 无可驱逐页、connector 传输失败、backend shape 不匹配、
  D 侧 allocate-partial 不支持。

禁用时必须保持原路径行为(包括退回不启用 PD offload,由 D 常规接收全量 c4a 到
HBM),最多输出一次 warning 或 telemetry 标记。

---

## 3. 架构

### 3.1 数据流

```text
P (prefill 实例):
  c4a page 写满 -> seal
  prefill sparse MLA 用原始 block_table 完成 gather -> 出首 token
  connector.save 按 layer_name 分流:
    c4a       -> NIXL DRAM xfer 到 D 的 CPU 权威池
    indexer key / C128A / SWA -> 常规 KV 传输到 D 的 HBM
  request_finished -> free-now, kv_transfer_params 携带 last-token Top-K 页
  P allocator 整请求释放 KV

D (decode 实例):
  connector 收齐 c4a latent -> 落 CPU 权威池, 标 cpu_ready_evicted
                其余层 -> 落 HBM
  last-token Top-K 页 seed -> per-layer hot pool
  allocate-partial: c4a 只在 HBM 分配 hot pool + tail

  decode 每层:
    indexer top-k
      -> adapter 将 top-k 压缩条目去重成 logical pages
      -> 查 residency: hit 直接用;miss 批量 CPU->GPU load 到 hot pool
      -> patch 本层 layer-local block_table
      -> wait 真实 miss
      -> sparse MLA gather
```

CPU DRAM 是只读权威池,驻留在 D 本机 NUMA/Grace 内存;GPU hot pool 是每层缓存。
evict 不写回,只删除 residency 映射并释放 GPU slot。页在当前 decode step 中所有
消费者完成前不能释放 GPU 槽。

PD 分离下主干验收不是"复制到 CPU 后仍保留原 GPU cache",而是 **D 从一开始就不为
c4a 全量分配 HBM**:c4a 的 HBM survivor set 收敛到 last-token Top-K latent 集合、
hot pool 和 mutable tail。

last-token Top-K latent 可按页粒度承载:P 从最后一个有效 token 的 `topk_indices`
取出去重 logical pages 随 KV 一起发送,D 用它 seed hot pool;页内未被 Top-K 命中的
latent 是页粒度 carrier 的副作用,不计入逻辑 survivor set。若后续 backend 支持
compact physical top-k buffer,可把 survivor set 降到条目粒度,但 Phase 1 不要求改
kernel gather 语义。

### 3.2 MLA page offload 模块

selected-page offload 的 D 侧控制面放在 MLA attention backend 附近:
`vllm/v1/attention/backends/mla/page_offload/`。

该目录承载 offload 生命周期、驻留状态、hot pool、传输描述、观测指标和模型
adapter。P→D 的填充由 `SparsePageConnector`(在 `vllm/distributed/kv_transfer/`
下)负责,复用 NIXL 传输原语;D 侧 page-in 的 host↔device copy 复用 NIXL 的
`CopyBlocksOp`。page_offload 模块不拥有跨实例传输,只拥有 D 侧 selected-page 控制面。

```text
vllm/v1/attention/backends/mla/page_offload/
├── __init__.py
├── config.py            SparsePageOffloadConfig
├── selected_pages.py    SparsePageAdapter / SparsePageSelection
├── paging.py            SparsePageState / SparseHotPagePool
├── cpu_pool.py          CPUAuthoritativePagePool
├── manager.py           SparsePageOffloadManager
├── block_table.py       LayerLocalBlockTables
├── transfer.py          页 page-in copy 描述符(复用 CopyBlocksOp)
├── coordinator.py       SparsePageOffloadCoordinator
├── telemetry.py         SparseSelectionCollector
└── adapters/
    └── deepseek_v4_c4a.py
```

文件职责建议:

| 文件 | 职责 | 不负责 |
|---|---|---|
| `__init__.py` | 暴露最小公共入口,例如 coordinator/config/adapter 类型;避免让模型侧 import 内部状态类 | 不做注册副作用,不初始化全局状态 |
| `config.py` | 解析和校验 page offload 配置,包括开关、每层 hot pool 页数、lookahead、层选择、带宽预算、观测模式、传输后端 | 不读取 HF model config 的模型语义,不决定某个 layer 是否可 offload |
| `selected_pages.py` | 定义 adapter 抽象和一次 top-k 选择的结构化结果,例如 `(request, layer, logical_page)` 去重集合、真实 miss 集合、tail pin 标记 | 不维护长期 residency,不分配 GPU/CPU 槽 |
| `paging.py` | 维护 GPU hot pool 和页驻留状态,包括 `LogicalPage -> PhysSlot`、in-flight 引用、tail pin、本步 protected 页、hot_score 和淘汰候选 | 不提交传输,不 patch block table,不关心具体模型 top-k 张量布局 |
| `cpu_pool.py` | 管理 D 侧 CPU DRAM 权威副本的页槽和生命周期,接收 connector 送来的页,记录 ready/ref_cnt 状态,按请求结束释放 | 不做 GPU hot pool 淘汰,不拥有 P→D 传输 |
| `manager.py` | 聚合 `paging.py` 与 `cpu_pool.py` 的状态机,提供 receive(connector 填充)、seed、promote/load、evict、complete、request cleanup 等 D worker-side 控制面 | 不实现 scheduler-side `OffloadingManager`,不拥有 connector |
| `block_table.py` | 管理固定地址、固定形状的 layer-local compressed block table buffer,并提供原地 patch 接口 | 不修改共享 `attn_metadata.block_table`,不 patch SWA block table |
| `transfer.py` | 把 D 侧 page-in(CPU→GPU hot pool)翻译成可提交给 `CopyBlocksOp` 的批量页 copy 描述符 | 不实现跨实例传输,不改变现有 CPU KV offload 的 block-level copy 语义 |
| `coordinator.py` | 每层 decode 的编排入口:接收 adapter selection,查询 residency,触发 page-in,等待真实 miss,调用 block table patch,向 telemetry 记账 | 不包含 DeepSeek V4 kernel 细节,不直接调用 FlashMLA kernel |
| `telemetry.py` | 只读观测和 go/no-go 指标,包括 unique selected pages、miss(H)、复用率、搬运字节、wait 预算、hot-set drift | 不改变调度行为,不提交传输,不影响 logits |
| `adapters/deepseek_v4_c4a.py` | DeepSeek V4 c4a 适配层:识别 c4a 层,解释 top-k indices 和 compressed block table 列语义,提供页大小/页列/尾页规则,供 connector 分流与 coordinator 复用 | 不实现通用状态机,不处理 GLM 5.2 或其他模型 |

依赖方向保持单向:DeepSeek/MLA 执行路径调用 `coordinator.py`;coordinator 依赖
manager/paging/cpu_pool/block_table/transfer;adapter 只提供模型语义;
`SparsePageConnector` 从 P→D 填充 `cpu_pool`,并调用 manager 的 receive/seed 入口。
`transfer.py` 与 connector 复用 NIXL 的 `CopyBlocksOp`,但 `kv_transfer` 不反向
import page offload 内部状态类。

page offload 框架只处理 D 侧 selected-page offload 生命周期。模型差异放在
adapter 中:

- 哪些层可 offload。
- top-k indices 如何映射到 logical pages。
- 每页字节数、页列语义、GPU KV tensor 寻址方式。
- 是否需要 layer-local block table。
- 如何登记层名和当前请求行。

DeepSeek V4 首个 adapter 是 `DeepSeekV4C4AAdapter`;GLM 5.2 若使用相同
selected-page 语义,后续单独新增 `GLM52DSAAdapter`,不改通用
page offload manager/coordinator。首个 PR 不应提交空的预留 adapter 文件。

### 3.2.1 最小数据结构契约

实现前需要先固定以下轻量结构,避免 request row、request id、logical page 和
physical GPU slot 混用:

```text
LogicalPage:
  request_id: str | int        # engine request id,不能只用 batch row
  layer_name: str
  page_idx: int               # compressed logical page, tok // 64

SelectedPage:
  logical: LogicalPage
  req_row: int                # 当前 batch 中的 row
  token_rows: int32[]         # 命中该页的 top-k row,可选;观测用
  is_tail: bool

CPUPageRef:
  cpu_block_id: int
  ready: bool                 # connector 传输完成后置位
  ref_cnt: int

HotPageSlot:
  slot_id: int                # hot pool 内部槽
  phys_block_id: int          # 写入 block_table 的 GPU block id
  in_flight_job_id: int | None
  protected_step: int | None

SparsePageSelection:
  selected_pages: list[SelectedPage]
  unique_pages: list[LogicalPage]
  tail_pages: set[LogicalPage]
  miss_pages: list[LogicalPage]
```

`request_id` 必须来自请求生命周期中稳定且不会和 batch row 混淆的 id,且必须能在
P 与 D 之间对齐(connector 用它路由 CPU 权威页)。若 engine 回收 request id,
manager 必须在 request cleanup 后移除所有相关 CPU 和 GPU state,或额外引入
generation/epoch。

### 3.2.2 每层调用时序(D 侧 decode)

```text
indexer forward
  -> writes topk_indices_buffer
attention backend forward_mqa
  -> if disabled: 原路径
  -> adapter.extract_selection(req_id_per_token, topk_indices_buffer, seq_lens)
  -> coordinator.prepare_layer(layer_name, selection, source_block_table)
       - update telemetry
       - classify hit/miss/tail
       - submit CPU->GPU page-in for real miss
       - build patched block table or patched physical top-k buffer
       - wait only on real miss required by this layer
  -> backend physical-index conversion / sparse decode kernel
coordinator.finish_layer(layer_name)
```

只读观测 PR 只执行 `extract_selection` 和 telemetry,不得 submit copy、patch
block table 或改变传给 kernel 的 tensor。

### 3.2.3 PD 主干时序

同步原型必须覆盖 3.1 的完整主干:P connector 发送 + D allocate-partial + decode
同步 reload(单请求、`compress_ratio == 4` c4a 层)。D 每层的 miss 判定为:

```text
real_miss = selected_pages - (hot_pool/resident ∪ tail_pages)
real_miss 同步 CPU->GPU load 到 hot pool -> patch layer-local block_table
  -> wait load 完成 -> sparse MLA gather
```

主干输出必须能被测试直接观察:

- D 侧 c4a resident GPU block 数 = `hot_pool + tail + last_token_topk_carrier`,
  不等于完整 prompt 的 c4a 页数。
- P 侧 `request_finished` 后 KV 整体释放,没有生命周期中途的单页 free。
- CPU 权威池包含所有 sealed c4a 页且 ready。
- 下一个 decode step 的 logits 与 offload 关闭时一致。

若 D 侧 block allocator 不能安全地为单层 c4a 只分配部分页(allocate-partial),
同步原型应先补 allocator 接口或 fail closed;不得让 D 常规分配全量 c4a HBM 后再
依赖聚合式中途释放。

### 3.3 控制策略

每个 `(request, layer)` 维护:

- `residency: LogicalPage -> PhysSlot`
- `hot_score`
- `last_selected_step`
- `pinned_tail`
- `CPUPageRef`

准入和淘汰:

- top-k 选中且不在 hot pool 的页触发 promote(CPU→GPU page-in)。
- hot pool 满时,只能逐出非尾页、非本步选中、无在途 load 引用的页。
- 逐出候选按 `hot_score` 低者优先。
- 带宽预算超限时停止预取或延后 promote,但真实 miss 在 MLA 前必须 wait。

`hot_score` 可先用简单 EMA + recency:

```text
hot_score = EMA(select_count) + beta * gamma ** (step - last_selected_step)
```

这只是 hot pool 的局部策略,不应塞进现有 `CachePolicy`。

状态机(D 侧):

```text
mutable_tail                       # D 上随 decode 延伸写入的尾页
sealed(P) --connector xfer-->
cpu_ready_evicted                  # 到达 D 的 CPU 权威池,尚未进 hot pool
  -> loading_to_gpu
  -> cpu_ready_gpu_resident
  -> (evict) -> cpu_ready_evicted
```

关键规则:

- `mutable_tail` 永不 offload,只 pin 在 D 的 GPU cache。
- `cpu_ready_evicted` 是 CPU 权威副本;load 到 hot pool 前不能被 MLA gather。
- `loading_to_gpu` 不能被选为淘汰目标;load 完成后进入 `cpu_ready_gpu_resident`。
- `cpu_ready_gpu_resident` 的页可被淘汰回 `cpu_ready_evicted`,evict 纯丢弃 GPU
  slot,不写回(CPU 权威副本已存在)。
- request finished/aborted/preempted 时,必须取消或 drain in-flight job,释放 CPU
  page、hot slot 和 residency 映射。
- 任何状态异常都应 fail closed:保留 GPU resident 或禁用本请求 offload,不能使用
  陈旧 CPU 副本,也不能用 NIXL 整层 rehydrate 静默把 c4a 灌回 HBM。

### 3.4 SparsePageConnector 与传输层边界

P→D 的 c4a latent 传输走自研 `SparsePageConnector`,只复用 NIXL 的传输原语,
控制面自建。与附录 A 的决策一致:复用传输,不复用控制面。

#### 3.4.1 connector 接口核对(当前 commit)

在 `KVConnectorBase_V1` 与 `NixlConnector` 上核对三点:

| 要点 | 结论 | 证据 |
|---|---|---|
| 逐层分流 | base 提供逐层 hook `save_kv_layer(layer_name, kv_layer, ...)`;但 NIXL 实现是 no-op,按整请求 block 搬、不认层型,分流逻辑须自写 | `nixl/connector.py` `save_kv_layer`;`multi_connector.py` 广播 dispatch |
| DRAM 落地 | 支持。`nixl_memory_type="DRAM"`、逐层 `host_xfer_buffers[layer_name]` CPU tensor、`CopyBlocksOp` 的 `d2h`/`h2d` 原语齐备 | `nixl/base_worker.py` |
| P free-now + sideband | 支持。`request_finished -> (async_free, kv_transfer_params)`,P 天然释放并可携带 last-token Top-K 页列表 | `base.py` `request_finished` |

两个必须绕开的 NIXL 语义:

- **host buffer 是整层全量镜像,不是压缩池。** `host_xfer_buffers[layer_name]` 按
  设备 KV cache 完整 shape 分配。c4a 落地必须指向自建 `CPUAuthoritativePagePool`,
  不复用该镜像。
- **NIXL load 会自动 `h2d` 把整层灌回 HBM。** 对 c4a 恰好相反,c4a 必须留在 CPU、
  由 coordinator 逐步 page-in。stock load 路径对 c4a 不可用;`h2d` copy 原语复用,
  但触发权在 hot pool coordinator。

#### 3.4.2 SparsePageConnector 形态

单个薄 connector 拥有全部 V4 层,只复用 NIXL 传输原语,控制面自建:

```text
SparsePageConnector(自研,拥有全部 V4 层)
  复用 NIXL worker 原语: agent 握手 / DRAM+VRAM 注册 / CopyBlocksOp(d2h,h2d) / notif 轮询

  save (P 侧, 按 layer_name 分流):
    c4a      -> NIXL DRAM xfer 到 D 的紧凑 CPU 权威池
    其余层    -> 常规 HBM block xfer(委托 NIXL 现有 block 路径)
  request_finished -> free-now + kv_transfer_params 携带 last-token Top-K 页

  load (D 侧):
    c4a      -> 落 CPU 权威池, 标 cpu_ready_evicted, 交 manager; 不自动 h2d
    其余层    -> 正常灌回 HBM
  decode hot pool page-in 的 h2d 由 coordinator 逐步驱动, 复用 CopyBlocksOp
```

架构取舍:

- **不用 `MultiConnector` 拆 c4a/rest。** 它把每个调用广播给所有子 connector、不按
  层 filter,NIXL 又无"排除层"开关,会双重搬运。
- **不 fork NIXL 整块 worker。** 它是 request-block 粒度、load 时自动 rehydrate,
  硬塞逐层 + 留 CPU 是对抗其模型。只取传输原语,薄壳持有 `NixlWrapper` 直接调底层。
- **不接 LMCache / Mooncake 控制面。** 其 key 语义(前缀 / 内容哈希)与
  `(request, layer, logical_page)` 冲突。Mooncake transfer engine 可作为 Phase 2
  跨节点传输替换 NIXL,控制面不变。

#### 3.4.3 传输层复用边界

复用:

- NIXL 的 agent 握手、DRAM/VRAM 注册、`CopyBlocksOp`(`d2h`/`h2d`)、notif 轮询。
- `request_finished -> (async_free, kv_transfer_params)` 的 free-now 与 sideband。
- `BlockStatus` 的 ready/ref_cnt 语义可借鉴到 `CPUPageRef`。

需要新增或严格验证:

- c4a 单层 tensor + 单页 + 批量多页的 DRAM xfer 描述符,不能误扩展到
  SWA/C128A 或其他 layer。
- D 侧 page-in 的 layer-local sparse page copy spec(CPU→GPU hot pool),复用
  `CopyBlocksOp` 但只覆盖选中页:

```text
SparsePageCopySpec:
  tensor_idx: int
  page_size_bytes: int
  layer_name: str
  src_page_ids: int64[num_pages]   # CPU blocks
  dst_page_ids: int64[num_pages]   # GPU hot pool phys blocks
  direction: cpu_to_gpu
```

不复用:

- NIXL 的整层 host mirror 与自动 rehydrate 语义,不作为 c4a 的 CPU 权威池或
  page-in 路径。
- `CPUOffloadingManager` 不作为 selected-page offload 控制面。
- `cpu/policies/CachePolicy` 不作为 GPU hot pool 淘汰策略。

原因详见附录 A。

---

## 4. 容量与性能模型

### 4.1 D 侧 HBM 常驻底线

D 的常驻底线必须逐层求和。以下模型按 `fp8_ds_mla` 的 584B c4a/C128A 条目估算;
plain bf16/per-tensor fp8 cache dtype 需要替换页大小:

```text
HBM_floor(D) =
  Σ_requests [
    Σ_C4A    ceil(S / 4)   * 132B       # indexer key
  + Σ_C4A    1 page        * 37,376B    # 尾页
  + Σ_C4A    H_L           * 37,376B    # GPU hot pool,per (request, layer)
  + Σ_C128A  ceil(S / 128) * 584B       # C128A latent
  + Σ_SWA    window        * M_swa      # SWA KV
  ]
```

hot pool 是 per (request, layer):c4a page 是请求私有 KV,跨请求无页复用,复用只来自
同请求同层跨 decode step。因此 hot pool 随并发线性增长,收益来自把 c4a 全量池
`ceil(S/256)` 页替换成 `H_L` 个 hot pool 页。PD 分离下 D 从一开始就按 allocate-partial
分配,没有 prefill 全量峰值;若 `H_L` 接近全量页数,显存收益趋零。

### 4.2 `H=512` 的上下文收益

设 21 个 c4a 层:

| 上下文 | 每层页数 | `H=512` 常驻比例 | 21 层理论节省(/req/GPU) |
|---:|---:|---:|---:|
| 128K | 512 | 100% | 约 0 |
| 256K | 1024 | 50% | 约 383 MiB |
| 512K | 2048 | 25% | 约 1.12 GiB |
| 1M | 4096 | 12.5% | 约 2.62 GiB |

所以 128K + `H=512` 基本没有容量收益。主战场是 256K-1M 长上下文或高并发,这正是
D 实例的画像。

### 4.3 带宽墙

页级 miss 才是核心指标:

```text
B_miss,step = Σ_{request,layer} U_miss(request, layer) * B_page
T_added     ≈ max(0, B_miss / BW_eff + submit/event - overlap)
```

必须按**去重后的页数**统计,不能按 top-k token 数统计。单页约 37 KiB,逐页 copy
的提交/event 开销会很重,所以必须批量:

```text
GPU unique pages -> residency bitmap -> miss list/copy descriptors
  -> 一次或少数几次 CPU->GPU copy
  -> GPU scatter / patch block_table
```

平台判断:

- DGX/HGX B300 仍走 PCIe/NUMA,只有 exact miss 极低时才可能生产化。
- GB300 NVL72 的 Grace↔B300 C2C 带宽更适合作为首发平台。
- D 侧 CPU 权威池必须放本地 NUMA/Grace 内存,避免跨 socket 或跨 Superchip 回载。
- P→D 传输带宽是额外一环:P 出 KV 后一次性把 sealed c4a 送到 D 的 CPU 池,这段
  走 connector/NIXL,不计入 decode step 的分页附加,但要计入首 token→首 decode 的
  端到端时延预算。

---

## 5. 实现集成点

| 位置 | 改动 |
|---|---|
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` / `flashinfer_mla_sparse.py` | D 侧 decode 接入点。c4a 且 selected-page offload 已开启(consumer + `SparsePageConnector`)时,在 logical top-k 转 physical top-k 前调用 coordinator,并改用 layer-local patched block table 或 patched physical top-k buffer |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | 仅在目标 commit 确认 DeepSeek V4 使用 FlashMLA sparse decode 时接入;否则只作为 `fp8_ds_mla` layout 和 page size 核对来源 |
| DeepSeek V4 c4a compressor / KV 写入路径(P 侧) | 页封存时标 sealed;prefill sparse MLA 完成后由 connector 按层分流发送 sealed c4a 到 D 的 CPU 权威池;从 last-token `topk_indices` 提取 seed 页;不做生命周期中途 GPU 释放,由 `request_finished` free-now 整请求释放 |
| `vllm/distributed/kv_transfer/.../SparsePageConnector`(新增) | PD 部署下自研薄 connector,拥有全部 V4 层;复用 NIXL 传输原语;`save_kv_layer` 按 layer 分流(c4a→D CPU 权威池,其余→D HBM);`request_finished` 返回 free-now 并携带 last-token Top-K seed;load 侧对 c4a 不自动 h2d |
| D 侧 KV block manager / allocate-partial | c4a 层只在 HBM 分配 hot pool + tail,不排满全量 c4a;evicted 逻辑页不会被共享 `block_table` 读到;若不支持,fail closed |
| `vllm/v1/attention/backends/mla/page_offload/` | D 侧 selected-page coordinator、adapter、驻留/热页状态、观测和 page-in 辅助 |
| NIXL 传输原语 / `CopyBlocksOp` | 仅复用 P→D DRAM xfer 与 D 侧 page-in 的 host↔device copy;不复用整层 host mirror 与自动 rehydrate |
| `triton_convert_req_index_to_global_index` / 等价函数 | 不改 kernel 语义;只保证传入的 block table 已 patch,或由 coordinator 直接产出等价 physical top-k |
| `cpu/policies/` | 不改 |

`vllm/v1/kv_offload/` 不承载 selected-page 控制面。P→D 传输由 `SparsePageConnector`
复用 NIXL 完成,D 侧 selected-page 逻辑由 MLA page offload 模块拥有,避免把
attention 访问模式和 FlashMLA 调度语义混入通用 KV offload/connector。

CUDA graph 约束:

- layer-local block table buffer 必须固定地址、固定形状。
- `patch_block_table` 原地写,不换 tensor。
- 初期可让 c4a 层 eager 保正确性;性能版应把 unique page、residency bitmap、miss
  list、copy descriptor、hot_score、block table patch 尽量留在 GPU 侧,减少 host 尾巴。

公开配置建议:

selected-page offload 的开启不再是单独开关,而是由 KV 传输配置推导:D 实例选用
`SparsePageConnector` 且 `kv_role` 为 `kv_consumer` 时自动开启;其余情况(P 侧、
非该 connector、无 PD)一律关闭。所有旋钮都放在该 connector 的
`kv_connector_extra_config` 里,不再经由 `--hf-overrides`。

| 旋钮(`kv_connector_extra_config`) | 状态 | 说明 |
|---|---|---|
| `sparse_page_cpu_pool_size_gib` | 提案 | D 侧 CPU 权威池预算;独立于现有 KVTransfer offload |
| `sparse_page_transfer_backend` | 提案 | `nixl`(首选)或后续 `mooncake`(Phase 2 跨节点);只复用传输原语,不启用其控制面 |
| `sparse_page_hot_pool_blocks` | 提案 | 每可卸载层 hot pool 页数 `H`(单个整数) |
| `sparse_page_prefetch_lookahead` | 提案 | 跨层/跨步预取深度 |
| `sparse_page_offload_layers` | 提案 | `auto` 或显式层范围 |

配置校验建议:

- 未选用 `SparsePageConnector`(或非 consumer 角色)时不初始化 manager、不分配
  CPU pool、不改变 metadata。
- 选用了 `SparsePageConnector` 作为 consumer 但模型/backend 不满足支持范围时
  fail closed,报清晰错误,不能静默跑半套 offload。
- `sparse_page_hot_pool_blocks` 以每个可卸载 layer 为单位;residency 按 (request, layer)
  维护,hot pool 随并发线性增长,telemetry 需按 (request, layer) 记账。
- `sparse_page_cpu_pool_size_gib` 必须能推导出最大 CPU page 数,并在启动时打印
  `page_size_bytes`、`num_cpu_pages`、`num_c4a_layers`。

不要把 `--kv-offloading-size` 当作 selected-page CPU pool 预算。当前 vLLM 会用它
自动配置 `KVTransferConfig` 和 `OffloadingConnector`,这会启用现有 CPU KV offload
路径,违反本设计"现有 KV connector 与 CPU KV offload 路径不受影响"的约束。

示例为提案占位;PD 分离需分别启动 P 与 D 实例并配置 `SparsePageConnector`
(以下 `--kv-transfer-config` 为占位,落地时接入正式配置):

```bash
# Prefill 实例(发送端)
CUDA_VISIBLE_DEVICES=0 vllm serve /data/public_models/Deepseek-V4-Flash \
    --kv-cache-dtype fp8 --trust-remote-code --max-model-len 1000000 \
    --enforce-eager --port 8001 \
    --kv-transfer-config '{"kv_connector": "SparsePageConnector", "kv_role": "kv_producer"}'

# Decode 实例(接收端 + selected-page offload)
# consumer 角色 + SparsePageConnector 即自动开启 offload;旋钮全部走
# kv_connector_extra_config,不再使用 --hf-overrides。
CUDA_VISIBLE_DEVICES=1 vllm serve /data/public_models/Deepseek-V4-Flash \
    --kv-cache-dtype fp8 --trust-remote-code --max-model-len 1000000 \
    --enforce-eager --port 8002 \
    --kv-transfer-config '{"kv_connector": "SparsePageConnector", "kv_role": "kv_consumer", "kv_connector_extra_config": {"sparse_page_cpu_pool_size_gib": 128, "sparse_page_transfer_backend": "nixl", "sparse_page_hot_pool_blocks": 512, "sparse_page_prefetch_lookahead": 2, "sparse_page_offload_layers": "auto"}}'
```

---

### 6 测试矩阵

单元测试优先,只在行为型 offload PR 中加入端到端 logits 测试:

| 范围 | 测试点 |
|---|---|
| adapter | `tok // 64` 映射到 logical page;top-k 去重;尾页识别;无效 `-1` token 忽略 |
| block table | layer-local buffer 固定地址;只 patch c4a 层;共享 `block_table` 不变;shape/dtype/device 校验 |
| connector 分流 | `save_kv_layer` 只把 c4a 发到 CPU 池,indexer/C128A/SWA 走 HBM;load 侧对 c4a 不自动 h2d;`request_finished` free-now 且携带 seed |
| state machine | connector receive/seed/page-in/evict/cleanup;in-flight ref 保护;request abort;request id 复用与 P/D 对齐 |
| D allocate-partial | c4a 层只分配 hot pool + tail;evicted 逻辑页不被共享 `block_table` 引用;不全量分配 c4a |
| transfer | batch 多页 copy descriptor;page size 584B * 64;不触发 SWA/C128A copy |
| fallback | CPU pool OOM、hot pool 满、无可驱逐页、connector 传输 fail、allocate-partial 不支持、unsupported backend/非 PD |
| correctness | offload on/off 对同一 prompt 的 logits 无损或在量化容差内一致 |
| memory accounting | D 侧 c4a GPU resident block 数和 `torch.cuda.memory_allocated`/allocator 统计低于全量;不能只增加 CPU mirror 或 hot pool |
| CUDA graph | patched buffer 地址稳定;observe-only 不引入 data-dependent allocation |

推荐命令按实际改动收敛到具体文件,例如:

```bash
.venv/bin/python -m pytest tests/v1/attention/test_sparse_page_offload.py -v
.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_gpu_worker.py -v
pre-commit run ruff-check --files <changed files>
```

若改动影响模型输出、索引语义或 serving 行为,PR 描述必须附 logits 对比或
model eval 结果;只读观测 PR 可明确说明没有改变 logits 路径。

---

## 附录 A:为什么控制面自建

selected-page offload 是 D worker 侧、逐层、由真实 top-k 驱动的 GPU hot pool 分页。
它复用现有传输原语,但控制面必须自建。

**为什么不用 `CPUOffloadingManager` 做控制面:**

`CPUOffloadingManager` 面向 scheduler 侧的前缀缓存/CPU KV 缓存管理。

**为什么不用 NIXL / LMCache / Mooncake 的 connector 做控制面:**

1. **NIXL:** `save_kv_layer` 是 no-op、按整请求 block 搬,load 时自动整层 rehydrate 回
   HBM,host buffer 是全量镜像;这些都和"c4a 留 CPU、逐页 page-in"相反。只取其传输原语。
2. **LMCache / Mooncake:** key 语义是前缀/内容哈希,与 `(request, layer, logical_page)`
   冲突;其 CPU 后端是整块 KV 复用,不是逐层逐步 top-k 分页进 GPU hot pool。
3. **Mooncake transfer engine** 可作为跨节点传输后端,但仍只是传输层。

因此只复用 NIXL 传输原语、`CopyBlocksOp` 和 `request_finished` 的 free-now/sideband
语义;控制面由 `SparsePageOffloadCoordinator` / `SparsePageOffloadManager` /
`SparseHotPagePool` 与 `SparsePageConnector` 自建。
