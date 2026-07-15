# DeepSeek V4 c4a CPU offload 与 selected-page GPU reload 设计

> **范围:** 本文只验证 DeepSeek V4 c4a 压缩 latent 在 `fp8_ds_mla`
> layout 下的本机 CPU DRAM offload。接口按通用 sparse selected-page
> offload 设计,后续可通过 adapter 扩展到 GLM 5.2 DSA 等模型。
>
> **场景:** 聚合推理(prefill/decode 同实例)。不讨论跨节点/远端内存层。
>
> **实现边界:** Phase 1 只面向 DeepSeek V4 sparse MLA decode 路径。
> 目标 commit 落地前必须确认当前模型实际选择的 backend。若使用
> `flashinfer_mla_sparse_sm120.py` / `flashinfer_mla_sparse.py`,offload hook
> 应放在 `topk_indices_buffer` 生成之后、`triton_convert_req_index_to_global_index`
> 或等价转换之前。`flashmla_sparse.py` 只作为 FlashMLA layout/metadata 参考,
> 除非目标 commit 的 DeepSeek V4 确认走该路径。

---

## 1. 目标与结论

DeepSeek V4 的 CSA 层按 `compress_ratio` 分成三类:

| 层型 | 语义 | offload 结论 |
|---|---|---|
| `swaonly` | SWA 窗口 KV,每步必读 | 常驻 HBM |
| `c4a` | 压缩池约 `S/4`,由 indexer top-k 选择 | **唯一 offload 候选** |
| `c128a` / HCA | 压缩池约 `S/128`,当前语义接近全量枚举 | 常驻 HBM |

目标是在 decode 阶段把 c4a 中冷的、已封存的压缩页从 HBM 下沉到本机 CPU DRAM,在同卡上容纳更长上下文或更高并发。正确性要求是**逐 token logits 无损**;性能收益必须由页级观测数据证明。

核心结论:

1. c4a latent 封存后只读,CPU 副本可作为权威副本;GPU evict 是纯丢弃,不写回。
2. indexer key、SWA、C128A 必须常驻 HBM,不进入 offload 候选。
3. MLA kernel 只能从 HBM gather,所以 CPU miss 页必须先 load 到 GPU hot pool,再 patch 本层压缩 `block_table`。
4. 不能直接复用 `CPUOffloadingManager` 作为控制面;只能复用 `CPUOffloadingWorker` 的传输原语。
5. 第一批社区 PR 应先提交只读观测能力,再提交 offload 行为。

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

早期草案中两个容易出错的点:

- indexer key 容量应按 `S/4` 压缩候选算,不是 `S` 原始 token。
- hot pool 页数应和 `ceil(S/256)` 比较,不是 `ceil(S/4)`。

落地前还必须核对当前 backend 的真实 KV cache shape。当前代码中
`flashmla_sparse.py` 的注释写 DeepSeek V4 `fp8_ds_mla` 条目是 584B,但
`FlashMLASparseBackend.get_kv_cache_shape()` 在 `fp8_ds_mla` 下仍返回 656B。
这可能是 V3.2 旧路径、另一个 V4 backend 覆盖,或目标 commit 的待修正点。
实现 selected-page copy 前必须把最终使用的 c4a tensor shape、page size 和
alignment 写成断言,不能只依赖文档常量。

### 2.2 必须保持的正确性不变式

- **top-k 始终在全量逻辑池上计算。** indexer key 全驻留,offload 只影响被选中 latent 的物理驻留位置。
- **MLA 前必须保证选中页在 HBM。** 对真实 top-k 命中的页直接 gather;对 miss 页先 CPU→GPU load,再等待完成。
- **`block_table` 必须按层 patch。** V4 metadata 的原始 `attn_metadata.block_table` 没有 layer 维度,而 hot pool 是 per c4a layer 的,不能原地修改共享表。
- **尾页常驻。** 当前正在写入的尾部压缩页可变,不能使用 CPU 陈旧副本。
- **关闭开关时零行为变化。** dense attention、SWA、C128A、现有 CPU KV offload 路径不受影响。

### 2.3 `block_table` 映射

V4 decode 路径中,`compute_global_topk_indices_and_lens` 将 indexer 选出的压缩条目 `tok` 转成物理槽位:

```text
slot = block_table[req, tok // 64] * 64 + tok % 64
```

因此 c4a 的一个 offload page 对应本层压缩 `block_table` 的一列。`patch_block_table` 是 `LogicalPage -> PhysSlot` 的 1:1 标量或 scatter 写,没有 4 倍展开。

实现要求:

- 为每个 c4a 层准备固定地址的 layer-local compressed block table buffer。
- 每步从原始压缩表拷贝或初始化,只 patch 本层选中列。
- 不能 patch SWA 的 64-token `swa_metadata.block_table`。
- 落地前重新核对目标 commit 的 `attn_metadata.block_size`、`compress_ratio`、`storage_block_size` 和列语义。

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

- 非 DeepSeek V4、非 `fp8_ds_mla`、非 `compress_ratio == 4` c4a 层。
- DCP / decode context parallel。当前 indexer 对 `compress_ratio > 1` + DCP
  仍是 `NotImplementedError`,offload 不应绕过该限制。
- C128A、SWA、dense MLA、ROCm/XPU backend。
- MTP/native spec decode、多 token decode、DSpark non-causal decode,除非另有
  logits 无损验证。
- CPU pool 不足、hot pool 无可驱逐页、DMA submit 失败、backend shape 不匹配。

禁用时必须保持原路径行为,最多输出一次 warning 或 telemetry 标记。

---

## 3. 架构

### 3.1 数据流

```text
prefill:
  c4a page 写满 -> seal -> batch store GPU page 到 CPU 权威池
    -> 等当前 step 消费完成后才可释放 GPU 槽

decode 每层:
  indexer top-k
    -> adapter 将 top-k 压缩条目去重成 logical pages
    -> 查 residency: hit 直接用;miss 批量 load 到 hot pool
    -> patch 本层 layer-local block_table
    -> wait 真实 miss
    -> sparse MLA gather
```

CPU DRAM 是只读权威池;GPU hot pool 是每层缓存。evict 不写回,只删除
residency 映射并释放 GPU slot。页写满后不能立即释放 GPU 槽,必须等该页
在当前 prefill/decode step 中所有消费者都完成;Phase 1 可先只在 decode 后释放,
避免破坏 prefill gather。

### 3.2 MLA page offload 模块

selected-page offload 的基础设施放在 MLA attention backend 附近:
`vllm/v1/attention/backends/mla/page_offload/`。

该目录承载 offload 生命周期、驻留状态、hot pool、传输描述、观测指标和模型
adapter。现有 `vllm/v1/kv_offload/cpu/` 只作为 CPU DRAM 传输原语的复用
来源,不拥有 selected-page 控制面。

```text
vllm/v1/attention/backends/mla/page_offload/
├── __init__.py
├── config.py            SparsePageOffloadConfig
├── selected_pages.py    SparsePageAdapter / SparsePageSelection
├── paging.py            SparsePageState / SparseHotPagePool
├── cpu_pool.py          CPUAuthoritativePagePool
├── manager.py           SparsePageOffloadManager
├── block_table.py       LayerLocalBlockTables
├── transfer.py          OffloadingTransfer / WorkerOffloadingTransfer
├── coordinator.py       SparsePageOffloadCoordinator
├── telemetry.py         SparseSelectionCollector
└── adapters/
    └── deepseek_v4_c4a.py
```

文件职责建议:

| 文件 | 职责 | 不负责 |
|---|---|---|
| `__init__.py` | 暴露最小公共入口,例如 coordinator/config/adapter 类型;避免让模型侧 import 内部状态类 | 不做注册副作用,不初始化全局状态 |
| `config.py` | 解析和校验 page offload 配置,包括开关、每层 hot pool 页数、lookahead、层选择、带宽预算、观测模式 | 不读取 HF model config 的模型语义,不决定某个 layer 是否可 offload |
| `selected_pages.py` | 定义 adapter 抽象和一次 top-k 选择的结构化结果,例如 `(request, layer, logical_page)` 去重集合、真实 miss 集合、tail pin 标记 | 不维护长期 residency,不分配 GPU/CPU 槽 |
| `paging.py` | 维护 GPU hot pool 和页驻留状态,包括 `LogicalPage -> PhysSlot`、in-flight 引用、tail pin、本步 protected 页、hot_score 和淘汰候选 | 不提交 DMA,不 patch block table,不关心具体模型 top-k 张量布局 |
| `cpu_pool.py` | 管理 CPU DRAM 权威副本的页槽和生命周期,按请求结束释放 CPU page,记录 ready/ref_cnt 状态 | 不做 GPU hot pool 淘汰,不复用 `CPUOffloadingManager` 的 `OffloadKey` 语义 |
| `manager.py` | 聚合 `paging.py` 与 `cpu_pool.py` 的状态机,提供 seal/store、promote/load、evict、complete、request cleanup 等 worker-side 控制面 | 不实现 scheduler-side `OffloadingManager`,不接管现有 KV connector |
| `block_table.py` | 管理固定地址、固定形状的 layer-local compressed block table buffer,并提供原地 patch 接口 | 不修改共享 `attn_metadata.block_table`,不 patch SWA block table |
| `transfer.py` | 将 sparse page copy 请求翻译成可提交给 CPU worker/底层 copy handler 的描述符,必要时定义 layer-local sparse page copy spec | 不实现新的 offload backend,不改变现有 CPU KV offload 的 block-level copy 语义 |
| `coordinator.py` | 每层 decode 的编排入口:接收 adapter selection,查询 residency,触发 load/store,等待真实 miss,调用 block table patch,向 telemetry 记账 | 不包含 DeepSeek V4 kernel 细节,不直接调用 FlashMLA kernel |
| `telemetry.py` | 只读观测和 go/no-go 指标,包括 unique selected pages、miss(H)、复用率、搬运字节、wait 预算、hot-set drift | 不改变调度行为,不提交 DMA,不影响 logits |
| `adapters/deepseek_v4_c4a.py` | DeepSeek V4 c4a 适配层:识别 c4a 层,解释 top-k indices 和 compressed block table 列语义,提供页大小/页列/尾页规则 | 不实现通用状态机,不处理 GLM 5.2 或其他模型 |

这组文件的依赖方向应保持单向:DeepSeek/MLA 执行路径调用
`coordinator.py`;coordinator 依赖 manager/paging/cpu_pool/block_table/transfer;
adapter 只提供模型语义。`transfer.py` 可以复用 `vllm/v1/kv_offload/cpu/`
中的 worker 原语,但 `kv_offload` 不反向 import page offload 模块。

page offload 框架只处理 selected-page offload 生命周期。模型差异放在
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
  ready: bool
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

`request_id` 必须来自请求生命周期中稳定且不会和 batch row 混淆的 id。若
engine 回收 request id,manager 必须在 request cleanup 后移除所有相关 CPU 和
GPU state,或额外引入 generation/epoch。

### 3.2.2 每层调用时序

```text
indexer forward
  -> writes topk_indices_buffer
attention backend forward_mqa
  -> if disabled: 原路径
  -> adapter.extract_selection(req_id_per_token, topk_indices_buffer, seq_lens)
  -> coordinator.prepare_layer(layer_name, selection, source_block_table)
       - update telemetry
       - classify hit/miss/tail
       - submit CPU->GPU load for real miss
       - build patched block table or patched physical top-k buffer
       - wait only on real miss required by this layer
  -> backend physical-index conversion / sparse decode kernel
coordinator.finish_layer(layer_name)
```

只读观测 PR 只执行 `extract_selection` 和 telemetry,不得 submit DMA、patch
block table 或改变传给 kernel 的 tensor。

### 3.3 控制策略

每个 `(request, layer)` 维护:

- `residency: LogicalPage -> PhysSlot`
- `hot_score`
- `last_selected_step`
- `pinned_tail`
- `CPUPageRef`

准入和淘汰:

- top-k 选中且不在 hot pool 的页触发 promote。
- hot pool 满时,只能逐出非尾页、非本步选中、无在途 load/store 引用的页。
- 逐出候选按 `hot_score` 低者优先。
- 带宽预算超限时停止预取或延后 promote,但真实 miss 在 MLA 前必须 wait。

`hot_score` 可先用简单 EMA + recency:

```text
hot_score = EMA(select_count) + beta * gamma ** (step - last_selected_step)
```

这只是 hot pool 的局部策略,不应塞进现有 `CachePolicy`。

状态机:

```text
mutable_tail
  -> sealed_gpu_only
  -> storing_to_cpu
  -> cpu_ready_gpu_resident
  -> cpu_ready_evicted
  -> loading_to_gpu
  -> cpu_ready_gpu_resident
```

关键规则:

- `mutable_tail` 永不 offload,只 pin 在原 GPU cache。
- `sealed_gpu_only` 可以提交 GPU->CPU store,但 GPU 槽在本 step 消费完成前
  不能释放。
- `storing_to_cpu` 不能 evict;store 完成后才进入 `cpu_ready_*`。
- `cpu_ready_evicted` 是 CPU 权威副本;load 到 hot pool 前不能被 MLA gather。
- request finished/aborted/preempted 时,必须取消或 drain in-flight job,释放 CPU
  page、hot slot 和 residency 映射。
- 任何状态异常都应 fail closed:保留 GPU resident 或禁用本请求 offload,不能
  使用陈旧 CPU 副本。

### 3.4 传输层复用边界

复用:

- `CPUOffloadingWorker` 的 copy stream、event、完成通知。
- `GPULoadStoreSpec` / `CPULoadStoreSpec` 可复用的部分。
- `BlockStatus` 的 ready/ref_cnt 语义可借鉴。

需要扩展或严格验证:

- 现有 worker 是否支持**单层 tensor + 单页 + 批量多页**寻址。
- 当前 `GPULoadStoreSpec` / `CPULoadStoreSpec` 是 KV block/group 语义,大概率
  需要新增 layer-local sparse page copy spec,不要改坏现有 CPU KV offload 路径。

建议新增的 copy spec 不复用 `OffloadKey` 语义,只表达一次批量页 copy:

```text
SparsePageCopySpec:
  tensor_idx: int
  page_size_bytes: int
  layer_name: str
  src_page_ids: int64[num_pages]   # GPU phys blocks or CPU blocks
  dst_page_ids: int64[num_pages]
  direction: gpu_to_cpu | cpu_to_gpu
```

如果继续复用 `CPUOffloadingWorker`,需要先证明它能接受
`CanonicalKVCaches` 中的单层 c4a tensor view,并且 `group_data_refs` 不会把
同一个 copy 误扩展到 SWA/C128A 或其他 layer。若做不到,应抽出底层
batched pointer copy handler,由 selected-page transfer 自己构造
`src_ptr/dst_ptr/size`。

不复用:

- `CPUOffloadingManager` 不作为 selected-page offload 控制面。
- `cpu/policies/CachePolicy` 不作为 GPU hot pool 淘汰策略。

原因详见附录 A。

---

## 4. 容量与性能模型

### 4.1 HBM 常驻底线

V4 的常驻底线必须逐层求和。以下模型按 `fp8_ds_mla` 的 584B c4a/C128A
条目估算;plain bf16/per-tensor fp8 cache dtype 需要替换页大小:

```text
HBM_floor =
  Σ_requests [
    Σ_C4A    ceil(S / 4)   * 132B       # indexer key
  + Σ_C4A    1 page        * 37,376B    # 尾页
  + Σ_C128A  ceil(S / 128) * 584B       # C128A latent
  + Σ_SWA    window        * M_swa      # SWA KV
  ]
+ Σ_C4A H_L * 37,376B                  # GPU hot pool,跨请求共享
```

收益来自把 c4a 全量池 `ceil(S/256)` 页替换成 `H_L` 个 hot pool 页。若 `H_L` 接近全量页数,显存收益趋零。

### 4.2 `H=512` 的上下文收益

设 21 个 c4a 层:

| 上下文 | 每层页数 | `H=512` 常驻比例 | 21 层理论节省(/req/GPU) |
|---:|---:|---:|---:|
| 128K | 512 | 100% | 约 0 |
| 256K | 1024 | 50% | 约 383 MiB |
| 512K | 2048 | 25% | 约 1.12 GiB |
| 1M | 4096 | 12.5% | 约 2.62 GiB |

所以 128K + `H=512` 基本没有容量收益。主战场是 256K-1M 长上下文或高并发。

### 4.3 带宽墙

页级 miss 才是核心指标:

```text
B_miss,step = Σ_{request,layer} U_miss(request, layer) * B_page
T_added     ≈ max(0, B_miss / BW_eff + submit/event - overlap)
```

必须按**去重后的页数**统计,不能按 top-k token 数统计。单页约 37 KiB,逐页 DMA 的提交/event 开销会很重,所以必须批量:

```text
GPU unique pages -> residency bitmap -> miss list/copy descriptors
  -> 一次或少数几次 CPU->GPU copy
  -> GPU scatter / patch block_table
```

平台判断:

- DGX/HGX B300 仍走 PCIe/NUMA,只有 exact miss 极低时才可能生产化。
- GB300 NVL72 的 Grace↔B300 C2C 带宽更适合作为首发平台。
- CPU 权威池必须放本地 NUMA/Grace 内存,避免跨 socket 或跨 Superchip 回载。

建议初始 go/no-go:

| 平台 | 平均 exact miss | P99 分页附加 TPOT | 并发/上下文收益 |
|---|---|---|---|
| DGX/HGX B300 | ≤ 4-8 页 / c4a 层 / step | < 基线 5% | ≥ 1.5x |
| GB300 NVL72 | 可适度放宽 | < 基线 5% | ≥ 1.5x |

---

## 5. 实现集成点

| 位置 | 改动 |
|---|---|
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` / `flashinfer_mla_sparse.py` | NVIDIA V4 Phase 1 候选接入点。c4a 且 `sparse_page_offload` 时,在 logical top-k 转 physical top-k 前调用 coordinator,并改用 layer-local patched block table 或 patched physical top-k buffer |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | 仅在目标 commit 确认 DeepSeek V4 使用 FlashMLA sparse decode 时接入;否则只作为 `fp8_ds_mla` layout 和 page size 核对来源 |
| DeepSeek V4 c4a compressor / KV 写入路径 | 页封存时通知 coordinator seal/store;尾页保持 pinned;释放 GPU 槽前必须等当前 step 消费完成 |
| `vllm/v1/attention/backends/mla/page_offload/` | selected-page coordinator、adapter、驻留/热页状态、观测和传输辅助 |
| `CPUOffloadingWorker` / load-store spec | 仅复用 CPU DRAM 传输框架;预计新增 layer-local sparse page batch copy spec |
| `triton_convert_req_index_to_global_index` / 等价函数 | 不改 kernel 语义;只保证传入的 block table 已 patch,或由 coordinator 直接产出等价 physical top-k |
| `cpu/policies/` | 不改 |

`vllm/v1/kv_offload/` 不承载 selected-page 控制面;它只提供可复用的
CPU offload worker、load/store spec 和引用保护语义。selected-page 逻辑由
MLA page offload 模块拥有,避免把 `kv_offload` 的 backend/medium 目录混入
attention 访问模式和 FlashMLA 调度语义。

CUDA graph 约束:

- layer-local block table buffer 必须固定地址、固定形状。
- `patch_block_table` 原地写,不换 tensor。
- 初期可让 c4a 层 eager 保正确性;性能版应把 unique page、residency bitmap、miss list、copy descriptor、hot_score、block table patch 尽量留在 GPU 侧,减少 host 尾巴。

公开配置建议:

| 开关 | 状态 | 说明 |
|---|---|---|
| `sparse_page_offload` | 提案 | 打开 selected-page offload;不能由 `--kv-offloading-size` 隐式打开 |
| `sparse_page_cpu_pool_size_gib` | 提案 | CPU 权威池预算;独立于现有 KVTransfer offload |
| `sparse_page_transfer_backend` | 提案 | 初期可选 `native_cpu`;只复用传输原语,不启用现有 KV offload 控制面 |
| `sparse_page_hot_pool_blocks` | 提案 | 每可卸载层 hot pool 页数 `H` |
| `sparse_page_prefetch_lookahead` | 提案 | 跨层/跨步预取深度 |
| `sparse_page_offload_layers` | 提案 | `auto` 或显式层范围 |

配置校验建议:

- `sparse_page_offload=false` 时不初始化 manager、不分配 CPU pool、不改变 metadata。
- `sparse_page_offload=true` 但模型/backend 不满足支持范围时 fail closed,报清晰错误
  或自动降级为 `observe_only`,不能静默跑半套 offload。
- `sparse_page_hot_pool_blocks` 以每个可卸载 layer 为单位;若支持跨请求共享,文档和
  telemetry 必须把 per-layer 与 global pool 区分清楚。
- `sparse_page_cpu_pool_size_gib` 必须能推导出最大 CPU page 数,并在启动时打印
  `page_size_bytes`、`num_cpu_pages`、`num_c4a_layers`。
- `sparse_page_observe_only` 应作为 Phase 1 独立开关,用于只读采集。

不要把 `--kv-offloading-size` 当作 selected-page CPU pool 预算。当前 vLLM
会用它自动配置 `KVTransferConfig` 和 `OffloadingConnector`,这会启用现有 CPU
KV offload 路径,违反本设计“现有 CPU KV offload 路径不受影响”的约束。

示例为提案占位;落地时应先接入正式 vLLM 配置,不要依赖现有
`--kv-offloading-size`:

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve /data/public_models/Deepseek-V4-Flash \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --max-model-len 262144 \
    --port 8003 \
    --hf-overrides '{"sparse_page_offload": true, "sparse_page_cpu_pool_size_gib": 128, "sparse_page_transfer_backend": "native_cpu", "sparse_page_hot_pool_blocks": 512, "sparse_page_prefetch_lookahead": 2, "sparse_page_offload_layers": "auto"}'
```

---

## 6. 分阶段落地与 PR 拆分

### 6.1 Phase 1:单请求验证

1. **只读观测。** `SparseSelectionCollector` 采集:
   - 每层每步 unique selected pages / missed pages。
   - `miss(H)` 曲线。
   - 页在未来 1/2/4/8 step 的复用率。
   - 每层搬运字节估算、exact wait 预算、hot-set drift。
   - 观测实现必须从 logical `topk_indices_buffer` 推导页号,模拟不同 `H`
     的 `miss(H)` 曲线,不得实际搬运或 patch。
   - 指标至少包含 `layer_name`、`step`、`num_unique_pages`、
     `num_tail_pages`、`simulated_miss_pages`、`estimated_bytes`。
2. **同步 offload 原型。** 单请求、同步 load+wait,先证明 logits 无损。
3. **异步流水化。** 批量 DMA + lookahead 预取,用事件等待残余 miss。

Phase 1 go/no-go:页级 miss 和搬运字节无法覆盖 HBM 收益时,停止 offload,保留观测能力。

### 6.2 Phase 2:多请求生产化

- 跨请求共享 hot pool。
- 带宽预算闸。
- thrash 检测和降级到更多常驻。
- NUMA/Grace 本地内存绑定。

### 6.3 社区 PR 顺序

1. **重复工作检查。** 提 PR 前检查相关 issue/open PR;PR 描述中说明不重复、列出测试和 AI assistance。
2. **先提交观测能力。** 只读 metrics,默认关闭或 debug 开关,不改 logits、不调度 DMA、不 patch block table。
3. **再提交 page offload 辅助设施。** `SparsePageAdapter`、`LayerLocalBlockTables`、copy spec 验证或小扩展。
4. **再提交 DeepSeek V4 c4a offload。** 默认关闭,先单请求同步无损,再异步。
5. **GLM 5.2 单独后续 PR。** 若 DSA 语义匹配,新增 `GLM52DSAAdapter`,不混入首个 V4 PR。

### 6.4 测试矩阵

单元测试优先,只在行为型 offload PR 中加入端到端 logits 测试:

| 范围 | 测试点 |
|---|---|
| adapter | `tok // 64` 映射到 logical page;top-k 去重;尾页识别;无效 `-1` token 忽略 |
| block table | layer-local buffer 固定地址;只 patch c4a 层;共享 `block_table` 不变;shape/dtype/device 校验 |
| state machine | seal/store/load/evict/cleanup;in-flight ref 保护;request abort;request id 复用 |
| transfer | batch 多页 copy descriptor;page size 584B * 64;不触发 SWA/C128A copy |
| fallback | CPU pool OOM、hot pool 满、无可驱逐页、DMA submit fail、unsupported backend |
| correctness | offload on/off 对同一 prompt 的 logits 无损或在量化容差内一致 |
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

## 附录 A:为什么不能用 `CPUOffloadingManager` 做控制面

`CPUOffloadingManager` 面向 scheduler 侧的前缀缓存/CPU KV 缓存管理;selected-page offload 是 worker 侧、逐层、由真实 top-k 驱动的 GPU hot pool 分页。关键不匹配:

1. **运行位置:** scheduler 看不到每层真实 top-k 和 miss 页。
2. **键语义:** 现有 manager 用内容哈希 `OffloadKey`;c4a 页应按 `(request, layer, logical_page)` 管。
3. **调度节律:** 现有 manager 每调度步 prepare/commit;selected-page offload 每层每步 promote/wait。
4. **淘汰对象:** 现有 manager 管 CPU 缓存块;这里淘汰 GPU hot pool 槽,且 evict 无写回。
5. **策略接口:** `CachePolicy.touch/evict` 不表达带宽闸、推测预取、本步 protected、跨请求 hot pool 和 `hot_score(step)`。
6. **生命周期:** CPU 权威池需要请求结束即时释放;现有 manager 更偏缓存淘汰释放。

因此只复用 `CPUOffloadingWorker`、load/store spec 和引用保护语义;控制面由 `SparsePageOffloadCoordinator` / `SparsePageOffloadManager` / `SparseHotPagePool` 自建。
