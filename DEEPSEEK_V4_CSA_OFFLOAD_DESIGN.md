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
> 支持 TP,以及启用 EP 的 DP 部署(DP+EP);DCP 仍不支持。

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
5. 支持 TP 和 DP+EP。c4a MLA KV 在 TP rank 上按复制语义处理,每个 D worker 独立
   维护 CPU 权威池、GPU hot pool 和 layer-local block table;DP request 只归属一个
   D replica,EP 只作用于 MoE 层,不参与 c4a page 的寻址、传输或驻留。

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
- **并行 rank 之间不共享可变驻留状态。** CPU page、hot slot、in-flight job 和 patched
  block table 都是 D worker-local;TP rank 之间不借用物理 block id,DP replica 之间不
  迁移活动请求的 page state,EP rank 不进入 page key。

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

### 2.4 并行拓扑契约

支持以下静态拓扑,并要求在 offload on/off 的相同拓扑上验证 logits:

| 拓扑 | 支持结论 | page ownership 与传输规则 |
|---|---|---|
| `TP > 1, DP = 1, EP off/on` | 支持 | DeepSeek V4 MLA c4a KV 在 TP rank 上复制;每个 D TP worker 有独立 CPU 权威副本和 hot pool,按本 rank 的 block table page-in;开启 EP 只改变 MoE expert ownership |
| `TP >= 1, DP > 1, EP off` | 支持 | 每个请求固定归属一个 D DP replica;该 replica 的 TP workers 独占请求 page state |
| `TP >= 1, DP > 1, EP on` | 支持 | 先按 DP replica 隔离请求,再在该 replica 的 TP workers 上按复制 KV 语义各自 page-in;EP group 不改变 KV ownership |

TP 传输复用 NIXL `TransferTopology` / TP mapping,而不是在
`SparsePageConnector` 中另写 rank 算法:

- P/D TP size 相同或一方是另一方的整数倍;其他比例启动时 fail closed。
- MLA KV 是复制布局。每个 D TP rank 只从 TP mapping 选出的一个 P TP rank 读取
  c4a page;P TP 较大时不拼接多个副本,P TP 较小时允许多个 D rank 读取同一 P rank。
- 每个 D TP rank 本地运行 indexer、miss 判定、page-in 和 block table patch。正常 TP
  执行下 top-k 和 logical page 集合应一致,但 GPU phys block id 和 residency 状态允许
  不同;热路径不增加 TP collective。
- CPU 权威副本先按 worker-local 方案实现,因此 host 容量和 P→D c4a 流量会随 D TP
  size 复制。跨 TP rank 共享一份 CPU pool 属于后续优化,不能作为正确性依赖。

DP+EP 遵循 request ownership,不按 EP expert ownership 路由 page:

- router/connector 必须把每个请求显式绑定到一个 D DP replica,并在请求完成、abort
  或 preempt 前保持亲和性。P DP rank 与 D DP rank 不要求编号或数量相同。
- connector endpoint identity 至少区分 producer engine、producer DP rank、consumer
  engine 和 consumer DP rank;进入具体 worker 后再用 NIXL TP mapping 选择远端 TP rank。
- EP 只改变 MoE expert 的放置和通信。MoE all-to-all 完成后,attention 仍使用请求所属
  DP replica 的本地 KV;不得按 `ep_rank` 复制、迁移或查找 c4a page。
- 活动请求跨 DP replica 迁移和 elastic DP/EP resize 不在本设计内;发生时必须先 drain
  请求,否则对该请求 fail closed。

状态的全局定位可写为:

```text
WorkerRoute = (consumer_engine_id, consumer_dp_rank, consumer_tp_rank)
PageOwner   = (WorkerRoute, request_id, generation, layer_name, page_idx)
```

`WorkerRoute` 只用于 connector 路由、日志和一致性校验。worker 内部 manager 已由进程
隔离,仍可使用紧凑的 `(request_id, generation, layer_name, page_idx)` 作为 page key。

### 2.5 暂不支持范围

以下场景不启用行为型 offload:

- 纯聚合部署(prefill/decode 同实例)。本设计只支持 PD 分离。
- 非 DeepSeek V4、非 `fp8_ds_mla`、非 `compress_ratio == 4` c4a 层。
- DCP / decode context parallel。当前 indexer 对 `compress_ratio > 1` + DCP
  仍是 `NotImplementedError`,offload 不应绕过该限制。
- P/D TP size 既不相同、也不存在整数倍关系;活动请求跨 DP replica 迁移;活动请求期间
  elastic DP/EP resize。
- C128A、SWA、dense MLA、ROCm/XPU backend。
- MTP/native spec decode、多 token decode、DSpark non-causal decode,除非另有
  logits 无损验证。

禁用时必须保持原路径行为(包括退回不启用 PD offload,由 D 常规接收全量 c4a 到
HBM),最多输出一次 warning 或 telemetry 标记。

---

## 3. 架构

### 3.1 数据流

```text
P[producer_dp_rank, producer_tp_rank] (prefill worker):
  c4a page 写满 -> seal
  prefill sparse MLA 用原始 block_table 完成 gather -> 出首 token
  connector.save 按 layer_name 分流:
    c4a       -> NIXL DRAM xfer 到 D 的 CPU 权威池
    indexer key / C128A / SWA -> 常规 KV 传输到 D 的 HBM
  request_finished -> free-now, kv_transfer_params 携带 transfer 描述符与 tail 页引用
  P allocator 整请求释放 KV

D[consumer_dp_rank, consumer_tp_rank] (decode worker):
  request route 固定 consumer_dp_rank
  NIXL TP mapping 选择 producer_tp_rank
  connector 收齐 c4a latent -> 落 CPU 权威池, 标 cpu_ready_evicted
                c4a tail 页 -> 从 CPU restore 到 D HBM
                其余层 -> 落 HBM
  allocate-partial: c4a 只在 HBM 分配 hot pool + tail;hot pool 起始为空

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

以上状态全部是 D worker-local。TP 的每个 worker 持有相同 logical c4a 内容的独立
CPU/GPU 副本,但物理 slot 可以不同;DP replica 只持有路由到自己的请求;EP collective
不访问 page offload manager。connector sideband 在 scheduler 聚合 tail page 与 transfer
描述符,但下发给 worker 时必须保留 consumer DP ownership,不能广播到其他 DP replica。

PD 分离下主干验收不是"复制到 CPU 后仍保留原 GPU cache",而是 **D 从一开始就不为
c4a 全量分配 HBM**:c4a 的 HBM survivor set 收敛到 hot pool 和 mutable tail。

D 侧 hot pool 起始为空,不在 prefill/接收阶段预热。进入 decode 后,hot pool 完全由
每层 top-k 的真实 miss 驱动 CPU->GPU page-in 逐步填充,不从 P 传输 last-token Top-K
页来 seed。代价是首个 decode step 会对选中页产生一次冷启动 miss;收益是去掉一条
"P 产生、传输、但 D 接收侧未消费"的 sideband 链路。是否重新引入 seed 预热应由
first-decode-step 的 miss 遥测证明,而不是默认开启。

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
├── selection.py         SparsePageAdapter / SparsePageSelection
├── hot_pool.py          SparsePageHotPool / SparsePageStagingPlan
├── staging.py           SparsePageStagingManager
├── block_table_cache.py LayerLocalBlockTableCache
├── protocol.py          sideband / route wire 数据模型
├── route_tracker.py     producer generation / consumer route 生命周期
├── coordinator.py       SparsePageOffloadCoordinator
├── selection_metrics.py SparsePageSelectionCollector
└── adapters/
    └── deepseek_v4_c4a.py
```

文件职责建议:

| 文件 | 职责 | 不负责 |
|---|---|---|
| `__init__.py` | 暴露最小公共入口,例如 coordinator/config/adapter 类型;避免让模型侧 import 内部状态类 | 不做注册副作用,不初始化全局状态 |
| `config.py` | 解析和校验 page offload 配置,包括开关、每层 hot pool 页数、lookahead、层选择、带宽预算、传输后端 | 不读取 HF model config 的模型语义,不决定某个 layer 是否可 offload |
| `selection.py` | 定义 adapter 抽象和一次 top-k 选择的结构化结果,例如 `(request, layer, logical_page)` 去重集合、真实 miss 集合、tail pin 标记 | 不维护长期 residency,不分配 GPU/CPU 槽 |
| `hot_pool.py` | 维护 layer-local GPU hot pool、LRU 驻留和每请求配额,把一次选择转换为不可变 `SparsePageStagingPlan` | 不提交传输,不 patch block table,不持有 CPU 权威页 |
| `staging.py` | 执行 staging plan,管理 Phase 1 同步 CPU mirror、批量 page-in/tail copy、prefill seal 和 request cleanup | 不决定淘汰策略,不实现 scheduler-side `OffloadingManager`,不拥有 connector |
| `block_table_cache.py` | 管理固定地址、固定形状的 layer-local compressed block table buffer,并提供原地 patch 接口 | 不修改共享 `attn_metadata.block_table`,不 patch SWA block table |
| `protocol.py` | 定义 P→D sideband、page reference 和包含 TP/DP/EP identity 的 route wire model | 不维护活跃请求状态,不做 page staging |
| `route_tracker.py` | 封装 producer generation、consumer route 唯一绑定、stale 检查和完成清理 | 不解析 NIXL 参数,不持有 KV tensor 或 page residency |
| `coordinator.py` | 每层 decode 的编排入口:接收 adapter selection,查询 residency,触发 page-in,等待真实 miss,调用 block table patch,向 telemetry 记账 | 不包含 DeepSeek V4 kernel 细节,不直接调用 FlashMLA kernel |
| `selection_metrics.py` | 记录实际 staging 的 go/no-go 指标,包括 unique selected pages、miss(H)、复用率、搬运字节、wait 预算、hot-set drift | 不改变调度行为,不提交传输,不影响 logits |
| `adapters/deepseek_v4_c4a.py` | DeepSeek V4 c4a 适配层:识别 c4a 层,解释 top-k indices 和 compressed block table 列语义,提供页大小/页列/尾页规则,供 connector 分流与 coordinator 复用 | 不实现通用状态机,不处理 GLM 5.2 或其他模型 |

依赖方向保持单向:DeepSeek/MLA 执行路径调用 `coordinator.py`;coordinator 依赖
staging/hot_pool/block_table_cache,adapter 只提供模型语义。`SparsePageConnector` 只依赖
config、protocol、route_tracker 和 coordinator 公共入口,不 import hot pool 或 staging
内部数据面状态。后续异步传输可在 staging 下增加 transfer/CPU pool 实现,不改变
staging plan 或 connector route 生命周期接口。

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
  generation: int              # request id 复用时递增;不复用时可固定为 0
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
P 与 D 之间对齐(connector 用它路由 CPU 权威页)。`generation` 防止 engine 回收
request id 后命中旧页;manager 在 request cleanup 后仍必须移除所有相关 CPU 和 GPU
state。DP 路由信息不塞入 worker-local `LogicalPage`,但 connector metadata 必须携带并
校验 producer/consumer engine 与 DP rank;TP rank 由 NIXL worker topology 决定。

### 3.2.2 每层调用时序(D 侧 decode)

```text
indexer forward
  -> writes topk_indices_buffer
attention backend forward_mqa
  -> if disabled: 原路径
  -> adapter.extract_selection(req_id_per_token, topk_indices_buffer, seq_lens)
  -> coordinator.stage_decode_layer(layer_name, selection, source_block_table)
       - update telemetry
       - classify hit/miss/tail
       - submit CPU->GPU page-in for real miss
       - build patched block table or patched physical top-k buffer
       - wait only on real miss required by this layer
  -> backend physical-index conversion / sparse decode kernel
coordinator.finish_layer(layer_name)
```

### 3.2.3 PD 主干时序

同步原型必须覆盖 3.1 的完整主干:P connector 发送 + D allocate-partial + decode
同步 reload(每个 D DP replica 先覆盖单请求、该 replica 内覆盖全部 TP rank、
`compress_ratio == 4` c4a 层)。D 每层的 miss 判定为:

```text
real_miss = selected_pages - (hot_pool/resident ∪ tail_pages)
real_miss 同步 CPU->GPU load 到 hot pool -> patch layer-local block_table
  -> wait load 完成 -> sparse MLA gather
```

主干输出必须能被测试直接观察:

- D 侧 c4a resident GPU block 数 = `hot_pool + tail`,不等于完整 prompt 的 c4a 页数。
- P 侧 `request_finished` 后 KV 整体释放,没有生命周期中途的单页 free。
- CPU 权威池包含所有 sealed c4a 页且 ready。
- 下一个 decode step 的 logits 与 offload 关闭时一致。
- 同一请求只出现在一个 D DP replica;该 replica 的每个 TP rank 都能独立完成 page-in,
  且 patched phys block id 只引用本 rank hot pool。
- 开启 EP 前后 page key、connector route 和 c4a resident block 统计不因 expert
  placement 改变。

若 D 侧 block allocator 不能安全地为单层 c4a 只分配部分页(allocate-partial),
同步原型应先补 allocator 接口或 fail closed;不得让 D 常规分配全量 c4a HBM 后再
依赖聚合式中途释放。

### 3.3 控制策略

每个 D worker 为本 DP replica 拥有的请求按 `(request, generation, layer)` 维护:

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
| P free-now + sideband | 支持。`request_finished -> (async_free, kv_transfer_params)`,P 天然释放并可携带 tail 页引用与 transfer 描述符 | `base.py` `request_finished` |

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
  复用 NIXL worker 原语和拓扑:
    agent 握手 / TransferTopology / TP mapping
    DRAM+VRAM 注册 / CopyBlocksOp(d2h,h2d) / notif 轮询

  save (P 侧, 按 layer_name 分流):
    c4a      -> NIXL DRAM xfer 到 D 的紧凑 CPU 权威池
    其余层    -> 常规 HBM block xfer(委托 NIXL 现有 block 路径)
  request_finished -> free-now + kv_transfer_params 携带 route、generation、
                      tail 页引用与 transfer 描述符

  load (D 侧):
    c4a      -> 落 CPU 权威池, 标 cpu_ready_evicted, 交 manager; 不自动 h2d
    其余层    -> 正常灌回 HBM
  decode hot pool page-in 的 h2d 由 coordinator 逐步驱动, 复用 CopyBlocksOp
```

并行部署中 connector 需要分两级路由:

1. scheduler/router 先选定唯一的 `consumer_dp_rank`,并在 sideband 中携带 producer 与
   consumer engine/DP identity。DP rank 是独立 engine endpoint,不能把请求 metadata
   广播给整个 DP/EP group。
2. D worker 使用 NIXL 已有 TP topology 选择远端 producer TP rank。由于 MLA KV
   复制,每个 D TP worker 拉取一份完整 c4a page 到自己的 CPU pool;不得按普通多头
   attention 对 latent 内容做 head slice 或 concat。

P/D TP size 不同但可整除时,沿用 NIXL MLA mapping。任何 rank 在握手后看到的
`is_mla`、page shape、page size、compression ratio 或 layer range 不一致,整个请求
fail closed。EP rank 不参与以上两级路由;即使 EP group 跨 DP/TP rank,KV endpoint
仍由 request DP owner 和 attention TP worker 唯一确定。

架构取舍:

- **不用 `MultiConnector` 拆 c4a/rest。** 它把每个调用广播给所有子 connector、不按
  层 filter,NIXL 又无"排除层"开关,会双重搬运。
- **不 fork NIXL 整块 worker。** 它是 request-block 粒度、load 时自动 rehydrate,
  硬塞逐层 + 留 CPU 是对抗其模型。只取传输原语,薄壳持有 `NixlWrapper` 直接调底层。
- **不接 LMCache / Mooncake 控制面。** 其 key 语义(前缀 / 内容哈希)与
  `(request, layer, logical_page)` 冲突。

#### 3.4.3 传输层复用边界

复用:

- NIXL 的 agent 握手、DRAM/VRAM 注册、`CopyBlocksOp`(`d2h`/`h2d`)、notif 轮询。
- NIXL `TransferTopology` 和 MLA TP mapping;支持相同 TP size 以及可整除的异构 TP,
  不自行推导远端 TP rank。
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

### 4.1 D worker 的 HBM 常驻底线

D 的常驻底线必须逐 worker、逐层求和。以下模型是一个 D TP worker/GPU 的预算,
按 `fp8_ds_mla` 的 584B c4a/C128A 条目估算;plain bf16/per-tensor fp8 cache dtype
需要替换页大小:

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

并行维度的核算规则:

- 上式是 **per D TP worker/GPU**;`H_L` 不除以 TP size。MLA c4a KV 和 selected page
  在 TP rank 上复制,所以同一请求的集群级 HBM/CPU page 容量约再乘 D TP size。
- DP 按请求分片,单请求不乘 DP size;集群总量对所有 DP replica 的活动请求求和。
- EP 不增加 attention KV 副本。不能用 `ep_size` 乘 page 容量,也不能从某个 expert
  rank 的空闲 HBM 借用 hot slot。
- worker-local CPU 权威池原型的最坏 host 容量为
  `sealed_c4a_bytes_per_request * D_TP`;若 host 容量不足则按请求 fail closed。

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
- worker-local TP 副本会让 c4a P→D 总流量近似乘 D TP size。性能验收必须同时报告
  per-rank 与整 TP group 的传输字节;不得只报告单 rank 带宽。

---

## 5. 实现集成点

| 位置 | 改动 |
|---|---|
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` / `flashinfer_mla_sparse.py` | D 侧 decode 接入点。c4a 且 selected-page offload 已开启(consumer + `SparsePageConnector`)时,在 logical top-k 转 physical top-k 前调用 coordinator,并改用 layer-local patched block table 或 patched physical top-k buffer |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | 仅在目标 commit 确认 DeepSeek V4 使用 FlashMLA sparse decode 时接入;否则只作为 `fp8_ds_mla` layout 和 page size 核对来源 |
| DeepSeek V4 c4a compressor / KV 写入路径(P 侧) | 页封存时标 sealed;prefill sparse MLA 完成后由 connector 按层分流发送 sealed c4a 到 D 的 CPU 权威池;不做生命周期中途 GPU 释放,由 `request_finished` free-now 整请求释放 |
| `vllm/distributed/kv_transfer/.../SparsePageConnector`(新增) | PD 部署下自研薄 connector,拥有全部 V4 层;复用 NIXL 传输原语;`save_kv_layer` 按 layer 分流(c4a→D CPU 权威池,其余→D HBM);`request_finished` 返回 free-now 并携带 tail 页引用与 transfer 描述符;load 侧对 c4a 不自动 h2d,仅把 tail restore 回 HBM |
| NIXL `TransferTopology` / TP mapping | 直接复用 MLA replicated-KV 的 P/D TP rank 映射;相同或可整除异构 TP 均由握手结果驱动;selected-page 路径不得另建一套 TP 映射 |
| scheduler / PD request router | 选择并固定 `consumer_dp_rank`;sideband 携带 producer/consumer engine、DP identity 和 request generation;不把 page state 广播到 EP group |
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
| `sparse_page_transfer_backend` | 提案 | `nixl`;只复用传输原语,不启用其控制面 |
| `sparse_page_hot_pool_blocks` | 提案 | 每请求、每可卸载层的 resident hot 页上限 `H`(单个整数) |
| `sparse_page_prefetch_lookahead` | 提案 | 跨层/跨步预取深度 |
| `sparse_page_offload_layers` | 提案 | `auto` 或显式层范围 |

配置校验建议:

- 未选用 `SparsePageConnector`(或非 consumer 角色)时不初始化 manager、不分配
  CPU pool、不改变 metadata。
- 选用了 `SparsePageConnector` 作为 consumer 但模型/backend 不满足支持范围时
  fail closed,报清晰错误,不能静默跑半套 offload。
- `sparse_page_hot_pool_blocks` 以每个请求、每个可卸载 layer 为单位;residency 按
  (request, layer) 维护,worker 的 resident slot capacity 为
  `H * max_num_seqs`,telemetry 需按 (request, layer) 记账。
- `sparse_page_cpu_pool_size_gib` 必须能推导出最大 CPU page 数,并在启动时打印
  `page_size_bytes`、`num_cpu_pages`、`num_c4a_layers`。
- 启动握手必须打印 producer/consumer engine 与 DP rank、local/remote TP size、当前
  TP mapping、MLA replicated 标记。P/D TP size 不可整除时拒绝开启。
- 每个 DP replica 独立创建 connector endpoint、CPU pool 和 coordinator。配置中的
  CPU pool 预算是 **per D worker**,不能在 DP/TP/EP world size 间静默均分。
- `enable_expert_parallel` 不改变 page offload 配置和 page key。若请求在活动期间改变
  DP owner 或发生 elastic resize,该请求 fail closed;drain 后的新请求可按新拓扑创建
  state。

不要把 `--kv-offloading-size` 当作 selected-page CPU pool 预算。当前 vLLM 会用它
自动配置 `KVTransferConfig` 和 `OffloadingConnector`,这会启用现有 CPU KV offload
路径,违反本设计"现有 KV connector 与 CPU KV offload 路径不受影响"的约束。

示例为提案占位;PD 分离需分别启动 P 与 D 实例并配置 `SparsePageConnector`
(以下 `--kv-transfer-config` 为占位,落地时接入正式配置):

```bash
# Prefill 实例(发送端,TP=2)
CUDA_VISIBLE_DEVICES=0,1 vllm serve /data/public_models/Deepseek-V4-Flash \
    --kv-cache-dtype fp8 --trust-remote-code --max-model-len 1000000 \
    --tensor-parallel-size 2 --enforce-eager --port 8001 \
    --kv-transfer-config '{"kv_connector": "SparsePageConnector", "kv_role": "kv_producer"}'

# Decode 实例(接收端 + selected-page offload,TP=2)
# consumer 角色 + SparsePageConnector 即自动开启 offload;旋钮全部走
# kv_connector_extra_config,不再使用 --hf-overrides。
CUDA_VISIBLE_DEVICES=2,3 vllm serve /data/public_models/Deepseek-V4-Flash \
    --kv-cache-dtype fp8 --trust-remote-code --max-model-len 1000000 \
    --tensor-parallel-size 2 --enforce-eager --port 8002 \
    --kv-transfer-config '{"kv_connector": "SparsePageConnector", "kv_role": "kv_consumer", "kv_connector_extra_config": {"sparse_page_cpu_pool_size_gib": 128, "sparse_page_transfer_backend": "nixl", "sparse_page_hot_pool_blocks": 512, "sparse_page_prefetch_lookahead": 2, "sparse_page_offload_layers": "auto"}}'
```

DP+EP 部署在 P、D 各自实例上增加 `--data-parallel-size <DP>` 和
`--enable-expert-parallel`;TP 可同时大于 1。PD router 必须把 producer engine/DP rank
写入 transfer params,并为请求选择唯一 D DP replica。不能仅靠 P/D 的 ordinal DP rank
相等来隐式配对,因为两侧 DP size 可以不同。

---

## 6. 测试矩阵

单元测试优先,只在行为型 offload PR 中加入端到端 logits 测试:

| 范围 | 测试点 |
|---|---|
| adapter | `tok // 64` 映射到 logical page;top-k 去重;尾页识别;无效 `-1` token 忽略 |
| block table | layer-local buffer 固定地址;只 patch c4a 层;共享 `block_table` 不变;shape/dtype/device 校验 |
| connector 分流 | `save_kv_layer` 只把 c4a 发到 CPU 池,indexer/C128A/SWA 走 HBM;load 侧对 c4a 不自动 h2d,仅 restore tail;`request_finished` free-now 且携带 tail 页引用 |
| TP topology | TP=2 同构 P/D;P TP=1→D TP=2;P TP=2→D TP=1;验证每个 D rank 只拉一个完整 MLA page、不 slice/concat、只 patch 本 rank phys block;不可整除 TP fail closed |
| DP routing | 两个并发请求路由到不同 D DP replica;CPU page/hot slot/cleanup 不串 replica;P/D DP size 不同仍按显式 engine/DP identity 正确路由 |
| EP orthogonality | `enable_expert_parallel` on/off 使用相同 page key 和 c4a selection;MoE all-to-all 后仍访问 request owner 的本地 KV;EP rank 不出现在 page lookup 中 |
| state machine | connector receive/tail restore/page-in/evict/cleanup;in-flight ref 保护;request abort;request id 复用与 P/D 对齐 |
| D allocate-partial | c4a 层只分配 hot pool + tail;evicted 逻辑页不被共享 `block_table` 引用;不全量分配 c4a |
| transfer | batch 多页 copy descriptor;page size 584B * 64;不触发 SWA/C128A copy |
| fallback | CPU pool OOM、hot pool 满、无可驱逐页、connector 传输 fail、allocate-partial 不支持、unsupported backend/非 PD |
| correctness | offload on/off 对同一 prompt 的 logits 无损或在量化容差内一致;至少覆盖 TP=2、TP=2+EP、DP=2+EP、TP=2+DP=2+EP |
| memory accounting | 每个 D TP worker 的 c4a GPU resident block 数和 `torch.cuda.memory_allocated`/allocator 统计低于全量;同时报告 TP group 汇总,不能只增加 CPU mirror 或 hot pool |
| CUDA graph | patched buffer 地址稳定;staging 路径不引入 data-dependent allocation |

推荐命令按实际改动收敛到具体文件,例如:

```bash
.venv/bin/python -m pytest tests/v1/attention/test_sparse_page_offload.py -v
.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_gpu_worker.py -v
pre-commit run ruff-check --files <changed files>
```

若改动影响模型输出、索引语义或 serving 行为,PR 描述必须附 logits 对比或
model eval 结果。

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
语义;控制面由 `SparsePageOffloadCoordinator` / `SparsePageStagingManager` /
`SparsePageHotPool` / `SparsePageRouteTracker` 与 `SparsePageConnector` 自建。
