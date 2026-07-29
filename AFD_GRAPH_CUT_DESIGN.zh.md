# 切图式 AFD：面向 vLLM 的模型无关 Attention–FFN 分离方案

- **相关：** RFC [#22799](https://github.com/vllm-project/vllm/issues/22799)（ATTN-FFN 分离）、RFC [#27584](https://github.com/vllm-project/vllm/issues/27584)（弹性 AFD）、PR [#25162](https://github.com/vllm-project/vllm/pull/25162)、[#29772](https://github.com/vllm-project/vllm/pull/29772)、[#30945](https://github.com/vllm-project/vllm/pull/30945)

---

## 1. 摘要

Attention–FFN 分离（AFD）把 MoE 模型的注意力和 FFN/MoE 放到两个资源池中。注意力主要受显存和
KV cache 限制，FFN/MoE 更依赖算力，因此分开部署后可以分别扩容。这个方法主要适合 MoE 模型；对稠密
模型来说，来回传输激活值的成本通常太高。

vLLM 之前的方案（#25162、#29772、#30945）把通信代码写进每个模型的 decoder layer `forward`。
这种做法需要逐个适配模型，还会干扰 CUDA graph，并可能因过小或为空的微批导致 kernel 崩溃。

本方案提出**切图式 AFD（Graph-Cut AFD）**。它利用 vLLM 共享的 `Attention`、`MLAAttention` 和
`FusedMoE` 模块，以及 decoder 中稳定的 `mlp`/`ffn` 边界，把模型自动切成可独立运行的计算片段。
当前实例不负责的模块会在构图时替换成空操作，数据收发则由计算图外的事件循环和可插拔 connector
完成。事件循环还会动态组批并补齐到固定大小，避免产生空批。

因此，满足标准分段 decoder 接口的模型不需要修改源码就能使用 AFD。需要额外必选状态的非标准层会在
启动时直接报错，避免运行到一半才失败。

---

## 2. 架构

先说明本章会用到的几个概念：

- **计算片段（segment）**：切图后可以独立运行的一段模型计算。
- **切点（boundary）**：Attention 和 FFN 的分界，只标记在哪里切图，不负责通信。
- **后续状态（continuation）**：跨过切点后还会用到的数据，例如 residual 和 positions。这些数据留在
  Attention 实例，不通过网络传输。
- **事件循环（event loop）**：在计算图外收发数据、组批，并推动各个计算片段继续运行。
- **连接器（connector）**：事件循环使用的传输接口，负责在 Attention 和 FFN 实例之间搬运激活值。

整体上，AFD 把模型分成 Attention 和 FFN 两类实例。计算图只负责计算，事件循环通过 connector
负责通信。两者在 boundary 处交接。

### 2.1 整体结构

引擎通过 `--afd-config` 接收 JSON 配置，并将其转换为 `AFDConfig`。其中 `role` 决定当前实例做什么：
`attention`、`ffn` 或 `none`，默认是 `none`。

Attention 和 FFN 两种实例使用同一份模型定义，区别只在于保留哪些模块：

- **Attention 实例**保留真正的注意力模块和 KV cache，并在多个副本间运行 DP-Attention。
  本地的 FFN/MoE 会被换成 `AFDAttnBoundary`。程序运行到这里时，会把 MoE 的输入交给 FFN 实例，
  同时在本地保存后续计算需要的状态。
- **FFN 实例**保留真正的 `FusedMoE`，并在多个副本间运行 EP。注意力模块会被换成
  `AFDNoOpAttention`，因此不会分配 KV cache，也不会执行注意力计算。MoE 则由
  `AFDFfnBoundary` 包装。

标准 MoE decoder 层可以简化为：

```text
h = h + attn( input_layernorm(h) )       # 注意力块  -> Attention 角色
h = h + moe(  post_attn_layernorm(h) )   # FFN/MoE 块 -> FFN 角色
```

AFD 在两个残差块之间切开模型。两边只传输切点处的 `hidden_states`，流程如下：

1. Attention 计算片段运行到 `AFDAttnBoundary`，产出 MoE 输入。
2. `residual`、`positions` 和 `metadata` 等后续还要使用的数据留在 Attention 实例，并按
   `transfer_id` 保存。
3. Attention 事件循环异步发送 MoE 输入。FFN 事件循环收到数据后组批，再调用
   `AFDFfnBoundary` 计算 MoE。
4. FFN 事件循环异步返回结果。Attention 事件循环找到之前保存的状态，继续运行下一段。

### 2.2 模型切分与执行

模块替换由 `apply_afd_roles(model, vllm_config)` 完成，代码位于
`vllm/model_executor/models/afd.py`。它在权重加载后直接修改已经创建好的模型，不需要逐个修改模型的
`forward()`。

这个 pass 根据常见的属性名识别 decoder 层：

- 注意力模块通常叫 `self_attn` 或 `attn`；
- FFN 模块通常叫 `mlp` 或 `ffn`；
- 只处理包含 `FusedMoE`、`MoERunner` 或 `experts` 路由子模块的 MoE 层。

`AFDNoOpAttention` 是一个真正的空操作：它没有参数，输入什么就返回什么。它不继承
`AttentionLayerBase`，也不会加入 `static_forward_context`，所以 KV-cache 管理器不会为它分配显存。
`_pop_attention_kv` 还会移除原注意力模块内部所有需要 KV cache 的子模块，例如 DeepSeek-V4 的
indexer、compressor 和 SWA 缓存。这样，FFN 实例的计算图中不再包含注意力模块，也不会启动注意力
kernel。

模块替换本身与具体模型无关，但当前分段运行时只支持下面这种标准接口：

```text
layer(positions, hidden, residual, optional...) -> (hidden, residual)
```

远端 MoE 只能把 `hidden` 作为必需输入。如果某个模型还要求没有默认值的额外参数，系统会在启动时直接
报错，而不是冒险运行。FFN worker 也不能假设自己拥有请求专属的 `input_ids` 或其他路由状态。

boundary 只负责标记切点或执行本地 MoE，不负责通信：

```python
class AFDAttnBoundary(nn.Module):
    def forward(self, hidden, *args, **kwargs):
        return torch.ops.vllm.afd_cut(hidden, self.layer_id)

class AFDFfnBoundary(nn.Module):
    def forward(self, hidden, *args, **kwargs):
        return self.mlp(hidden, *args, **kwargs)
```

这两个 boundary 不持有 connector，不读取全局模式，不调用 `poll()`，也不会等待远端结果。

`vllm::afd_cut` 是一个保持输入不变的自定义算子。在 eager 模式下，它直接返回原 tensor；在
Dynamo/FX 图中，它会留下一个明确的切点。它的名称被加入 `AFD_SPLITTING_OPS`，因此普通的
piecewise partition 和 AFD 专用 partitioner 都能找到它。

`partition_afd_graph()` 位于 `vllm/distributed/afd_transfer/graph_cut.py`。它会扫描所有
`afd_cut`，找出切点之后仍要使用的数据，并生成不包含切点标记的 `GraphModule` 片段：

```text
AttentionSegment_i(..., local_state) -> (moe_input, continuation)
MoeGraph_i(moe_input)                 -> moe_output
AttentionSegment_i+1(moe_output, continuation) -> ...
```

运行时的主要步骤是：

- `AFDGraphCutProgram.start()` 运行到第一个切点，返回
  `(layer_id, moe_input, state)`。
- `resume(moe_output)` 收到远端 MoE 结果后，把它交给下一个计算片段。
- `residual`、原始输入、`SymInt` 等仍然有效的数据保存在本地 `continuation` 中，不通过网络发送。
- `AFDFXAttentionExecutor` 把 `start()`、`resume()` 和 `finalize()` 接入 Attention 事件循环。

正式运行时，`AFDSegmentedModel` 先构造“decoder 片段 → `afd_cut` → decoder 片段”的 FX 图，然后
立即切图。切图完成后，原 boundary 变成本地 identity，任何可执行片段里都不会再出现 cut marker。
事件循环只通过 `AFDFXAttentionExecutor` 执行这些片段，没有另一套同步或 eager fallback 流程。
片段内部在 dispatcher 选择 `FULL` 时，使用按 `(segment, BatchDescriptor)` 缓存的 CUDA graph。
每个片段和 bucket 都持有固定地址的输入 buffer；运行时先在图外复制动态输入，再 capture/replay
固定地址的计算。dispatcher 不支持 `FULL` 或 batch 超出最大 capture size 时，保持 eager 执行。

这里有一条必须遵守的规则：**一个 connector 只能由所属的事件循环调用 `poll()`。** 模型模块、测试
helper 或其他 fallback 流程都不能读取同一个完成队列，否则可能提前取走本应由事件循环处理的传输结果。

### 2.3 调度与传输

PR #25162 和 #29772 曾因流水线产生空批而崩溃。这里用两条简单规则避免这个问题：

- 队列为空时不执行任何计算，因此不会产生 `(0, H)` tensor 或 `gridDim=0`。
- batch 大小不匹配时，使用 vLLM 现有的 `cudagraph_capture_sizes` bucket padding。补齐的 token 会在
  采样前删除，也不会进入残差计算，不需要额外的 ghost batch 机制。

Phase 0 的两侧使用不同调度策略。Attention 每次接收一个完整的 vLLM scheduler batch，并按
`A0→F0→A1→F1→…` 同步逐层推进；本层 FFN 结果全部返回前，不拆分 batch，也不执行其他层或其他
batch。FFN 侧仍按层维护异步队列，可以从任意就绪层取任务，并合并同一层中来自不同 Attention rank
的完整 batch item。

Attention 和 FFN 之间的传输统一通过 AFD connector 完成。它借用了 KV connector 的注册和工厂模式：
`AFDConnectorFactory` 的结构与 `KVConnectorFactory` 相似，可以按名称加载不同实现。

但两者只在工厂和注册方式上相似，方法接口并不相同。`KVConnectorBase_V1` 用来读写每个请求的 KV
block，而 AFD connector 传输的是层间激活值。直接复用 KV connector 会把两种不同的生命周期绑在
一起。因此，AFD 定义了独立的 `AFDConnectorBase` 接口、具体 connector 实现和
`AFDConnectorFactory` 工厂。

> 命名约定：接口基类固定叫 `AFDConnectorBase`，具体实现叫 `XxxAFDConnector`。文中小写
> "AFD connector" 泛指这一整套抽象，并非某个类名。

## 3. 数据流

对单个请求来说，每层的数据会这样流动：

```text
Attention 池                            FFN 池
------------                            ------
计算 AttentionSegment_i
保存本层后续计算需要的状态
发送 MoE 输入 ─────────────────────▶  计算 MoeGraph_i
恢复本地状态 ◀─────────────────────  返回 MoE 输出
继续下一段计算
```

两个池之间只传输 MoE 的输入和输出。residual、positions、metadata 等后续计算需要的状态留在
Attention 侧，不经过网络。每次发送都有唯一的 `transfer_id`；MoE 输出返回后，Attention 侧用它找到
对应的本地状态，再继续执行。这样既减少了传输量，也不会把不同请求或不同层的结果接错。

### 3.1 调度：Attention 同步逐层，FFN 异步队列

A↔F 边界按 `(layer, attn_rank)` 建立队列。Attention 侧一次只推进一个完整 scheduler batch：
执行第 `i` 层 Attention、发送完整 batch、等待该 batch 的第 `i` 层 FFN 结果全部返回，再进入第
`i+1` 层。FFN 侧按层接收来自各个 Attention rank 的任务，可以从任意非空层队列中选择工作。

调度时需要遵守三条规则：

- **Attention 有状态，FFN 无状态。** KV cache 保存在原来的 Attention rank 上，所以 FFN 算完后，
  结果必须发回这个 rank。FFN 只根据输入做计算，任何 FFN rank 都可以接任务，因此容易做负载均衡和扩缩容。
- **Attention batch 保持完整并同步逐层推进。** Attention 不拆分 scheduler batch，也不在等待 FFN
  返回时穿插其他 batch。FFN 可以交错处理不同 Attention rank 的任务，但每个 item 仍按照
  `A0→F0→A1→F1→…` 前进。
- **没有数据就不计算。** 队列为空时，本轮不启动 kernel；数据量不足标准 bucket 时再用 padding 补齐，
  因此不需要 ghost token，也不会产生空批。

Attention 侧把完整 scheduler batch 放入第 0 层，并把 FFN 返回结果作为同一 batch 的下一层
continuation；该 continuation 的优先级高于后续 batch。FFN 侧根据 `layer_id` 把收到的数据放入对应
层队列，具体策略见 §3.2。

代码中的收发由单线程事件循环驱动。两侧都用非阻塞的 `poll()` 和 `isend()`，但调度语义不同：
Attention 发出一个完整 batch 后只轮询它的返回结果；FFN 则可继续选择其他就绪任务。驱动循环、流和
背压的细节见 §3.3。

### 3.2 Attention 完整 batch 与 FFN 跨层动态批

Attention 侧不做跨层乱序动态批。一个完整 scheduler batch 进入后，严格按下面的顺序执行：

```text
完整 batch：A0 → 等待 F0 全部返回 → A1 → 等待 F1 全部返回 → … → An
```

本层结果返回后，FX continuation 恢复同一个 batch 的本地 residual、positions、slot mapping 和
attention metadata，再执行下一层。后续 scheduler batch 即使已经排队，也必须等当前 batch 完整走完。

FFN worker 持有所有层的 `FusedMoE` 权重。一次 FFN 计算只需要激活值和 `layer_id`，没有跨层状态，
所以 FFN 侧仍然每层各有一个队列，并从所有非空队列中选择一层执行。多个 Attention rank 的完整
scheduler batch 会自然错峰到达；FFN 可以合并同一层中已经到达的多个完整 item，算完后再按 item
长度和 `transfer_id` 分开发回原来的 Attention rank。

具体实现中，队列和 bucket 操作由 `AFDDynamicBatchScheduler`
（`vllm/distributed/afd_transfer/scheduler.py`）提供：

- 每层对应一个 `LayerQueue`，记录排队的 token 数和队首等待时间。
- `pick_ready_layer(now)` 从所有非空队列中选择一层；所有非空队列都可以立即执行，不必等 token
  数达到某个门槛。
- 如果没有非空队列，函数返回 `None`，本轮不发射计算，因此不会产生空批。
- `pick_bucket`、`pad_to_bucket` 和 `drop_padding` 负责选择 bucket、补齐输入和移除 padding。

Attention 每次只取一个完整 scheduler batch item，且返回的 continuation 必须优先于后续 item。
FFN 可以合并同一层的多个完整 item。两侧都不会拆分单个 item 内的 token。

**必须避免冷门层一直等不到执行。** 如果总是选择数据最多的层，数据较少的层可能长期得不到处理，
对应的 Attention batch 也会一直等待返回。为此，FFN 调度器会同时考虑排队时间：某层的队首等待超过
`AFDConfig.age_limit_ms` 毫秒后，必须优先执行该层。

**CUDA graph 的处理方式。** 形状相同的 MoE 层可以共用按 batch bucket 捕获的图，replay 时根据
`layer_id` 切换权重指针；DeepSeek 开头的 dense 层需要单独捕获。Attention 侧则按
`(attention backend/层类型, batch bucket)` 捕获图，并在 replay 时传入当前批次的 positions、
slot mapping 和 metadata。Attention 依赖这些额外状态，不能直接照搬 FFN 侧只处理张量的方式。
所有层的权重已经常驻显存；每层的图只记录 kernel 启动序列，不会再复制一份权重。

### 3.3 异步队列的执行模型：驱动循环、流与背压

这里不使用任务线程池。Attention 侧的事件循环由 `gpu_model_runner.forward` 推进；没有请求入口的
FFN worker 则由一个后台线程持续推进 `AFDFfnEventLoop`
（`vllm/distributed/afd_transfer/worker_loop.py`）。

FFN 每推进一轮（tick），事件循环只做三件事，而且都不会等待网络：

1. 调用 `poll()`，取回已经完成的传输并放入对应层的队列。
2. 从所有就绪层中选一层，补齐到合适的 bucket 后执行这一层的计算。
3. 调用 `isend()` 发出结果，然后立即开始下一轮。

FFN 选层由 `AFDDynamicBatchScheduler` 决定。Attention tick 只发送当前完整 batch，或轮询当前层的
返回结果；结果到达后优先恢复同一 batch 的下一层。连接器负责接收和发送，
`replay_fn(layer, padded)` 负责执行选中的 MoE 层。FFN 回传结果时，会把消息阶段改为
`STAGE_F2A`（combine，§5.1）。

**当前实现状态。** Attention 完整 batch 同步逐层事件循环、FFN 跨层动态事件循环、runner 分段执行、
动态 P2P 接收以及 credit 背压都已接通，并通过 CPU 协议和执行单测。FX continuation program
驱动每个片段按 bucket capture/replay CUDA graph；静态输入地址复用和 eager fallback 已有单测。
CUDA graph kernel capture 和 NCCL 多卡性能仍需要在 GPU 硬件上验证。

**计算和通信可以同时进行。** 计算使用默认 CUDA 流，`isend` 和 `irecv` 使用独立的通信流。
计算完成后记录一个 CUDA event，通信流等到该 event 后再发送数据。FFN 可以一边发送第 `i` 层的结果，
一边计算其他已就绪层；Attention Phase 0 则等待当前完整 batch 返回后再推进。connector 只在图外由
事件循环调用，计算片段内没有通信操作。

**credit 用来限制队列长度。** 如果没有上限，较快的一侧会不断向较慢的一侧发送数据，最终占满显存。

- Attention Phase 0 每个 rank 只有一个可用 credit。每发出一个完整 batch 的 `A→F` 就占用该
  credit，收到对应的 `F→A` 后再归还；归还前不能发送其他 batch。

**单个节点故障不会卡住其他节点。** `poll()` 本身不阻塞，队列又按 `attn_rank` 分开。某个 Attention
节点掉线时，只有它对应的队列停止前进并最终超时回收；其他 rank 仍可继续收发和计算。

**CUDA graph 只包含固定形状的计算。** `replay_graph` 属于可捕获的计算子图（§4）；`poll()`、
`isend()`、补齐 bucket 和选层仍是图外的 Python 或通信逻辑。两部分通过 `AFDHandle` 连接。因此，
运行时 batch 形状发生变化，也不会破坏 CUDA graph 捕获。

---

## 4. 与编译 / CUDA graph 的交互

AFD 把计算和通信分开处理：CUDA graph 只记录固定形状的计算，收发数据和动态调度都留在图外。

`vllm::afd_cut` 会被加入 `splitting_ops`，告诉编译器“从这里切开”。它本身只是一个标记，输入什么就
返回什么，不会读取 connector，也不会收发或轮询数据。因此，编译器可以安全地在这里把完整计算图拆成
多个 FX 计算片段。

普通的 `split_graph()` 负责按标记切图。AFD 使用 `partition_afd_graph()` 再做一步处理：去掉切点
标记，并生成事件循环可以分别调用的计算片段。这样，Attention 算完后可以暂停，等远端 MoE 结果回来
再继续，而不需要重新运行前面的计算。

CUDA graph 的使用方式如下：

- 每个角色只捕获自己负责的计算，不把网络通信放进图中。
- batch size 会补齐到 `cudagraph_capture_sizes` 中的固定大小。固定形状可以让同一张图反复使用。
- 结构相同的 MoE 层可以共用同一组图，运行时通过 `layer_id` 选择对应的层。
- `poll()`、选层、补齐 batch、`isend()` 和超时处理仍由图外的 Python 事件循环完成。

因此，CUDA graph 不会跨节点，也不会在 replay 时重复执行 Python 或 connector 操作。即使每次收到的
batch 大小不同，补齐后的形状仍来自固定集合，所以已经捕获的图不会失效。

目前 CPU 测试已经覆盖切点保留、暂停后继续执行、并发状态、远端输出格式、事件循环推进、bucket
dispatch 和 CUDA graph 静态输入地址复用。实际 kernel capture 以及多卡环境下的 replay，仍需在
GPU 硬件上验证。

---

## 5. 与 DP / EP 的交互

- **Attention 侧继续使用 DP。** 每个 DP rank 处理自己的请求，KV cache 的分片方式保持不变。
- **FFN 侧使用 EP。** 不同专家分布在不同的 EP rank 上，由 `FusedMoE` 负责分片和计算。

### 5.1 A↔F 传输如何融进 EP all-to-all（Phase 2）

MoE 使用 EP 时，一层中通常有两次 all-to-all 通信：

1. **dispatch：** gate 为每个 token 选出 top-k 专家，再把 token 发送到这些专家所在的 rank。一个
   token 可能因此被复制 k 份。
2. **combine：** 专家完成计算后，把结果送回 token 原来的 rank，并把 k 份结果合并成一份。

Phase 2 的目标是直接利用这两次通信完成 Attention 和 FFN 之间的数据传输：

- **A→F 与 dispatch 合并。** Attention rank 不再先把 `hidden_states` 发给 FFN 的入口 rank，而是
  直接通过 dispatch all-to-all 发给对应的专家 rank。
- **F→A 与 combine 合并。** 专家计算完成后，combine 直接把结果送回原来的 Attention rank。
  §3.2 中 `AFDMeta(layer, attn_rank)` 的 `attn_rank` 就是这个返回地址。

不做融合时，数据要经过四个步骤：A→F、dispatch、combine、F→A。FFN 的入口 rank 只负责转发数据，
还会带来 #25162 提到的冗余 `all_gather`。融合后只剩两个步骤：A→F 与 dispatch 一起完成，combine
与 F→A 一起完成。这也是 §2.3 为 connector 预留 `all_to_all` 原语的原因。Phase 0 先使用独立的 P2P
传输；Phase 2 再把它们并入 EP all-to-all。

这项优化需要先解决三个问题，因此放在 Phase 2：

1. **Attention 侧必须提前知道 token 要去哪个专家。** dispatch 开始前必须已经有 gate 的 top-k
   结果。可以把 gate 计算移到 Attention 侧，也可以让 `AFDMeta` 携带 `(expert_ids, weights)`。
   但这两种做法都会改变 §2.1 中只传 `hidden_states` 的简单边界。
2. **需要处理 DP 和 EP rank 数不一致的情况。** Attention 侧有 `N_attn` 个 rank，FFN 侧有
   `N_ffn` 个 rank，两边不一定一一对应。需要通过 TP/DP/EP 组合测试验证数据能发到正确位置
   （见 §6 Phase 2）。
3. **需要权衡通信量。** 融合后，A→F 会直接发送 top-k 产生的 k 份 token，而不是只发送一份；
   但它省掉了 FFN 池内单独的 dispatch 和中转 rank。也就是说，复制没有消失，减少的是一次中转和
   一次独立的 shuffle。

---

## 6. 实施计划

### 6.1 Phase 0 — 基线功能

Phase 0 先让整个系统正常运行起来，包括角色参数、纯 `afd_cut`、FX continuation partition、图外
Attention/FFN 事件循环、connector 数据投递和 FFN 跨层乱序动态批调度；不在 model forward 内通信。
Attention 每次接收一个完整 scheduler batch，并按
`A0→F0→A1→F1→…` 同步逐层推进：本层 FFN 结果全部返回前不拆 batch、不执行其他层或其他 batch。

#### 6.1.1 测试例子：标准 residual decoder MoE

**(a) 在同一个 `torch.distributed` WORLD 中拉起双池（P2P 连接器）：**

> 下面命令展示两个 rank 的角色参数，必须由能建立共享 WORLD 的部署 launcher 启动；两个互不相关的
> 单进程 `vllm serve` 不会自动组成 P2P 进程组。Phase 0 拓扑固定为 1 Attention ↔ 1 FFN。

```bash
MODEL=/data/public_models/DeepSeek-V3
# 约定：ffn 池为 rank 0、attention 池为 rank 1，peer_rank 指向对端
CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL" --port 9000 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --afd-config '{"role":"ffn","afd_connector":"P2PAFDConnector",
    "afd_connector_extra_config":{"peer_rank":1}}' &

# Attention 池：真实注意力 + KV；纯 boundary 产出 MoE 输入，event loop 发送到 FFN 池
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL" --port 9001 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --afd-config '{"role":"attention","afd_connector":"P2PAFDConnector",
    "afd_connector_extra_config":{"peer_rank":0}}'

# 冒烟：打 attention 池（9001）；一次补全应正确返回（残差流经双池 ping-pong 往返，§3）
curl -s http://localhost:9001/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"9.11 and 9.9, which is bigger?","max_tokens":32}'
```

**(b) `nsys` 系统断言（FFN 池无注意力 kernel）：**

```bash
export TMPDIR=/data1/shuiquan.lin/nsys/tmp
nsys profile -o afd_ffn --trace=cuda --duration=180  {vLLM}

# 生成并查看 sqlite 报告
nsys stats afd_ffn.nsys-rep                              # 总览
nsys stats --report cuda_gpu_kern_sum afd_ffn.nsys-rep   # CUDA kernel 排名
nsys stats --report cuda_gpu_kern_sum afd_ffn.sqlite \
  | grep -Ei 'attn|attention|flash|paged|mla|csa|hca'    # 断言：FFN 池应无匹配
```

### 6.2 Phase 1 — Attention DBO 或者跨层乱序动态调度

Phase 1 属于未来优化项，Attention 使用 DBO 或者跨层乱序动态调度重叠计算

### 6.3 Phase 2 — EP 融合

Phase 2 属于未来优化项：把 A→F 传输折进 EP all-to-all。

---

## 7. 先例

**机制本身**（FX/图级自动切、零模型改动）已被 PyTorch **PiPPy → `torch.distributed.pipelining`**
证明并产品化，其宣称目标就是把"model code as-is … without heavyweight modifications"并行化。但
pipelining 在层*之间*切（粗粒度、单向），AFD 需要*层内、双向*，那些 API 不提供。

**"在注意力算子处自动切以避免侵入式改动"这一具体想法**出现在研究中（Model-Attention Disaggregation，
arXiv 2405.01814），用符号执行在注意力 op 处切片——最接近的公开先例，但是研究原型、不是服务框架。

生产级 AFD 系统（StepFun Step-3、ByteDance MegaScale-Infer、华为 xDeepServe）全部手工切。**没有人在
服务框架内落地过切图式 AFD。** 因此本设计是把一个已被验证的原语应用到一个未验证位置的综合。
