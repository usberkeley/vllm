# vllm-frontend-rs

This is a Rust drop-in alternative frontend for vLLM. The current goal is to rebuild the northbound serving layer in Rust while still talking to the core Python vLLM engine process(es) via ZMQ over the existing engine boundary.

It should still be considered experimental, and is not feature-complete. We are working to add more functionality from the python front-end.

See <https://github.com/Inferact/vllm-frontend-rs> for the original commit history before it was moved into the main vllm repo.

## Architecture

The component is organized as a Cargo workspace with several crates, layered bottom-up:

```text
┌─────────────────────────────────┐
│  vllm-cmd / vllm-rs             │  CLI entrypoint:
│                                 │  Python vLLM frontend subprocess
│                                 │  Rust managed-engine serve mode
│                                 │  Engine-free render mode
├─────────────────────────────────┤
│  vllm-server                    │  OpenAI-compatible HTTP API (axum)
├─────────────────────────────────┤
│  vllm-chat                      │  Chat completions: template rendering,
│                                 │  structured assistant events,
│                                 │  reasoning & tool parsing
├─────────────────────────────────┤
│  vllm-text                      │  Tokenizer & incremental detokenizer
├─────────────────────────────────┤
│  vllm-llm                       │  Thin token-in/token-out facade over
│                                 │  the engine client
├─────────────────────────────────┤
│  vllm-engine-core-client        │  ZMQ transport + MessagePack protocol
│                                 │  for the headless vLLM engine
└─────────────────────────────────┘
```

`vllm-rs` integrates into Python `vllm` as a Rust frontend subprocess.
Python owns process startup and launches the Rust API server as a Python-supervised worker, while
passing the inherited listening socket and transport addresses into `vllm-rs`.

For example:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B
```

### External Engine

`vllm-rs serve` can be run standalone with `--data-parallel-size-local 0` when the Python engines
are started elsewhere and this node should run only the Rust frontend. The frontend still uses
the global `--data-parallel-size` to determine how many engines it expects to join the shared handshake.

```bash
vllm serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1
```

Then start the Rust frontend-only server:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 0
```

To build the `vllm-rs` in isolation:

```bash
# from the local checkout
./build_rust.sh
```

### Engine-free renderer

`vllm-rs render` serves text-only request preprocessing without starting or
connecting to a Python inference engine:

```bash
cargo run --manifest-path rust/Cargo.toml -p vllm-cmd --release -- \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768
```

It exposes `/v1/chat/completions/render` and `/v1/completions/render`. Only
tokenizer and model configuration files are loaded; model weights, PyTorch,
and vLLM kernels are not required.

To serve HTTPS, pass `--ssl-certfile`. If the certificate and private key are
in separate files, pass `--ssl-keyfile` for the key; otherwise the key is read from
the certificate file. For mTLS, pass `--ssl-ca-certs` with a CA
bundle and set `--ssl-cert-reqs` to `1` (optional) or `2` (required) to
verify client certificates. Use `--ssl-ciphers` to override the default
OpenSSL cipher list.

```bash
# Combined PEM (cert + key in one file):
vllm-rs \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768 \
  --ssl-certfile /path/to/combined.pem

# Separate cert and key files:
vllm-rs \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768 \
  --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

The render endpoints return the public token-in `GenerateRequest` consumed by
the Rust `/inference/v1/generate` endpoint. A chat render response, or one item
from a completion render response, can be submitted to that endpoint without
changing its fields.

The render and inference paths use the same `vllm-chat` and `vllm-text`
request-preparation logic; render mode stops before engine submission.
Tool-call and reasoning parsers use model-based auto-detection by default. Use
`--tool-call-parser` and `--reasoning-parser` to override either selection;
unified parsers require the same selection for both options.

```bash
curl http://127.0.0.1:8000/v1/chat/completions/render \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 16
  }'
```

### Example Request

After either full-frontend startup path, you can use any OpenAI-compatible
client against the inference endpoints:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```

### Pooling APIs

The Rust frontend exposes the following pooling routes on top of
`TextLlm::encode_batch` and `Llm::encode`:

| Route | Result |
| --- | --- |
| `/pooling` | Sequence or token pooling tensors, preserving their dimensions |
| `/classify` | Class probabilities, number of classes and optional model label |

Pooling and classification accept a text string, a batch of strings,
token IDs or a batch of token IDs in `input`. They support prompt truncation,
LoRA model selection, priority and cache salt. Embedding dimensions and pooler
activation defaults are resolved by engine-core. `/pooling` accepts `embed`,
`classify`, `token_embed` and `token_classify`; when omitted, the first supported
task in that order is selected.

```bash
curl http://127.0.0.1:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"input": "hello", "task": "embed"}'

curl http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"input": "a useful result"}'
```

The configured model must support the requested task.

For multimodal pooling, `/v1/embeddings`, `/pooling` and `/classify` accept a
conversation in `messages` (or a conversation in `input`). The model's chat
renderer and media processor prepare the expanded tokens and features together.
`chat_template`, `chat_template_kwargs` and `add_generation_prompt` control
conversation rendering; `add_generation_prompt` defaults to false.

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    ]
  }]
}
```

An `input` object with `content` is treated as one user message. A batch can mix
these content objects and text strings. Supported media depend on the loaded
Rust model processor, as with chat. Precomputed image embeddings are not accepted.
`truncate_prompt_tokens` is rejected for media inputs to preserve media positions.

Batched chat conversations, IOProcessor plugins, composite outputs,
padding, non-float32 output dtypes and binary HTTP responses are not supported.
Unsupported request options return an error rather than being silently ignored.

### Scoring

Pooling models are served at `/score` (and `/v1/score` for compatibility).
One query is scored against every document, and a query list of the same
length is paired positionally instead.

```bash
curl http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-base",
    "text_1": "What is the capital of France?",
    "text_2": ["Paris is the capital of France.", "Berlin is in Germany."]
  }'
```

`/rerank` ranks the documents instead of returning them in request order, and
answers with the JinaAI rerank shape: each result keeps the `index` it had in
the request, echoes its `document`, and carries a `relevance_score`. Use
`top_n` to keep only the highest-scoring documents; token usage still covers
every document that was scored. `/v1/rerank` and `/v2/rerank` are aliases.

```bash
curl http://127.0.0.1:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-base",
    "query": "What is the capital of France?",
    "documents": ["Berlin is in Germany.", "Paris is the capital of France."],
    "top_n": 1
  }'
```

How a pair becomes a score depends on what the model's pooler supports.
Cross-encoders (`classify`) read the pair as one prompt joined by the
tokenizer's pair template and return the classifier output. Embedding models
(`embed`) encode each side separately and the frontend takes the cosine
similarity; only that path accepts pre-tokenized inputs, since a cross-encoder
needs the text to apply its pair template. Late-interaction scoring is not
supported yet.

For bi-encoder 1:N scoring, the query is encoded once and its embedding is
reused across documents. Usage still counts the query tokens for every pair.

Both scoring endpoints also accept `{"content": [...]}` on either side, with
the same image, video and audio URL/content parts as chat. Document batches can
mix strings and content objects. Rerank echoes structured documents in
`document.multi_modal`.

Bi-encoders tokenize each side independently and reuse query preprocessing and
embeddings for 1:N requests. Cross-encoders apply the tokenizer's pair template
before expanding media placeholders; the query/document token-type boundary
is adjusted to the expanded tokens. Models requiring a scoring prompt template
can use an explicit `chat_template`, with `query` and `document` message roles,
plus `chat_template_kwargs` or `instruction`. This template option applies to
cross-encoders. Model-specific scoring templates are not selected automatically.
Multimodal scoring also rejects `truncate_prompt_tokens`.

```json
{
  "queries": "A red car",
  "documents": [{
    "content": [{
      "type": "image_url",
      "image_url": {"url": "https://example.com/car.png"}
    }]
  }]
}
```
