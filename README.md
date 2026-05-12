# Buoy SIGMOD Artifact

This repository contains the artifacts used to evaluate Buoy, a KV-cache
replacement policy for prefix-caching LLM serving. The artifact has two
separate code paths:

1. **Standalone simulator** (`kvcache_cachesim/`): a lightweight Python
   simulator for reproducing cache-hit-rate experiments on JSONL traces. It
   does not run model inference.
2. **Real serving-engine integration** (`vllm/`): a fork of vLLM with native KV
   offloading and configurable KV replacement policies, including Buoy. This is
   the path for end-to-end serving experiments driven by `trace-replayer/`.

The repository also includes the traces used by both paths in `traces/`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `kvcache_cachesim/` | Python cache simulator and policy implementations (`fifo`, `lru`, `lfu`, `slru`, `s3fifo`, `pgdsf`, `wa`, `hotprefix`, `buoy`). |
| `vllm/` | Modified vLLM serving engine with the `--kv-replacement-policy` option for native KV offloading. |
| `trace-replayer/` | Rust client used to replay traces against a running OpenAI-compatible vLLM endpoint. |
| `traces/` | JSONL traces, including `qwen_traceA_blksz_16.jsonl` and `qwen_traceB_blksz_16.jsonl`. |
| `scripts/prepare_vllm_build_deps.sh` | Helper script that downloads pinned build-time dependencies for compiling the vLLM fork. |

## 1. Simulator experiments

Use this path to reproduce cache-hit-rate experiments without launching vLLM.
The simulator consumes traces from `traces/` and reports per-policy statistics
as a Markdown table.

### 1.1 Install the simulator

From the repository root:

```bash
cd kvcache_cachesim
uv venv --python=3.12
source .venv/bin/activate
uv pip install .
```

### 1.2 Run one or more policies

General command:

```bash
python run.py \
  --algo <policy> \
  --trace_path <trace.jsonl> \
  --chunk_size <N> \
  --model_name <hf-model> \
  --dtype <float16|float32|bfloat16|int8> \
  --cap_hbm <GB> \
  --cap_dram <GB> \
  --cap_ssd <GB>
```

`--algo` accepts one or more space-separated policies. Supported simulator
policies are:

```text
fifo lru lfu slru s3fifo pgdsf wa hotprefix buoy
```

Example: evaluate all simulator policies on `qwen_traceA_blksz_16.jsonl` with
Llama-3.1-8B-Instruct, 16 GiB HBM, 64 GiB DRAM, and no SSD tier:

```bash
python run.py \
  --algo fifo lru lfu slru s3fifo pgdsf wa hotprefix buoy \
  --trace_path ../traces/qwen_traceA_blksz_16.jsonl \
  --chunk_size 16 \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dtype float16 \
  --cap_hbm 16 \
  --cap_dram 64 \
  --cap_ssd 0
```

## 2. End-to-end vLLM experiments

Use this path to reproduce serving experiments with a real vLLM engine. The
workflow is:

1. build/install the modified vLLM engine;
2. start `vllm serve` with a selected KV replacement policy;
3. replay a trace with `trace-replayer`; and
4. inspect `<output-path>.summary.json`.

### 2.1 Prepare pinned vLLM build dependencies

The vLLM fork expects local source checkouts for FlashAttention, CUTLASS,
FlashMLA, and Triton kernels. The helper script downloads the same pinned
versions used in our environment and writes an environment file that points
vLLM's build to those checkouts.

From the repository root:

```bash
./scripts/prepare_vllm_build_deps.sh
source .deps/vllm-build/vllm_build_env.sh
```

By default, dependencies are cloned under `.deps/vllm-build/`. To use another
location, pass it as the first argument:

```bash
./scripts/prepare_vllm_build_deps.sh /path/to/vllm-build-deps
source /path/to/vllm-build-deps/vllm_build_env.sh
```

This repository's `vllm/` fork is based on vLLM 0.15.1, and vLLM 0.15.1
corresponds to the pinned dependency revisions below:

| Dependency | Revision |
| --- | --- |
| `vllm-project/flash-attention` | `188be16520ceefdc625fdf71365585d2ee348fe2` |
| `nvidia/cutlass` | `v4.2.1` |
| `vllm-project/FlashMLA` | `c2afa9cb93e674d5a9120a170a6da57b89267208` |
| `triton-lang/triton` | `v3.5.0` |

### 2.2 Install the modified vLLM engine

Use a Python environment suitable for building vLLM on your GPU host. After
sourcing the environment file from the previous step:

```bash
cd vllm
uv pip install --editable .
```

If the shell is restarted before building, source the generated environment file
again so that `VLLM_FLASH_ATTN_SRC_DIR`, `VLLM_CUTLASS_SRC_DIR`,
`FLASH_MLA_SRC_DIR`, and `TRITON_KERNELS_SRC_DIR` are set.

### 2.3 Start vLLM with a replacement policy

Select the policy with `--kv-replacement-policy`. The vLLM integration supports
`lru`, `fifo`, `lfu`, `slru`, `s3fifo`, and `buoy` for native KV offloading.

Example: serve Llama-3.1-8B-Instruct with Buoy replacement and a 64 GiB native KV
offloading buffer:

```bash
vllm serve /path/to/Llama-3.1-8B-Instruct \
  --port 8080 \
  --max-num-seqs 16 \
  --attention-backend FLASH_ATTN \
  --enable-prefix-caching \
  --kv_offloading_backend native \
  --kv_offloading_size 64 \
  --disable-hybrid-kv-cache-manager \
  --kv-replacement-policy buoy
```

To evaluate LRU or another policy, change only the final argument, for example
`--kv-replacement-policy lru`.

### 2.4 Build the trace replayer

Install Rust if it is not already available:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then build the client:

```bash
cd trace-replayer
cargo build \
  -p request-sim \
  --bin client \
  --release \
  -j64
```

### 2.5 Replay a trace against vLLM

With vLLM running on `localhost:8080`, replay `qwen_traceA_blksz_16.jsonl`:

```bash
./target/release/client \
  --tokenizer /path/to/Llama-3.1-8B-Instruct/tokenizer.json \
  --tokenizer-config /path/to/Llama-3.1-8B-Instruct/tokenizer_config.json \
  --endpoint http://localhost:8080/v1/chat/completions \
  --api openai \
  --dataset bailian \
  --dataset-path ../traces/qwen_traceA_blksz_16.jsonl \
  --output-path <output-path> \
  --scale-factor 1.0 \
  --model-name /path/to/Llama-3.1-8B-Instruct \
  --stream \
  --sequential
```

After the replay finishes, the summary metrics are written to:

```text
<output-path>.summary.json
```

Repeat Sections 2.3 and 2.5 for each replacement policy and trace you want to
compare.

## Notes for reviewers

- The simulator and the vLLM engine integration are intentionally separate. The
  simulator is for policy-level hit-rate studies; the `vllm/` fork is the real
  serving-engine implementation used for end-to-end experiments.
- Simulator traces are consumed directly as JSONL files. End-to-end experiments
  use the same trace files through `trace-replayer/` against a live vLLM server.
- Replace `/path/to/Llama-3.1-8B-Instruct` with a local model directory that
  contains the model weights and tokenizer files.
