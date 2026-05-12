# Prefix KV Cache Simulator

A Python-based simulator for evaluating tiered prefix KV cache eviction algorithms under realistic LLM serving traces. Supports multi-tier cache architectures (HBM, DRAM, SSD) and multiple popular eviction policies.

## Usage

Trace files are stored in the repository-level `traces/` directory. The default trace path points there automatically; when running from `kvcache_cachesim/`, pass custom traces as `../traces/<file>.jsonl`.

Run the simulator via CLI:

```bash
python run.py --algo <policy> --trace_path <trace.jsonl> --chunk_size <N> --model_name <hf-model> --dtype <float16|float32|bfloat16|int8> --cap_hbm <GB> --cap_dram <GB> --cap_ssd <GB>
```

Example:

```bash
python run.py \
    --algo fifo \
    --trace_path ../traces/qwen_traceA_blksz_16.jsonl \
    --chunk_size 16 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --cap_hbm 16 \
    --cap_dram 64 \
    --cap_ssd 0
```
