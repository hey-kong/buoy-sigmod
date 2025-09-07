## Traces

This folder contains multiple production-grade KVCache traces for research on cache management in LLM serving systems.

### 1. [Qwen Trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)

Released by Alibaba Cloud, from the paper: **[KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider](https://www.usenix.org/system/files/atc25-wang-jiahao.pdf)**.

- To-C trace, e.g., ChatGPT-like service ([./qwen_traceA_blksz_16.jsonl](./qwen_traceA_blksz_16.jsonl)).
- To-B trace, e.g., task automation with API calling ([./qwen_traceB_blksz_16.jsonl](./qwen_traceB_blksz_16.jsonl)).

### 2. [Mooncake Trace](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake_trace.jsonl)

Released by Kimi, from the paper: **[Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot](https://www.usenix.org/system/files/fast25-qin.pdf)**.
