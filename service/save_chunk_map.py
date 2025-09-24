from gen_request import save_chunk_map

file = "tmp_traces/qwen_traceB_blksz_256.jsonl"
dst_file = "traces/qwen_traceB_blksz_256_cm.json"

save_chunk_map(file, dst_file)
