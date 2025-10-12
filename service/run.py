import argparse
import json
import os
from tqdm import tqdm

from model_runner import ModelRunner
import ttft_timer

DEVICE = 'cuda'
SSD_PATH = '' # fill in your SSD path here
BYTES_PER_TOKEN = 204800 # 131072 for llama3.1-8b, 204800 for mistral-24b

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn conversation test for 3-level cache eviction policies")
    parser.add_argument("--input", type=str, default="tmp_traces/qwen_traceB_blksz_256.jsonl", help="Input file path")
    parser.add_argument("--chunkmap", type=str, default="tmp_traces/qwen_traceB_blksz_256_cm.json", help="Chunk map file path")
    parser.add_argument("--output", type=str, default="stat.jsonl", help="Statistics file path")
    parser.add_argument("--model", type=str, default="/data/llm/Llama-3.1-8B-Instruct", help="Model name or local path")
    parser.add_argument("--hbm", type=int, default=12, help="HBM size in GB")
    parser.add_argument("--dram", type=int, default=32, help="DRAM size in GB")
    parser.add_argument("--ssd", type=int, default=512, help="SSD size in GB")
    parser.add_argument("--policy", type=str, required=True, help="Eviction policy")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(args.chunkmap, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)

    runner = ModelRunner(
        model=args.model,
        device=DEVICE,
        ssd_path=SSD_PATH,
        bytes_per_token=BYTES_PER_TOKEN,
        hbm_size=args.hbm * 1024**3,
        dram_size=args.dram * 1024**3,
        ssd_size=args.ssd * 1024**3,
        policy=args.policy,
        chunk_map=chunk_map
    )

    with open(args.input, "r", encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as stat_f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            chat_id = item["chat_id"]
            hash_ids = item["hash_ids"]
            if item["input_length"] > 16000:
                continue

            with ttft_timer.timer():
                token_cnt, hit_string = runner.process_request(
                    chat_id=chat_id,
                    hash_ids=hash_ids
                )
            stat_item = {
                "chat_id": chat_id,
                "ttft": ttft_timer.elapsed(),
                "token_count": token_cnt,
                "hit_string": hit_string
            }
            stat_f.write(json.dumps(stat_item, ensure_ascii=False) + "\n")
            stat_f.flush()
