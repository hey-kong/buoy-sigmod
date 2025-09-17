import argparse
import json
from tqdm import tqdm

from gen_request import build_chunk_map
from model_runner import ModelRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn conversation test for 3-level cache eviction policies")
    parser.add_argument("--input", type=str, default="traces/qwen_traceB_blksz_16.jsonl", help="Input file path")
    parser.add_argument("--output", type=str, default="stat.jsonl", help="Statistics file path")
    parser.add_argument("--model", type=str, default="/data/llm/Llama-3.1-8B-Instruct", help="Model name or local path")
    parser.add_argument("--hbm", type=int, default=12, help="HBM size in GB")
    parser.add_argument("--dram", type=int, default=32, help="DRAM size in GB")
    parser.add_argument("--ssd", type=int, default=512, help="SSD size in GB")
    parser.add_argument("--policy", type=str, required=True, help="Eviction policy")
    args = parser.parse_args()

    chunk_map = build_chunk_map(args.input)

    runner = ModelRunner(
        model=args.model,
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

            ttft, token_cnt, hit_string = runner.process_request(
                chat_id=chat_id,
                hash_ids=hash_ids
            )
            stat_item = {
                "chat_id": chat_id,
                "ttft": ttft,
                "token_count": token_cnt,
                "hit_string": hit_string
            }
            stat_f.write(json.dumps(stat_item, ensure_ascii=False) + "\n")
            stat_f.flush()
