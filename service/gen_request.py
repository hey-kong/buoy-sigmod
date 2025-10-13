import json
import random
import string

from tqdm import tqdm

jsonl_file = "traces/qwen_traceA_blksz_16.jsonl"

def random_word():
    return random.choice(string.ascii_lowercase)


def random_chunk(num_words):
    return ' '.join(random_word() for _ in range(num_words))


def build_chunk_map(jsonl_file):
    chunk_size = 256
    chunk_map = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)
            input_len = item["input_length"]
            hash_ids = item["hash_ids"]

            full_chunks = input_len // chunk_size
            remainder = input_len % chunk_size

            for idx, h in enumerate(hash_ids):
                if h not in chunk_map:
                    if idx < full_chunks:
                        num_words = chunk_size
                    elif idx == full_chunks and remainder > 0:
                        num_words = remainder
                    else:
                        num_words = chunk_size
                    chunk_map[h] = random_chunk(num_words)
    return chunk_map


def save_chunk_map(jsonl_file, dst_file):
    chunk_map = build_chunk_map(jsonl_file)
    json.dump(chunk_map, open(dst_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def build_requests(jsonl_file, chunk_map):
    requests = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for rid, line in enumerate(f):
            item = json.loads(line)
            chunks = [chunk_map[h] for h in item["hash_ids"]]
            request_str = " ".join(chunks)
            requests.append(request_str)
    return requests
