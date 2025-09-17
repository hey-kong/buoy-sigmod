import json
import random
import string

jsonl_file = "traces/qwen_traceA_blksz_16.jsonl"

def random_word():
    vocab = string.ascii_lowercase
    length = random.randint(1, 1)
    return ''.join(random.choice(vocab) for _ in range(length))


def random_chunk(num_words):
    return ' '.join(random_word() for _ in range(num_words))


def build_chunk_map(jsonl_file):
    chunk_size = 16
    chunk_map = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
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


def build_requests(jsonl_file, chunk_map):
    requests = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for rid, line in enumerate(f):
            item = json.loads(line)
            chunks = [chunk_map[h] for h in item["hash_ids"]]
            request_str = " ".join(chunks)
            requests.append(request_str)
    return requests


chunk_map = build_chunk_map(jsonl_file)
requests = build_requests(jsonl_file, chunk_map)
