import json
import argparse

from tqdm import tqdm
from tabulate import tabulate

from kv_calculator import calc_kv_cache_bytes
from eviction.fifo import TieredTrieFIFOCache
from eviction.lru import TieredTrieLRUCache
from eviction.lfu import TieredTrieLFUCache
from eviction.slru import TieredTrieSLRUCache
from eviction.s3fifo import TieredTrieS3FIFOCache
from eviction.pgdsf import TieredPGDSFCache
from eviction.wa import TieredWACache, collect_stats, build_reuse_estimator
from eviction.hotprefix import HotPrefixCache
from eviction.buoy import TieredTrieBuoyCache


def get_requests(trace_path, chunk_size):
    result = []
    with open(trace_path, "r") as f:
        for line in f:
            item = json.loads(line)
            timestamp = item["timestamp"]
            input_length = item["input_length"]
            hash_ids = item["hash_ids"]
            n_chunk = len(hash_ids)
            token_counts = [chunk_size] * (n_chunk - 1)
            last_chunk_len = input_length - chunk_size * (n_chunk - 1)
            token_counts.append(last_chunk_len)
            assert sum(token_counts) == input_length
            result.append((timestamp, hash_ids, token_counts))
    return result


def get_cache(algo, kv_bytes, cap_hbm, cap_dram, cap_ssd, trace_path, chunk_size):
    if algo == "fifo":
        return TieredTrieFIFOCache(kv_bytes, cap_hbm, cap_dram, cap_ssd)
    elif algo == "lru":
        return TieredTrieLRUCache(kv_bytes, cap_hbm, cap_dram, cap_ssd)
    elif algo == "lfu":
        return TieredTrieLFUCache(kv_bytes, cap_hbm, cap_dram, cap_ssd)
    elif algo == "slru":
        return TieredTrieSLRUCache(kv_bytes, cap_hbm, cap_dram, cap_ssd)
    elif algo == "s3fifo":
        return TieredTrieS3FIFOCache(kv_bytes, cap_hbm, cap_dram, cap_ssd, chunk_size)
    elif algo == "pgdsf":
        return TieredPGDSFCache(kv_bytes, cap_hbm, cap_dram, cap_ssd)
    elif algo == "wa":
        freq_counter, mean_gap = collect_stats(trace_path)
        reuse_estimator = build_reuse_estimator(freq_counter)
        return TieredWACache(kv_bytes, cap_hbm, cap_dram, cap_ssd, reuse_estimator, mean_gap)
    elif algo == "hotprefix":
        return HotPrefixCache(kv_bytes, cap_hbm, cap_dram, cap_ssd)
    elif algo == "buoy":
        return TieredTrieBuoyCache(kv_bytes, cap_hbm, cap_dram, cap_ssd, chunk_size)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        nargs="+",
        choices=["fifo", "lru", "lfu", "slru", "s3fifo", "pgdsf", "wa", "hotprefix", "buoy"],
        default=["fifo"],
        help="Eviction algorithm(s) to use (space separated for multiple)",
    )
    parser.add_argument("--trace_path", type=str, default="traces/qwen_traceA_blksz_16.jsonl",
                        help="Path to the trace JSONL file")
    parser.add_argument("--chunk_size", type=int, default=16, help="Chunk size")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name for KV cache calculation")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16", "int8"], default="float16",
                        help="Data type")
    parser.add_argument("--cap_hbm", type=float, default=16, help="HBM capacity in GB")
    parser.add_argument("--cap_dram", type=float, default=64, help="DRAM capacity in GB")
    parser.add_argument("--cap_ssd", type=float, default=0, help="SSD capacity in GB")
    args = parser.parse_args()

    algo_list = args.algo

    kv_bytes = calc_kv_cache_bytes(
        model_name=args.model_name,
        seq_len=1,
        dtype=args.dtype
    )

    cap_hbm = args.cap_hbm * 1024 ** 3
    cap_dram = args.cap_dram * 1024 ** 3
    cap_ssd = args.cap_ssd * 1024 ** 3

    trace_path = args.trace_path
    chunk_size = args.chunk_size

    requests = list(get_requests(trace_path, chunk_size))

    results = []
    for algo in algo_list:
        cache = get_cache(algo, kv_bytes, cap_hbm, cap_dram, cap_ssd, trace_path, chunk_size)
        for timestamp, hash_ids, token_counts in tqdm(requests, desc=algo):
            cache.access_prefix(chunk_ids=hash_ids, token_counts=token_counts)
        results.append(cache.get_stats())

    print(tabulate(results, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()
