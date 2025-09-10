import os
import json
import math


def merge_trace_chunk(base_name, orig_chunk_size, target_chunk_size):
    if target_chunk_size % orig_chunk_size != 0:
        raise ValueError("target_chunk_size must be a multiple of orig_chunk_size.")

    input_file = f'traces/{base_name}_{orig_chunk_size}.jsonl'
    output_file = f'tmp_traces/{base_name}_{target_chunk_size}.jsonl'
    chunk_group = target_chunk_size // orig_chunk_size

    chunk_to_id = {}
    next_id = 0

    def get_chunk_tuple(ids, i, group_size):
        return tuple(ids[i:i + group_size])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line)
            hash_ids = item['hash_ids']
            new_hash_ids = []
            i = 0
            while i < len(hash_ids):
                group_size = min(chunk_group, len(hash_ids) - i)
                chunk = get_chunk_tuple(hash_ids, i, group_size)
                if chunk not in chunk_to_id:
                    chunk_to_id[chunk] = next_id
                    next_id += 1
                new_hash_ids.append(chunk_to_id[chunk])
                i += chunk_group
            item['hash_ids'] = new_hash_ids
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_trace_chunk(base_name, orig_chunk_size, target_chunk_size):
    if orig_chunk_size % target_chunk_size != 0:
        raise ValueError("orig_chunk_size must be a multiple of target_chunk_size.")

    input_file = f'traces/{base_name}_{orig_chunk_size}.jsonl'
    output_file = f'tmp_traces/{base_name}_{target_chunk_size}.jsonl'

    chunk_to_id = {}
    next_id = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line)
            hash_ids = item['hash_ids']
            input_length = item['input_length']
            new_hash_ids = []

            num_full_chunks = input_length // orig_chunk_size
            last_chunk_size = input_length % orig_chunk_size
            if last_chunk_size > 0:
                total_chunks = num_full_chunks + 1
            else:
                total_chunks = num_full_chunks

            for idx, hid in enumerate(hash_ids):
                if idx == total_chunks - 1 and last_chunk_size > 0:
                    this_chunk_size = last_chunk_size
                else:
                    this_chunk_size = orig_chunk_size
                small_chunk_num = math.ceil(this_chunk_size / target_chunk_size)
                for offset in range(small_chunk_num):
                    key = (hid, offset)
                    if key not in chunk_to_id:
                        chunk_to_id[key] = next_id
                        next_id += 1
                    new_hash_ids.append(chunk_to_id[key])
            item['hash_ids'] = new_hash_ids
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    merge_trace_chunk("qwen_traceA_blksz", 16, 256)
    merge_trace_chunk("qwen_traceB_blksz", 16, 256)
    split_trace_chunk("mooncake_trace", 512, 256)
