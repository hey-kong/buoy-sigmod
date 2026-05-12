# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id


def test_s3fifo_reinserts_positive_frequency_hbm_pages():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=4,
        prefix_cache_policy="s3fifo",
    )

    candidates = pool.free_block_queue.popleft_n(pool.get_num_free_blocks())
    hot_block, cold_block = candidates[:2]
    hot_block.block_hash = make_block_hash_with_group_id(BlockHash(b"hot"), 0)
    hot_block.access_count = 2
    pool.free_block_queue.append(hot_block)
    pool.free_block_queue.append(cold_block)
    for block in candidates[2:]:
        pool.free_block_queue.append(block)

    allocated = pool.get_new_blocks(1)[0]

    # The hot cached block receives a second chance and an uncached block is
    # allocated first. Its frequency is decremented while it remains cached.
    assert allocated is cold_block
    assert hot_block.access_count == 1
    assert hot_block.block_hash is not None


def test_s3fifo_frequency_is_capped_on_prefix_hits():
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=4,
        prefix_cache_policy="s3fifo",
    )

    block = pool.get_new_blocks(1)[0]
    for _ in range(5):
        pool.touch([block])
        pool.free_blocks([block])

    assert block.access_count == 3
