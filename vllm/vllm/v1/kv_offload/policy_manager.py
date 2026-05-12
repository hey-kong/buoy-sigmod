# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable
from typing import Literal

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.ghost import Ghost

EvictionPolicy = Literal["lru", "fifo", "lfu", "s3fifo", "buoy"]


class PolicyOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable backend and configurable eviction.

    The manager supports LRU, FIFO, LFU, and S3-FIFO eviction. Ties in LFU
    are broken by insertion order. S3-FIFO uses FIFO eviction in the backing
    store and maintains a bounded ghost queue of recently evicted block hashes.
    Buoy is a quick-demotion policy that uses a reinsert FIFO for DRAM pages
    without an HBM copy, a ghost queue, and a dynamic prefix-depth threshold
    to avoid storing cold deep-prefix pages.
    """

    def __init__(
        self,
        backend: Backend,
        enable_events: bool = False,
        eviction_policy: EvictionPolicy = "lru",
        enable_hbm_residency_tracking: bool = False,
    ):
        if eviction_policy not in ("lru", "fifo", "lfu", "s3fifo", "buoy"):
            raise ValueError(f"Unknown eviction policy: {eviction_policy}")
        self.backend: Backend = backend
        self.eviction_policy = eviction_policy
        # block_hash -> BlockStatus for all blocks stored in the backing medium.
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        # When quick demotion is enabled, keep two recency chains:
        # - blocks_without_hbm_copy (A): DRAM pages with no HBM copy; evictable.
        # - blocks_with_hbm_copy (B): DRAM pages with an HBM copy; protected.
        self.enable_hbm_residency_tracking = enable_hbm_residency_tracking
        self.blocks_without_hbm_copy: OrderedDict[BlockHash, None] = OrderedDict()
        self.blocks_with_hbm_copy: OrderedDict[BlockHash, None] = OrderedDict()
        self.visited: dict[BlockHash, bool] = {}
        self.access_counts: dict[BlockHash, int] = {}
        self.ghost_capacity = getattr(backend, "num_blocks", 0) * (
            4 if eviction_policy == "buoy" else 1
        )
        self.ghost: Ghost[BlockHash] = Ghost(self.ghost_capacity)
        self.depth_threshold = 0.0
        self._depth_threshold_samples = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        block_hashes = list(block_hashes)
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for block_hash in reversed(list(block_hashes)):
            if block_hash not in self.blocks:
                continue
            if self.eviction_policy == "lru":
                self.blocks.move_to_end(block_hash)
                self._move_residency_chain_to_end(block_hash)
            elif self.eviction_policy == "lfu":
                self.access_counts[block_hash] = (
                    self.access_counts.get(block_hash, 0) + 1
                )

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def mark_blocks_loaded_to_hbm(self, block_hashes: Iterable[BlockHash]):
        if not self.enable_hbm_residency_tracking:
            return
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                continue
            self.blocks_without_hbm_copy.pop(block_hash, None)
            self.visited.pop(block_hash, None)
            self.blocks_with_hbm_copy[block_hash] = None

    def mark_blocks_evicted_from_hbm(self, block_hashes: Iterable[BlockHash]):
        if not self.enable_hbm_residency_tracking:
            return
        for block_hash in block_hashes:
            if block_hash not in self.blocks:
                self.blocks_with_hbm_copy.pop(block_hash, None)
                self.blocks_without_hbm_copy.pop(block_hash, None)
                self.visited.pop(block_hash, None)
                continue
            if block_hash in self.blocks_with_hbm_copy:
                self.blocks_with_hbm_copy.pop(block_hash)
                self.blocks_without_hbm_copy[block_hash] = None
                if self.eviction_policy == "buoy":
                    self.visited[block_hash] = True

    def _move_residency_chain_to_end(self, block_hash: BlockHash):
        if not self.enable_hbm_residency_tracking:
            return
        if block_hash in self.blocks_with_hbm_copy:
            self.blocks_with_hbm_copy.move_to_end(block_hash)
        elif block_hash in self.blocks_without_hbm_copy:
            self.blocks_without_hbm_copy.move_to_end(block_hash)

    def _drop_residency_state(self, block_hash: BlockHash):
        self.blocks_with_hbm_copy.pop(block_hash, None)
        self.blocks_without_hbm_copy.pop(block_hash, None)
        self.visited.pop(block_hash, None)

    def _get_blocks_to_evict(self, num_blocks_to_evict: int) -> list[BlockHash] | None:
        if num_blocks_to_evict <= 0:
            return []

        eviction_order = (
            self.blocks_without_hbm_copy
            if self.enable_hbm_residency_tracking
            else self.blocks
        )

        if self.eviction_policy == "buoy" and self.enable_hbm_residency_tracking:
            num_evictable = sum(
                1
                for block_hash in self.blocks_without_hbm_copy
                if self.blocks[block_hash].ref_cnt == 0
            )
            if num_evictable < num_blocks_to_evict:
                return None

            to_evict: list[BlockHash] = []
            max_iterations = len(self.blocks_without_hbm_copy) * 2
            iterations = 0
            while len(to_evict) < num_blocks_to_evict:
                if not self.blocks_without_hbm_copy or iterations >= max_iterations:
                    return None
                iterations += 1
                block_hash = next(iter(self.blocks_without_hbm_copy))
                if self.blocks[block_hash].ref_cnt != 0:
                    self.blocks_without_hbm_copy.move_to_end(block_hash)
                    continue
                if self.visited.get(block_hash, False):
                    self.visited[block_hash] = False
                    self.blocks_without_hbm_copy.move_to_end(block_hash)
                    continue
                to_evict.append(block_hash)
                self.blocks_without_hbm_copy.pop(block_hash)
                self.visited.pop(block_hash, None)
                max_iterations = len(self.blocks_without_hbm_copy) * 2
                iterations = 0
            return to_evict

        evictable = [
            block_hash
            for block_hash in eviction_order
            if self.blocks[block_hash].ref_cnt == 0
        ]
        if len(evictable) < num_blocks_to_evict:
            return None

        if self.eviction_policy == "lfu":
            evictable.sort(key=lambda block_hash: self.access_counts.get(block_hash, 0))
        # OrderedDict order is FIFO for FIFO/S3-FIFO/Buoy and LRU order for LRU.
        return evictable[:num_blocks_to_evict]

    def prepare_store(
        self,
        block_hashes: Iterable[BlockHash],
        *,
        start_depth: int = 0,
        request_hit_block_count: int | None = None,
    ) -> PrepareStoreOutput | None:
        block_hashes = list(block_hashes)
        ghost_hit_block_hashes = {
            block_hash
            for block_hash in block_hashes
            if block_hash not in self.blocks and block_hash in self.ghost
        }
        if self.eviction_policy == "buoy" and request_hit_block_count is not None:
            self._update_depth_threshold(
                request_hit_block_count + len(ghost_hit_block_hashes)
            )

        # filter out blocks that are already stored or that Buoy chooses not to
        # materialize in DRAM because they are cold pages deeper than the
        # dynamic prefix-depth threshold.
        ghost_only_block_hashes: set[BlockHash] = set()
        block_hashes_to_store = []
        for idx, block_hash in enumerate(block_hashes):
            if block_hash in self.blocks:
                continue
            depth = start_depth + idx
            if (
                self.eviction_policy == "buoy"
                and block_hash not in ghost_hit_block_hashes
                and not self.ghost.is_empty()
                and depth >= self.depth_threshold
            ):
                self._put_ghost(block_hash)
                ghost_only_block_hashes.add(block_hash)
                continue
            block_hashes_to_store.append(block_hash)

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        to_evict = self._get_blocks_to_evict(num_blocks_to_evict)
        if to_evict is None:
            return None

        # evict blocks
        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))
            self.access_counts.pop(block_hash, None)
            self._drop_residency_state(block_hash)
            self._put_ghost(block_hash)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        ghost_hit_block_hashes &= set(block_hashes_to_store)

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block
            if self.enable_hbm_residency_tracking:
                self.blocks_without_hbm_copy[block_hash] = None
                if self.eviction_policy == "buoy":
                    self.visited[block_hash] = block_hash in ghost_hit_block_hashes
            self.access_counts[block_hash] = 1

        # build store specs for allocated blocks
        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
            ghost_hit_block_hashes=ghost_hit_block_hashes,
            ghost_only_block_hashes=ghost_only_block_hashes,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]
                    self.access_counts.pop(block_hash, None)
                    self._drop_residency_state(block_hash)

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def _update_depth_threshold(self, hit_block_count: int) -> None:
        self._depth_threshold_samples += 1
        self.depth_threshold += (
            hit_block_count - self.depth_threshold
        ) / self._depth_threshold_samples

    def _put_ghost(self, block_hash: BlockHash) -> None:
        if self.eviction_policy not in ("s3fifo", "buoy") or self.ghost_capacity <= 0:
            return
        self.ghost.put(block_hash)

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
