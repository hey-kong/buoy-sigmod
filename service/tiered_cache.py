import abc
import os
import shutil
from typing import List, Optional

from transformers.cache_utils import DynamicCache

import kvcache_io
import ttft_timer


class TrieNode:
    def __init__(
            self,
            chunk_id_hash: Optional[str],
            parent = None,
        ):
        self.chunk_id_hash = chunk_id_hash
        self.parent = parent
        self.children = {}
        self.linked_node = None
        self.location = None
        self.cache = None
        self.has_cache = False
        self.kv_bytes = 0


class LinkedNode:
    def __init__(self, trie_node=None):
        self.prev = None
        self.next = None
        self.trie_node: Optional[TrieNode] = trie_node


class TieredCache(abc.ABC):
    def __init__(
            self,
            device: str,
            bytes_per_token: int,
            ssd_path: str,
            hbm_capacity: int,
            dram_capacity: int,
            ssd_capacity: int
        ):
        self.tiers = ['hbm', 'dram', 'ssd']
        self.device = device
        self.bytes_per_token = bytes_per_token
        if os.path.exists(ssd_path):
            shutil.rmtree(ssd_path)
        os.makedirs(ssd_path)
        self.ssd_path = ssd_path
        self.max_bytes = {
            'hbm': hbm_capacity,
            'dram': dram_capacity,
            'ssd': ssd_capacity
        }
        self.cur_bytes = {tier: 0 for tier in self.tiers}
        self.node_map = {}
    
    def _move_data(self, trie_node: TrieNode, dest_tier: str, modify_bytes: bool = True):
        source_tier = trie_node.location

        if source_tier == dest_tier:
            return

        trie_node.location = dest_tier
        if modify_bytes:
            self.cur_bytes[dest_tier] += trie_node.kv_bytes
        if source_tier not in self.tiers or trie_node.kv_bytes == 0:
            return
        if modify_bytes:
            self.cur_bytes[source_tier] -= trie_node.kv_bytes
        if trie_node.has_cache is False:
            return
        
        path = os.path.join(self.ssd_path, trie_node.chunk_id_hash + '.pt')
        if source_tier == 'hbm':
            with ttft_timer.without_timing():
                buffer = kvcache_io.alloc_cpu_buffer(trie_node.cache)
            kvcache_io.move_cache_to_cpu(trie_node.cache, buffer)
            with ttft_timer.without_timing():
                kvcache_io.pin_kvcache(trie_node.cache)
        if dest_tier == 'ssd':
            kvcache_io.save_kvcache(trie_node.cache, path)
            trie_node.cache = None
        if source_tier == 'ssd':
            trie_node.cache = kvcache_io.load_kvcache(path)
            with ttft_timer.without_timing():
                kvcache_io.pin_kvcache(trie_node.cache)
        if dest_tier == 'hbm':
            kvcache_io.move_cache_to_gpu(trie_node.cache, self.device)
    
    def _remove_data(self, trie_node: TrieNode):
        tier = trie_node.location
        assert tier in self.tiers
        if tier == 'ssd':
            path = os.path.join(self.ssd_path, trie_node.chunk_id_hash + '.pt')
            if os.path.exists(path):
                os.remove(path)
        else:
            trie_node.cache = None
        self.cur_bytes[tier] -= trie_node.kv_bytes
        trie_node.has_cache = False
        trie_node.location = None
    
    def _make_cache_list(self, nodes: List[TrieNode]) -> List[Optional[DynamicCache]]:
        copied: List[Optional[DynamicCache]] = []

        for n in nodes:
            if not n.has_cache:
                copied.append(None)
                continue

            assert n.location in self.tiers
            cache = n.cache

            if n.location == 'ssd':
                path = os.path.join(self.ssd_path, n.chunk_id_hash + '.pt')
                cache = kvcache_io.load_kvcache(path)
                with ttft_timer.without_timing():
                    kvcache_io.pin_kvcache(cache)
            
            new_cache = DynamicCache()
            for i, layer in enumerate(cache.layers):
                if n.location == 'hbm':
                    k = layer.keys.detach().clone()
                    v = layer.values.detach().clone()
                    new_cache.update(k, v, i)
                else:
                    k = layer.keys.detach().clone().to(self.device)
                    v = layer.values.detach().clone().to(self.device)
                    new_cache.update(k, v, i)
            copied.append(new_cache)
        
        return copied
    
    def store(self, chunk_id_hash: str, kv_data: DynamicCache):
        node: TrieNode = self.node_map.get(chunk_id_hash, None)
        if node is None:
            return
        assert node.kv_bytes > 0 and not node.has_cache
        node.cache = kv_data
        node.has_cache = True

        with ttft_timer.without_timing():
            if node.location in ['dram', 'ssd']:
                buffer = kvcache_io.alloc_cpu_buffer(node.cache)
                kvcache_io.move_cache_to_cpu(node.cache, buffer)
                kvcache_io.pin_kvcache(node.cache)
            if node.location == 'ssd':
                path = os.path.join(self.ssd_path, chunk_id_hash + '.pt')
                kvcache_io.save_kvcache(node.cache, path)
                node.cache = None
    
    def print_status(self, chat_id: int):
        print(f'\n{chat_id + 1} chat requests processed, cache status:')
        for tier in self.tiers:
            print(f'{tier}: {self.cur_bytes[tier] / 2**30} GB/{self.max_bytes[tier] / 2**30} GB')

    @abc.abstractmethod
    def on_access(self, chunk_id_hashes: List[str], token_counts: List[int]) -> List[TrieNode]:
        raise NotImplementedError
