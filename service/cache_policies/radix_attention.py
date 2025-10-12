import heapq
from typing import List

from tiered_cache import TrieNode, TieredCache
import ttft_timer


class LRUTrieNode(TrieNode):
    def __init__(self, chunk_id, parent=None):
        super().__init__(chunk_id, parent)

        self.clock = 0

    def __lt__(self, other):
        return self.clock < other.clock


class RadixAttentionCache(TieredCache):
    def __init__(self, device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity):
        super().__init__(device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity)

        self.root = LRUTrieNode(None)
    
    def _collect_leaves_device(self, tier: str) -> List[LRUTrieNode]:
        def is_leaf(tn: LRUTrieNode):
            if tn.location != tier:
                return False
            if tn == self.root:
                return False
            if len(tn.children) == 0:
                return True
            for child in tn.children.values():
                if child.location == tier:
                    return False
            return True
    
        tier_idx = self.tiers.index(tier)
        ret_list = []
        stack = [self.root]
        while stack:
            tn = stack.pop()
            if is_leaf(tn):
                ret_list.append(tn)
            else:
                for cur_child in tn.children.values():
                    if self.tiers.index(cur_child.location) <= tier_idx:
                        stack.append(cur_child)
        return ret_list
    
    def _demote_if_needed(self):
        for tier in self.tiers:
            while self.cur_bytes[tier] > self.max_bytes[tier]:
                leaves = self._collect_leaves_device(tier)
                heapq.heapify(leaves)
                
                while self.cur_bytes[tier] > self.max_bytes[tier] and len(leaves):
                    tn = heapq.heappop(leaves)
                    if tier == 'hbm':
                        self._move_data(tn, 'dram')
                    elif tier == 'dram':
                        self._move_data(tn, 'ssd')
                    else:
                        self._remove_data(tn)
                        assert tn.parent
                        del tn.parent.children[tn.chunk_id_hash]
                        tn.parent = None
                        del self.node_map[tn.chunk_id_hash]
    
    def on_access(self, chunk_ids: List[str], token_counts: List[int]):
        node = self.root
        visited_nodes = []
        for cid, token_count in zip(chunk_ids, token_counts):
            if cid not in node.children:
                child = LRUTrieNode(cid, node)
                node.children[cid] = child
                self.node_map[cid] = child
                child.kv_bytes = token_count * self.bytes_per_token

            node = node.children[cid]
            
            visited_nodes.append(node)
        
        for n in visited_nodes:
            self._move_data(n, 'hbm')
        
        caches = self._make_cache_list(visited_nodes)

        with ttft_timer.without_timing():
            self._demote_if_needed()
        
        return caches
