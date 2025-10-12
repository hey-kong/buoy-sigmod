import math
from typing import List

from tiered_cache import TrieNode, LinkedNode, TieredCache
import ttft_timer


class WALinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.reuse_prob = 0
        self.offset = 0
        self.priority = (self.reuse_prob, -self.offset)
        self.access_times = []
    
    def update_priority(self):
        if len(self.access_times) < 2:
            # Assign default reuse probability to protect new nodes.
            self.reuse_prob = 0.1 / self.offset
        else:
            waited = self.access_times[-1] - self.access_times[-2]
            gaps = [
                self.access_times[i] - self.access_times[i - 1]
                for i in range(1, len(self.access_times))
            ]
            mean_gap = sum(gaps) / len(gaps)
            lambda_val = 1.0 / mean_gap
            prob = 1.0 - math.exp(-lambda_val * waited)
            lifespan = self.access_times[-1] - self.access_times[0]
            life_factor = min(1.0, lifespan / (mean_gap * 10))
            self.reuse_prob = prob * life_factor
        self.priority = (self.reuse_prob, -self.offset)


class TieredWACache(TieredCache):
    def __init__(self, device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity):
        super().__init__(device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity)

        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = TrieNode(None)
            head_trie_node.location = t
            tail_trie_node = TrieNode(None)
            tail_trie_node.location = t

            self.head[t] = WALinkedNode(head_trie_node)
            self.tail[t] = WALinkedNode(tail_trie_node)
            
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]
        self.root = TrieNode(None)

        self.total_access = 0
        self.lambda_val = 0
        self.life_val = 0
    
    def _add_to_head(self, node: WALinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        self._move_data(node.trie_node, tier)
    
    def _add_by_priority(self, node: WALinkedNode):
        if self.cur_bytes['hbm'] >= self.max_bytes['hbm'] and \
           node.priority < self.tail['hbm'].prev.priority:
            cur_tier = 'dram'
        else:
            cur_tier = 'hbm'
        
        cn = self.head[cur_tier].next
        while cn is not self.tail[cur_tier] and cn.priority > node.priority:
            cn = cn.next
        
        node.prev = cn.prev
        node.next = cn
        cn.prev.next = node
        cn.prev = node
        
        self._move_data(node.trie_node, cur_tier)
    
    # without remove data, must have reference when called
    def _temp_remove_node(self, node: WALinkedNode):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
    
    def _pop_tail(self, tier: str) -> WALinkedNode:
        tail = self.tail[tier]
        if tail.prev is self.head[tier]:
            return None
        node = tail.prev
        self._temp_remove_node(node)
        return node
    
    def _delete_subtree(self, trie_node: TrieNode):
        for child in list(trie_node.children.values()):
            self._delete_subtree(child)
        
        if trie_node.linked_node:
            ln = trie_node.linked_node
            if ln.prev is not None and ln.next is not None:
                self._temp_remove_node(ln)
        self._remove_data(trie_node)
        trie_node.linked_node = None

        if trie_node.parent:
            del trie_node.parent.children[trie_node.chunk_id_hash]
        trie_node.parent = None

        del self.node_map[trie_node.chunk_id_hash]

    def _demote_if_needed(self):
       for tier in self.tiers:
            while self.cur_bytes[tier] > self.max_bytes[tier]:
                node = self._pop_tail(tier)

                tn = node.trie_node

                if tier == 'hbm':
                    self._add_to_head(node, 'dram')
                elif tier == 'dram':
                    self._add_to_head(node, 'ssd')
                else:
                    self._delete_subtree(tn)
    
    def on_access(self, chunk_ids: List[str], token_counts: List[int]):
        self.total_access += 1
        node = self.root
        visited_nodes = []
        for idx, (cid, token_count) in enumerate(zip(chunk_ids, token_counts)):
            if cid not in node.children:
                child = TrieNode(cid, node)
                node.children[cid] = child
                self.node_map[cid] = child
                child.kv_bytes = token_count * self.bytes_per_token

                linked_node = WALinkedNode(child)
                linked_node.offset = idx + 1
                linked_node.access_times = [self.total_access]
                child.linked_node = linked_node

                node = child
            else:
                node = node.children[cid]
                self._temp_remove_node(node.linked_node)
                node.linked_node.access_times.append(self.total_access)
            
            visited_nodes.append(node)
        
        for n in visited_nodes:
            n.linked_node.update_priority()
            self._add_by_priority(n.linked_node)
        
        caches = self._make_cache_list(visited_nodes)

        with ttft_timer.without_timing():
            self._demote_if_needed()

        return caches
