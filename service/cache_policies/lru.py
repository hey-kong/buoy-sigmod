from typing import List

from tiered_cache import TrieNode, LinkedNode, TieredCache
import ttft_timer


class LRULinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)


class TieredLRUCache(TieredCache):
    def __init__(self, device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity):
        super().__init__(device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity)
        
        self.head = {t: LRULinkedNode() for t in self.tiers}
        self.tail = {t: LRULinkedNode() for t in self.tiers}
        for t in self.tiers:
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]
        self.root = TrieNode(None)
    
    def _add_to_head(self, node: LRULinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        self._move_data(node.trie_node, tier)

    # without remove data, must have reference when called
    def _temp_remove_node(self, node: LRULinkedNode):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
    
    def _pop_tail(self, tier: str) -> LRULinkedNode:
        tail = self.tail[tier]
        if tail.prev is self.head[tier]:
            return None
        node = tail.prev
        self._temp_remove_node(node)
        return node
    
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
                    self._remove_data(tn)
                    if tn.parent:
                        del tn.parent.children[tn.chunk_id_hash]
                    tn.linked_node = None
                    tn.parent = None
                    del self.node_map[tn.chunk_id_hash]
                    node.prev = node.next = None
    
    def on_access(self, chunk_id_hashes: List[str], token_counts: List[int]):
        node = self.root
        visited_nodes = []
        for cid, token_count in zip(chunk_id_hashes, token_counts):
            if cid not in node.children:
                child = TrieNode(cid, node)
                node.children[cid] = child
                self.node_map[cid] = child
                child.kv_bytes = token_count * self.bytes_per_token

                linked_node = LRULinkedNode(child)
                child.linked_node = linked_node

                node = child
            else:
                node = node.children[cid]
                if node.linked_node:
                    self._temp_remove_node(node.linked_node)
                else:
                    linked_node = LRULinkedNode(node)
                    node.linked_node = linked_node
            visited_nodes.append(node)
        
        caches = self._make_cache_list(visited_nodes)

        for n in reversed(visited_nodes):
            self._add_to_head(n.linked_node, self.tiers[0])
        
        with ttft_timer.without_timing():
            self._demote_if_needed()

        return caches
