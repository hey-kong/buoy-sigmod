from typing import List

from tiered_cache import TrieNode, LinkedNode, TieredCache
import ttft_timer


class FIFOLinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)


class TieredFIFOCache(TieredCache):
    def __init__(self, device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity):
        super().__init__(device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity)
        
        self.head = {t: FIFOLinkedNode() for t in self.tiers}
        self.tail = {t: FIFOLinkedNode() for t in self.tiers}
        for t in self.tiers:
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]
        self.root = TrieNode(None)
    
    def _add_to_head(self, node: FIFOLinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        self._move_data(node.trie_node, tier)
    
    # without remove data, must have reference when called
    def _temp_remove_node(self, node: FIFOLinkedNode):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
    
    def _pop_tail(self, tier: str) -> FIFOLinkedNode:
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
    
    def on_access(self, chunk_id_hashes: List[str], token_counts: List[int]):
        node = self.root
        visited_nodes = []
        new_nodes = []
        for cid, token_count in zip(chunk_id_hashes, token_counts):
            if cid not in node.children:
                child = TrieNode(cid, node)
                node.children[cid] = child
                self.node_map[cid] = child
                child.kv_bytes = token_count * self.bytes_per_token

                linked_node = FIFOLinkedNode(child)
                child.linked_node = linked_node

                new_nodes.append(child)

            node = node.children[cid]

            visited_nodes.append(node)

        for n in reversed(new_nodes):
            self._add_to_head(n.linked_node, 'hbm')
        
        caches = self._make_cache_list(visited_nodes)
        
        with ttft_timer.without_timing():
            self._demote_if_needed()
        
        return caches
