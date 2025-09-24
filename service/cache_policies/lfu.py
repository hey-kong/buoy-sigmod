from typing import List

from tiered_cache import TrieNode, LinkedNode, TieredCache
import ttft_timer


class LFULinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.freq = 1


class TieredLFUCache(TieredCache):
    def __init__(self, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity):
        super().__init__(bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity)
        
        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = TrieNode(None)
            head_trie_node.tier = t
            tail_trie_node = TrieNode(None)
            tail_trie_node.tier = t

            self.head[t] = LFULinkedNode(head_trie_node)
            self.tail[t] = LFULinkedNode(tail_trie_node)
            
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]
        self.root = TrieNode(None)

        self.freq_buckets = {}
        self.dummy = LFULinkedNode(TrieNode(None))
        self.dummy.freq = 0
        self._add_to_head(self.dummy, self.tiers[0])
        self.freq_buckets[0] = self.dummy
    
    def _add_to_head(self, node: LFULinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        self._move_data(node.trie_node, tier)

    def _insert_by_freq(self, node: LFULinkedNode):
        freq = node.freq
        for f in sorted(self.freq_buckets.keys(), reverse=True):
            if f <= freq:
                first = self.freq_buckets[f]
                node.prev = first.prev
                node.next = first
                first.prev.next = node
                first.prev = node
                tier = first.trie_node.tier
                node.trie_node.tier = tier

                self._move_data(node.trie_node, first.trie_node.tier)
                self.freq_buckets[freq] = node
                return
    
    def _get_next(self, ln: LFULinkedNode):
        if ln.next is not self.tail[ln.trie_node.tier]:
            return ln.next
        if ln.trie.node.tier == 'hbm' and self.head['dram'].next is not self.tail['dram']:
            return self.head['dram'].next
        if ln.trie.node.tier in ['hbm', 'dram'] and self.head['ssd'].next is not self.tail['ssd']:
            return self.head['ssd'].next
        return None
    
    # without remove data, must have reference when called
    def _temp_remove_node(self, node: LFULinkedNode):
        freq = node.freq
        if freq != 0 and freq in self.freq_buckets and node is self.freq_buckets[freq]:
            cur = self._get_next(node)
            if cur is not None and cur.freq == freq:
                self.freq_buckets[freq] = cur
            else:
                del self.freq_buckets[freq]

        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
    
    def _pop_tail(self, tier: str) -> LFULinkedNode:
        tail = self.tail[tier]
        node = tail.prev
        if node is self.dummy:
            node = node.prev
        if node is self.head[tier]:
            return None
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

                linked_node = LFULinkedNode(child)
                child.linked_node = linked_node

                node = child
            else:
                node = node.children[cid]
                if node.linked_node:
                    self._temp_remove_node(node.linked_node)
                    node.linked_node.freq += 1
                else:
                    linked_node = LFULinkedNode(node)
                    node.linked_node = linked_node
            
            visited_nodes.append(node)
        
        caches = self._make_cache_list(visited_nodes)

        for n in reversed(visited_nodes):
            self._insert_by_freq(n.linked_node)
        
        self._demote_if_needed()

        return caches
