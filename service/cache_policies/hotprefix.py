from typing import List

from tiered_cache import TrieNode, LinkedNode, TieredCache
import ttft_timer


class HPLinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.priority = 0
        self.clock = 0
        self.freq = 0
        self.length = 0

    def update_priority(self, clock, length):
        self.clock = clock
        self.freq += 1
        self.length = length
        self.priority = self.freq + (self.clock / self.length)


class HotPrefixCache(TieredCache):
    def __init__(self, device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity,
                 max_age: int = 15, freq_threshold: int = 1, aging_interval: int = 64):
        super().__init__(device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity)

        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = TrieNode(None)
            head_trie_node.location = t
            tail_trie_node = TrieNode(None)
            tail_trie_node.location = t

            self.head[t] = HPLinkedNode(head_trie_node)
            self.tail[t] = HPLinkedNode(tail_trie_node)
            
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]
        self.root = TrieNode(None)

        self.max_age = max_age
        self.freq_threshold = freq_threshold
        self.aging_interval = aging_interval
        self.total_access = 0

    def _add_to_head(self, node: HPLinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        self._move_data(node.trie_node, tier)
    
    def _add_by_priority(self, node: HPLinkedNode, tier: str):
        cn = self.head[tier].next
        while cn is not self.tail[tier] and cn.priority > node.priority:
            cn = cn.next

        node.prev = cn.prev
        node.next = cn
        cn.prev.next = node
        cn.prev = node

        self._move_data(node.trie_node, tier)
    
    def _insert_before(self, node: HPLinkedNode, before: HPLinkedNode):
        assert node.trie_node.location == before.trie_node.location
        node.prev = before.prev
        node.next = before
        before.prev.next = node
        before.prev = node

    # without remove data, must have reference when called
    def _temp_remove_node(self, node: HPLinkedNode):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _pop_tail(self, tier: str) -> HPLinkedNode:
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
                    if node.freq < self.freq_threshold:
                        self._add_to_head(node, 'ssd')
                        continue
                    if self.cur_bytes['dram'] >= self.max_bytes['dram']:
                        comp_node = self.tail['dram'].prev
                        if node.freq * node.clock < comp_node.freq * comp_node.clock:
                            self._add_to_head(node, 'ssd')
                            continue
                    self._add_by_priority(node, 'dram')
                elif tier == 'dram':
                    self._add_to_head(node, 'ssd')
                else:
                    self._delete_subtree(tn)
    
    def _aging(self):
        for tier in ['hbm', 'dram']:
            node = self.head[tier].next
            while node is not self.tail[tier]:
                old_clock = node.clock
                node.clock = max(old_clock - 1, 0)
                node.priority = node.freq + (node.clock / node.length)

                next_node = node.next
                cur = node.prev
                if cur is not self.head[tier] and cur.priority < node.priority:
                    self._temp_remove_node(node)
                    while cur is not self.head[tier] and cur.priority < node.priority:
                        cur = cur.prev
                    self._insert_before(node, cur.next)
                node = next_node

    def on_access(self, chunk_ids: List[str], token_counts: List[int]):
        node = self.root
        visited_nodes = []
        for idx, (cid, token_count) in enumerate(zip(chunk_ids, token_counts)):
            if cid not in node.children:
                child = TrieNode(cid, node)
                node.children[cid] = child
                self.node_map[cid] = child
                child.kv_bytes = token_count * self.bytes_per_token

                linked_node = HPLinkedNode(child)
                child.linked_node = linked_node

                node = child
            else:
                node = node.children[cid]
                self._temp_remove_node(node.linked_node)
            
            node.linked_node.update_priority(self.max_age, idx + 1)
            
            visited_nodes.append(node)
        
        for n in reversed(visited_nodes):
            self._add_by_priority(n.linked_node, 'hbm')
        
        caches = self._make_cache_list(visited_nodes)

        with ttft_timer.without_timing():
            self._demote_if_needed()
            if self.total_access % self.aging_interval == 0:
                self._aging()

        return caches
