from typing import List

from tiered_cache import TrieNode, LinkedNode, TieredCache
import ttft_timer


class PGDSFTrieNode(TrieNode):
    def __init__(self, chunk_id, parent=None):
        super().__init__(chunk_id, parent)

        self.m = 0
        self.total_cost = 0
        self.avg_cost = 0


class PGDSFLinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.priority = 0
        self.clock = 0
        self.freq = 0

    def update_priority(self, clock, cost, new_size, is_cached):
        self.clock = clock
        self.freq += 1
        if not is_cached:
            assert new_size > 0
            self.trie_node.total_cost += cost / new_size
            self.trie_node.m += 1
            self.trie_node.avg_cost = self.trie_node.total_cost / self.trie_node.m
        self.priority = self.clock + self.freq * self.trie_node.avg_cost


class TieredPGDSFCache(TieredCache):
    def __init__(self, device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity):
        super().__init__(device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity)

        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = PGDSFTrieNode(None)
            head_trie_node.location = t
            tail_trie_node = PGDSFTrieNode(None)
            tail_trie_node.location = t

            self.head[t] = PGDSFLinkedNode(head_trie_node)
            self.tail[t] = PGDSFLinkedNode(tail_trie_node)
            
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]
        self.root = PGDSFTrieNode(None)

        self.global_clock = 0
    
    def _add_to_head(self, node: PGDSFLinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        self._move_data(node.trie_node, tier)
    
    def _add_by_priority(self, node: PGDSFLinkedNode):
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
    def _temp_remove_node(self, node: PGDSFLinkedNode):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
    
    def _pop_tail(self, tier: str) -> PGDSFLinkedNode:
        tail = self.tail[tier]
        if tail.prev is self.head[tier]:
            return None
        node = tail.prev
        self._temp_remove_node(node)
        return node
    
    def _delete_subtree(self, trie_node: PGDSFTrieNode):
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
        hit_count = 0
        chunk_count = len(chunk_ids)
        node = self.root
        visited_nodes = []
        for idx, (cid, token_count) in enumerate(zip(chunk_ids, token_counts)):
            if cid not in node.children:
                child = PGDSFTrieNode(cid, node)
                node.children[cid] = child
                self.node_map[cid] = child
                child.kv_bytes = token_count * self.bytes_per_token

                linked_node = PGDSFLinkedNode(child)
                linked_node.update_priority(self.global_clock, chunk_count * (chunk_count - hit_count), idx + 1 - hit_count, False)
                child.linked_node = linked_node

                node = child
            else:
                hit_count += 1
                node = node.children[cid]
                self._temp_remove_node(node.linked_node)
                node.linked_node.update_priority(self.global_clock, 0, 0, True)
            
            visited_nodes.append(node)
        
        for n in reversed(visited_nodes):
            self._add_by_priority(n.linked_node)
        
        caches = self._make_cache_list(visited_nodes)

        with ttft_timer.without_timing():
            self._demote_if_needed()

        return caches
