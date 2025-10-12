from collections import OrderedDict
from typing import List, Optional, Set, Tuple

from tiered_cache import TrieNode, LinkedNode, TieredCache
import ttft_timer


class Ghost:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: OrderedDict[int, Set[Optional[Tuple[int, ...]]]] = OrderedDict()

    def put(self, key: int, value: Optional[List[int]] = None):
        if key not in self.data:
            self.data[key] = set()
        self.data[key].add(Tuple(value) if value is not None else None)

        if len(self.data) > self.capacity:
            self.data.popitem(last=False)

    def get(self, key: int) -> Optional[Set[Optional[Tuple[int, ...]]]]:
        return self.data.get(key, None)

    def exists(self, key: int, value: Optional[List[int]] = None) -> bool:
        if key not in self.data:
            return False
        v = Tuple(value) if value is not None else None
        return v in self.data[key]


class S3FIFOLinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.freq = 0


class TieredS3FIFOCache(TieredCache):
    def __init__(self, device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity, chunk_size: int = 256):
        super().__init__(device, bytes_per_token, ssd_path, hbm_capacity, dram_capacity, ssd_capacity)
        
        self.head = {t: S3FIFOLinkedNode() for t in self.tiers}
        self.tail = {t: S3FIFOLinkedNode() for t in self.tiers}
        for t in self.tiers:
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]
        self.root = TrieNode(None)

        self.ghost = Ghost(dram_capacity // bytes_per_token // chunk_size)
    
    def _add_to_head(self, node: S3FIFOLinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        self._move_data(node.trie_node, tier)
    
    def _temp_remove_node(self, node: S3FIFOLinkedNode):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
    
    def _pop_tail(self, tier: str) -> S3FIFOLinkedNode:
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
    
    def _demote_if_needed(self):
        for tier in self.tiers:
            while self.cur_bytes[tier] > self.max_bytes[tier]:
                node = self._pop_tail(tier)

                tn = node.trie_node

                if tier == 'hbm':
                    if node.freq > 1:
                        self._add_to_head(node, 'dram')
                    else:
                        node.freq = 0
                        self._add_to_head(node, 'ssd')
                elif tier == 'dram':
                    if node.freq > 0:
                        node.freq -= 1
                        self._add_to_head(node, 'dram')
                    else:
                        self._add_to_head(node, 'ssd')
                else:
                    self._delete_subtree(tn)
    
    def on_access(self, chunk_ids: List[str], token_counts: List[int]):
        node = self.root
        visited_nodes = []
        pending_nodes =  []
        destinations = []
        for cid, token_count in zip(chunk_ids, token_counts):
            if cid not in node.children:
                child = TrieNode(cid, node)
                node.children[cid] = child
                self.node_map[cid] = child
                child.kv_bytes = token_count * self.bytes_per_token

                linked_node = S3FIFOLinkedNode(child)
                child.linked_node = linked_node
                if self.ghost.exists(cid):
                    destinations.append('dram')
                else:
                    destinations.append('hbm')

                pending_nodes.append(child)

                node = child
            else:
                node = node.children[cid]
                if node.location == 'ssd':
                    self._temp_remove_node(node.linked_node)
                    destinations.append('dram')
                    pending_nodes.append(node)
                node.linked_node.freq = min(node.linked_node.freq + 1, 3)
            
            visited_nodes.append(node)
        
        for n, dst in zip(reversed(pending_nodes), destinations):
            self._add_to_head(n.linked_node, dst)
        
        caches = self._make_cache_list(visited_nodes)

        with ttft_timer.without_timing():
            self._demote_if_needed()

        return caches
