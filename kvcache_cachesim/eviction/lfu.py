from typing import List

from .base import TrieNode, LinkedNode, TieredCache


class LFULinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.freq = 1


class TieredTrieLFUCache(TieredCache):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int):
        super().__init__(bytes_per_token, cap_hbm, cap_dram, cap_ssd)
        self.name = "LFU"

        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = TrieNode(chunk_id=None)
            head_trie_node.tier = t
            tail_trie_node = TrieNode(chunk_id=None)
            tail_trie_node.tier = t
            # Create head and tail linked nodes with associated trie nodes
            self.head[t] = LFULinkedNode(trie_node=head_trie_node)
            self.tail[t] = LFULinkedNode(trie_node=tail_trie_node)
            # Link head and tail
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

        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

    def _add_by_freq(self, node: LFULinkedNode):
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
                self.cur_bytes[tier] += node.trie_node.kv_bytes
                self.freq_buckets[freq] = node
                return

    def _get_next(self, cur: LFULinkedNode) -> LFULinkedNode:
        cur_tier = cur.trie_node.tier
        tier_tail = self.tail[cur_tier]
        if cur.next is not tier_tail:
            return cur.next
        next_tier = self.next_tier.get(cur_tier, None)
        while next_tier:
            next_head = self.head[next_tier]
            next_tail = self.tail[next_tier]
            next_node = next_head.next
            if next_node is not next_tail:
                return next_node
            next_tier = self.next_tier.get(next_tier, None)
        return None

    def _remove_node(self, node: LFULinkedNode):
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

        tn = node.trie_node
        self.cur_bytes[tn.tier] -= tn.kv_bytes

    def _pop_tail(self, tier: str) -> LFULinkedNode:
        tail = self.tail[tier]
        node = tail.prev
        if node is self.dummy:
            node = node.prev
        if node is self.head[tier]:
            return None
        self._remove_node(node)
        return node

    def _demote_if_needed(self, tier: str):
        while self.cur_bytes[tier] > self.max_bytes[tier]:
            node = self._pop_tail(tier)
            if not node:
                break
            tn = node.trie_node

            # Determine next tier
            next_tier = self.next_tier.get(tier, None)

            if next_tier:
                # Demote to next tier
                self._add_to_head(node, next_tier)
            else:
                # Evict from the last tier
                parent = tn.parent
                if parent:
                    del parent.children[tn.chunk_id]
                tn.linked_node = None
                tn.parent = None
                node.prev = node.next = None

    def access_prefix(self,
                      chunk_ids: List[int],
                      token_counts: List[int]):
        self.total_access += 1
        self.total_chunks += len(chunk_ids)

        node = self.root
        hits = 0
        hit_chunks = {t: 0 for t in self.tiers}
        for cid, cnt in zip(chunk_ids, token_counts):
            if cid in node.children:
                node = node.children[cid]
                hits += 1
                hit_chunks[node.tier] += 1
            else:
                break
        for t, c in hit_chunks.items():
            self.tier_hit_chunks[t] += c

        node = self.root
        visited = []
        for cid, cnt in zip(chunk_ids, token_counts):
            sz = cnt * self.bytes_per_token

            if cid not in node.children:
                child = TrieNode(cid, parent=node)
                node.children[cid] = child
                child.token_count = cnt
                child.kv_bytes = sz

                ln = LFULinkedNode(child)
                child.linked_node = ln
                node = child
            else:
                node = node.children[cid]
                if node.linked_node:
                    # Remove; move to HBM later
                    self._remove_node(node.linked_node)
                    node.linked_node.freq += 1
                else:
                    ln = LFULinkedNode(node)
                    node.linked_node = ln

            visited.append(node.linked_node)

        # Reorder visited nodes to preserve prefix order
        for ln in reversed(visited):
            self._add_by_freq(ln)

        # Demotion and eviction
        for tier in self.tiers:
            self._demote_if_needed(tier)

    def print_cache(self):
        for tier in self.tiers:
            lst = []
            cur = self.head[tier].next
            while cur is not self.tail[tier] and cur is not self.dummy:
                lst.append(f"{cur.trie_node.chunk_id}")
                cur = cur.next
            print(f"{tier.upper()}:", lst)

    def print_trie(self):
        def dfs(node, depth=0):
            indent = "  " * depth
            if node.chunk_id is not None:
                print(f"{indent}- {node.chunk_id} (tokens={node.token_count}, bytes={getattr(node, 'kv_bytes', '?')})")
            for child in node.children.values():
                dfs(child, depth + 1)

        print("Trie structure:")
        dfs(self.root)


if __name__ == "__main__":
    cache = TieredTrieLFUCache(bytes_per_token=1, cap_hbm=512, cap_dram=512, cap_ssd=512)
    cache.access_prefix([0, 1], [256, 256])
    cache.print_trie()
    cache.print_cache()  # ➜ [0, 1]

    cache.access_prefix([1, 2, 3], [256, 256, 256])
    cache.print_trie()
    cache.print_cache()  # ➜ [1, 2], [3, 0], [1]

    cache.access_prefix([0, 2], [256, 256])
    cache.print_trie()
    cache.print_cache()  # ➜ [0, 2], [1, 2], [3, 1]

    cache.access_prefix([1, 2, 4, 3], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()  # ➜ [1, 2], [0, 4], [3, 2]

    cache.print_stats()
