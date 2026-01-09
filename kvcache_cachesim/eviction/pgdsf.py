from typing import List

from .base import TrieNode, LinkedNode, TieredCache


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

    def init_priority(self, clock):
        self.clock = clock
        self.priority = self.clock

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
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int):
        super().__init__(bytes_per_token, cap_hbm, cap_dram, cap_ssd)
        self.name = "PGDSF"

        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = PGDSFTrieNode(chunk_id=None)
            head_trie_node.tier = t
            tail_trie_node = PGDSFTrieNode(chunk_id=None)
            tail_trie_node.tier = t
            # Create head and tail linked nodes with associated trie nodes
            self.head[t] = PGDSFLinkedNode(trie_node=head_trie_node)
            self.tail[t] = PGDSFLinkedNode(trie_node=tail_trie_node)
            # Link head and tail
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]

        self._global_clock = 0

        self.root = PGDSFTrieNode(None)

    def _add_to_head(self, node: PGDSFLinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

    def _add_by_priority(self, node: PGDSFLinkedNode):
        cur_tier = self.tiers[0]
        next_tier = self.tiers[1]
        if self.cur_bytes[next_tier] > 0 and self.head[next_tier].next.priority > node.priority:
            cur_tier = next_tier

        cur = self.tail[cur_tier].prev
        while cur is not self.head[cur_tier] and cur.priority <= node.priority:
            cur = cur.prev
        node.next = cur.next
        node.prev = cur
        cur.next.prev = node
        cur.next = node
        node.trie_node.tier = cur_tier
        self.cur_bytes[cur_tier] += node.trie_node.kv_bytes

    def _remove_node(self, node: PGDSFLinkedNode):
        if node.prev is None or node.next is None:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

        tn = node.trie_node
        self.cur_bytes[tn.tier] -= tn.kv_bytes

    def _pop_tail(self, tier: str) -> PGDSFLinkedNode:
        tail = self.tail[tier]
        node = tail.prev
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
                tn.linked_node = None
                self._global_clock = max(self._global_clock, node.priority)

    def access_prefix(self,
                      chunk_ids: List[int],
                      token_counts: List[int]):
        self.total_access += 1
        self.total_chunks += len(chunk_ids)

        tn = self.root
        hits = 0
        hit_chunks = {t: 0 for t in self.tiers}
        for cid, cnt in zip(chunk_ids, token_counts):
            if cid in tn.children:
                tn = tn.children[cid]
                if tn.linked_node is None:
                    break
                hits += 1
                hit_chunks[tn.tier] += 1
            else:
                break
        for t, c in hit_chunks.items():
            self.tier_hit_chunks[t] += c

        tn = self.root
        depth = 0
        for cid, cnt in zip(chunk_ids, token_counts):
            depth += 1
            sz = cnt * self.bytes_per_token

            if cid not in tn.children:
                child = PGDSFTrieNode(cid, parent=tn)
                tn.children[cid] = child
                child.token_count = cnt
                child.kv_bytes = sz

                ln = PGDSFLinkedNode(child)
                ln.update_priority(self._global_clock, len(chunk_ids) * (len(chunk_ids) - hits), depth - hits, False)
                child.linked_node = ln
                tn = child
            else:
                tn = tn.children[cid]
                if tn.linked_node:
                    # Remove; add by priority later
                    self._remove_node(tn.linked_node)
                    tn.linked_node.update_priority(self._global_clock, 0, 0, True)
                else:
                    ln = PGDSFLinkedNode(tn)
                    ln.init_priority(self._global_clock)
                    tn.linked_node = ln
                    tn.linked_node.update_priority(self._global_clock, len(chunk_ids) * (len(chunk_ids) - hits),
                                                   depth - hits, False)

            self._add_by_priority(tn.linked_node)

        # Demotion and eviction
        for tier in self.tiers:
            self._demote_if_needed(tier)

    def print_cache(self):
        for tier in self.tiers:
            lst = []
            cur = self.head[tier].next
            while cur is not self.tail[tier]:
                chunk_id = getattr(cur.trie_node, "chunk_id", "?")
                priority = getattr(cur, "priority", "?")
                lst.append(f"{chunk_id}({priority:.4f})")
                cur = cur.next
            print(f"{tier.upper()}: {lst}")

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
    cache = TieredPGDSFCache(bytes_per_token=1, cap_hbm=512, cap_dram=512, cap_ssd=512)
    cache.access_prefix([0, 1], [256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([3, 4, 5], [256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([0, 2], [256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([3, 4, 6, 7], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.print_stats()
