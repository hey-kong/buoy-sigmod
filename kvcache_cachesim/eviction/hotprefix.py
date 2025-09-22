from typing import List

from .base import TrieNode, LinkedNode, TieredCache


class HPLinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.priority = 0
        self.freq = 0
        self.clock = 0
        self.length = 0

    def update_priority(self, clock, length):
        self.freq += 1
        self.clock = clock
        self.length = length
        self.priority = self.freq + (self.clock / self.length)


class HotPrefixCache(TieredCache):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int,
                 max_age: int = 15,
                 freq_threshold: int = 1,
                 aging_interval: int = 64):
        super().__init__(bytes_per_token, cap_hbm, cap_dram, cap_ssd)
        self.name = "HotPrefix"

        if cap_hbm > 0 and cap_dram > 0 and cap_ssd == 0:
            self.num_tiers = 2
        elif cap_hbm > 0 and cap_dram > 0 and cap_ssd > 0:
            self.num_tiers = 3
        else:
            raise ValueError(
                "This simulator only supports two-tier (HBM+DRAM) or three-tier (HBM+DRAM+SSD) configurations."
            )

        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = TrieNode(chunk_id=None)
            head_trie_node.tier = t
            tail_trie_node = TrieNode(chunk_id=None)
            tail_trie_node.tier = t
            # Create head and tail linked nodes with associated trie nodes
            self.head[t] = HPLinkedNode(trie_node=head_trie_node)
            self.tail[t] = HPLinkedNode(trie_node=tail_trie_node)
            # Link head and tail
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]

        self.max_age = max_age
        self.freq_threshold = freq_threshold
        self.aging_interval = aging_interval

        self.root = TrieNode(None)

    def _add_to_head(self, node: HPLinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

    def _add_by_priority(self, node: HPLinkedNode, tier: str):
        cur = self.head[tier].next
        while cur is not self.tail[tier] and cur.priority > node.priority:
            cur = cur.next

        node.prev = cur.prev
        node.next = cur
        cur.prev.next = node
        cur.prev = node
        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

    def _remove_node(self, node: HPLinkedNode):
        if node.prev is None or node.next is None:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

        tn = node.trie_node
        self.cur_bytes[tn.tier] -= tn.kv_bytes

    def _pop_tail(self, tier: str) -> HPLinkedNode:
        tail = self.tail[tier]
        node = tail.prev
        if node is self.head[tier]:
            return None
        self._remove_node(node)
        return node

    def _get_last(self, tier: str) -> HPLinkedNode:
        last = self.tail[tier].prev
        if last is not self.head[tier]:
            return last
        return None

    def _demote_if_needed(self, tier: str):
        def evict(trie_node):
            # Iteratively removes trie_node subtree and linked nodes.
            stack = [trie_node]
            while stack:
                trie_node = stack.pop()
                stack.extend(list(trie_node.children.values()))
                linked_node = trie_node.linked_node
                if linked_node is not None:
                    self._remove_node(linked_node)
                    trie_node.linked_node = None
                if trie_node.parent is not None:
                    del trie_node.parent.children[trie_node.chunk_id]
                trie_node.parent = None

        while self.cur_bytes[tier] > self.max_bytes[tier]:
            node = self._pop_tail(tier)
            if not node:
                break

            tn = node.trie_node
            next_tier = self.next_tier.get(tier)

            # Selective admission
            if tier is self.tiers[0]:
                # Frequency threshold filtering
                if node.freq < self.freq_threshold:
                    if self.num_tiers == 2:
                        evict(tn)
                    elif self.num_tiers == 3:
                        self._add_to_head(node, self.tiers[2])
                    continue
                if self.cur_bytes[next_tier] >= self.max_bytes[next_tier]:
                    # Hotness comparison
                    comp_node = self._get_last(next_tier)
                    if node.freq * node.clock < comp_node.freq * comp_node.clock:
                        if self.num_tiers == 2:
                            evict(tn)
                        elif self.num_tiers == 3:
                            self._add_to_head(node, self.tiers[2])
                        continue

            if next_tier:
                # Demote to next tier
                if next_tier == self.tiers[1]:
                    self._add_by_priority(node, next_tier)
                else:
                    self._add_to_head(node, next_tier)
            else:
                # Evict from the last tier
                evict(tn)

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
        length = 0
        for cid, cnt in zip(chunk_ids, token_counts):
            length += 1
            sz = cnt * self.bytes_per_token

            if cid not in node.children:
                child = TrieNode(cid, parent=node)
                node.children[cid] = child
                child.token_count = cnt
                child.kv_bytes = sz

                ln = HPLinkedNode(child)
                child.linked_node = ln
                node = child
                ln.update_priority(self.max_age, length)
            else:
                node = node.children[cid]
                if node.linked_node:
                    # Remove; add by priority later
                    self._remove_node(node.linked_node)
                    node.linked_node.update_priority(self.max_age, length)
                else:
                    ln = HPLinkedNode(node)
                    node.linked_node = ln
                    ln.update_priority(self.max_age, length)

            visited.append(node.linked_node)

        for ln in reversed(visited):
            self._add_by_priority(ln, self.tiers[0])

        # Demotion and eviction
        for tier in self.tiers:
            self._demote_if_needed(tier)

        if self.total_access % self.aging_interval == 0:
            self.aging()

    def aging(self):
        def _unlink(node):
            node.prev.next = node.next
            node.next.prev = node.prev

        def _insert_before(node, ref):
            node.prev = ref.prev
            node.next = ref
            ref.prev.next = node
            ref.prev = node

        for cur_tier in (self.tiers[0], self.tiers[1]):
            node = self.head[cur_tier].next
            while node != self.tail[cur_tier]:
                old_clock = node.clock
                node.clock = max(old_clock - 1, 0)
                node.priority = node.freq + (node.clock / node.length)

                next_node = node.next
                cur = node.prev
                if cur is not self.head[cur_tier] and node.priority > cur.priority:
                    _unlink(node)
                    while cur is not self.head[cur_tier] and node.priority > cur.priority:
                        cur = cur.prev
                    _insert_before(node, cur.next)
                node = next_node

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
    cache = HotPrefixCache(bytes_per_token=1, cap_hbm=512, cap_dram=512, cap_ssd=512)
    cache.access_prefix([0, 1], [256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([1, 2, 3], [256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([0, 2], [256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([1, 2, 4, 3], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.print_stats()
