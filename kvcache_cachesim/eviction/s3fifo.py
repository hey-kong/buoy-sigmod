from typing import List

from .base import TrieNode, LinkedNode, TieredCache
from .ghost import Ghost


class S3FIFOLinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.freq = 0
        self.depth = 0


class TieredTrieS3FIFOCache(TieredCache):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int,
                 chunk_size: int):
        super().__init__(bytes_per_token, cap_hbm, cap_dram, cap_ssd)
        self.name = "S3-FIFO"

        if cap_hbm > 0 and cap_dram > 0 and cap_ssd == 0:
            self.num_tiers = 2
        elif cap_hbm > 0 and cap_dram > 0 and cap_ssd > 0:
            self.num_tiers = 3
        else:
            raise ValueError(
                "This simulator only supports two-tier (HBM+DRAM) or three-tier (HBM+DRAM+SSD) configurations."
            )

        self.head = {t: S3FIFOLinkedNode() for t in self.tiers}
        self.tail = {t: S3FIFOLinkedNode() for t in self.tiers}
        for t in self.tiers:
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]

        self.root = TrieNode(None)
        # Only store the evicted first prefix chunk
        self.ghost = Ghost(cap_dram // bytes_per_token // chunk_size)

    def _add_to_head(self, node: S3FIFOLinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

    def _remove_node(self, node: S3FIFOLinkedNode):
        if node.prev is None or node.next is None:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

        tn = node.trie_node
        self.cur_bytes[tn.tier] -= tn.kv_bytes

    def _pop_tail(self, tier: str) -> S3FIFOLinkedNode:
        tail = self.tail[tier]
        if tail.prev is self.head[tier]:
            return None
        node = tail.prev
        self._remove_node(node)
        return node

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

            if tier is self.tiers[0]:
                if node.freq > 1:
                    self._add_to_head(node, next_tier)
                else:
                    if self.num_tiers == 2:
                        if node.depth == 1:
                            self.ghost.put(node.trie_node.chunk_id)
                        evict(tn)
                    elif self.num_tiers == 3:
                        node.freq = 0
                        self._add_to_head(node, self.tiers[2])
            elif tier is self.tiers[1]:
                if node.freq == 0:
                    if self.num_tiers == 2:
                        evict(tn)
                    elif self.num_tiers == 3:
                        self._add_to_head(node, self.tiers[2])
                else:
                    node.freq = max(node.freq - 1, 0)
                    self._add_to_head(node, tier)
            else:
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
        depth = 0
        tier = self.tiers[0]
        for cid, cnt in zip(chunk_ids, token_counts):
            depth += 1
            sz = cnt * self.bytes_per_token

            if cid not in node.children:
                child = TrieNode(cid, parent=node)
                node.children[cid] = child
                child.token_count = cnt
                child.kv_bytes = sz

                ln = S3FIFOLinkedNode(child)
                ln.depth = depth
                child.linked_node = ln
                node = child
                if depth == 1 and self.ghost.exists(cid):
                    tier = self.tiers[1]
                node.tier = tier
                visited.append(node.linked_node)
            else:
                node = node.children[cid]
                if node.linked_node:
                    if node.tier == self.tiers[2]:
                        self._remove_node(node.linked_node)
                        node.tier = self.tiers[1]
                        visited.append(node.linked_node)
                else:
                    ln = S3FIFOLinkedNode(node)
                    node.linked_node = ln
                    if node.tier == self.tiers[2]:
                        node.tier = self.tiers[1]
                    visited.append(node.linked_node)
                node.linked_node.freq = min(node.linked_node.freq + 1, 3)

        # Reorder visited nodes to preserve prefix order
        for ln in reversed(visited):
            self._add_to_head(ln, ln.trie_node.tier)

        # Demotion and eviction
        for tier in self.tiers:
            self._demote_if_needed(tier)

    def print_cache(self):
        for tier in self.tiers:
            lst = []
            cur = self.head[tier].next
            while cur is not self.tail[tier]:
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
    cache = TieredTrieS3FIFOCache(bytes_per_token=1, cap_hbm=1024, cap_dram=512, cap_ssd=0)
    cache.access_prefix([0, 1], [256, 256])
    cache.print_trie()
    cache.print_cache()  # ➜ [0, 1]

    cache.access_prefix([1, 2, 3], [256, 256, 256])
    cache.print_trie()
    cache.print_cache()  # ➜ [1, 2, 3, 0], [1]

    cache.access_prefix([0, 2], [256, 256])
    cache.print_trie()
    cache.print_cache()  # ➜ [2, 1, 2, 3], [0, 1]

    cache.access_prefix([1, 2, 4, 3], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()  # ➜ [4, 3, 2, 1], [2, 0]

    cache.print_stats()
