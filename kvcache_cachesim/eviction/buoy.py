from typing import List

from .base import TrieNode, LinkedNode, TieredCache


class BuoyLinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.grace = 0


class TieredTrieBuoyCache(TieredCache):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int,
                 max_grace: int = 65535):
        super().__init__(bytes_per_token, cap_hbm, cap_dram, cap_ssd)
        self.name = "Buoy"

        if cap_hbm > 0 and cap_dram > 0 and cap_ssd == 0:
            self.num_tiers = 2
        elif cap_hbm > 0 and cap_dram > 0 and cap_ssd > 0:
            self.num_tiers = 3
        else:
            raise ValueError(
                "This simulator only supports two-tier (HBM+DRAM) or three-tier (HBM+DRAM+SSD) configurations."
            )

        self.head = {t: BuoyLinkedNode() for t in self.tiers}
        self.tail = {t: BuoyLinkedNode() for t in self.tiers}
        for t in self.tiers:
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]

        self.max_grace = max_grace

        self.root = TrieNode(None)

    def _add_to_head(self, node: BuoyLinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

    def _remove_node(self, node: BuoyLinkedNode):
        if node.prev is None or node.next is None:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

        tn = node.trie_node
        self.cur_bytes[tn.tier] -= tn.kv_bytes

    def _pop_tail(self, tier: str) -> BuoyLinkedNode:
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

        def _bulk_move_chain_to_tier(root_tn, target_tier):
            # For root nodes not reused, only one root-to-leaf path (no branching)
            chain = []
            tn = root_tn
            while tn is not None and tn.tier != target_tier:
                if tn.linked_node is not None:
                    chain.append(tn.linked_node)
                tn = next(iter(tn.children.values()), None)
            for ln in reversed(chain):
                self._remove_node(ln)
                self._add_to_head(ln, target_tier)

        def is_tier_leaf(tn: TrieNode):
            if tn.tier != tier:
                return False
            if tn == self.root:
                return False
            if len(tn.children) == 0:
                return True
            for child in tn.children.values():
                if child.tier == tier:
                    return False
            return True

        while self.cur_bytes[tier] > self.max_bytes[tier]:
            node = self._pop_tail(tier)
            if not node:
                break

            tn = node.trie_node
            if tier is self.tiers[0]:
                if node.grace == 0:
                    if self.num_tiers == 2:
                        evict(tn)
                    elif self.num_tiers == 3:
                        _bulk_move_chain_to_tier(tn, self.tiers[2])
                else:
                    self._add_to_head(node, self.tiers[1])
                continue

            # HBM + DRAM
            if self.num_tiers == 2:
                if node.grace == 0 and is_tier_leaf(tn):
                    evict(tn)
                    continue
                node.grace = max(node.grace - 1, 0)
                self._add_to_head(node, tier)

            # HBM + DRAM + SSD
            if self.num_tiers == 3:
                if tier is self.tiers[1]:
                    if node.grace == 0 and is_tier_leaf(tn):
                        self._add_to_head(node, self.tiers[2])
                        continue
                    node.grace = max(node.grace - 1, 0)
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
        depth = 0
        for cid, cnt in zip(chunk_ids, token_counts):
            sz = cnt * self.bytes_per_token

            if cid not in node.children:
                child = TrieNode(cid, parent=node)
                node.children[cid] = child
                child.token_count = cnt
                child.kv_bytes = sz
                child.tier = self.tiers[0]
                if depth > 0:
                    child.tier = self.tiers[1]

                ln = BuoyLinkedNode(child)
                child.linked_node = ln
                if child.parent is not self.root and len(child.parent.children) == 1:
                    ln.grace = min(256, child.parent.linked_node.grace)
                node = child
            else:
                node = node.children[cid]
                if node.linked_node:
                    # Remove; promote later
                    self._remove_node(node.linked_node)
                else:
                    ln = BuoyLinkedNode(node)
                    node.linked_node = ln
                node.tier = self.tiers[0]
                node.linked_node.grace = min(node.linked_node.grace + 256, self.max_grace)

            self._add_to_head(node.linked_node, node.tier)
            depth += 1

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
    cache = TieredTrieBuoyCache(bytes_per_token=1, cap_hbm=512, cap_dram=512, cap_ssd=512)
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
