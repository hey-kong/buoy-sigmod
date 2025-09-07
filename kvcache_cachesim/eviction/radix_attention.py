import heapq
from typing import List

from .base import TrieNode, TieredCache


class LRUTrieNode(TrieNode):
    def __init__(self, chunk_id, parent=None):
        super().__init__(chunk_id, parent)

        self.clock = 0

    def __lt__(self, other):
        return self.clock < other.clock


class RadixAttention(TieredCache):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int):
        super().__init__(bytes_per_token, cap_hbm, cap_dram, cap_ssd)
        self.name = "RadixAttention"

        self.root = LRUTrieNode(None)

    def _collect_leaves_device(self, tier):
        def is_leaf(tn: LRUTrieNode):
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

        tier_idx = self.tiers.index(tier)
        ret_list = []
        stack = [self.root]
        while stack:
            tn = stack.pop()
            if is_leaf(tn):
                ret_list.append(tn)
            else:
                for cur_child in tn.children.values():
                    if self.tiers.index(cur_child.tier) <= tier_idx:
                        stack.append(cur_child)
        return ret_list

    def _demote_if_needed(self, tier: str):
        next_tier = self.next_tier.get(tier, None)

        while self.cur_bytes[tier] > self.max_bytes[tier]:
            leaves = self._collect_leaves_device(tier)
            heapq.heapify(leaves)

            while self.cur_bytes[tier] > self.max_bytes[tier] and len(leaves):
                tn = heapq.heappop(leaves)
                self.cur_bytes[tier] -= tn.kv_bytes
                if next_tier:
                    # Demote to next tier
                    tn.tier = next_tier
                    self.cur_bytes[next_tier] += tn.kv_bytes
                else:
                    # Evict
                    parent = tn.parent
                    assert parent is not None
                    assert parent and tn.chunk_id in parent.children
                    del parent.children[tn.chunk_id]
                    tn.linked_node = None
                    tn.parent = None

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
        tier = self.tiers[0]
        for cid, cnt in zip(chunk_ids, token_counts):
            sz = cnt * self.bytes_per_token

            if cid not in node.children:
                child = LRUTrieNode(cid, parent=node)
                child.token_count = cnt
                child.kv_bytes = sz
                child.tier = tier
                node.children[cid] = child
                self.cur_bytes[tier] += sz
                node = child
            else:
                node = node.children[cid]
                self.cur_bytes[node.tier] -= sz
                node.tier = tier
                self.cur_bytes[node.tier] += sz

            node.clock = self.total_access

        # Demotion and eviction
        for tier in self.tiers:
            self._demote_if_needed(tier)

    def print_trie(self):
        def dfs(node, depth=0):
            indent = "  " * depth
            if node.chunk_id is not None:
                tier = getattr(node, 'tier', 'None')
                print(
                    f"{indent}- {node.chunk_id} (tokens={node.token_count}, bytes={getattr(node, 'kv_bytes', '?')}, tier={tier})")
            for child in node.children.values():
                dfs(child, depth + 1)

        print("Trie structure:")
        dfs(self.root)


if __name__ == "__main__":
    cache = RadixAttention(bytes_per_token=1, cap_hbm=512, cap_dram=512, cap_ssd=512)
    cache.access_prefix([0, 1], [256, 256])
    cache.print_trie()

    cache.access_prefix([1, 2, 3], [256, 256, 256])
    cache.print_trie()

    cache.access_prefix([0, 2], [256, 256])
    cache.print_trie()

    cache.access_prefix([1, 2, 4, 3], [256, 256, 256, 256])
    cache.print_trie()

    cache.print_stats()
