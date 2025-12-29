from typing import List

from .base import TrieNode, LinkedNode, TieredCache
from .ghost import Ghost


class BuoyLinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.visited = False


class DepthMean:
    def __init__(self):
        self._sum = 0
        self._cnt = 0

    def add(self, depth: int, inc: int = 1):
        if inc <= 0:
            return
        self._sum += depth * inc
        self._cnt += inc

    def get(self) -> int:
        return self._sum // self._cnt if self._cnt else 0


class TieredTrieBuoyCache(TieredCache):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int,
                 chunk_size: int):
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

        self.root = TrieNode(None)
        self.root.linked_node = BuoyLinkedNode()
        self.ghost = Ghost(cap_dram * 4 // bytes_per_token // chunk_size)
        self.depth_mean = DepthMean()

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

    def _evict(self, trie_node):
        linked_node = trie_node.linked_node
        if linked_node is not None:
            self._remove_node(linked_node)
            trie_node.linked_node = None
        if trie_node.parent is not None:
            del trie_node.parent.children[trie_node.chunk_id]
            trie_node.parent = None

    def _demote_if_needed(self, tier: str):
        while self.cur_bytes[tier] > self.max_bytes[tier]:
            node = self._pop_tail(tier)
            if not node:
                break

            tn = node.trie_node
            if tier == self.tiers[0]:
                self._add_to_head(node, self.tiers[1])
                continue

            # HBM + DRAM
            if self.num_tiers == 2:
                if not node.visited:
                    self.ghost.put(tn.chunk_id, tn.get_path())
                    self._evict(tn)
                    continue
                node.visited = False
                self._add_to_head(node, tier)

            # HBM + DRAM + SSD
            if self.num_tiers == 3:
                if tier == self.tiers[1]:
                    if not node.visited:
                        self._add_to_head(node, self.tiers[2])
                        continue
                    node.visited = False
                    self._add_to_head(node, tier)
                else:
                    self._evict(tn)

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

        # Depth-based cache admission
        depth = 0
        deepest_hit_depth = 0
        depth_threshold = self.depth_mean.get()

        tn = self.root
        admit_nodes = []
        reject_nodes = []
        for cid, cnt in zip(chunk_ids, token_counts):
            depth += 1
            sz = cnt * self.bytes_per_token

            if cid not in tn.children:
                child = TrieNode(cid, parent=tn)
                tn.children[cid] = child
                child.token_count = cnt
                child.kv_bytes = sz
                child.tier = self.tiers[1]

                node = BuoyLinkedNode(child)
                child.linked_node = node
                if self.ghost.exists(child.chunk_id, child.get_path()):
                    node.visited = True
                    deepest_hit_depth = depth
                    self.ghost.remove(child.chunk_id, child.get_path())
                    admit_nodes.append(node)
                elif depth >= depth_threshold and not self.ghost.is_empty():
                    reject_nodes.append(node)
                else:
                    admit_nodes.append(node)
                tn = child
            else:
                tn = tn.children[cid]
                ln = tn.linked_node
                # Remove; promote later
                self._remove_node(ln)
                if tn.tier == self.tiers[2]:
                    tn.tier = self.tiers[1]
                else:
                    tn.tier = self.tiers[0]
                ln.visited = True
                deepest_hit_depth = depth
                admit_nodes.append(ln)

        if deepest_hit_depth > 0:
            self.depth_mean.add(deepest_hit_depth)

        # Reverse nodes to preserve prefix order
        for node in reversed(admit_nodes):
            self._add_to_head(node, node.trie_node.tier)
        for node in reversed(reject_nodes):
            if self.num_tiers == 2:
                self.ghost.put(node.trie_node.chunk_id, node.trie_node.get_path())
                self._evict(node.trie_node)
            elif self.num_tiers == 3:
                self._add_to_head(node, self.tiers[2])

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
    cache = TieredTrieBuoyCache(bytes_per_token=1, cap_hbm=512, cap_dram=1024, cap_ssd=0, chunk_size=256)
    cache.access_prefix([0, 1, 2, 3], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([0, 4, 5, 6], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([7, 8, 9, 10], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([7, 11, 12, 13], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([0, 1, 2, 14], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.access_prefix([7, 8, 9, 15], [256, 256, 256, 256])
    cache.print_trie()
    cache.print_cache()

    cache.print_stats()
