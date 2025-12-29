import json
from collections import Counter
from typing import List

from .base import TrieNode, LinkedNode, TieredCache


def collect_frequency(jsonl_path):
    freq = Counter()
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            for hid in record["hash_ids"]:
                freq[hid] += 1
    return freq


def build_reuse_prob(freq_counter):
    hist = Counter(freq_counter.values())
    if not hist:
        return {}

    max_f = max(hist.keys())
    tail = [0] * (max_f + 2)
    for f, c in hist.items():
        if f <= max_f:
            tail[f] += c

    for k in range(max_f, -1, -1):
        tail[k] += tail[k + 1]

    reuse_prob = {}
    for k in range(0, max_f + 1):
        denom = tail[k]
        if denom == 0:
            reuse_prob[k] = 0.0
        else:
            reuse_prob[k] = tail[k + 1] / denom
    reuse_prob[max_f + 1] = 0.0
    return reuse_prob


class WALinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.reuse_prob = 0
        self.offset = 0
        self.priority = (self.reuse_prob, -self.offset)
        self.freq = 0

    def update_priority(self, reuse_prob):
        self.freq += 1
        self.reuse_prob = reuse_prob.get(self.freq, 0.0)
        self.priority = (self.reuse_prob, -self.offset)


class TieredWACache(TieredCache):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int,
                 reuse_prob):
        super().__init__(bytes_per_token, cap_hbm, cap_dram, cap_ssd)
        self.name = "WA"

        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = TrieNode(chunk_id=None)
            head_trie_node.tier = t
            tail_trie_node = TrieNode(chunk_id=None)
            tail_trie_node.tier = t
            # Create head and tail linked nodes with associated trie nodes
            self.head[t] = WALinkedNode(trie_node=head_trie_node)
            self.tail[t] = WALinkedNode(trie_node=tail_trie_node)
            # Link head and tail
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]

        self.reuse_prob = reuse_prob

        self.root = TrieNode(None)

    def _add_to_head(self, node: WALinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

    def _add_by_priority(self, node: WALinkedNode):
        cur_tier = self.tiers[0]
        if self.cur_bytes[cur_tier] >= self.max_bytes[cur_tier] and node.priority < self.tail[cur_tier].prev.priority:
            cur_tier = self.next_tier[cur_tier]

        cur = self.head[cur_tier].next
        while cur is not self.tail[cur_tier] and cur.priority > node.priority:
            cur = cur.next

        node.prev = cur.prev
        node.next = cur
        cur.prev.next = node
        cur.prev = node
        node.trie_node.tier = cur_tier
        self.cur_bytes[cur_tier] += node.trie_node.kv_bytes

    def _remove_node(self, node: WALinkedNode):
        if node.prev is None or node.next is None:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

        tn = node.trie_node
        self.cur_bytes[tn.tier] -= tn.kv_bytes

    def _pop_tail(self, tier: str) -> WALinkedNode:
        tail = self.tail[tier]
        node = tail.prev
        if node is self.head[tier]:
            return None
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

            # Determine next tier
            next_tier = self.next_tier.get(tier, None)

            if next_tier:
                # Demote to next tier
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
        depth = 0
        for cid, cnt in zip(chunk_ids, token_counts):
            depth += 1
            sz = cnt * self.bytes_per_token

            if cid not in node.children:
                child = TrieNode(cid, parent=node)
                node.children[cid] = child
                child.token_count = cnt
                child.kv_bytes = sz

                ln = WALinkedNode(child)
                ln.offset = depth
                child.linked_node = ln
                node = child
            else:
                node = node.children[cid]
                if node.linked_node:
                    # Remove; add by priority later
                    ln = node.linked_node
                    self._remove_node(ln)
                else:
                    ln = WALinkedNode(node)
                    ln.offset = depth
                    node.linked_node = ln

            visited.append(node.linked_node)

        for ln in visited:
            ln.update_priority(self.reuse_prob)
            self._add_by_priority(ln)

        # Demotion and eviction
        for tier in self.tiers:
            self._demote_if_needed(tier)

    def print_cache(self):
        for tier in self.tiers:
            lst = []
            cur = self.head[tier].next
            while cur is not self.tail[tier]:
                chunk_id = getattr(cur.trie_node, "chunk_id", "?")
                reuse_prob = getattr(cur, "reuse_prob", "?")
                offset = getattr(cur, "offset", "?")
                lst.append(f"{chunk_id}({reuse_prob}, {offset})")
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
