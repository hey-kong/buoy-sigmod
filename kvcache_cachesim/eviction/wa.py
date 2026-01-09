import json
import math
from collections import Counter, defaultdict
from typing import List

from .base import TrieNode, LinkedNode, TieredCache


def collect_stats(jsonl_path):
    freq_counter = Counter()
    access_times = defaultdict(list)

    with open(jsonl_path, "r") as f:
        for t, line in enumerate(f):
            record = json.loads(line)
            for hid in record["hash_ids"]:
                freq_counter[hid] += 1
                access_times[hid].append(t)

    gaps = []
    for times in access_times.values():
        if len(times) < 2:
            continue
        for i in range(1, len(times)):
            gaps.append(times[i] - times[i - 1])
    mean_gap = sum(gaps) / len(gaps)

    return freq_counter, mean_gap


def build_reuse_estimator(freq_counter):
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

    reuse_estimator = {}
    for k in range(0, max_f + 1):
        denom = tail[k]
        if denom == 0:
            reuse_estimator[k] = 0.0
        else:
            reuse_estimator[k] = tail[k + 1] / denom
    reuse_estimator[max_f + 1] = 0.0
    return reuse_estimator


class WATrieNode(TrieNode):
    def __init__(self, chunk_id, parent=None):
        super().__init__(chunk_id, parent)

        self.freq = 0


class WALinkedNode(LinkedNode):
    def __init__(self, trie_node=None):
        super().__init__(trie_node)

        self.last_access = 0
        self.reuse_prob = 0
        self.offset = 0
        self.priority = (self.reuse_prob, -self.offset)

    def update_priority(self, total_access, mean_gap):
        waited = total_access - self.last_access
        w = math.exp(- waited / mean_gap)
        self.priority = (self.reuse_prob * w, -self.offset)


class TieredWACache(TieredCache):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int,
                 reuse_estimator,
                 mean_gap):
        super().__init__(bytes_per_token, cap_hbm, cap_dram, cap_ssd)
        self.name = "WA"

        self.head = {}
        self.tail = {}
        for t in self.tiers:
            head_trie_node = WATrieNode(chunk_id=None)
            head_trie_node.tier = t
            tail_trie_node = WATrieNode(chunk_id=None)
            tail_trie_node.tier = t
            # Create head and tail linked nodes with associated trie nodes
            self.head[t] = WALinkedNode(trie_node=head_trie_node)
            self.tail[t] = WALinkedNode(trie_node=tail_trie_node)
            # Link head and tail
            self.head[t].next = self.tail[t]
            self.tail[t].prev = self.head[t]

        self.reuse_estimator = reuse_estimator
        self.mean_gap = int(mean_gap)

        self.root = WATrieNode(None)

    def _add_to_head(self, node: WALinkedNode, tier: str):
        node.next = self.head[tier].next
        node.prev = self.head[tier]
        self.head[tier].next.prev = node
        self.head[tier].next = node

        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

    def _add_by_priority(self, node: WALinkedNode, tier: str):
        cur = self.tail[tier].prev
        if cur.priority > node.priority:
            # quickly insert the lowest priority nodes at tail
            node.next = cur.next
            node.prev = cur
            cur.next.prev = node
            cur.next = node
        else:
            cur = self.head[tier].next
            while cur is not self.tail[tier] and cur.priority > node.priority:
                cur = cur.next
            node.prev = cur.prev
            node.next = cur
            cur.prev.next = node
            cur.prev = node
        node.trie_node.tier = tier
        self.cur_bytes[tier] += node.trie_node.kv_bytes

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
                child = WATrieNode(cid, parent=tn)
                tn.children[cid] = child
                child.token_count = cnt
                child.kv_bytes = sz

                ln = WALinkedNode(child)
                child.linked_node = ln
                tn = child
            else:
                tn = tn.children[cid]
                if tn.linked_node:
                    # Remove; reinsert later by prefix order
                    ln = tn.linked_node
                    self._remove_node(ln)
                else:
                    ln = WALinkedNode(tn)
                    tn.linked_node = ln

            tn.freq += 1
            ln.last_access = self.total_access
            ln.reuse_prob = self.reuse_estimator.get(tn.freq, 0.0)
            ln.offset = depth
            ln.update_priority(self.total_access, self.mean_gap)
            self._add_by_priority(ln, self.tiers[0])

        # Demotion and eviction
        for tier in self.tiers:
            self._demote_if_needed(tier)

        if self.total_access % self.mean_gap == 0:
            self.refresh(self.tiers[0])

    def refresh(self, tier: str):
        def _unlink_all():
            head = self.head[tier]
            tail = self.tail[tier]
            head.next = tail
            tail.prev = head

        def _insert_before(node, ref):
            node.prev = ref.prev
            node.next = ref
            ref.prev.next = node
            ref.prev = node

        head = self.head[tier]
        tail = self.tail[tier]

        nodes = []
        node = head.next
        while node is not tail:
            next = node.next
            node.update_priority(self.total_access, self.mean_gap)
            node.prev = None
            node.next = None
            nodes.append(node)
            node = next

        nodes.sort(key=lambda n: n.priority, reverse=True)
        _unlink_all()
        for n in nodes:
            _insert_before(n, tail)

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
