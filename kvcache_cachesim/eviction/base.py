import abc


class TrieNode:
    def __init__(self, chunk_id, parent=None):
        self.chunk_id = chunk_id
        self.parent = parent
        self.children = {}
        self.linked_node = None
        self.token_count = 0
        self.kv_bytes = 0
        self.tier = None

    def get_path(self) -> list[int]:
        path = []
        node = self
        while node is not None:
            if node.chunk_id is not None:
                path.append(node.chunk_id)
            node = node.parent
        return path[::-1]


class LinkedNode:
    def __init__(self, trie_node=None):
        self.prev = self.next = None
        self.trie_node = trie_node


class TieredCache(abc.ABC):
    def __init__(self,
                 bytes_per_token: int,
                 cap_hbm: int,
                 cap_dram: int,
                 cap_ssd: int):
        self.name = ""
        self.bytes_per_token = bytes_per_token
        self.tiers = ['hbm', 'dram', 'ssd']
        self.max_bytes = {'hbm': cap_hbm, 'dram': cap_dram, 'ssd': cap_ssd}
        self.cur_bytes = {t: 0 for t in self.tiers}
        self.total_access = 0
        self.total_chunks = 0
        self.tier_hit_chunks = {tier: 0 for tier in self.tiers}

        if cap_dram > 0:
            self.next_tier = {'hbm': 'dram'}
            if cap_ssd > 0:
                self.next_tier['dram'] = 'ssd'
        elif cap_ssd > 0:
            self.next_tier = {'hbm': 'ssd'}
        else:
            self.next_tier = {}

        self.last_tier = 'hbm'
        while self.last_tier in self.next_tier:
            self.last_tier = self.next_tier[self.last_tier]

    def print_stats(self):
        print(f"Algorithm: {self.name}")
        if self.total_access == 0:
            print("No access records yet.")
            return

        overall_rate = 0.0
        for tier in self.tiers:
            hit_chunks = self.tier_hit_chunks[tier]
            tier_rate = hit_chunks / self.total_chunks if self.total_chunks else 0.0

            overall_rate += tier_rate
            print(f"{tier.upper():4} hit rate: {hit_chunks}/{self.total_chunks} = {tier_rate:.2%}")
        print(f"Overall hit rate: {overall_rate:.2%}")

    def get_stats(self):
        stat = {"Algorithm": self.name}
        if self.total_access == 0:
            for tier in self.tiers:
                stat[f"{tier.upper()} hit rate"] = "N/A"
            stat["Overall hit rate"] = "N/A"
            return stat

        overall_rate = 0.0
        for tier in self.tiers:
            hit_chunks = self.tier_hit_chunks[tier]
            tier_rate = hit_chunks / self.total_chunks if self.total_chunks else 0.0
            stat[f"{tier.upper()} hit rate"] = f"{tier_rate:.2%}"
            overall_rate += tier_rate
        stat["Overall hit rate"] = f"{overall_rate:.2%}"
        return stat
