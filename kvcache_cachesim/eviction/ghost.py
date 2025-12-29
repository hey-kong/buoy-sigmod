from collections import OrderedDict
from typing import List, Optional, Tuple, Set


class Ghost:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue: OrderedDict[Tuple[int, Optional[Tuple[int, ...]]], None] = OrderedDict()
        self.index: dict[int, Set[Optional[Tuple[int, ...]]]] = {}

    def put(self, key: int, value: Optional[List[int]] = None):
        v: Optional[Tuple[int, ...]] = tuple(value) if value is not None else None
        entry = (key, v)

        if entry not in self.queue:
            self.queue[entry] = None
            if key not in self.index:
                self.index[key] = set()
            self.index[key].add(v)

        while len(self.queue) > self.capacity:
            old_entry, _ = self.queue.popitem(last=False)
            old_k, old_v = old_entry
            self.index[old_k].remove(old_v)
            if not self.index[old_k]:
                del self.index[old_k]

    def remove(self, key: int, value: Optional[List[int]] = None):
        if key not in self.index:
            return

        if value is None:
            vals_to_remove = list(self.index[key])
            for v in vals_to_remove:
                entry = (key, v)
                self.queue.pop(entry, None)
            del self.index[key]
            return

        v_tuple = tuple(value)
        entry = (key, v_tuple)

        self.queue.pop(entry, None)

        if v_tuple in self.index[key]:
            self.index[key].remove(v_tuple)
            if not self.index[key]:
                del self.index[key]

    def exists(self, key: int, value: Optional[List[int]] = None) -> bool:
        if key not in self.index:
            return False
        v = tuple(value) if value is not None else None
        return v in self.index[key]

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def __contains__(self, key: int) -> bool:
        return key in self.index

    def __len__(self):
        return len(self.queue)

    def print_ghost(self):
        if not self.queue:
            print("Ghost contents: None")
            return
        print("Ghost contents (FIFO order):")
        for (k, v) in self.queue.keys():
            print(f"  ({k}, {list(v) if v is not None else None})")

    def __repr__(self):
        return f"{list(self.queue.keys())}"
