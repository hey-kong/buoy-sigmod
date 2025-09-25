from collections import OrderedDict
from typing import List, Optional, Set


class Ghost:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: OrderedDict[int, Set[Optional[tuple[int, ...]]]] = OrderedDict()

    def put(self, key: int, value: Optional[List[int]] = None):
        if key not in self.data:
            self.data[key] = set()
        self.data[key].add(tuple(value) if value is not None else None)

        if len(self.data) > self.capacity:
            self.data.popitem(last=False)

    def get(self, key: int) -> Optional[Set[Optional[tuple[int, ...]]]]:
        return self.data.get(key, None)

    def exists(self, key: int, value: Optional[List[int]] = None) -> bool:
        if key not in self.data:
            return False
        v = tuple(value) if value is not None else None
        return v in self.data[key]

    def print_ghost(self):
        if not self.data:
            print("Ghost contents: None")
            return

        print("Ghost contents:")
        for k, v in self.data.items():
            vals = list(v) if v else []
            print(f"  {k}: {vals}")

    def __contains__(self, key: int) -> bool:
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{ {k: list(v) for k, v in self.data.items()} }"
