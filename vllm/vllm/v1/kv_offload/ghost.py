# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

GhostKey = TypeVar("GhostKey", bound=Hashable)


class Ghost(Generic[GhostKey]):
    """Bounded FIFO ghost list for recently evicted KV pages.

    The ghost list stores only page identities, not KV payloads. The optional
    value lets callers disambiguate multiple entries for the same key.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue: OrderedDict[tuple[GhostKey, tuple[int, ...] | None], None] = (
            OrderedDict()
        )
        self.index: dict[GhostKey, set[tuple[int, ...] | None]] = {}

    def put(self, key: GhostKey, value: list[int] | None = None):
        v: tuple[int, ...] | None = tuple(value) if value is not None else None
        entry = (key, v)

        if entry in self.queue:
            self.queue.pop(entry)
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

    def remove(self, key: GhostKey, value: list[int] | None = None):
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

    def exists(self, key: GhostKey, value: list[int] | None = None) -> bool:
        if key not in self.index:
            return False
        v = tuple(value) if value is not None else None
        return v in self.index[key]

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def __contains__(self, key: GhostKey) -> bool:
        return key in self.index

    def __len__(self):
        return len(self.queue)

    def print_ghost(self):
        if not self.queue:
            print("Ghost contents: None")
            return
        print("Ghost contents (FIFO order):")
        for k, v in self.queue:
            print(f"  ({k}, {list(v) if v is not None else None})")

    def __repr__(self):
        return f"{list(self.queue.keys())}"
