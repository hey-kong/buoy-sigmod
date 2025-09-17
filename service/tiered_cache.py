import abc
import os
import shutil
from typing import List, Optional, Tuple

import torch

class CacheChunk:
    def __init__(
            self,
            chunk_id_hash: str,
            location: str,
            kv_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        ):
        self.chunk_id_hash = chunk_id_hash
        self.location = location
        self.kv_data = kv_data
        self.prev = None
        self.next = None
        if kv_data is not None:
            self.token_count = kv_data[0][0].shape[-2]
            self.kv_bytes = sum(k.element_size() * k.nelement() + v.element_size() * v.nelement() for k, v in kv_data)
        else:
            self.token_count = 0
            self.kv_bytes = 0

class ChunkLinkedList:
    def __init__(self):
        self.head = CacheChunk("head", None)
        self.tail = CacheChunk("tail", None)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def add_to_head(self, chunk: CacheChunk):
        chunk.prev = self.head
        chunk.next = self.head.next
        self.head.next.prev = chunk
        self.head.next = chunk
    
    def remove(self, chunk: CacheChunk):
        chunk.prev.next = chunk.next
        chunk.next.prev = chunk.prev
        chunk.prev = None
        chunk.next = None
    
    def get_tail(self) -> Optional[CacheChunk]:
        if self.tail.prev == self.head:
            return None
        tail_chunk = self.tail.prev
        return tail_chunk

class TieredCache(abc.ABC):
    def __init__(self, ssd_path, hbm_capacity, dram_capacity, ssd_capacity):
        self.tiers = ['hbm', 'dram', 'ssd']
        if os.path.exists(ssd_path):
            shutil.rmtree(ssd_path)
        os.makedirs(ssd_path)
        self.ssd_path = ssd_path
        self.max_bytes = {
            'hbm': hbm_capacity,
            'dram': dram_capacity,
            'ssd': ssd_capacity
        }
        self.cur_bytes = {tier: 0 for tier in self.tiers}
        self.kv_chunks = {tier: ChunkLinkedList() for tier in self.tiers}
    
    # only move data, should not be called without kv_chunks and cur_bytes change
    def move_chunk_data(self, chunk: CacheChunk, dest_tier):
        source_tier = chunk.location
        path = os.path.join(self.ssd_path, chunk.chunk_id_hash + '.pt')
        if source_tier in ['hbm', 'dram'] and dest_tier == 'ssd':
            torch.save(chunk.kv_data, path)
            chunk.kv_data = None
        elif source_tier == 'hbm' and dest_tier == 'dram':
            chunk.kv_data = [ (k.cpu(), v.cpu()) for k, v in chunk.kv_data ]
        elif source_tier == 'dram' and dest_tier == 'hbm':
            chunk.kv_data = [ (k.cuda(), v.cuda()) for k, v in chunk.kv_data ]
        elif source_tier == 'ssd' and dest_tier == 'hbm':
            chunk.kv_data = torch.load(path, map_location='cuda')
            os.remove(path)
        elif source_tier == 'ssd' and dest_tier == 'dram':
            chunk.kv_data = torch.load(path)
            os.remove(path)
        chunk.location = dest_tier
    
    def move_chunk_to_head(self, chunk: CacheChunk, dest_tier):
        source_tier = chunk.location
        self.move_chunk_data(chunk, dest_tier)
        self.kv_chunks[source_tier].remove(chunk)
        self.kv_chunks[dest_tier].add_to_head(chunk)
        self.cur_bytes[source_tier] -= chunk.kv_bytes
        self.cur_bytes[dest_tier] += chunk.kv_bytes
    
    def get_chunk(self, chunk_id_hash) -> Optional[CacheChunk]:
        for tier in self.tiers:
            c = self.kv_chunks[tier].head.next
            while c != self.kv_chunks[tier].tail:
                if c.chunk_id_hash == chunk_id_hash:
                    return c
                c = c.next
        return None
    
    def add_chunk(self, chunk_id_hash, location, kv_data):
        chunk = CacheChunk(chunk_id_hash, location, kv_data)
        self.kv_chunks[location].add_to_head(chunk)
        self.cur_bytes[location] += chunk.kv_bytes
        return chunk
    
    def remove_chunk(self, chunk: CacheChunk):
        self.kv_chunks[chunk.location].remove(chunk)
        self.cur_bytes[chunk.location] -= chunk.kv_bytes
        if chunk.location == 'ssd':
            path = os.path.join(self.ssd_path, chunk.chunk_id_hash + '.pt')
            os.remove(path)
    
    @abc.abstractmethod
    def on_access(self, chunk_id_hash):
        raise NotImplementedError
    
    @abc.abstractmethod
    def store(self, chunk_id_hash, kv_data):
        raise NotImplementedError
