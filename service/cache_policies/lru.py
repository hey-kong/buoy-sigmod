from tiered_cache import TieredCache

class TieredLRUCache(TieredCache):
    def __init__(self, ssd_path, hbm_capacity, dram_capacity, ssd_capacity):
        super().__init__(ssd_path, hbm_capacity, dram_capacity, ssd_capacity)
    
    def on_access(self, chunk_id_hash):
        chunk = self.get_chunk(chunk_id_hash)
        if chunk:
            self.move_chunk_to_head(chunk, 'hbm')
            self._evict_if_needed()
            return chunk
        return None
    
    def store(self, chunk_id_hash, kv_data):
        self.add_chunk(chunk_id_hash, 'hbm', kv_data)
        self._evict_if_needed()
    
    def _evict_if_needed(self):
        while self.cur_bytes['hbm'] > self.max_bytes['hbm']:
            tail_chunk = self.kv_chunks['hbm'].get_tail()
            assert tail_chunk is not None
            self.move_chunk_to_head(tail_chunk, 'dram')
        while self.cur_bytes['dram'] > self.max_bytes['dram']:
            tail_chunk = self.kv_chunks['dram'].get_tail()
            assert tail_chunk is not None
            self.move_chunk_to_head(tail_chunk, 'ssd')
        while self.cur_bytes['ssd'] > self.max_bytes['ssd']:
            tail_chunk = self.kv_chunks['ssd'].get_tail()
            assert tail_chunk is not None
            self.remove_chunk(tail_chunk)
