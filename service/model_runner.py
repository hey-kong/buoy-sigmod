from hashlib import sha256
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from cache_policies.lru import TieredLRUCache

class ModelRunner:
    def __init__(self, model, hbm_size, dram_size, ssd_size, policy: str, chunk_map):
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype="auto").to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if policy.lower() == "lru":
            self.cache = TieredLRUCache(
                ssd_path="tmp",
                hbm_capacity=hbm_size,
                dram_capacity=dram_size,
                ssd_capacity=ssd_size
            )
        else:
            raise ValueError(policy)
        self.hbm_size = hbm_size
        self.dram_size = dram_size
        self.ssd_size = ssd_size
        self.policy = policy
        self.chunk_map = chunk_map
    
    @torch.inference_mode()
    def process_request(self, chat_id, hash_ids):
        start_time = time.perf_counter()

        dynamic_cache = DynamicCache()
        prefix_ids = []
        tokens_cnt = 0
        hit_string = ""

        for cid in hash_ids:
            prefix_ids.append(str(cid))
            hash_input = " ".join(prefix_ids)
            hash_key = sha256(hash_input.encode('utf-8')).hexdigest()

            kv_chunk = self.cache.on_access(hash_key)

            if kv_chunk is not None:
                self._add_to_cache(dynamic_cache, kv_chunk.kv_data)
                tokens_cnt += kv_chunk.token_count
                hit_string += "1"
                continue

            request_str = self.chunk_map[cid]
            inputs = self.tokenizer(
                request_str,
                return_tensors="pt",
                add_special_tokens=False
            ).to('cuda')
            input_len = inputs['input_ids'].shape[1]
            self.model(
                **inputs,
                past_key_values=dynamic_cache,
                use_cache=True
            )
            new_kv = self._extract_from_cache(dynamic_cache, input_len)

            self.cache.store(hash_key, new_kv)

            tokens_cnt += input_len
            hit_string += "0"

        new_token = self.tokenizer(
            " ",
            return_tensors="pt",
            add_special_tokens=False
        ).to('cuda')
        self.model(
            **new_token,
            past_key_values=dynamic_cache,
            use_cache=True
        )

        ttft = time.perf_counter() - start_time

        return ttft, tokens_cnt, hit_string
    
    def _add_to_cache(self, dynamic_cache: DynamicCache, kv_data):
        while len(dynamic_cache.layers) < len(kv_data):
            dynamic_cache.layers.append(DynamicLayer())
        
        for layer_id, (key, value) in enumerate(kv_data):
            dynamic_cache.layers[layer_id].update(key, value)
    
    def _extract_from_cache(self, dynamic_cache: DynamicCache, length):
        extracted = []
        for layer in dynamic_cache.layers:
            key = layer.keys[:, :, -length:, :].clone()
            value = layer.values[:, :, -length:, :].clone()
            extracted.append((key, value))
        return extracted
