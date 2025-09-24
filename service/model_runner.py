from hashlib import sha256
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

class ModelRunner:
    def __init__(self, model, device, ssd_path, bytes_per_token, hbm_size, dram_size, ssd_size, policy: str, chunk_map):
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype='auto').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if policy.lower() == 'lru':
            from cache_policies.lru import TieredLRUCache
            self.cache = TieredLRUCache(
                device=device,
                bytes_per_token=bytes_per_token,
                ssd_path=ssd_path,
                hbm_capacity=hbm_size,
                dram_capacity=dram_size,
                ssd_capacity=ssd_size
            )
        elif policy.lower() == 'lfu':
            from cache_policies.lfu import TieredLFUCache
            self.cache = TieredLFUCache(
                device=device,
                bytes_per_token=bytes_per_token,
                ssd_path=ssd_path,
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
        dynamic_cache = DynamicCache()

        prefix_ids = []
        hash_keys = []
        token_cnts = []
        chunk_inputs = []

        hit_string = ""
        full_input_ids = None
        full_attention_mask = None

        for cid in hash_ids:
            prefix_ids.append(str(cid))
            hash_input = " ".join(prefix_ids)
            hash_key = sha256(hash_input.encode('utf-8')).hexdigest()
            hash_keys.append(hash_key)

            request_str = self.chunk_map[str(cid)]
            inputs = self.tokenizer(request_str, return_tensors="pt", add_special_tokens=False).to('cuda')
            input_len = inputs['input_ids'].shape[1]
            token_cnts.append(input_len)
            chunk_inputs.append(inputs)

            if full_input_ids is None:
                full_input_ids = inputs['input_ids']
                full_attention_mask = inputs['attention_mask']
            else:
                full_input_ids = torch.cat([full_input_ids, inputs['input_ids']], dim=1)
                full_attention_mask = torch.cat([full_attention_mask, inputs['attention_mask']], dim=1)
        
        caches = self.cache.on_access(hash_keys, token_cnts)

        miss = False
        for idx, cache in enumerate(caches):
            if not miss and cache is not None:
                self._add_to_cache(dynamic_cache, cache)
                hit_string += "1"
            else:
                miss = True
                inputs = chunk_inputs[idx]
                self.model(**inputs, past_key_values=dynamic_cache, use_cache=True)
                new_kv = self._extract_from_cache(dynamic_cache, token_cnts[idx])
                self.cache.store(hash_keys[idx], new_kv)
                hit_string += "0"
        
        new_token = self.tokenizer(" ", return_tensors="pt", add_special_tokens=False).to('cuda')
        full_input_ids = torch.cat([full_input_ids, new_token['input_ids']], dim=1)
        self.model.generate(
            full_input_ids,
            attention_mask=full_attention_mask,
            past_key_values=dynamic_cache,
            use_cache=True,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        if chat_id % 1000 == 999:
            self.cache.print_status(chat_id)
        
        return sum(token_cnts), hit_string
    
    @staticmethod
    def _add_to_cache(dst_cache: DynamicCache, cache: DynamicCache):
        while len(dst_cache.layers) < len(cache.layers):
            dst_cache.layers.append(DynamicLayer())
        
        for i, (dst_layer, inc_layer) in enumerate(zip(dst_cache.layers, cache.layers)):
            k = inc_layer.keys
            v = inc_layer.values
            if k is not None and v is not None:
                dst_layer.update(k.to('cuda', non_blocking=True), v.to('cuda', non_blocking=True))
    
    @staticmethod
    def _extract_from_cache(dynamic_cache: DynamicCache, length):
        extracted = DynamicCache()
        for i, layer in enumerate(dynamic_cache.layers):
            k = layer.keys[:, :, -length:, :]
            v = layer.values[:, :, -length:, :]
            extracted.update(k, v, i)
        return extracted
