import time

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM

from quantization.cachegen_basics import CacheGenConfig
from quantization.kv_cache_quant import quantize_dynamic_cache, dequantize_dynamic_cache
from transfer.kvcache_io import extract_kvcache, save_kvcache_quantized, load_kvcache_quantized

MODEL_PATH = "/data/llm/Llama-3.1-8B-Instruct"
DEVICE = "cuda"

INITIAL_PROMPT = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
"""
default_prompt = "To be or not to be, that is the question. " * 1000
SUFFIX = """<|eot_id|>\n"
<|start_header_id|>assistant<|end_header_id|>
"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16).to(DEVICE)

inputs = tokenizer(INITIAL_PROMPT + default_prompt, return_tensors="pt").to(DEVICE)
cfg = CacheGenConfig.from_model_name(MODEL_PATH)

# 1) 得到原生 prefix_cache
prefix_cache = DynamicCache()
with torch.no_grad():
    prefix_cache = model(**inputs, past_key_values=prefix_cache, use_cache=True).past_key_values

# 2) 量化保存
kvcache_file_path = "./quant_cache.safetensors"

# Step 1: 提取（在CPU上保留张量）
layers = extract_kvcache(prefix_cache, dtype=torch.float16, device="cpu")

# Step 2: 量化（CPU）
pack = quantize_dynamic_cache(prefix_cache, cfg)

# Step 3: 保存
save_kvcache_quantized(pack, kvcache_file_path)

# 3) 反量化 -> legacy -> DynamicCache

# Step 1: 只读量化数据
pack = load_kvcache_quantized(kvcache_file_path)

# Step 2: 反量化（GPU）
kv_layers = dequantize_dynamic_cache(pack, device=torch.device(DEVICE))

# Step 3: 还原 DynamicCache
quant_cache = DynamicCache.from_legacy_cache(kv_layers)

# test: with cache
new_inputs = tokenizer(INITIAL_PROMPT + default_prompt + SUFFIX, return_tensors="pt").to(model.device.type)
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    output1 = model.generate(**new_inputs, past_key_values=prefix_cache, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.eos_token_id, max_new_tokens=10)
if torch.cuda.is_available():
    torch.cuda.synchronize()
end = time.perf_counter()
generated_ids = output1[0]
input_length = new_inputs.input_ids.shape[1]
response = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
print("time:", end - start)
print("text:", response)

# test: with quant_cache
new_inputs = tokenizer(INITIAL_PROMPT + default_prompt + SUFFIX, return_tensors="pt").to(model.device.type)
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    output2 = model.generate(**new_inputs, past_key_values=quant_cache, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.eos_token_id, max_new_tokens=10)
if torch.cuda.is_available():
    torch.cuda.synchronize()
end = time.perf_counter()
generated_ids = output2[0]
input_length = new_inputs.input_ids.shape[1]
response = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
print("time:", end - start)
print("text:", response)

# test: without cache
new_inputs = tokenizer(INITIAL_PROMPT + default_prompt + SUFFIX, return_tensors="pt").to(model.device.type)
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    output3 = model.generate(**new_inputs, past_key_values=None, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.eos_token_id, max_new_tokens=10)
if torch.cuda.is_available():
    torch.cuda.synchronize()
end = time.perf_counter()
generated_ids = output3[0]
input_length = new_inputs.input_ids.shape[1]
response = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
print("time:", end - start)
print("text:", response)
