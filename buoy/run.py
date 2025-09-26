import time

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM

from transfer.quantization.cachegen_basics import CacheGenConfig
from transfer.quantization.kvcache_quant import quantize_dynamic_cache, dequantize_dynamic_cache
from transfer.kvcache_io import alloc_cpu_buffer, move_cache_to_cpu, move_cache_to_gpu, save_kvcache_quantized, load_kvcache_quantized, pin_kvcache_quantized, move_pack_to_gpu

MODEL_PATH = "/data/llm/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
DTYPE = torch.float16

PROMPT_PREFIX_MAP = {
    'llama': (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
    ),
    'mistral': (
        "<s>[INST] "
    ),
}

PROMPT_SUFFIX_MAP = {
    'llama': (
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    ),
    'mistral': (
        " [/INST]"
    ),
}

if "llama" in MODEL_PATH.lower():
    model_key = "llama"
elif "mistral" in MODEL_PATH.lower():
    model_key = "mistral"
else:
    raise ValueError(f"Unknown model type from MODEL_PATH: {MODEL_PATH}")
prefix = PROMPT_PREFIX_MAP[model_key]
suffix = PROMPT_SUFFIX_MAP[model_key]
default_prompt = "Hi " * 4076

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE).to(DEVICE)

inputs = tokenizer(prefix + default_prompt, return_tensors="pt").to(DEVICE)
cfg = CacheGenConfig.from_model_name(MODEL_PATH)

# 1) 得到原生 prefix_cache
prefix_cache = DynamicCache()
with torch.no_grad():
    prefix_cache = model(**inputs, past_key_values=prefix_cache, use_cache=True).past_key_values

# 2) 量化保存
kvcache_file_path = "./quant_cache.safetensors"

# Step 1: 提取（在CPU上保留张量）
buffer = alloc_cpu_buffer(prefix_cache)
move_cache_to_cpu(prefix_cache, buffer)

# Step 2: 量化（CPU）
pack = quantize_dynamic_cache(prefix_cache, cfg)

# Step 3: 保存
save_kvcache_quantized(pack, kvcache_file_path)

# 3) 反量化 -> legacy -> DynamicCache

# Step 1: 只读量化数据
pack = load_kvcache_quantized(kvcache_file_path)
pin_kvcache_quantized(pack)

# Step 2: 传输到 GPU
move_pack_to_gpu(pack, device=torch.device(DEVICE))

# Step 3: 反量化
kv_layers = dequantize_dynamic_cache(pack)

# Step 4: 还原 DynamicCache
quant_cache = DynamicCache.from_legacy_cache(kv_layers)
move_cache_to_gpu(prefix_cache, device=torch.device(DEVICE))

# test: with cache
new_inputs = tokenizer(prefix + default_prompt + suffix, return_tensors="pt").to(DEVICE)
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
new_inputs = tokenizer(prefix + default_prompt + suffix, return_tensors="pt").to(DEVICE)
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
new_inputs = tokenizer(prefix + default_prompt + suffix, return_tensors="pt").to(DEVICE)
num_prompt_tokens = new_inputs["input_ids"].shape[1]
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
