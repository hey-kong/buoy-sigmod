import torch
from safetensors.torch import save_file, load_file
from transformers.cache_utils import DynamicCache


def alloc_cpu_buffer(dynamic_cache):
    buffer = []
    for layer in dynamic_cache.layers:
        K_host = torch.empty_like(layer.keys, device="cpu", pin_memory=True)
        V_host = torch.empty_like(layer.values, device="cpu", pin_memory=True)
        buffer.append((K_host, V_host))
    return buffer


def move_cache_to_cpu(dynamic_cache, buffer):
    for (K_host, V_host), layer in zip(buffer, dynamic_cache.layers):
        K = layer.keys.contiguous()
        V = layer.values.contiguous()
        K_host.copy_(K)
        V_host.copy_(V)
        layer.keys = K_host
        layer.values = V_host


def move_cache_to_gpu(dynamic_cache, device):
    for layer in dynamic_cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)


def save_kvcache(c: DynamicCache, cache_file_path: str):
    tensor_dict = {}
    for i, layer in enumerate(c.layers):
        tensor_dict[f"key_cache_{i}"] = layer.keys.contiguous()
        tensor_dict[f"value_cache_{i}"] = layer.values.contiguous()
    save_file(tensor_dict, cache_file_path)


def load_kvcache(cache_file_path: str) -> DynamicCache:
    tensors = load_file(cache_file_path)
    layer_ids = sorted(set(int(k.split("_")[-1]) for k in tensors.keys()))
    cache = DynamicCache()
    for i in layer_ids:
        cache.update(
            key_states=tensors[f"key_cache_{i}"],
            value_states=tensors[f"value_cache_{i}"],
            layer_idx=i,
        )
    return cache


def pin_kvcache(cache: DynamicCache):
    for layer in cache.layers:
        layer.keys = layer.keys.pin_memory()
        layer.values = layer.values.pin_memory()


def save_kvcache_quantized(pack, cache_file_path):
    tensor_dict = {}
    for k, v in pack.items():
        if isinstance(v, torch.Tensor):
            tensor_dict[k] = v
    save_file(tensor_dict, cache_file_path)


def load_kvcache_quantized(cache_file_path: str) -> dict:
    pack = load_file(cache_file_path)
    return pack


def pin_kvcache_quantized(pack: dict):
    for k, v in pack.items():
        pack[k] = v.pin_memory()
