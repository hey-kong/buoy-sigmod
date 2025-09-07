import torch
from safetensors.torch import save_file, load_file
from transformers.cache_utils import DynamicCache


def extract_kvcache(dynamic_cache, device):
    layers_out = []
    for i, layer in enumerate(dynamic_cache.layers):
        K = layer.keys.detach().to(device, non_blocking=True).contiguous()
        V = layer.values.detach().to(device, non_blocking=True).contiguous()
        layers_out.append({"k": K, "v": V})
    return layers_out


def move_dynamic_cache_to_device(dynamic_cache, device):
    for i, layer in enumerate(dynamic_cache.layers):
        layer.keys = layer.keys.to(device, non_blocking=True).contiguous()
        layer.values = layer.values.to(device, non_blocking=True).contiguous()


def save_kvcache(c: DynamicCache, cache_file_path: str):
    tensor_dict = {}
    for i, layer in enumerate(c.layers):
        tensor_dict[f"key_cache_{i}"] = layer.keys.cpu().contiguous()
        tensor_dict[f"value_cache_{i}"] = layer.values.cpu().contiguous()
    save_file(tensor_dict, cache_file_path)


def load_kvcache(cache_file_path: str) -> DynamicCache:
    tensors = load_file(cache_file_path)

    layer_ids = sorted(set(int(k.split("_")[-1]) for k in tensors.keys()))

    cache = DynamicCache()
    for i in layer_ids:
        cache.update(
            key_states=tensors[f"key_cache_{i}"].pin_memory(),
            value_states=tensors[f"value_cache_{i}"].pin_memory(),
            layer_idx=i,
        )
    return cache


def save_kvcache_quantized(pack, cache_file_path):
    tensor_dict = {}
    for k, v in pack.items():
        if isinstance(v, torch.Tensor):
            tensor_dict[k] = v
    save_file(tensor_dict, cache_file_path)


def load_kvcache_quantized(cache_file_path: str):
    return load_file(cache_file_path)
