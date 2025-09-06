import torch
from safetensors.torch import save_file, load_file
from transformers.cache_utils import DynamicCache


def extract_kvcache(c: DynamicCache, dtype=torch.float16, device="cpu"):
    layers_out = []
    for i, layer in enumerate(c.layers):
        k = layer.keys.detach().to(device).to(dtype).contiguous()
        v = layer.values.detach().to(device).to(dtype).contiguous()
        layers_out.append({"k": k, "v": v})
    return layers_out


def move_kv_layers_to_device(kv_layers, device):
    new_layers = []
    for k, v in kv_layers:
        new_layers.append((
            k.to(device, non_blocking=True),
            v.to(device, non_blocking=True)
        ))
    return new_layers


def save_kvcache_quantized(pack, cache_file_path):
    save_dict = {}
    for k, v in pack.items():
        if isinstance(v, torch.Tensor):
            save_dict[k] = v
    save_file(save_dict, cache_file_path)


def load_kvcache_quantized(cache_file_path: str):
    return load_file(cache_file_path)
