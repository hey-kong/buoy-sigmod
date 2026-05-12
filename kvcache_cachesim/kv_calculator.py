import json


def calc_kv_cache_bytes(
        model_name: str,
        seq_len: int,
        dtype: str,
        batch_size: int = 1,
        config_path: str = "modelconfig.json"
) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' not found in {config_path}.")
    cfg = configs[model_name]

    required_keys = ["num_hidden_layers", "num_key_value_heads", "head_dim"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing key '{key}' in config for model '{model_name}'")

    num_hidden_layers = int(cfg["num_hidden_layers"])
    num_key_value_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])

    dtype_size_map = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1
    }
    if dtype not in dtype_size_map:
        raise ValueError(
            "Unsupported dtype, must be one of: float32, float16, bfloat16, int8"
        )
    dtype_size = dtype_size_map[dtype]

    total_elements = (
            num_hidden_layers
            * batch_size
            * num_key_value_heads
            * seq_len
            * head_dim
            * 2
    )
    total_bytes = total_elements * dtype_size
    return total_bytes


if __name__ == "__main__":
    model = "meta-llama/Llama-3.1-8B-Instruct"
    seq_len = 4096
    dtype = "float16"

    size_bytes = calc_kv_cache_bytes(model, seq_len, dtype)
    size_gb = size_bytes / (1024 ** 3)
    print(f"{model} KV cache: {size_bytes} bytes / {size_gb:.2f} GiB (seq_len={seq_len}, dtype={dtype})")
