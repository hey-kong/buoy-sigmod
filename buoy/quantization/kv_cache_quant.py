import torch
from typing import List, Tuple, Dict, Optional


def quant(bins: int, qA: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if bins == 256:
        C = bins // 2 - 1
        max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
        max1 = torch.clamp(max1, min=1e-12)
        xq = torch.round(qA * (C / max1)).to(torch.int8)
        xq = torch.clamp(xq, min=-C, max=C)
        return xq, max1
    elif bins == 65536:
        return qA.to(torch.float16).contiguous(), None
    else:
        raise ValueError(f"Unsupported bins={bins}, only 256 (int8) or 65536 (float16) allowed")


def dequant(bins: int, xq: torch.Tensor, max1: torch.Tensor) -> torch.Tensor:
    if bins == 256:
        C = bins // 2 - 1
        return (xq.to(torch.float32) / C) * max1
    elif bins == 65536:
        return xq.to(torch.float16)
    else:
        raise ValueError(f"Unsupported bins={bins}")


def make_key_bins(cfg) -> torch.Tensor:
    ret = torch.zeros(cfg.nlayers, dtype=torch.int32)
    for spec in cfg.kspecs:
        ret[spec.start_layer: spec.end_layer] = spec.bins
    return ret


def make_value_bins(cfg) -> torch.Tensor:
    ret = torch.zeros(cfg.nlayers, dtype=torch.int32)
    for spec in cfg.vspecs:
        ret[spec.start_layer: spec.end_layer] = spec.bins
    return ret


def _flatten_tokens_channels(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    assert x.dim() == 4, f"expect [B,H,T,D], got {tuple(x.shape)}"
    B, H, T, D = x.shape
    xf = x.permute(0, 2, 1, 3).contiguous().reshape(B * T, H * D)
    return xf, (B, H, T, D)


def _restore_from_flat(
        xf: torch.Tensor,
        shape_meta: Tuple[int, int, int, int],
        dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    B, H, T, D = shape_meta
    return (
        xf.to(dtype=dtype)
        .view(B, T, H, D)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def quantize_dynamic_cache(cache, config) -> Dict[str, torch.Tensor]:
    assert len(cache.layers) == config.nlayers, \
        f"layers mismatch: cache={len(cache.layers)} vs cfg={config.nlayers}"

    key_bins = make_key_bins(config)
    val_bins = make_value_bins(config)

    pack: Dict[str, torch.Tensor] = {
        "num_layers": torch.tensor([len(cache.layers)], dtype=torch.int32),
        "key_bins": key_bins.clone(),
        "val_bins": val_bins.clone(),
    }

    for i, layer in enumerate(cache.layers):
        k = layer.keys.detach().to(dtype=torch.float32).contiguous()
        v = layer.values.detach().to(dtype=torch.float32).contiguous()

        k_flat, k_meta = _flatten_tokens_channels(k)
        v_flat, v_meta = _flatten_tokens_channels(v)

        kb = int(key_bins[i].item())
        vb = int(val_bins[i].item())

        qk, maxk = quant(kb, k_flat)
        qv, maxv = quant(vb, v_flat)

        pack[f"k{i}_q"] = qk
        pack[f"v{i}_q"] = qv

        if maxk is not None:
            pack[f"k{i}_scale"] = maxk.squeeze(-1)
        pack[f"k{i}_shape"] = torch.tensor(k_meta, dtype=torch.int32)

        if maxv is not None:
            pack[f"v{i}_scale"] = maxv.squeeze(-1)
        pack[f"v{i}_shape"] = torch.tensor(v_meta, dtype=torch.int32)

    return pack


def dequantize_dynamic_cache(
        pack: Dict[str, torch.Tensor],
        device: torch.device = torch.device("cuda"),
        out_dtype: torch.dtype = torch.float16,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    nL = int(pack["num_layers"][0].item())

    kb_list = pack["key_bins"].int().tolist()
    vb_list = pack["val_bins"].int().tolist()

    k_shapes = [tuple(pack[f"k{i}_shape"].tolist()) for i in range(nL)]
    v_shapes = [tuple(pack[f"v{i}_shape"].tolist()) for i in range(nL)]

    kv_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(nL):
        kb, vb = kb_list[i], vb_list[i]

        qk_raw = pack[f"k{i}_q"]
        qv_raw = pack[f"v{i}_q"]

        if not qk_raw.is_pinned():
            qk_raw = qk_raw.pin_memory()
        qk = qk_raw.to(device)
        if kb == 256:
            sk = pack[f"k{i}_scale"]
            if not sk.is_pinned():
                sk = sk.pin_memory()
            sk = sk.to(device).to(torch.float32).unsqueeze(-1)
            nt_k = sk.numel()
            nc_k = qk.numel() // nt_k
            k_flat = dequant(kb, qk.view(nt_k, nc_k), sk)
        elif kb == 65536:
            k_flat = dequant(kb, qk, None)
        else:
            raise ValueError(f"Unsupported key bins={kb}")

        if not qv_raw.is_pinned():
            qv_raw = qv_raw.pin_memory()
        qv = qv_raw.to(device)
        if vb == 256:
            sv = pack[f"v{i}_scale"]
            if not sv.is_pinned():
                sv = sv.pin_memory()
            sv = sv.to(device).to(torch.float32).unsqueeze(-1)
            nt_v = sv.numel()
            nc_v = qv.numel() // nt_v
            v_flat = dequant(vb, qv.view(nt_v, nc_v), sv)
        elif vb == 65536:
            v_flat = dequant(vb, qv, None)
        else:
            raise ValueError(f"Unsupported value bins={vb}")

        K = _restore_from_flat(k_flat, k_shapes[i], out_dtype)
        V = _restore_from_flat(v_flat, v_shapes[i], out_dtype)

        kv_layers.append((K, V))

    return kv_layers
