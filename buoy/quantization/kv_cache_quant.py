import torch
from typing import List, Tuple, Dict, Any


def quant(bins: int, qA: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    C = bins // 2 - 1
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    max1 = torch.clamp(max1, min=1e-12)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    xq = torch.clamp(xq, min=-C, max=C)

    return xq, max1


def dequant(bins: int, xq: torch.Tensor, max1: torch.Tensor) -> torch.Tensor:
    C = bins // 2 - 1
    return (xq.to(torch.float32) / C) * max1


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
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    B, H, T, D = shape_meta
    return (
        xf.to(device=device, dtype=dtype, non_blocking=True)
        .view(B, T, H, D)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _pack_int4x2_to_int8(q_int8: torch.Tensor) -> Tuple[torch.Tensor, int]:
    flat = q_int8.view(-1).to(torch.int8)
    pad = 0
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8)], dim=0)
        pad = 1
    u4 = (flat & 0x0F).to(torch.uint8)
    hi, lo = u4[0::2], u4[1::2]
    packed = ((hi << 4) | lo).contiguous()
    return packed, pad


def _unpack_uint8_to_int4x2(packed: torch.Tensor, pad: int) -> torch.Tensor:
    hi = torch.bitwise_right_shift(packed, 4) & 0x0F
    lo = packed & 0x0F
    u4 = torch.stack((hi, lo), dim=-1).reshape(-1).to(torch.int8)
    if pad:
        u4 = u4[:-pad]
    u4 = torch.where(u4 >= 8, u4 - 16, u4)
    return u4


def quantize_dynamic_cache(cache, config) -> Dict[str, Any]:
    assert len(cache.layers) == config.nlayers, \
        f"layers mismatch: cache={len(cache.layers)} vs cfg={config.nlayers}"

    key_bins = make_key_bins(config)
    val_bins = make_value_bins(config)

    out: Dict[str, Any] = {
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

        if kb == 16:
            pk, padk = _pack_int4x2_to_int8(qk)
            out[f"k{i}_q"] = pk
            out[f"k{i}_pad"] = torch.tensor([padk], dtype=torch.int32)
        else:
            out[f"k{i}_q"] = qk
            out[f"k{i}_pad"] = torch.tensor([0], dtype=torch.int32)

        if vb == 16:
            pv, padv = _pack_int4x2_to_int8(qv)
            out[f"v{i}_q"] = pv
            out[f"v{i}_pad"] = torch.tensor([padv], dtype=torch.int32)
        else:
            out[f"v{i}_q"] = qv
            out[f"v{i}_pad"] = torch.tensor([0], dtype=torch.int32)

        out[f"k{i}_scale"] = maxk.squeeze(-1)
        out[f"k{i}_shape"] = torch.tensor(k_meta, dtype=torch.int32)
        out[f"v{i}_scale"] = maxv.squeeze(-1)
        out[f"v{i}_shape"] = torch.tensor(v_meta, dtype=torch.int32)

    return out


def dequantize_dynamic_cache(
        pack: Dict[str, Any],
        device: torch.device = torch.device("cuda"),
        out_dtype: torch.dtype = torch.float16,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    nL = int(pack["num_layers"][0].item())
    key_bins = pack["key_bins"]
    val_bins = pack["val_bins"]

    k_shapes, v_shapes = [], []
    k_scales, v_scales = [], []
    qk_list, qv_list = [], []
    kb_list, vb_list = [], []

    for i in range(nL):
        kb = int(key_bins[i].item())
        vb = int(val_bins[i].item())
        kb_list.append(kb)
        vb_list.append(vb)

        k_shapes.append(tuple(pack[f"k{i}_shape"].tolist()))
        v_shapes.append(tuple(pack[f"v{i}_shape"].tolist()))

        k_scales.append(pack[f"k{i}_scale"].to(torch.float32).unsqueeze(-1))
        v_scales.append(pack[f"v{i}_scale"].to(torch.float32).unsqueeze(-1))

        if kb == 16:
            qk = _unpack_uint8_to_int4x2(
                pack[f"k{i}_q"].to(device, non_blocking=True),
                int(pack[f"k{i}_pad"][0].item())
            )
        else:
            qk = pack[f"k{i}_q"].to(device, dtype=torch.int8, non_blocking=True).view(-1)
        qk_list.append(qk)

        if vb == 16:
            qv = _unpack_uint8_to_int4x2(
                pack[f"v{i}_q"].to(device, non_blocking=True),
                int(pack[f"v{i}_pad"][0].item())
            )
        else:
            qv = pack[f"v{i}_q"].to(device, dtype=torch.int8, non_blocking=True).view(-1)
        qv_list.append(qv)

    qk_cat = torch.cat(qk_list, dim=0)
    qv_cat = torch.cat(qv_list, dim=0)
    sk_cat = torch.cat(k_scales, dim=0).to(device, non_blocking=True)
    sv_cat = torch.cat(v_scales, dim=0).to(device, non_blocking=True)

    kv_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    offset_k, offset_v = 0, 0
    for i in range(nL):
        nt_k = k_scales[i].numel()
        nt_v = v_scales[i].numel()

        nc_k = qk_list[i].numel() // nt_k
        nc_v = qv_list[i].numel() // nt_v

        qk = qk_cat[offset_k: offset_k + nt_k * nc_k].view(nt_k, nc_k)
        qv = qv_cat[offset_v: offset_v + nt_v * nc_v].view(nt_v, nc_v)
        sk = sk_cat[offset_k // nc_k: offset_k // nc_k + nt_k]
        sv = sv_cat[offset_v // nc_v: offset_v // nc_v + nt_v]

        k_flat = dequant(kb_list[i], qk, sk)
        v_flat = dequant(vb_list[i], qv, sv)

        K = _restore_from_flat(k_flat, k_shapes[i], out_dtype, device)
        V = _restore_from_flat(v_flat, v_shapes[i], out_dtype, device)
        kv_layers.append((K, V))

        offset_k += nt_k * nc_k
        offset_v += nt_v * nc_v

    return kv_layers
