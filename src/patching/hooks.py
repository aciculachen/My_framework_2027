"""Capture and patch the four kinds of activation the 18 components live in.

Only the last spatial layer and the last temporal layer are touched:
    spatial_per_head    [B*L, H_s, N, hd_s]   per-head attention output before out_proj
    spatial_ffn_delta   [N, B*L, E_s]         spatial FFN output before the residual add
    temporal_per_head   [N*B, H_t, L, hd_t]   per-head attention output before out_proj
    temporal_mlp_delta  [L, N*B, E_t]         temporal FFN output before the residual add

`capture_all_activations(model, x)` runs one forward and returns them.
`patching_context(model, hook_names, source)` replaces the named components with the
source's values for every forward inside the block; hook names are those of
patching.components.SPEC_TO_HOOK (spatial_head_k, spatial_ffn_delta, head_k, temporal_ffn_delta).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActivationCache:
    spatial_per_head: torch.Tensor | None = None
    spatial_ffn_delta: torch.Tensor | None = None
    temporal_per_head: torch.Tensor | None = None
    temporal_mlp_delta: torch.Tensor | None = None


CACHE_FIELDS = tuple(f.name for f in fields(ActivationCache))


def capture_all_activations(model: nn.Module, x: torch.Tensor, device: torch.device) -> ActivationCache:
    """One forward of x ([1, L, N, 3]) with capture enabled; tensors are detached, on CPU, float32."""
    spatial, temporal = model.spatial_transformer, model.temporal_transformer
    s_last, t_last = spatial.layers[-1], temporal.layers[-1]
    spatial.enable_per_head_capture(); temporal.enable_per_head_capture()
    temporal.enable_mlp_probe_capture(); spatial.enable_residual_capture()
    try:
        with torch.inference_mode():
            model(x.to(device))
        cache = ActivationCache(
            spatial_per_head=s_last.per_head_output.clone(),
            spatial_ffn_delta=s_last.ffn_delta.clone(),
            temporal_per_head=t_last.per_head_output.clone(),
            temporal_mlp_delta=t_last.mlp_delta.clone(),
        )
    finally:
        spatial.disable_per_head_capture(); spatial.disable_residual_capture()
        temporal.disable_per_head_capture(); temporal.disable_mlp_probe_capture()
    for name in CACHE_FIELDS:
        if getattr(cache, name) is None:
            raise RuntimeError(f"capture failed: {name} is None")
    return cache


def _make_spatial_layer_patched_forward(layer, source_ffn_delta: torch.Tensor):
    def patched_forward(x, need_attn=False, average_attn_weights=False):
        h = layer.ln1(x)
        if layer._capture_per_head:
            attn_out, attn_w = layer._forward_capture_heads(h)
        else:
            attn_out, attn_w = layer.self_attn(h, h, h, need_weights=need_attn,
                                               average_attn_weights=average_attn_weights)
        x = x + layer.dropout(attn_out)
        rep = source_ffn_delta.to(x.device)
        if layer._capture_residual_stream:
            layer.ffn_delta = rep.detach().cpu()
        x = x + rep
        return x, attn_out, attn_w
    return patched_forward


def _make_head_patched_capture(layer, patch_map: dict[int, torch.Tensor]):
    """Replaces heads of a (spatial or temporal) layer inside its _forward_capture_heads."""
    def _patched(h):
        S, bsz, E = h.shape
        qkv = F.linear(h, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(S, bsz, layer.num_heads, layer.head_dim).permute(1, 2, 0, 3)
        k = k.view(S, bsz, layer.num_heads, layer.head_dim).permute(1, 2, 0, 3)
        v = v.view(S, bsz, layer.num_heads, layer.head_dim).permute(1, 2, 0, 3)
        drop_p = 0.0 if not layer.training else layer.dropout.p
        attn_out_heads = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)  # [bsz, H, S, hd]
        for h_idx, source_act in patch_map.items():
            attn_out_heads[:, h_idx, :, :] = source_act.to(attn_out_heads.device)
        layer.per_head_output = attn_out_heads.detach().cpu()
        attn_out = attn_out_heads.permute(2, 0, 1, 3).contiguous().view(S, bsz, E)
        attn_out = F.linear(attn_out, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
        return attn_out, None
    return _patched


def _make_temporal_layer_patched_forward(layer, source_ffn_delta: torch.Tensor):
    def patched_forward(x, need_attn=False, average_attn_weights=False):
        h = layer.ln1(x)
        if layer._capture_per_head:
            attn_out, attn_w = layer._forward_capture_heads(h)
        else:
            attn_out, attn_w = layer.self_attn(h, h, h, need_weights=need_attn,
                                               average_attn_weights=average_attn_weights)
        x = x + layer.dropout(attn_out)
        rep = source_ffn_delta.to(x.device)
        if layer._capture_mlp_probes:
            layer.mlp_delta = rep.detach().cpu()
        x = x + rep
        return x, attn_out, attn_w
    return patched_forward


def _head_index(name: str, prefix: str) -> int | None:
    if name.startswith(prefix) and name[len(prefix):].isdigit():
        return int(name[len(prefix):])
    return None


@contextmanager
def patching_context(model: nn.Module, hook_names: list[str], source: ActivationCache):
    """Inside the block, every forward uses `source` values for the named components.

    Unknown names raise instead of being ignored. Restores the model on exit.
    """
    spatial, temporal = model.spatial_transformer, model.temporal_transformer
    s_last, t_last = spatial.layers[-1], temporal.layers[-1]
    s_heads: dict[int, torch.Tensor] = {}
    t_heads: dict[int, torch.Tensor] = {}
    s_ffn = t_ffn = None
    for name in hook_names:
        if (k := _head_index(name, "spatial_head_")) is not None:
            s_heads[k] = source.spatial_per_head[:, k, :, :]
        elif (k := _head_index(name, "head_")) is not None:
            t_heads[k] = source.temporal_per_head[:, k, :, :]
        elif name == "spatial_ffn_delta":
            s_ffn = source.spatial_ffn_delta
        elif name == "temporal_ffn_delta":
            t_ffn = source.temporal_mlp_delta
        else:
            raise ValueError(f"unknown hook name: {name}")

    saved: list[tuple[object, str, object]] = []

    def swap(obj, attr, new):
        saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, new)

    try:
        if s_heads:
            swap(s_last, "_forward_capture_heads", _make_head_patched_capture(s_last, s_heads))
            swap(s_last, "_capture_per_head", True)
        if s_ffn is not None:
            swap(s_last, "forward", _make_spatial_layer_patched_forward(s_last, s_ffn))
        if t_heads:
            swap(t_last, "_forward_capture_heads", _make_head_patched_capture(t_last, t_heads))
            swap(t_last, "_capture_per_head", True)
        if t_ffn is not None:
            swap(t_last, "forward", _make_temporal_layer_patched_forward(t_last, t_ffn))
        yield
    finally:
        for obj, attr, original in reversed(saved):
            setattr(obj, attr, original)
