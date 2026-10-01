"""Patched DCS: M of the model when a component set C takes alpha-perturbed activations.

Forward the unperturbed window x_base while the components in C take their activations from the
forward of x_alpha of the same window (U \\ C is evaluated by the same call). With a `jmap`, the
source is the alpha-matched cache of another window j != i (source-randomized control).
`circuit.runtime` runs `patched_centroids` per GPU worker on a share of the windows.
"""
from __future__ import annotations

import numpy as np
import torch

from circuit.cache import alpha_slug, to_activation_cache
from dcs.metric import consistency_score, compute_centroid_power
from patching.components import to_hook_names
from patching.hooks import patching_context


def clean_score_from_cache(centroids: dict[str, np.ndarray], alpha_set: list[float], reference=None) -> float:
    """Clean-model score M_0 from the cached centroids (no forward), against `reference` (see consistency_score)."""
    return consistency_score(centroids["centroid_base"],
                             {a: centroids[f"centroid_alpha_{alpha_slug(a)}"] for a in alpha_set}, alpha_set, reference)


def patched_centroids(model, device, indices, get_window, hooks: list[str], alpha_slugs: list[str],
                      sxx_scaler, freq_axis, jmap=None) -> dict[tuple[int, int], float]:
    """Centroid of the patched forward for every (window index, alpha index).

    get_window(i) -> the cached window dict; jmap[i][a_idx] -> source window for the control
    (None: the window's own alpha cache). Only the cached fields the hooks touch are moved to the
    GPU (fp16); forwards are queued on the GPU and read back once.
    """
    needed = {f for f, prefixes in (("spatial_per_head", ("spatial_head_",)), ("spatial_ffn_delta", ("spatial_ffn_delta",)),
                                    ("temporal_per_head", ("head_",)), ("temporal_mlp_delta", ("temporal_ffn_delta",)))
              if any(h.startswith(pfx) for h in hooks for pfx in prefixes)}
    pending = []
    for i in indices:
        wd = get_window(i)
        x = wd["x_base"].unsqueeze(0).to(device)
        for a_idx, slug in enumerate(alpha_slugs):
            src = get_window(int(jmap[i][a_idx])) if jmap is not None else wd
            stored = src[f"cache_alpha_{slug}"]
            cache = to_activation_cache({k: (v.to(device) if k in needed else None) for k, v in stored.items()})   # blocking: no pinned staging build-up
            with patching_context(model, hooks, cache):
                with torch.inference_mode():
                    pending.append((i, a_idx, model(x)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {(i, a): float(compute_centroid_power(y.cpu().numpy(), sxx_scaler, freq_axis)[0]) for i, a, y in pending}
