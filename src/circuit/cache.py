"""Disk cache of inputs and activations for every (window, alpha), so a patched DCS
evaluation never recomputes the source forward.

    cache_dir/
      manifest.json      recording, alpha_set, n_windows, checkpoint hash
      centroids.npz      centroid_base[N], centroid_alpha_<slug>[N]  (clean model)
      window_{i:04d}.pt  x_base, cache_base, x_alpha_<slug>, cache_alpha_<slug>  (fp16 caches)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dcs.metric import compute_centroid_power
from patching.hooks import CACHE_FIELDS, ActivationCache, capture_all_activations


def alpha_slug(alpha: float) -> str:
    """0.1 -> 'p010', -1.0 -> 'm100'."""
    return f"{'p' if alpha >= 0 else 'm'}{int(round(abs(alpha) * 100)):03d}"


def strip_cache(cache: ActivationCache) -> dict[str, torch.Tensor]:
    """The four fields as fresh fp16 tensors (halves disk and RAM; ~3 significant digits)."""
    return {f: getattr(cache, f).detach().clone().contiguous().to(torch.float16) for f in CACHE_FIELDS}


def to_activation_cache(stripped: dict[str, torch.Tensor]) -> ActivationCache:
    return ActivationCache(**stripped)


def build_cache(model, device, base_windows: np.ndarray, alpha_windows: dict[float, np.ndarray],
                alpha_set: list[float], sxx_scaler, freq_axis, out_dir: Path, manifest_extra: dict) -> dict:
    """Forward every window at base and at each alpha; store inputs, caches and clean centroids."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = base_windows.shape[0]
    cents = {"centroid_base": np.zeros(n, np.float32),
             **{f"centroid_alpha_{alpha_slug(a)}": np.zeros(n, np.float32) for a in alpha_set}}

    def run(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0)
        cache = capture_all_activations(model, x, device)
        with torch.inference_mode():
            y = model(x.to(device)).cpu().numpy()
        return x.squeeze(0).clone().contiguous(), strip_cache(cache), float(compute_centroid_power(y, sxx_scaler, freq_axis)[0])

    for i in tqdm(range(n), desc="cache build"):
        x, c, cents["centroid_base"][i] = run(base_windows[i])
        wd = {"x_base": x, "cache_base": c}
        for a in alpha_set:
            x, c, cents[f"centroid_alpha_{alpha_slug(a)}"][i] = run(alpha_windows[a][i])
            wd[f"x_alpha_{alpha_slug(a)}"], wd[f"cache_alpha_{alpha_slug(a)}"] = x, c
        torch.save(wd, out_dir / f"window_{i:04d}.pt")
    np.savez(out_dir / "centroids.npz", **cents)
    manifest = {"n_windows": int(n), "alpha_set": [float(a) for a in alpha_set],
                "alpha_slugs": [alpha_slug(a) for a in alpha_set], **manifest_extra}
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)
    return manifest


def load_manifest(cache_dir: Path) -> dict:
    return json.load(open(cache_dir / "manifest.json"))


def load_centroids(cache_dir: Path) -> dict[str, np.ndarray]:
    d = np.load(cache_dir / "centroids.npz")
    return {k: d[k] for k in d.files}


def load_window(cache_dir: Path, i: int) -> dict:
    """Memory-mapped: tensors stay fp16 and file-backed (page cache), so several workers can hold the
    whole cache; PyTorch upcasts at the assignment inside the patched forward."""
    return torch.load(cache_dir / f"window_{i:04d}.pt", map_location="cpu", weights_only=True, mmap=True)
