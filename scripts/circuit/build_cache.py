#!/usr/bin/env python3
"""Activation cache for the circuit search: inputs, activations of the 18 components and clean
centroids for every window at alpha = 1 (base) and at every alpha of --alphas.

    python scripts/circuit/build_cache.py --cache-dir <dir>

Writes window_<i>.pt, centroids.npz and manifest.json into --cache-dir (about 580 GB for RandomWalk1
and the default 11 alphas).
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from circuit.cache import build_cache  # noqa: E402
from data import paths  # noqa: E402
from data.data_loader import load_meta, load_scalers  # noqa: E402
from dcs.metric import load_freq_axis  # noqa: E402
from dcs.perturb import scale_velocity, window_data  # noqa: E402
from model.load import DEFAULT_CKPT, load_backbone  # noqa: E402

SEARCH_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--recording", default="RandomWalk1")
    ap.add_argument("--alphas", nargs="+", type=float, default=SEARCH_ALPHAS)
    ap.add_argument("--max-windows", type=int, default=None, help="first n windows only (smoke test)")
    args = ap.parse_args()

    model, device = load_backbone()
    dp = paths.PROCESSED_DATA_DIR
    sc, freq = load_scalers(dp), load_freq_axis(load_meta(dp))
    cfg = yaml.safe_load(open(paths.CONFIG_PATH))["stft"]
    with open(dp / "eval_raw_full_sequences.pkl", "rb") as f:
        mocap = pickle.load(f)["mocap"][args.recording].astype(np.float32)
    base = window_data(mocap, sc["mocap"], cfg["nperseg"], cfg["noverlap"])[0]
    alpha_w = {a: window_data(scale_velocity(mocap, a), sc["mocap"], cfg["nperseg"], cfg["noverlap"])[0] for a in args.alphas}
    n = min(len(base), args.max_windows) if args.max_windows else len(base)
    extra = {"recording": args.recording, "ckpt_sha256": hashlib.sha256(DEFAULT_CKPT.read_bytes()).hexdigest(),
             "dataset": paths.HF_DATASET}
    manifest = build_cache(model, device, base[:n], {a: w[:n] for a, w in alpha_w.items()}, args.alphas,
                           sc["radar_sxx"], freq, args.cache_dir, extra)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
