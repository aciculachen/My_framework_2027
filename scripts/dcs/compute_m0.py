#!/usr/bin/env python3
"""Consistency score M_P(f, g) of the unpatched model: the model's slopes alpha_pred(alpha) against the
physics model's slopes beta_g(alpha) over the alpha set of each --spec.

Clean forwards at every alpha of --alpha-range -> <out>/centroids.npz (reused when it covers the grid);
beta_g -> <out>/beta_g.npz; one <out>/<name>.json per --spec name=start:stop:step[~excluded].

    python scripts/dcs/compute_m0.py --alpha-range -1.16 1.16 0.01 --spec paper=-1.16:1.16:0.01~1 --out results/dcs
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from data.data_loader import load_meta, load_scalers  # noqa: E402
from dcs.forward import batched_centroids  # noqa: E402
from dcs.kinematic import kinematic_centroids  # noqa: E402
from dcs.metric import alpha_pred, alpha_range, consistency_score, load_freq_axis, parse_alpha_spec  # noqa: E402
from dcs.perturb import scale_velocity, window_data  # noqa: E402
from model.load import DEFAULT_CKPT, load_backbone  # noqa: E402


def clean_centroids(alphas, recording, batch=64):
    model, device = load_backbone()
    dp = P / "data/processed"
    sc, freq = load_scalers(dp), load_freq_axis(load_meta(dp))
    cfg = yaml.safe_load(open(P / "configs/config.yaml"))["stft"]
    with open(dp / "eval_raw_full_sequences.pkl", "rb") as f:
        mocap = pickle.load(f)["mocap"][recording].astype(np.float32)

    def cents(x_all):
        return batched_centroids(model, device, x_all, sc["radar_sxx"], freq, batch)

    base = cents(window_data(mocap, sc["mocap"], cfg["nperseg"], cfg["noverlap"])[0])
    rows = [cents(window_data(scale_velocity(mocap, a), sc["mocap"], cfg["nperseg"], cfg["noverlap"])[0])
            for a in tqdm(alphas, desc="clean sweep")]
    return base, np.stack(rows)


def load_or_compute_beta_g(path, alphas, recording):
    """{alpha: beta_g} for every alpha of the grid; cached in `path`, missing alphas are computed and appended."""
    have = {}
    if path.exists():
        d = np.load(path); have = {round(float(a), 6): float(b) for a, b in zip(d["alphas"], d["beta_g"])}
    todo = [a for a in alphas if round(a, 6) not in have]
    if todo:
        cfg = yaml.safe_load(open(P / "configs/config.yaml"))["stft"]
        with open(P / "data/processed/eval_raw_full_sequences.pkl", "rb") as f:
            mocap = pickle.load(f)["mocap"][recording].astype(np.float64)
        gb = kinematic_centroids(mocap, cfg["nperseg"], cfg["noverlap"])
        for a in tqdm(todo, desc="beta_g"):
            have[round(a, 6)] = alpha_pred(gb, kinematic_centroids(scale_velocity(mocap, a), cfg["nperseg"], cfg["noverlap"]))
        keys = sorted(have)
        np.savez(path, alphas=np.array(keys), beta_g=np.array([have[k] for k in keys]), recording=recording)
    return have


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha-range", nargs=3, type=float, metavar=("START", "STOP", "STEP"), default=[-2.0, 2.0, 0.01])
    ap.add_argument("--spec", action="append", default=[], help="name=start:stop:step or name=a,b,c (repeatable)")
    ap.add_argument("--recording", default="RandomWalk1")
    ap.add_argument("--out", type=Path, default=P / "results/dcs")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    grid = alpha_range(*args.alpha_range)
    npz = args.out / "centroids.npz"

    if npz.exists() and all(any(abs(a - g) < 1e-9 for g in np.load(npz)["alphas"]) for a in grid):
        d = np.load(npz); alphas, base, cent = [float(a) for a in d["alphas"]], d["centroid_base"], d["centroids"]
        print(f"reusing {npz} ({len(alphas)} alphas)")
    else:
        base, cent = clean_centroids(grid, args.recording)
        alphas = grid
        np.savez(npz, alphas=np.array(alphas), centroid_base=base, centroids=cent, recording=args.recording)
        print(f"saved {npz}: {len(alphas)} alphas x {len(base)} windows")

    idx = {round(a, 6): k for k, a in enumerate(alphas)}
    beta_g = load_or_compute_beta_g(args.out / "beta_g.npz", alphas, args.recording)
    ckpt_sha = hashlib.sha256(Path(DEFAULT_CKPT).read_bytes()).hexdigest()
    for spec in args.spec:
        name, sel = parse_alpha_spec(spec)
        missing = [a for a in sel if round(a, 6) not in idx]
        assert not missing, f"{name}: alphas not in grid: {missing[:5]}"
        m = consistency_score(base, {a: cent[idx[round(a, 6)]] for a in sel}, sel, {a: beta_g[round(a, 6)] for a in sel})
        json.dump({"name": name, "consistency_score": m, "alphas": sel, "n_windows": int(len(base)),
                   "recording": args.recording, "ckpt_sha256": ckpt_sha,
                   "alpha_pred": {f"{a:g}": alpha_pred(base, cent[idx[round(a, 6)]]) for a in sel},
                   "beta_g": {f"{a:g}": beta_g[round(a, 6)] for a in sel}},
                  open(args.out / f"{name}.json", "w"), indent=2)
        print(f"{name}: M_P(f, g) = {m:.6f} ({len(sel)} alphas, {min(sel):g}..{max(sel):g})")


if __name__ == "__main__":
    main()
