#!/usr/bin/env python3
"""Accessibility and targeted erasure of a physical variable at one location of C†.

    --site heads   T:{h1, h2, h6, h7}: the heads' columns of the last temporal layer's 8-head concat z
    --site smlp    S:{MLP}: the last spatial layer's FFN output

Target: range (per-marker range at the token's own frame, rank-1) or v_rad (per-marker radial
velocity at the centre frame and at the token's own frame, rank-2). Erasers are fitted on the fit
recording and scored on the evaluation recording. Conditions: native, erase (LEACE), and random
erasers of the same rank. Per condition: held-out decoding R^2 of range and v_rad at the site
(centre frame, centered ridge, 5 time-contiguous folds) and the consistency score M_P over alpha
0.1..0.9 against the ideal law, with the native model's centroid on x as the base.

    python scripts/erasure/run_erase.py --site heads --target v_rad --out results/erasure
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from data.data_loader import load_meta, load_scalers  # noqa: E402
from dcs.metric import compute_centroid_power, consistency_score, load_freq_axis  # noqa: E402
from dcs.perturb import scale_velocity, window_data  # noqa: E402
from erasure.erase import EraseStats, erased_attn, erased_input, erased_smlp  # noqa: E402
from erasure.probe import marker_range, marker_targets, marker_vrad, ridge_cv_r2, time_folds, time_splits  # noqa: E402
from model.load import load_backbone  # noqa: E402

HEAD_DIM = 32


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, choices=["heads", "smlp"])
    ap.add_argument("--target", required=True, choices=["range", "v_rad"])
    ap.add_argument("--heads", nargs="+", type=int, default=[1, 2, 6, 7], help="temporal heads for --site heads")
    ap.add_argument("--recording", default="RandomWalk1")
    ap.add_argument("--fit-recording", default="RandomWalk2")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--out", type=Path, default=P / "results/erasure")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, device = load_backbone()
    dp = P / "data/processed"
    sc, freq = load_scalers(dp), load_freq_axis(load_meta(dp))
    cfg = yaml.safe_load(open(P / "configs/config.yaml"))["stft"]
    with open(dp / "eval_raw_full_sequences.pkl", "rb") as f:
        mocap = pickle.load(f)["mocap"]

    def win(p):
        return window_data(p, sc["mocap"], cfg["nperseg"], cfg["noverlap"])

    fit_base, _, fit_p = win(mocap[args.fit_recording].astype(np.float32))
    base, _, _ = win(mocap[args.recording].astype(np.float32))
    alpha_win = np.stack([win(scale_velocity(mocap[args.recording].astype(np.float32), a))[0] for a in args.alphas], 1)
    n, L = base.shape[:2]
    centre = L // 2
    cols = torch.as_tensor(np.concatenate([np.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in args.heads])).to(device)

    # site: (native forward recording the site's tensor [L, N, E], eraser context)
    if args.site == "heads":
        def native(holder):
            return erased_input(model, holder)

        def site_of(holder):
            return holder["z_concat"][:, :, cols]

        def erased(e, holder=None):
            return erased_attn(model, e, holder, cols=cols)
    else:
        def native(holder):
            return erased_smlp(model, None, holder)

        def site_of(holder):
            return holder["smlp"]

        def erased(e, holder=None):
            return erased_smlp(model, e, holder)

    # ---- 1. erasers, fitted on the fit recording
    stats = None
    for i in time_splits(len(fit_base)):
        holder = {}
        with native(holder), torch.inference_mode():
            model(torch.from_numpy(fit_base[i]).unsqueeze(0).to(device))
        X = site_of(holder)
        if args.target == "range":
            Z = marker_range(fit_p[i])[..., None]
        else:
            v = marker_vrad(fit_p[i])
            Z = np.stack([np.broadcast_to(v[centre], v.shape), v], -1)
        Z = torch.as_tensor(np.ascontiguousarray(Z), dtype=torch.float32, device=device)
        stats = stats or EraseStats(L, X.shape[-1], Z.shape[-1], device)
        stats.add(X, Z)
    erasers = {"native": None, f"erase_{args.target}": stats.concept_eraser().to(torch.float32)}
    for r in range(args.n_random):
        erasers[f"erase_random{r}"] = stats.random_eraser(seed=r).to(torch.float32)

    # ---- 2. score on the evaluation recording
    def cent(y):
        return compute_centroid_power(y, sc["radar_sxx"], freq)

    ref = np.zeros(n)
    out = {c: np.zeros((n, len(args.alphas))) for c in erasers}
    feats = {c: None for c in erasers}
    for i in range(n):
        xb = torch.from_numpy(base[i]).unsqueeze(0).to(device)
        xa = torch.from_numpy(alpha_win[i]).to(device)
        for c, e in erasers.items():
            holder = {}
            ctx = native(holder) if e is None else erased(e, holder)
            with ctx, torch.inference_mode():
                y = model(xb).cpu().numpy()
            f = site_of(holder)[centre].cpu().numpy()
            if feats[c] is None:
                feats[c] = np.zeros((n,) + f.shape, np.float32)
            feats[c][i] = f
            if e is None:
                ref[i] = cent(y)[0]
                with torch.inference_mode():
                    out[c][i] = cent(model(xa).cpu().numpy())
            else:
                with erased(e), torch.inference_mode():
                    out[c][i] = cent(model(xa).cpu().numpy())
        if i % 200 == 0:
            print(f"  {i}/{n} windows", flush=True)

    # ---- 3. table
    tg = marker_targets(base, sc["mocap"])
    folds = time_folds(n)
    rows = []
    for c in erasers:
        row = {"condition": c, "M": consistency_score(ref, {a: out[c][:, j] for j, a in enumerate(args.alphas)}, args.alphas)}
        for t in ("range", "v_rad"):
            row[f"R2_{t}"], row[f"R2_{t}_std"] = ridge_cv_r2(feats[c], tg[t], folds)
        rows.append(row)
    df = pd.DataFrame(rows)
    stem = f"{args.site}_{args.target}"
    df.to_csv(args.out / f"{stem}.csv", index=False)
    json.dump({"site": args.site, "heads": args.heads if args.site == "heads" else None, "target": args.target,
               "recording": args.recording, "fit_recording": args.fit_recording, "alphas": args.alphas,
               "n_scored": int(n)}, open(args.out / f"{stem}_run.json", "w"), indent=2)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


if __name__ == "__main__":
    main()
