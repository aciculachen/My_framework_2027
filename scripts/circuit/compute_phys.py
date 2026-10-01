#!/usr/bin/env python3
"""Kinematic reference g for the domain-of-validity sweep: alpha_pred of the BSA-weighted
kinematic Doppler centroid under the same perturbations (radar at the origin).
Writes results/dov/phys.json with the same curve layout as sweep.json ("phys (<mode>)").
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from dcs.kinematic import kinematic_centroids  # noqa: E402
from dcs.metric import alpha_pred  # noqa: E402
from dcs.perturb import RADIAL_ORIGIN, scale_velocity  # noqa: E402

DEFAULT_ALPHAS = [round(a, 2) for a in np.arange(-2.0, 2.0001, 0.1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--modes", nargs="+", default=["radial", "tangential_pure"])
    ap.add_argument("--recording", default="RandomWalk1")
    ap.add_argument("--out", type=Path, default=P / "results/dov")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(P / "configs/config.yaml"))["stft"]
    with open(P / "data/processed/eval_raw_full_sequences.pkl", "rb") as f:
        mocap = pickle.load(f)["mocap"][args.recording].astype(np.float64)
    c_base = kinematic_centroids(mocap, cfg["nperseg"], cfg["noverlap"])
    curves = {}
    for mode in args.modes:
        curves[f"phys ({mode})"] = {f"{a:.2f}": alpha_pred(c_base, kinematic_centroids(scale_velocity(mocap, a, mode), cfg["nperseg"], cfg["noverlap"]))
                                    for a in args.alphas}
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump({"alphas": sorted(args.alphas), "modes": args.modes, "recording": args.recording, "n_windows": int(len(c_base)),
               "weights": "Rule of Nines body-surface-area shares spread evenly over segment markers",
               "radial_origin": RADIAL_ORIGIN.tolist(),
               "results": curves}, open(args.out / "phys.json", "w"), indent=2)
    for k, v in curves.items():
        print(k, {a: round(x, 3) for a, x in v.items() if a in ("-2.00", "-1.00", "0.00", "1.00", "2.00")})


if __name__ == "__main__":
    main()
