#!/usr/bin/env python3
"""Full model outputs per window for a circuit under radial scaling -> results/spectrograms/<name>/

    gt                    measured spectrogram (training target) of the window
    clean_x               f(x)                    unpatched, clean input
    clean_xprime_a<α>     f(x_α)                  unpatched, perturbed input
    circuit_a<α>          patch(C <- x_α)(x)      C patched from x_α
    complement_a<α>       patch(U\\C <- x_α)(x)   U\\C patched from x_α

Arrays are [N, 256] in the model's normalized dB scale; meta.json carries the frequency axis and
the scaler so a plot needs nothing else.

    python scripts/circuit/dump_spectrograms.py --name cdagger --components mlp_t h_t_1 h_t_2 h_t_6 h_t_7 mlp_s --alphas 0.3
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from data.data_loader import load_meta, load_recording_targets, load_scalers  # noqa: E402
from dcs.metric import load_freq_axis  # noqa: E402
from data import paths  # noqa: E402
from dcs.perturb import RADIAL_ORIGIN, scale_velocity, window_data  # noqa: E402
from model.load import DEFAULT_CKPT, load_backbone  # noqa: E402
from patching.components import ALL_COMPONENTS, complement, format_circuit, to_hook_names  # noqa: E402
from patching.hooks import capture_all_activations, patching_context  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output subdirectory under results/spectrograms/")
    ap.add_argument("--components", nargs="+", required=True, help="the circuit C (spec names)")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3])
    ap.add_argument("--recording", default="RandomWalk1")
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    circuit = list(args.components)
    assert all(c in ALL_COMPONENTS for c in circuit), circuit
    hooks_c, hooks_v = to_hook_names(circuit), to_hook_names(complement(circuit))

    model, device = load_backbone()
    dp = P / "data/processed"
    sc, freq = load_scalers(dp), load_freq_axis(load_meta(dp))
    cfg = yaml.safe_load(open(P / "configs/config.yaml"))["stft"]
    with open(dp / "eval_raw_full_sequences.pkl", "rb") as f:
        mocap = pickle.load(f)["mocap"][args.recording].astype(np.float32)
    base = window_data(mocap, sc["mocap"], cfg["nperseg"], cfg["noverlap"])[0]
    gt = load_recording_targets(args.recording, "eval", dp)
    assert len(gt) == len(base), (len(gt), len(base))
    n = min(len(base), args.max_windows) if args.max_windows else len(base)
    xa = {a: window_data(scale_velocity(mocap, a), sc["mocap"], cfg["nperseg"], cfg["noverlap"])[0]
          for a in args.alphas}

    def fwd(x):
        with torch.inference_mode():
            return model(x.to(device)).cpu().numpy()[0]

    out = {"gt": gt[:n], "clean_x": np.empty((n, gt.shape[1]), np.float32)}
    for a in args.alphas:
        for k in ("clean_xprime", "circuit", "complement"):
            out[f"{k}_a{a:.2f}"] = np.empty((n, gt.shape[1]), np.float32)
    for i in tqdm(range(n), desc="dump"):
        xb = torch.from_numpy(base[i]).unsqueeze(0)
        out["clean_x"][i] = fwd(xb)
        for a in args.alphas:
            x_alpha = torch.from_numpy(xa[a][i]).unsqueeze(0)
            out[f"clean_xprime_a{a:.2f}"][i] = fwd(x_alpha)
            cache = capture_all_activations(model, x_alpha, device)
            with patching_context(model, hooks_c, cache):
                out[f"circuit_a{a:.2f}"][i] = fwd(xb)
            with patching_context(model, hooks_v, cache):
                out[f"complement_a{a:.2f}"][i] = fwd(xb)

    d = P / "results/spectrograms" / args.name
    d.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        np.save(d / f"{k}.npy", v)
    meta = {"recording": args.recording, "n_windows": int(n), "alphas": args.alphas, "circuit": circuit,
            "circuit_paper": format_circuit(circuit), "complement": complement(circuit),
            "freq_axis_hz": freq.tolist(), "sxx_scaler_mean": float(sc["radar_sxx"].mean_[0]),
            "sxx_scaler_scale": float(sc["radar_sxx"].scale_[0]), "nperseg": cfg["nperseg"], "noverlap": cfg["noverlap"],
            "seconds_per_window": (cfg["nperseg"] - cfg["noverlap"]) / 256.0,
            "ckpt_sha256": hashlib.sha256(Path(DEFAULT_CKPT).read_bytes()).hexdigest(),
            "dataset": paths.HF_DATASET, "radial_origin": RADIAL_ORIGIN.tolist()}
    json.dump(meta, open(d / "meta.json", "w"), indent=2)
    print(f"saved {len(out)} arrays [{n}, {gt.shape[1]}] + meta.json to {d}")


if __name__ == "__main__":
    main()
