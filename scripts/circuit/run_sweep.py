#!/usr/bin/env python3
"""Domain-of-validity sweep: alpha_pred of the model, a circuit and its complement under
radial and tangential-pure velocity scaling over alpha in [-2, 2].

For every (mode, alpha) and window: forward x_alpha (unpatched); capture its activations;
forward x_base with the circuit patched from them; the same with the complement. Per curve,
alpha_pred = least-squares slope of the centroids against the clean centroids of x_base.
One process per GPU takes (mode, alpha) pairs round-robin and writes a resumable part file
each; the parent merges them into results/dov/sweep.json (+ summary.csv).

    python scripts/circuit/run_sweep.py --components mlp_t h_t_1 h_t_2 h_t_6 h_t_7 mlp_s
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from data import paths  # noqa: E402
from dcs.metric import alpha_pred  # noqa: E402
from dcs.perturb import RADIAL_ORIGIN  # noqa: E402
from patching.components import ALL_COMPONENTS, complement, format_circuit  # noqa: E402

DEFAULT_ALPHAS = [round(a, 2) for a in np.arange(-2.0, 2.0001, 0.1)]


def _worker(rank, n_gpus, tasks, subjects, recording, max_windows, parts_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(P / "src"))
    import pickle
    import torch
    torch.set_num_threads(2)
    import yaml
    from data import paths
    from data.data_loader import load_meta, load_scalers
    from dcs.metric import compute_centroid_power, load_freq_axis
    from dcs.perturb import scale_velocity, window_data
    from model.load import load_backbone
    from patching.components import to_hook_names
    from patching.hooks import capture_all_activations, patching_context

    model, device = load_backbone(device=torch.device("cuda:0"))
    dp = paths.PROCESSED_DATA_DIR
    sc, freq = load_scalers(dp), load_freq_axis(load_meta(dp))
    cfg = yaml.safe_load(open(P / "configs/config.yaml"))["stft"]
    with open(dp / "eval_raw_full_sequences.pkl", "rb") as f:
        mocap = pickle.load(f)["mocap"][recording].astype(np.float32)
    base = window_data(mocap, sc["mocap"], cfg["nperseg"], cfg["noverlap"])[0]
    n = min(len(base), max_windows) if max_windows else len(base)
    hooks = {s: (to_hook_names(c) if c is not None else None) for s, c in subjects.items()}

    def cent(x):
        with torch.inference_mode():
            return float(compute_centroid_power(model(x.to(device)).cpu().numpy(), sc["radar_sxx"], freq)[0])

    c_base = [cent(torch.from_numpy(base[i]).unsqueeze(0)) for i in range(n)]
    for mode, alpha in tasks[rank::n_gpus]:
        part = parts_dir / f"{mode}_alpha{alpha:+.2f}.csv"
        if part.exists():
            continue
        xa_all = window_data(scale_velocity(mocap, alpha, mode), sc["mocap"], cfg["nperseg"], cfg["noverlap"])[0]
        rows = []
        for i in range(n):
            xb, xa = torch.from_numpy(base[i]).unsqueeze(0), torch.from_numpy(xa_all[i]).unsqueeze(0)
            c_unp = cent(xa)
            cache = capture_all_activations(model, xa, device)
            for subj, h in hooks.items():
                if h is None:
                    c = c_unp
                else:
                    with patching_context(model, h, cache):
                        c = cent(xb)
                rows.append({"mode": mode, "alpha": alpha, "window": i, "subject": subj, "c_base": c_base[i], "c_other": c})
        tmp = part.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.rename(part)
        print(f"[gpu{rank}] {mode} alpha={alpha:+.2f} done ({n} windows)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", nargs="+", required=True, help="the circuit C (spec names)")
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--modes", nargs="+", default=["radial", "tangential_pure"])
    ap.add_argument("--recording", default="RandomWalk1")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--n-gpus", type=int, default=None, help="default: all visible GPUs")
    ap.add_argument("--out", type=Path, default=P / "results/dov")
    args = ap.parse_args()
    if args.n_gpus is None:
        import torch
        args.n_gpus = torch.cuda.device_count()

    circuit = list(args.components)
    assert all(c in ALL_COMPONENTS for c in circuit), circuit
    subjects = {"unpatched": None, "circuit": circuit, "complement": complement(circuit)}
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    tasks = [(m, a) for m in args.modes for a in args.alphas]

    import torch.multiprocessing as tmp
    ctx = tmp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, args.n_gpus, tasks, subjects, args.recording, args.max_windows,
                                               parts))
             for r in range(args.n_gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    df = pd.concat([pd.read_csv(f) for f in sorted(parts.glob("*.csv"))], ignore_index=True)
    curves, summary = {}, []
    for (subj, mode), g1 in df.groupby(["subject", "mode"]):
        key = f"{subj} ({mode})"
        curves[key] = {}
        for alpha, g2 in g1.groupby("alpha"):
            ap_ = alpha_pred(g2["c_base"].values, g2["c_other"].values)
            curves[key][f"{alpha:.2f}"] = ap_
            summary.append({"subject": subj, "mode": mode, "alpha": alpha, "alpha_pred": ap_, "n_windows": len(g2)})
    from model.load import DEFAULT_CKPT as ckpt
    out = {"alphas": sorted(args.alphas), "modes": args.modes, "subjects": list(subjects),
           "circuit": circuit, "circuit_paper": format_circuit(circuit),
           "complement": subjects["complement"], "recording": args.recording,
           "n_windows": int(df["window"].nunique()), "results": curves,
           "run": {"ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                   "dataset": paths.HF_DATASET,
                   "radial_origin": RADIAL_ORIGIN.tolist(),
                   "centroid": "power", "metric": "alpha_pred (least squares through origin)"}}
    json.dump(out, open(args.out / "sweep.json", "w"), indent=2)
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    print(f"saved {args.out / 'sweep.json'}: {len(curves)} curves x {len(args.alphas)} alphas, {out['n_windows']} windows")


if __name__ == "__main__":
    main()
