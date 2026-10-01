#!/usr/bin/env python3
"""Consistency score of the unpatched model, a circuit and its complement, from a run_sweep.py
sweep.json (alpha_pred per alpha) and a compute_phys.py phys.json (the physics model's slopes g).
rho = M(patched) / M(unpatched).

    python scripts/dcs/circuit_score.py --sweep results/dov/sweep.json --phys results/dov/phys.json
"""
import argparse
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from dcs.metric import score_from_slopes  # noqa: E402

TABLE_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--phys", required=True, type=Path)
    ap.add_argument("--mode", default="radial")
    ap.add_argument("--alphas", nargs="+", type=float, default=TABLE_ALPHAS)
    ap.add_argument("--out", type=Path, default=P / "results/dcs/circuit_score.json")
    args = ap.parse_args()
    sweep, phys = json.load(open(args.sweep))["results"], json.load(open(args.phys))["results"]
    keys = [f"{a:.2f}" for a in args.alphas]
    g = [phys[f"phys ({args.mode})"][k] for k in keys]
    m = {s: score_from_slopes([sweep[f"{s} ({args.mode})"][k] for k in keys], g)
         for s in ("unpatched", "circuit", "complement")}
    out = {"mode": args.mode, "alphas": args.alphas, "M": m,
           "rho": {s: m[s] / m["unpatched"] for s in ("circuit", "complement")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out["M"]), json.dumps(out["rho"]))


if __name__ == "__main__":
    main()
