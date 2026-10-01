#!/usr/bin/env python3
"""results/circuit/search.json -> results/circuit/table_circuits.csv

One row per distinct circuit C: |C|, tau range, C, and the scores relative to the unpatched model,
rho = M / M_P(f, g): rho_C, rho_C^rand, rho_{U\\C}, rho_{U\\C}^rand. The randomized columns are the mean
and std over the source-randomized seeds.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", type=Path, default=P / "results/circuit/search.json")
    args = ap.parse_args()
    res = json.load(open(args.search))
    M0 = res["M_P"]
    rows = []
    for c in res["controls"]:
        taus = sorted(c["taus"])
        rows.append({"size": c["size"], "tau": f"{taus[0]:g}" if len(taus) == 1 else f"{taus[0]:g}--{taus[-1]:g}",
                     "circuit": c["circuit"],
                     "rho_C": 1 - c["Delta_suf"] / M0,
                     "rho_C_rand_mean": 1 - c["rand_suf"]["Delta_mean"] / M0, "rho_C_rand_std": c["rand_suf"]["Delta_std"] / M0,
                     "rho_comp": 1 - c["Delta_nec"] / M0,
                     "rho_comp_rand_mean": 1 - c["rand_nec"]["Delta_mean"] / M0, "rho_comp_rand_std": c["rand_nec"]["Delta_std"] / M0})
    df = pd.DataFrame(rows)
    df.to_csv(args.search.parent / "table_circuits.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
