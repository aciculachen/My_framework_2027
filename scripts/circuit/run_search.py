#!/usr/bin/env python3
"""Circuit search on an activation cache -> results/circuit/search.json.

M_P(f, g) and the physics model's slopes beta_g come from a compute_m0.py json over the cache's alpha
grid. For every tau of --taus (ascending, each circuit warm-started from the previous one) the greedy
search returns a minimal tau-consistent circuit C:

    Delta_suf(C) = M_P - M(C)       <= (1 - tau) M_P
    Delta_nec(C) = M_P - M(U \\ C)   >= tau M_P

then, for every distinct circuit, the source-randomized controls on C and on U \\ C over --seeds.
search.json is rewritten after every stage.

    python scripts/circuit/run_search.py --cache-dir <dir> --m0 results/dcs/search_grid.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from circuit.runtime import Evaluator  # noqa: E402
from circuit.search import greedy_minimal, single_deltas, source_randomized  # noqa: E402
from patching.components import ALL_COMPONENTS, complement, format_circuit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--m0", required=True, type=Path, help="compute_m0.py json over the cache's alpha grid")
    ap.add_argument("--n-gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--taus", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--n-prune-trials", type=int, default=8)
    ap.add_argument("--prune-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=P / "results/circuit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "search.json"

    m0 = json.load(open(args.m0))
    reference = {round(float(a), 6): v for a, v in m0["beta_g"].items()}
    ev = Evaluator(args.cache_dir, args.n_gpus, reference)
    assert set(round(a, 6) for a in m0["alphas"]) == set(round(a, 6) for a in ev.alpha_set), \
        f"--m0 alphas {m0['alphas']} differ from the cache's {ev.alpha_set}"
    assert abs(ev.M0 - m0["consistency_score"]) < 1e-3, "cache centroids disagree with --m0"
    M0 = m0["consistency_score"]
    res = {"M_P": M0, "alphas": ev.alpha_set, "recording": ev.manifest["recording"], "taus": sorted(args.taus),
           "seeds": args.seeds, "primary": [], "controls": []}

    def save():
        json.dump(res, open(path, "w"), indent=2)

    print(f"M_P = {M0:.6f}  (n_windows={ev.n_windows}, alphas={ev.alpha_set})")
    res["single_deltas"] = single_deltas(ev, M0)
    save()

    C_prev = None
    for tau in sorted(args.taus):
        r = greedy_minimal(ev, M0, tau, res["single_deltas"], C_init=C_prev,
                           n_prune_trials=args.n_prune_trials, prune_seed=args.prune_seed)
        r["circuit"] = format_circuit(r["C"])
        res["primary"].append(r)
        C_prev = list(r["C"])
        print(f"tau={tau}: |C|={r['size']} {r['circuit']}  Delta_suf={r['Delta_suf']:.4f}  Delta_nec={r['Delta_nec']:.4f}")
        save()

    seen = set()
    for r in res["primary"]:
        key = tuple(r["C"])
        if key in seen:
            continue
        seen.add(key)
        ctrl = {"C": r["C"], "circuit": r["circuit"], "size": r["size"],
                "taus": [p["tau"] for p in res["primary"] if tuple(p["C"]) == key],
                "Delta_suf": r["Delta_suf"], "Delta_nec": M0 - ev(complement(r["C"])),
                "rand_suf": source_randomized(ev, M0, r["C"], args.seeds),
                "rand_nec": source_randomized(ev, M0, complement(r["C"]), args.seeds)}
        res["controls"].append(ctrl)
        print(f"{ctrl['circuit']}: Delta_nec={ctrl['Delta_nec']:.4f}  rand_suf={ctrl['rand_suf']['Delta_mean']:.4f}  "
              f"rand_nec={ctrl['rand_nec']['Delta_mean']:.4f}")
        save()

    ev.close()
    print(f"saved {path}  ({ev.n_calls} evaluations)")


if __name__ == "__main__":
    main()
