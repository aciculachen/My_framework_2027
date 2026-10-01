#!/usr/bin/env python3
"""The DoV main figures: alpha_pred of f, C-dagger and its complement against the kinematic g.

The model curves come from a run_sweep.py sweep.json, g from a compute_phys.py phys.json.
Writes <name>.png and <name>.csv of the plotted values. See README for the commands.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 1.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": True,
    "legend.framealpha": 0.9, "legend.edgecolor": "0.7", "legend.borderpad": 0.3,
    "legend.handletextpad": 0.4, "legend.handlelength": 1.6, "legend.columnspacing": 0.8,
})

STYLE = {                       # (f, circuit, complement) colours per mode
    "radial": ("#009E73", "#CC0000", "#888888"),
    "tangential_pure": ("#009E73", "#CC0000", "#888888"),
}
COLOR_PHYS = "#0072B2"


def curve(results: dict, key: str, alphas: list[float]) -> np.ndarray:
    c = results[key]
    return np.array([c.get(f"{a:.2f}", c.get(f"{a:.1f}", c.get(f"{a:g}", np.nan))) for a in alphas], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, type=Path, help="run_sweep.py sweep.json")
    ap.add_argument("--phys", required=True, type=Path, help="phys json holding the kinematic g")
    ap.add_argument("--mode", default="radial", choices=list(STYLE))
    ap.add_argument("--resolvable", type=float, default=1.16,
                    help="half-width of the shaded radar-resolvable band; 0 hides it")
    ap.add_argument("--out", required=True, type=Path, help="output basename, without suffix")
    args = ap.parse_args()

    model = json.loads(args.sweep.read_text())["results"]
    phys = json.loads(args.phys.read_text())["results"]
    phys_key = f"phys ({args.mode})"
    alphas = [round(x, 2) for x in np.arange(-2.0, 2.05, 0.1)]
    a_phys = sorted(float(k) for k in phys[phys_key])
    c_f, c_c, c_x = STYLE[args.mode]

    fig, ax = plt.subplots(figsize=(2.25, 1.9))
    g = curve(phys, phys_key, a_phys)
    ax.plot(np.array(a_phys), g, "-", color=COLOR_PHYS, label=r"$g$")
    f = curve(model, f"unpatched ({args.mode})", alphas)
    ax.plot(alphas, f, "-", color=c_f, label=r"$f$")
    cdag = curve(model, f"circuit ({args.mode})", alphas)
    ax.plot(alphas, cdag, "-", color=c_c, label=r"$C^\dagger$")
    compl = curve(model, f"complement ({args.mode})", alphas)
    ax.plot(alphas, compl, "--", color=c_x, label=r"$U \backslash C^\dagger$")

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)
    if args.resolvable:
        r = args.resolvable
        ax.axvspan(-r, r, color="#2ca02c", alpha=0.07, zorder=0)
        for x in (-r, r):
            ax.axvline(x, color="#2ca02c", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(0, 1.25, "radar-resolvable domain", color="#1a6b3a", fontsize=5.5,
                family="Avenir Next", fontweight="light", ha="center", va="center")

    ax.set_xlim(-2.1, 2.1)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    if args.mode == "radial":
        ax.set_ylim(-1.7, 1.7); ax.set_yticks([-1.5, -0.75, 0, 0.75, 1.5])
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.legend(loc="lower right", handlelength=1.4)
    else:
        ax.set_ylim(-0.2, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55), handlelength=1.4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\hat{m}(\alpha)$")
    ax.grid(alpha=0.15, which="both", linewidth=0.4)
    fig.tight_layout(pad=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    pd.DataFrame({"alpha": alphas, "f": f, "circuit": cdag, "complement": compl}).to_csv(
        args.out.with_name(args.out.name + "_curves.csv"), index=False)
    pd.DataFrame({"alpha": a_phys, "g": g}).to_csv(
        args.out.with_name(args.out.name + "_g.csv"), index=False)
    print(f"saved {args.out}.png and the two csv companions")


if __name__ == "__main__":
    main()
