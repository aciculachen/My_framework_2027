#!/usr/bin/env python3
"""Composite from a spectrogram dump -> results/spectrograms/<name>/composite_a<α>_t<start>-<end>s.png
plus a csv of the plotted centroid lines. One row of five panels in two groups:

    Unperturbed:  Radar GT | Model f(x)      Counterfactual / Patching:  Model f(x_α) | Circuit | Complement

Every panel is the dB spectrogram of the clip; model panels carry the power-weighted Doppler
centroid as a dashed line.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "src"))
from dcs.metric import compute_centroid_power  # noqa: E402

PRED_COLOR = "#ff7f0e"


class _Scaler:
    def __init__(self, mean, scale):
        self.mean_, self.scale_ = np.array([mean]), np.array([scale])

    def inverse_transform(self, X):
        return X * self.scale_[0] + self.mean_[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="subdirectory under results/spectrograms/")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--start", type=int, default=0, help="first window")
    ap.add_argument("--windows", type=int, default=240, help="number of windows (240 = 30 s)")
    ap.add_argument("--font-size", type=float, default=18, help="base font size")
    ap.add_argument("--cmap", default="viridis")
    args = ap.parse_args()
    d = P / "results/spectrograms" / args.name
    assert d.is_dir(), f"no such dump directory: {d}"
    meta = json.load(open(d / "meta.json"))
    fa = np.asarray(meta["freq_axis_hz"]); scaler = _Scaler(meta["sxx_scaler_mean"], meta["sxx_scaler_scale"])
    # Convention (data.utils.iq_to_spectrogram): spectrogram bins are in raw FFT order [0..+, -..-1];
    # freq_axis_hz is the fftshifted axis. Display therefore always fftshifts the data, like the centroid.
    assert fa[0] < 0 < fa[-1], "freq_axis_hz must be the fftshifted axis"
    spw = meta["seconds_per_window"]
    s, e, a = args.start, args.start + args.windows, args.alpha
    panels = [(0, "Radar GT", "gt", False), (0, "Model", "clean_x", True),
              (1, "Model", f"clean_xprime_a{a:.2f}", True), (1, "Circuit", f"circuit_a{a:.2f}", True),
              (1, "Complement", f"complement_a{a:.2f}", True)]
    data = []
    for row, label, key, with_centroid in panels:
        arr = np.load(d / f"{key}.npy")[s:e]
        data.append((row, label, key, scaler.inverse_transform(arr), compute_centroid_power(arr, scaler, fa), with_centroid))
    flat = np.concatenate([x[3].ravel() for x in data]); vmin, vmax = np.percentile(flat[~np.isnan(flat)], [1, 99])
    n_t = e - s; duration = n_t * spw; t = np.arange(n_t) * spw
    stride = next(st for st in (1, 2, 5, 10, 20, 50) if duration / st <= 3)
    xticks = np.arange(0, duration + 1e-6, stride)

    fs = args.font_size
    mpl.rcParams.update({"font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs, "xtick.labelsize": fs * 0.9,
                         "ytick.labelsize": fs * 0.9, "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2})
    fig = plt.figure(figsize=(13.0, 3.4), constrained_layout=True)
    gs = GridSpec(1, 6, figure=fig, wspace=0.06, width_ratios=[1, 1, 0.25, 1, 1, 1])
    cols = [0, 1, 3, 4, 5]                          # column 2 is the spacer between the two groups
    axes, im = [], None
    for (row, label, key, arr_db, cent, with_centroid), c in zip(data, cols):
        ax = fig.add_subplot(gs[0, c], sharey=axes[0] if axes else None); axes.append(ax)
        img = np.fft.fftshift(arr_db.T, axes=0)                       # raw FFT order -> ascending frequency
        im = ax.imshow(img, origin="lower", aspect="auto", cmap=args.cmap, extent=[0, duration, fa[0], fa[-1]], vmin=vmin, vmax=vmax)
        ax.set_box_aspect(1.0)
        if with_centroid:
            ax.plot(t, cent, color=PRED_COLOR, linestyle="--", linewidth=2.2, alpha=0.95)
        ax.axhline(0, color="white", linewidth=0.9, alpha=0.35)
        ax.set_xticks(xticks)
        ax.text(0.5, 1.04, label, transform=ax.transAxes, ha="center", va="bottom")
        if c != 0:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Frequency (Hz)")
    fig.supxlabel("Time (s)", y=0.05)
    fig.colorbar(im, ax=axes, location="right", fraction=0.025, pad=0.02, shrink=0.92).set_label("Magnitude (dB)")
    fig.canvas.draw()                              # final axes positions for the group titles
    y_top = axes[0].get_position().y1               # panel labels sit at ~y_top + 0.04 .. 0.12 (fig fraction)
    for a0, a1, title in ((axes[0], axes[1], "Unperturbed"), (axes[2], axes[4], "Counterfactual / Patching")):
        x = 0.5 * (a0.get_position().x0 + a1.get_position().x1)
        fig.text(x, y_top + 0.17, title, ha="center", va="bottom", fontweight="bold", transform=fig.transFigure)
    out = d / f"composite_a{a:.2f}_t{s * spw:03.0f}-{e * spw:03.0f}s.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame({"t_s": t, **{key: cent for _, _, key, _, cent, _ in data}}).to_csv(out.with_suffix(".csv"), index=False)
    print(f"saved {out} and {out.with_suffix('.csv').name}")


if __name__ == "__main__":
    main()
