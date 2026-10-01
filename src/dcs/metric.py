"""DCS metric: power-weighted Doppler centroid, the consistency score consistency_score, and the
fftshifted frequency axis (load_freq_axis) the centroid is taken over.

Centroid weighting is power, 10^(dB/10) = |H(f)|^2 (first moment of the PSD).
consistency_score (the paper's M, DCS) is the R^2 of the least-squares slope alpha_pred(alpha)
against the imposed alpha over an alpha grid; the grid is an argument, not a constant.
"""
from __future__ import annotations

import numpy as np


_EPS = 1e-8


def compute_centroid_power(
    sxx_normalized: np.ndarray,
    sxx_scaler,
    freq_axis: np.ndarray,
    eps: float = 1e-12,
    db_divisor: float = 10.0,
) -> np.ndarray:
    """Power-weighted Doppler centroid (first moment of PSD).

      1. Denormalize via reshape(-1, 1) to handle n_features_in_=1 scaler
      2. Convert dB to power: 10^(dB/10) = |H(f)|^2
      3. fftshift to match freq_axis
      4. Weighted centroid

    Args:
        sxx_normalized: [N, F] model output in normalized scale.
        sxx_scaler:    Fitted StandardScaler for radar_sxx (n_features_in_=1).
        freq_axis:     [F] fftshifted frequency axis in Hz.

    Returns:
        centroids: [N] centroid frequencies in Hz.
    """
    N, F = sxx_normalized.shape
    sxx_db = sxx_scaler.inverse_transform(
        sxx_normalized.reshape(-1, 1)
    ).reshape(N, F)
    # db_divisor 10 = power (the convention here); 20 = amplitude
    power = np.power(10.0, sxx_db / db_divisor)
    power_shifted = np.fft.fftshift(power, axes=-1)
    total = np.sum(power_shifted, axis=-1, keepdims=True)
    return np.sum(freq_axis * power_shifted, axis=-1) / (
        total.squeeze(-1) + eps
    )


def alpha_pred(c_base: np.ndarray, c_other: np.ndarray) -> float:
    """Least-squares slope through the origin of c_other against c_base: the alpha the
    centroids imply. NaN entries are ignored."""
    c_base, c_other = np.asarray(c_base, np.float64), np.asarray(c_other, np.float64)
    ok = ~(np.isnan(c_base) | np.isnan(c_other))
    c_base, c_other = c_base[ok], c_other[ok]
    denom = float(np.sum(c_base ** 2))
    if denom < _EPS:
        return float("nan")
    return float(np.sum(c_base * c_other) / denom)


def consistency_score(
    centroid_base: np.ndarray,        # [N_windows]
    centroid_other_per_alpha: dict,   # {alpha: [N_windows]}
    alpha_values: list[float],
    reference: dict | None = None,    # {alpha: reference slope}; None -> the ideal law, slope = alpha
) -> float:
    """Consistency score M = 1 - SSR / SST: R^2 of alpha_pred(alpha) against a reference slope.

    reference=None compares against the ideal law (alpha_pred = alpha); passing the physics model's
    own slopes beta_g(alpha) gives the paper's M_P(f, g). NaN entries are dropped; NaN is returned
    if fewer than two alphas remain or the reference has no variance.
    """
    alphas_arr = np.array(alpha_values, dtype=np.float64)
    preds = np.array([alpha_pred(centroid_base, centroid_other_per_alpha[a]) for a in alpha_values], dtype=np.float64)
    if reference is None:
        ref = alphas_arr
    else:
        table = {round(float(k), 6): float(v) for k, v in reference.items()}
        ref = np.array([table[round(float(a), 6)] for a in alpha_values], dtype=np.float64)
    return score_from_slopes(preds, ref)


def score_from_slopes(preds, ref) -> float:
    """M = 1 - SSR / SST of the slopes `preds` against the reference slopes `ref` (same alphas)."""
    preds, ref = np.asarray(preds, np.float64), np.asarray(ref, np.float64)
    valid = ~(np.isnan(preds) | np.isnan(ref))
    if valid.sum() < 2:
        return float("nan")
    r, p_ = ref[valid], preds[valid]
    sst = float(np.sum((r - r.mean()) ** 2))
    if sst <= 0:
        return float("nan")
    return 1.0 - float(np.sum((r - p_) ** 2)) / sst


def load_freq_axis(meta: dict) -> np.ndarray:
    """[F] fftshifted STFT frequency axis in Hz, from meta.pkl."""
    return np.fft.fftshift(next(iter(meta["radar_f"].values())))


def alpha_range(start: float, stop: float, step: float) -> list[float]:
    """Inclusive grid start..stop in steps of `step`, rounded to 6 decimals (e.g. -2, 2, 0.01 -> 401 values)."""
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(n)]


def parse_alpha_spec(spec: str) -> tuple[str, list[float]]:
    """'name=start:stop:step' -> (name, alphas); 'name=a,b,c' lists explicit values;
    an optional '~a,b' suffix removes those values (e.g. 'full=-2:2:0.01~0,1')."""
    name, rhs = spec.split("=", 1)
    rhs, _, drop = rhs.partition("~")
    if ":" in rhs:
        start, stop, step = (float(x) for x in rhs.split(":"))
        alphas = alpha_range(start, stop, step)
    else:
        alphas = [round(float(x), 6) for x in rhs.split(",")]
    excluded = {round(float(x), 6) for x in drop.split(",")} if drop else set()
    return name, [a for a in alphas if round(a, 6) not in excluded]
