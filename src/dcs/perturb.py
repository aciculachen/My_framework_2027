"""Law-defined input perturbations for DCS and domain-of-validity sweeps, plus windowing.

Every marker's frame-to-frame displacement is split into a radial component (along the
line from RADIAL_ORIGIN to the marker's previous position) and a tangential remainder; the
trajectory is re-integrated from the first frame.

modes:  radial            alpha * radial + tangential          (the DCS perturbation)
        tangential        radial + alpha * tangential
        uniform           alpha * (radial + tangential)
        radial_pure       alpha * radial                        (tangential removed)
        tangential_pure   alpha * tangential                    (radial removed)
"""
from __future__ import annotations

import numpy as np

from data import constants
from data.utils import make_sequences

_EPS = 1e-8
MODES = ("radial", "tangential", "uniform", "radial_pure", "tangential_pure")

RADIAL_ORIGIN = constants.RADAR_POS_M.mean(axis=0)


def scale_velocity(positions, alpha: float, mode: str = "radial", origin=RADIAL_ORIGIN) -> np.ndarray:
    """positions [T, M, 3] -> perturbed positions [T, M, 3] (float32); `origin` is the point the
    radial direction is measured from."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    p = np.asarray(positions, dtype=np.float32)
    radial_vec = p[:-1] - np.asarray(origin, dtype=np.float32)
    radial_unit = radial_vec / (np.linalg.norm(radial_vec, axis=2, keepdims=True) + _EPS)
    delta = p[1:] - p[:-1]
    radial = np.sum(delta * radial_unit, axis=2, keepdims=True) * radial_unit
    tangential = delta - radial
    new_delta = {"radial": alpha * radial + tangential, "tangential": radial + alpha * tangential,
                 "uniform": alpha * (radial + tangential), "radial_pure": alpha * radial,
                 "tangential_pure": alpha * tangential}[mode]
    out = np.empty_like(p)
    out[0] = p[0]
    out[1:] = p[0:1] + np.cumsum(new_delta, axis=0, dtype=np.float32)
    return out


def window_data(mocap_full, mocap_scaler, nperseg, noverlap):
    """Window a [T, M, 3] trajectory and normalize with the train-split scaler.
    Returns (normalized windows [N, L, M, 3] float32, per-frame range [N, L, M],
    unnormalized windowed positions [N, L, M, 3]). Callers that only need the first
    two may ignore the third; range and positions both measure from the origin."""
    windowed = make_sequences({"seq": mocap_full}, nperseg, noverlap)["seq"]
    N_win, L, M, D = windowed.shape
    flat = windowed.reshape(N_win * L, M * D)
    norm = mocap_scaler.transform(flat).reshape(N_win, L, M, D).astype(np.float32)
    r = np.linalg.norm(windowed, axis=-1)
    return norm, r, windowed
