"""Kinematic Doppler reference g: what the physics predicts for a (perturbed) trajectory.

Per marker: range to the radar (at the origin) -> radial velocity -> Doppler 2 v_r / lambda.
Per window: time-mean of the per-marker Doppler, then a body-surface-area weighted mean over
markers (Rule of Nines, Wallace 1951: head 9 %, torso 18.5 %, pelvis 18.5 %, each arm 9 %,
each leg 18 %; a segment's share is spread evenly over its markers). Since alpha_pred is a
ratio of centroids, the sample rate and wavelength cancel in the sweeps.
"""
from __future__ import annotations

import numpy as np

from data import constants
from data.utils import compute_mocap_range, compute_mocap_vel, estimate_doppler_frequency, make_sequences

RULE_OF_NINES = {"Head": 0.09, "Torso": 0.185, "Pelvis": 0.185,
                 "LeftShoulder": 0.0225, "LeftUpperArm": 0.0225, "LeftForearm": 0.0225, "LeftHand": 0.0225,
                 "RightShoulder": 0.0225, "RightUpperArm": 0.0225, "RightForearm": 0.0225, "RightHand": 0.0225,
                 "LeftThigh": 0.06, "LeftShank": 0.06, "LeftFoot": 0.06,
                 "RightThigh": 0.06, "RightShank": 0.06, "RightFoot": 0.06}


def bsa_weights(nodes=constants.MOCAP_NODES, segments=constants.MOCAP_SEGMENTS) -> np.ndarray:
    """[M] weights summing to 1: each segment's Rule-of-Nines share split evenly over its markers."""
    w = np.zeros(len(nodes))
    idx = {n: i for i, n in enumerate(nodes)}
    for seg, markers in segments.items():
        for m in markers:
            w[idx[m]] = RULE_OF_NINES[seg] / len(markers)
    assert abs(w.sum() - 1.0) < 1e-6, w.sum()
    return w


def kinematic_centroids(mocap, nperseg: int, noverlap: int, fps: float = constants.RD_SP_HZ,
                        wavelength: float = constants.RADAR_WAVELENGTH, weights=None) -> np.ndarray:
    """mocap [T, M, 3] (radar at origin) -> BSA-weighted mean Doppler per window [N]."""
    w = bsa_weights() if weights is None else np.asarray(weights)
    rng = compute_mocap_range({"x": np.asarray(mocap, np.float64)}, centroid=np.zeros((1, 1, 3)))["x"]
    dop = estimate_doppler_frequency(compute_mocap_vel({"x": rng}, fps=fps), wavelength=wavelength)["x"]
    win = make_sequences({"x": dop}, nperseg, noverlap)["x"].squeeze(-1)       # [N, L, M]
    return win.mean(axis=1) @ (w / w.sum())
