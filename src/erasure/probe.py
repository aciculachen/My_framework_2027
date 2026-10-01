"""Per-marker physical targets and the ridge probe (centered ridge, time-contiguous CV)."""
from __future__ import annotations

import numpy as np


def marker_range(p_win: np.ndarray) -> np.ndarray:
    """[L, M, 3] raw positions (radar at origin) -> [L, M] range per frame."""
    return np.linalg.norm(np.asarray(p_win, np.float32), axis=-1)


def marker_vrad(p_win: np.ndarray) -> np.ndarray:
    """[L, M, 3] raw positions (radar at origin) -> [L, M] radial velocity per frame."""
    p = np.asarray(p_win, np.float32)
    rng = np.linalg.norm(p, axis=-1, keepdims=True)
    return np.sum(np.gradient(p, axis=0) * (p / np.maximum(rng, 1e-8)), axis=-1)


def marker_targets(inputs: np.ndarray, mocap_scaler) -> dict[str, np.ndarray]:
    """inputs [W, L, N, 3] normalised -> per-marker centre-frame targets {"v_rad", "range"} [W, N]."""
    W, L, N, _ = inputs.shape
    raw = mocap_scaler.inverse_transform(inputs.reshape(-1, N * 3)).reshape(W, L, N, 3).astype(np.float32)
    rng = np.linalg.norm(raw, axis=-1, keepdims=True)
    r_hat = raw / np.maximum(rng, 1e-8)
    v_rad = np.sum(np.gradient(raw, axis=1) * r_hat, axis=-1)
    c = L // 2
    return {"v_rad": v_rad[:, c].astype(np.float32), "range": rng[..., 0][:, c].astype(np.float32)}


def time_splits(n_windows: int, fit_end: float = 0.60, stride: int = 8) -> np.ndarray:
    """Window indices used to fit erasers: the first `fit_end` of the recording, every `stride`-th
    window (8 = the smallest step at which windows share no samples)."""
    return np.arange(0, int(n_windows * fit_end), stride)


def time_folds(n: int, k: int = 5, gap: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    """k contiguous test blocks over 0..n-1; `gap` windows either side are held out of training."""
    block = n // k
    folds = []
    for i in range(k):
        ts, te = i * block, ((i + 1) * block if i < k - 1 else n)
        train = np.ones(n, bool)
        train[max(0, ts - gap):min(n, te + gap)] = False
        folds.append((np.where(train)[0], np.arange(ts, te)))
    return folds


def ridge_cv_r2(X: np.ndarray, y: np.ndarray, folds, alpha: float = 1.0) -> tuple[float, float]:
    """X [n_win, N, E], y [n_win, N] -> (mean R^2 over folds, std). Windows are the CV unit."""
    per_fold = []
    for tr, te in folds:
        Xtr = X[tr].reshape(-1, X.shape[-1]).astype(np.float64)
        ytr = y[tr].reshape(-1).astype(np.float64)
        Xte = X[te].reshape(-1, X.shape[-1]).astype(np.float64)
        yte = y[te].reshape(-1).astype(np.float64)
        mx, my = Xtr.mean(0), ytr.mean()
        Xc, yc = Xtr - mx, ytr - my
        coef = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ yc)
        pred = (Xte - mx) @ coef + my
        ss_res = float(np.sum((yte - pred) ** 2))
        ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        per_fold.append(float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot)
    a = np.asarray(per_fold, float)
    return float(a.mean()), float(a.std(ddof=0))
