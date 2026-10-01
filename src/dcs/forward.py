"""Clean forwards of the model turned into Doppler centroids, batched."""
from __future__ import annotations

import numpy as np
import torch

from dcs.metric import compute_centroid_power


def batched_centroids(model, device, x_all, sxx_scaler, freq_axis, batch: int = 64) -> np.ndarray:
    """x_all [N, L, M, 3] float32 -> power-weighted centroid of model(x) for every window, [N]."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(x_all), batch):
            y = model(torch.as_tensor(x_all[i:i + batch]).to(device)).cpu().numpy()
            out.append(compute_centroid_power(y, sxx_scaler, freq_axis))
    return np.concatenate(out)
