"""Readers for data/processed (written by data.preproc)."""
import pickle
from pathlib import Path

import numpy as np

from data import paths


def _load(name, data_path=None):
    path = Path(data_path or paths.PROCESSED_DATA_DIR) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/preprocess.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_meta(data_path=None) -> dict:
    return _load("meta.pkl", data_path)


def load_scalers(data_path=None) -> dict:
    return _load("scalers.pkl", data_path)


def load_recording_targets(recording, split="eval", data_path=None) -> np.ndarray:
    """radar_sxx_normalized rows of one recording, in window order. Returns [N, F] float32."""
    d = _load(f"{split}_final_data.pkl", data_path)
    rows = sorted((p["original_idx"], i) for i, p in enumerate(d["provenance"]) if p["filename"] == recording)
    assert rows and [r for r, _ in rows] == list(range(len(rows))), f"{recording}: windows missing or unordered"
    return np.asarray(d["radar_sxx_normalized"][[i for _, i in rows]], dtype=np.float32)
