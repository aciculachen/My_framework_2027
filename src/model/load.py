"""Load the released BaselineMeanPoolAgg as a frozen eval-mode backbone."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from data import paths
from model.models import BaselineMeanPoolAgg

DEFAULT_CKPT = paths.PROJECT_ROOT / "models/baseline_mean_pool_agg.pt"


def load_backbone(ckpt_path: str | Path = DEFAULT_CKPT, device: torch.device | None = None):
    """Return (model, device). Weights are loaded strictly: a missing or extra tensor is an error."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_path)
    model = BaselineMeanPoolAgg(**json.load(open(ckpt_path.parent / "model_params.json")))
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, device
