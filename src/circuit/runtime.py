"""Multi-GPU evaluator: `Evaluator(cache_dir, n_gpus)(components, seed=None) -> M`.

One spawned process per GPU. Each worker loads the model and a round-robin share of the
window cache (fp16, memory-mapped) and, per task, runs the patched forward for every (window, alpha)
it owns; the parent assembles the centroids and returns consistency_score against the cached clean
centroids. With `seed`, each (window, alpha) takes its source from a random other window of
the same worker's share (j != i); with n_gpus=1 that is a global random window.
"""
from __future__ import annotations

import os
import queue
import sys
import time
from pathlib import Path

import numpy as np
import torch.multiprocessing as tmp
from tqdm import tqdm

from circuit.cache import load_centroids, load_manifest
from circuit.patched_m import clean_score_from_cache
from dcs.metric import consistency_score

SRC = Path(__file__).resolve().parents[1]


def _worker(rank, n_gpus, cache_dir, n_windows, alpha_set, in_q, out_q, ready):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(SRC))
    import torch
    torch.set_num_threads(2)
    from circuit.cache import alpha_slug, load_window
    from circuit.patched_m import patched_centroids
    from data import paths
    from data.data_loader import load_meta, load_scalers
    from dcs.metric import load_freq_axis
    from model.load import load_backbone
    from patching.components import to_hook_names

    model, device = load_backbone(device=torch.device("cuda:0"))
    dp = paths.PROCESSED_DATA_DIR
    sxx_scaler, freq = load_scalers(dp)["radar_sxx"], load_freq_axis(load_meta(dp))
    own = list(range(rank, n_windows, n_gpus))
    windows = {i: load_window(Path(cache_dir), i) for i in own}
    slugs = [alpha_slug(a) for a in alpha_set]
    ready.set()
    while (task := in_q.get()) is not None:
        jmap = None
        if task["seed"] is not None:
            rng = np.random.default_rng(task["seed"] * 991 + rank)
            jmap = {i: rng.choice([j for j in own if j != i], size=len(slugs)) for i in own}
        out_q.put(patched_centroids(model, device, own, windows.__getitem__, to_hook_names(task["components"]),
                                    slugs, sxx_scaler, freq, jmap=jmap))


class Evaluator:
    def __init__(self, cache_dir: Path, n_gpus: int, reference: dict | None = None):
        self.cache_dir = Path(cache_dir)
        self.manifest = load_manifest(self.cache_dir)
        self.centroids = load_centroids(self.cache_dir)
        self.n_windows, self.alpha_set = self.manifest["n_windows"], self.manifest["alpha_set"]
        self.reference = reference
        self.M0 = clean_score_from_cache(self.centroids, self.alpha_set, reference)
        self.n_gpus = n_gpus
        ctx = tmp.get_context("spawn")
        self.in_qs = [ctx.Queue() for _ in range(n_gpus)]
        self.out_qs = [ctx.Queue() for _ in range(n_gpus)]
        events = [ctx.Event() for _ in range(n_gpus)]
        self.procs = [ctx.Process(target=_worker, args=(r, n_gpus, str(self.cache_dir), self.n_windows,
                                                        self.alpha_set, self.in_qs[r], self.out_qs[r], events[r]))
                      for r in range(n_gpus)]
        for p in self.procs:
            p.start()
        for r in tqdm(range(n_gpus), desc="workers loading"):
            while not events[r].wait(timeout=30):
                if not self.procs[r].is_alive():
                    raise RuntimeError(f"GPU worker {r} died while loading (exit code {self.procs[r].exitcode})")
        self.n_calls, self.t0 = 0, time.time()

    def __call__(self, components: list[str], seed: int | None = None) -> float:
        task = {"components": list(components), "seed": seed}
        for q in self.in_qs:
            q.put(task)
        cent = np.zeros((self.n_windows, len(self.alpha_set)), np.float32)
        for r, q in enumerate(self.out_qs):
            while True:                                   # wait, but fail loudly if the worker died
                try:
                    part = q.get(timeout=30)
                    break
                except queue.Empty:
                    if not self.procs[r].is_alive():
                        raise RuntimeError(f"GPU worker {r} died (exit code {self.procs[r].exitcode}); check host memory")
            for (i, a), c in part.items():
                cent[i, a] = c
        self.n_calls += 1
        if self.n_calls % 10 == 0:
            print(f"    [evaluator] {self.n_calls} evaluations, {(time.time() - self.t0) / self.n_calls:.1f} s each", flush=True)
        return consistency_score(self.centroids["centroid_base"], {a: cent[:, k] for k, a in enumerate(self.alpha_set)}, self.alpha_set, self.reference)

    def close(self):
        for q in self.in_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
