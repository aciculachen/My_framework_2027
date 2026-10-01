"""Circuit-search procedures over an `evaluate(components, seed=None) -> M` callable.

A set C is tau-consistent iff
    Delta_suf(C) = M_0 - M(C)      <= (1 - tau) * M_0      (sufficiency: patching C alone recovers the behaviour)
    Delta_nec(C) = M_0 - M(U \\ C)  >= tau * M_0            (necessity: patching everything but C does not)
`greedy_minimal` is the conditional greedy build (add the component with the smallest marginal
Delta_suf until tau-consistent) followed by multi-ordering minimality pruning; components in C_init
are locked, which keeps circuits nested across an ascending tau grid.
`source_randomized` evaluates a component set with a random other window as source.
The callable hides where forwards run (circuit.runtime), so these functions are pure Python.
"""
from __future__ import annotations

import numpy as np

from patching.components import ALL_COMPONENTS, complement


def single_deltas(evaluate, M0: float) -> dict[str, float]:
    """Delta_suf({c}) for every component."""
    return {c: M0 - evaluate([c]) for c in ALL_COMPONENTS}


def greedy_minimal(evaluate, M0: float, tau: float, singles: dict[str, float], *,
                   C_init: list[str] | None = None, n_prune_trials: int = 8, prune_seed: int = 0) -> dict:
    thr_suf, thr_nec = (1.0 - tau) * M0, tau * M0
    locked = set(C_init or [])
    C = list(C_init or [])
    memo: dict[tuple, float] = {}
    n_evals = 0

    def M(S):
        nonlocal n_evals
        key = tuple(sorted(S))
        if key not in memo:
            memo[key] = evaluate(list(key)); n_evals += 1
        return memo[key]

    def d_suf(S):
        return M0 - M(S)

    def d_nec(S):
        return M0 - M(complement(S))

    def consistent(S):
        return bool(S) and d_suf(S) <= thr_suf and d_nec(S) >= thr_nec

    while not consistent(C):
        candidates = [c for c in ALL_COMPONENTS if c not in C]
        if not candidates:
            break
        if not C:   # marginal from the empty set equals the single-component delta
            best = min(candidates, key=lambda c: (singles[c], ALL_COMPONENTS.index(c)))
        else:
            best = min(((d_suf(C + [c]), ALL_COMPONENTS.index(c), c) for c in candidates))[2]
        C.append(best)

    rng = np.random.default_rng(prune_seed)
    best_C, best_delta = list(C), d_suf(C)
    for _ in range(n_prune_trials):
        order = [c for c in C if c not in locked]
        rng.shuffle(order)
        C_try = list(C)
        removed = True
        while removed:
            removed = False
            for c in order:
                if c not in C_try or len(C_try) == 1:
                    continue
                C_minus = [x for x in C_try if x != c]
                if consistent(C_minus):
                    C_try, removed = C_minus, True
                    break
        d_try = d_suf(C_try)
        if len(C_try) < len(best_C) or (len(C_try) == len(best_C) and d_try < best_delta):
            best_C, best_delta = list(C_try), d_try

    C = best_C
    return {"tau": tau, "C": C, "size": len(C), "Delta_suf": best_delta, "M_suf": M0 - best_delta,
            "Delta_nec": d_nec(C), "M_nec": M(complement(C)), "consistent": consistent(C),
            "threshold_suf": thr_suf, "threshold_nec": thr_nec,
            "C_init": list(C_init or []), "n_prune_trials": n_prune_trials, "n_evals": n_evals}


def source_randomized(evaluate, M0: float, C: list[str], seeds: list[int]) -> dict:
    """Delta_suf of C when every (window, alpha) takes its source from a random other window."""
    per_seed = [float(evaluate(C, seed=s)) for s in seeds]
    arr = np.asarray(per_seed)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"C": list(C), "seeds": list(seeds), "M_per_seed": per_seed,
            "Delta_mean": float(M0 - arr.mean()), "Delta_std": std,
            "Delta_se": std / np.sqrt(arr.size) if arr.size > 1 else 0.0, "n_seeds": int(arr.size)}
