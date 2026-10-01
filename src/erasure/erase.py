"""Linear concept erasure (LEACE, Belrose et al. 2023), one eraser per position, and the two sites
it is applied at.

For a k-dim concept Z, with Sigma the pooled covariance and C_t = Cov_t(X, Z) at position t, remove
the oblique rank-k component

    r_t(x) = x - A_t B_t^T (x - mu_t),   A_t = Sigma^{1/2} Q_t,  B_t = Sigma^{-1/2} Q_t,
    Q_t = orth(Sigma^{-1/2} C_t)

so that no linear probe fitted at position t can read Z. A random eraser uses a random Q_t of the
same rank. Tensors are laid out [L, T, E]: L frames, T markers, E features.

Sites:  erased_attn   last temporal layer, 8-head concat z before W_O (a column subset = a head subset)
        erased_smlp   last spatial layer, FFN output before the residual addition
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@contextmanager
def swapped(pairs):
    saved = [(obj, attr, getattr(obj, attr)) for obj, attr, _ in pairs]
    for obj, attr, new in pairs:
        setattr(obj, attr, new)
    try:
        yield
    finally:
        for obj, attr, old in reversed(saved):
            setattr(obj, attr, old)


@dataclass
class Eraser:
    mu: torch.Tensor      # [L, E]
    A: torch.Tensor       # [L, E, k]
    B: torch.Tensor       # [L, E, k]

    def __call__(self, zc: torch.Tensor) -> torch.Tensor:
        d = torch.einsum("lte,lek->ltk", zc - self.mu[:, None], self.B)
        return zc - torch.einsum("ltk,lek->lte", d, self.A)

    def to(self, dtype):
        return Eraser(self.mu.to(dtype), self.A.to(dtype), self.B.to(dtype))


class EraseStats:
    """Sufficient statistics: pooled Sigma over every token, per-position means and Cov(X, Z)."""

    def __init__(self, L: int, E: int, k: int, device):
        f = dict(dtype=torch.float64, device=device)
        self.sxx, self.sx_all, self.n_all = torch.zeros(E, E, **f), torch.zeros(E, **f), 0
        self.sx, self.sz = torch.zeros(L, E, **f), torch.zeros(L, k, **f)
        self.sxz, self.n = torch.zeros(L, E, k, **f), 0

    def add(self, X: torch.Tensor, Z: torch.Tensor):
        """X [L, T, E], Z [L, T, k]"""
        X, Z = X.double(), Z.double()
        Xf = X.reshape(-1, X.shape[-1])
        self.sxx += Xf.T @ Xf; self.sx_all += Xf.sum(0); self.n_all += Xf.shape[0]
        self.sx += X.sum(1); self.sz += Z.sum(1); self.sxz += torch.einsum("lte,ltk->lek", X, Z); self.n += X.shape[1]

    def whitening(self, eps: float = 1e-6):
        mu = self.sx_all / self.n_all
        cov = self.sxx / self.n_all - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(eps * evals.max())
        return (evecs * evals.sqrt()) @ evecs.T, (evecs / evals.sqrt()) @ evecs.T

    def _eraser(self, Q: torch.Tensor) -> Eraser:
        Sh, Sih = self.whitening()
        return Eraser(self.sx / self.n, torch.einsum("ef,lfk->lek", Sh, Q), torch.einsum("ef,lfk->lek", Sih, Q))

    def concept_eraser(self) -> Eraser:
        mx, mz = self.sx / self.n, self.sz / self.n
        C = self.sxz / self.n - torch.einsum("le,lk->lek", mx, mz)
        _, Sih = self.whitening()
        Q, _ = torch.linalg.qr(torch.einsum("ef,lfk->lek", Sih, C))
        return self._eraser(Q)

    def random_eraser(self, seed: int) -> Eraser:
        g = torch.Generator(device="cpu").manual_seed(seed)
        G = torch.randn(self.sx.shape[0], self.sx.shape[1], self.sz.shape[1], generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(G.to(self.sx.device))
        return self._eraser(Q)


def erased_input(model, holder: dict):
    """Native forward that records the last temporal layer's 8-head concat z [L, T, E] in `holder`."""
    layer = model.temporal_transformer.layers[-1]
    heads, d = layer.self_attn.num_heads, layer.head_dim

    def patched(xx, need_attn=False, average_attn_weights=False):
        h = layer.ln1(xx)
        attn_out, attn_w, v = layer._forward_qkov_split(h, capture=True)
        z = attn_w @ v                                                   # [T, heads, L, d]
        holder["z_concat"] = z.permute(2, 0, 1, 3).reshape(z.shape[2], z.shape[0], heads * d).detach()
        xx = xx + layer.dropout(attn_out)
        return xx + layer.ffn(layer.ln2(xx)), attn_out, None

    return swapped([(layer, "forward", patched)])


def erased_attn(model, eraser: Eraser, holder: dict | None = None, cols: torch.Tensor | None = None):
    """The last temporal layer's 8-head concat z passes through `eraser` before W_O; `cols` restricts it
    to those columns (head h owns columns h*d .. (h+1)*d). `holder` receives `z_concat` (post-erasure)."""
    layer = model.temporal_transformer.layers[-1]
    heads, d = layer.self_attn.num_heads, layer.head_dim

    def patched(xx, need_attn=False, average_attn_weights=False):
        h = layer.ln1(xx)
        _, attn_w, v = layer._forward_qkov_split(h, capture=True)          # [T, heads, L, L], [T, heads, L, d]
        z = attn_w @ v
        L, T = z.shape[2], z.shape[0]
        zc = z.permute(2, 0, 1, 3).reshape(L, T, heads * d)
        if cols is None:
            zc = eraser(zc)
        else:
            zc = zc.clone()
            zc[:, :, cols] = eraser(zc[:, :, cols])
        if holder is not None:
            holder["z_concat"] = zc.detach()
        fz = zc.reshape(L, T, heads, d).permute(1, 2, 0, 3)
        attn_out, _, _ = layer._forward_qkov_split(h, frozen_z=fz)
        xx = xx + layer.dropout(attn_out)
        return xx + layer.ffn(layer.ln2(xx)), attn_out, None

    return swapped([(layer, "forward", patched)])


def erased_smlp(model, eraser: Eraser | None, holder: dict | None = None):
    """The last spatial layer's FFN output ([N, B*L, E]) passes through `eraser` (None: untouched)
    before the residual addition, laid out [L, B*N, E]. `holder` receives `smlp` [L, B*N, E]."""
    layer = model.spatial_transformer.layers[-1]

    def patched(x, need_attn=False, average_attn_weights=False):
        h = layer.ln1(x)
        attn_out, attn_w = layer.self_attn(h, h, h, need_weights=need_attn, average_attn_weights=average_attn_weights)
        x = x + layer.dropout(attn_out)
        d = layer.ffn(layer.ln2(x))                                      # [N, B*L, E]
        N, BL, E = d.shape
        L = eraser.mu.shape[0] if eraser is not None else BL
        B = BL // L
        d = d.view(N, B, L, E).permute(2, 1, 0, 3).reshape(L, B * N, E)
        if eraser is not None:
            d = eraser(d)
        if holder is not None:
            holder["smlp"] = d.detach()
        d = d.view(L, B, N, E).permute(2, 1, 0, 3).reshape(N, BL, E)
        return x + d, attn_out, attn_w

    return swapped([(layer, "forward", patched)])
