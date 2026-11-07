# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""C0: does this task actually need a high-rank memory set?

The spin problem turned out degenerate -- at high quality it has one solution
(3 configurations below -480, pairwise overlap 0.947, effective rank 1.1) -- so
"diversity versus quality" there is a property of the problem and no architecture
can break it.  Before asking whether self-referential learning destroys a memory
set's diversity, establish that the task needs diversity at all.

Test: take a memory set of real MNIST images, truncate to rank r by SVD, measure
inpainting quality against r.  Saturation at small r means MNIST is degenerate
too and the question is unaskable here.

Two details matter and the first version got both wrong.  Raw MNIST in [-1,1] is
dominated by its mean (background), giving every pair an overlap near 0.9 and an
effective rank of 2 -- so retrieval returns the mean image regardless.  Memories
are therefore centred and unit-normalised.  And with half the pixels occluded,
the overlap must be computed on the OBSERVED pixels only; using all of them lets
the missing region vote on its own reconstruction.
"""
from __future__ import annotations

import numpy as np

from pathlib import Path

_CANDIDATES = (
    Path(__file__).parent / "data" / "mnist.npz",
    Path.home() / ".keras" / "datasets" / "mnist.npz",
)
MNIST = str(next(p for p in _CANDIDATES if p.exists()))
N = 784


def load():
    d = np.load(MNIST)
    f = lambda a: a.reshape(-1, N).astype(np.float64) / 255.0
    return f(d["x_train"]), d["y_train"], f(d["x_test"]), d["y_test"]


def retrieve(Vq, mask, Xi, mu, n=20.0, steps=1):
    """Masked modern-Hopfield retrieval.

    Overlap is a cosine on the observed pixels only; the weight on memory mu is
    max(m,0)^(n-1), so n controls how sharply the closest memories dominate.
    """
    Xc = Xi - mu
    out = np.empty_like(Vq)
    for b in range(Vq.shape[0]):
        ob = mask[b]
        q = Vq[b, ob] - mu[ob]
        Xo = Xc[:, ob]
        qn = np.linalg.norm(q) + 1e-12
        xn = np.linalg.norm(Xo, axis=1) + 1e-12
        m = (Xo @ q) / (xn * qn)
        w = np.maximum(m, 0.0) ** (n - 1.0)
        s = w.sum()
        out[b] = mu + (w @ Xc) / s if s > 0 else mu
    return out


def eff_rank(X):
    sv = np.linalg.svd(X, compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    return float(1.0 / np.sum(p ** 2))


def rank_truncate(Xi, mu, r):
    """Truncate the CENTRED memory set, then add the mean back."""
    Xc = Xi - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S2 = S.copy(); S2[r:] = 0.0
    return mu + (U * S2) @ Vt


def score(R, Vtrue, mask, ytrue, xr, yr):
    hid = ~mask
    err = np.array([np.sqrt(((R[i, hid[i]] - Vtrue[i, hid[i]]) ** 2).mean())
                    for i in range(len(R))])
    d = ((R[:, None, :] - xr[None, :, :]) ** 2).sum(-1)
    acc = (yr[d.argmin(1)] == ytrue).mean()
    return float(err.mean()), float(acc)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xtr, ytr, xte, yte = load()
    mu = xtr.mean(0)

    K, nq = 100, 400
    idx = np.concatenate([rng.choice(np.flatnonzero(ytr == c), K // 10, replace=False)
                          for c in range(10)])
    Xi0 = xtr[idx]

    q = rng.choice(len(xte), nq, replace=False)
    Vtrue, ytrue = xte[q], yte[q]
    mask = rng.random(Vtrue.shape) > 0.5          # True = observed
    Vq = np.where(mask, Vtrue, 0.0)

    ref = rng.choice(len(xtr), 3000, replace=False)
    xr, yr = xtr[ref], ytr[ref]

    print(f"C0 trên MNIST — inpainting, K={K} ký ức thật, che 50%, {nq} truy vấn")
    print(f"hạng hiệu dụng bộ nhớ (đã tâm hoá): {eff_rank(Xi0 - mu):.1f} / {K}")
    print(f"[mốc] 1-NN trên chính {K} ký ức, ảnh SẠCH: "
          f"{(ytr[idx][(((Vtrue[:,None,:]-Xi0[None])**2).sum(-1)).argmin(1)]==ytrue).mean():.3f}\n")

    e0, a0 = score(np.tile(mu, (nq, 1)), Vtrue, mask, ytrue, xr, yr)
    print(f"{'hạng r':>8}{'hạng hd':>10}{'RMSE vùng che':>16}{'acc 1-NN':>11}")
    print(f"{'ảnh TB':>8}{1.0:>10.1f}{e0:>16.4f}{a0:>11.3f}   <- sàn")
    for r in (1, 2, 3, 5, 8, 12, 20, 35, 60, K):
        Xr = rank_truncate(Xi0, mu, r)
        R = retrieve(Vq, mask, Xr, mu, n=20.0)
        e, a = score(R, Vtrue, mask, ytrue, xr, yr)
        print(f"{r:>8}{eff_rank(Xr-mu):>10.1f}{e:>16.4f}{a:>11.3f}", flush=True)
