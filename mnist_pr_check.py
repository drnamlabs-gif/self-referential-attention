# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 1 of the pre-registered protocol (proposal_dam.tex): validate the PR
measure on known cases BEFORE trusting it.

PR(xi) = (sum c+)^2 / sum (c+)^2 over the correlation profile c_i of memory xi
with the training images. Required ordering: real image -> LOW (explains ~its
own neighbourhood), class mean -> higher, global mean -> HIGH (explains
everything), random -> near the floor. If the ordering fails, stop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mnist_c0 import load, N


def pr_profile(Xi: np.ndarray, X: np.ndarray) -> np.ndarray:
    """PR of each row of Xi against reference images X (both raw [0,1])."""
    Xc = Xi - Xi.mean(axis=1, keepdims=True)
    Rc = X - X.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    Rn = Rc / (np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-12)
    C = np.maximum(Xn @ Rn.T, 0.0)
    num = C.sum(axis=1) ** 2
    den = (C ** 2).sum(axis=1) + 1e-12
    return num / den


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xtr, ytr, _, _ = load()
    ref = xtr[rng.choice(len(xtr), 3000, replace=False)]

    real = xtr[rng.choice(len(xtr), 50, replace=False)]
    class_means = np.stack([xtr[ytr == c].mean(0) for c in range(10)])
    global_mean = xtr.mean(0)[None]
    rand = rng.random((50, N))

    rows = [("real images (n=50)", pr_profile(real, ref)),
            ("class means (n=10)", pr_profile(class_means, ref)),
            ("global mean", pr_profile(global_mean, ref)),
            ("random vectors (n=50)", pr_profile(rand, ref))]
    print(f"{'case':>24}{'PR mean':>10}{'PR sd':>8}")
    for name, v in rows:
        print(f"{name:>24}{v.mean():>10.0f}{v.std():>8.0f}")

    ok = (rows[0][1].mean() < rows[1][1].mean() < rows[2][1].mean())
    print("\nordering real < class mean < global mean:", "PASS" if ok else "FAIL")
