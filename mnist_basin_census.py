# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Census of the energy landscape during self-reference (MNIST banks).

For every stored bank snapshot of the showcase cells, start the retrieval map
at 100 fresh real images and iterate it to convergence; the set of distinct
fixed points reached is the set of basins the landscape still has.  The claim
"self-reference flattens the landscape" predicts a staircase: the count falls
from many to ONE as minima merge.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mnist_c0 import load

HERE = Path(__file__).parent
SNAPS = HERE / "data" / "mnist_snaps"
OUT = HERE / "data" / "mnist_basin_census.npz"
CELLS = [("power", 2.0, "self"), ("power", 20.0, "self"),
         ("softmax", 4.0, "self"), ("softmax", 32.0, "self"),
         ("softmax", 32.0, "biased")]
N_PROBE, ITERS, TOL, DEDUP = 100, 300, 1e-9, 0.999


def weights_batch(Q: np.ndarray, Xc: np.ndarray, rule: str,
                  sharp: float) -> np.ndarray:
    qn = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    xn = np.linalg.norm(Xc, axis=1) + 1e-12
    m = (Q @ Xc.T) / (qn * xn)
    if rule == "power":
        w = np.maximum(m, 0.0) ** (sharp - 1.0)
    else:
        z = sharp * m
        z -= z.max(axis=1, keepdims=True)
        w = np.exp(z)
    s = w.sum(axis=1, keepdims=True)
    return np.where(s > 0, w / np.where(s > 0, s, 1.0), 1.0 / Xc.shape[0])


def attractors(P0: np.ndarray, Xi: np.ndarray, mu: np.ndarray, rule: str,
               sharp: float) -> int:
    Xc = Xi - mu
    X = P0.copy()
    for _ in range(ITERS):
        Xn = mu + weights_batch(X - mu, Xc, rule, sharp) @ Xc
        if np.abs(Xn - X).max() < TOL:
            X = Xn
            break
        X = Xn
    A = X - X.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    reps: list[np.ndarray] = []
    for a in A:
        if not any(float(a @ r) > DEDUP for r in reps):
            reps.append(a)
    return len(reps)


def main() -> None:
    xtr, _, xte, _ = load()
    mu = xtr.mean(0)
    probes = xte[np.random.default_rng(42).choice(len(xte), N_PROBE,
                                                  replace=False)]
    t0 = time.time()
    counts, labels = [], []
    rounds = None
    for rule, sharp, mode in CELLS:
        z = np.load(SNAPS / f"{rule}_{sharp:g}_{mode}_s0.npz")
        S = z["snaps"].astype(np.float64)          # (T, K, 784)
        rounds = np.arange(S.shape[0]) * 150
        row = [attractors(probes, S[t], mu, rule, sharp)
               for t in range(S.shape[0])]
        counts.append(row)
        labels.append(f"{rule}_{sharp:g}_{mode}")
        print(f"{rule:>8} {sharp:>5g} {mode:>7}: "
              f"{row[0]:>3} -> {row[-1]:>2} attractors  "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(OUT, counts=np.array(counts),
                        labels=np.array(labels), rounds=rounds)
    print("saved", OUT)


if __name__ == "__main__":
    main()
