# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 3d: the relaxation cycle at sharp retrieval (beta = 4, K = 1280, alpha = 0.0005).

Records phi, the median |x_1| of the stored patterns and the basin count every
5 generations, from a consensus start and from a diverse start.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sweep_K import census, env, relax

K, BETA, ALPHA, GEN, EVERY = 1280, 4.0, 0.0005, 600, 5
OUT = Path(__file__).parent / "data" / "diag_cycle.npz"


def one(job):
    st, seed = job
    rng = np.random.default_rng(seed + 100)
    X = env(rng, K)
    if st == "cons":
        X[:, 0] = np.abs(X[:, 0])
    rows = []
    for g in range(1, GEN + 1):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1)[0]
            X[j] = q if rng.random() < ALPHA else relax(q, X, BETA)
        if g % EVERY == 0:
            rows.append((g, abs(2 * (X[:, 0] > 0).mean() - 1), np.median(np.abs(X[:, 0])), census(X, BETA)))
    return job, np.array(rows)


if __name__ == "__main__":
    jobs = [("cons", 0), ("sym", 0), ("cons", 1), ("sym", 1)]
    with ProcessPoolExecutor(4) as ex:
        out = {f"{st}_s{s}": r for (st, s), r in ex.map(one, jobs)}
    np.savez_compressed(OUT, **out)
    print("saved", OUT)
