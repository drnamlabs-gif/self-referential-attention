# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 2b: what ORDER is the consensus transition (beta = 2)?

sweep_fss.py showed a susceptibility peak that grows about in proportion to K
and a Binder cumulant that dips far below zero near alpha_c. Both are
textbook signs of a FIRST-ORDER transition: two coexisting states (symmetric
and consensus) with switching between them. A continuous transition shows
neither. Notebook 01 found no hysteresis at K = 20, but a first-order
transition only shows hysteresis once switching becomes slower than the run,
i.e. at large K.

Test, beta = 2, fine alpha grid around alpha_c ~ 0.375:
  K in {80, 160, 320, 640}, two starts:
    'sym'  : two balanced clusters (alive)
    'cons' : every point on ONE cluster (consensus)
  400 generations; the full phi time series of the second half is kept, so
  the distribution of phi (one hump or two) can be read directly.
Hysteresis at large K + bimodal phi = first order.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sweep_K import env, relax

BETA = 2.0
ALPHAS = np.round(np.linspace(0.33, 0.42, 10), 3)
KS = [80, 160, 320, 640]
STARTS = ["sym", "cons"]
SEEDS = {80: 6, 160: 6, 320: 6, 640: 4}
GEN = 400
MAX_WORKERS = 16
OUT = Path(__file__).parent / "data" / "sweep_order.npz"


def one(args):
    K, ai, start, seed = args
    alpha = float(ALPHAS[ai])
    rng = np.random.default_rng(seed * 2654435761 % 2**32 + K * 7 + ai * 101 + (start == "cons"))
    X = env(rng, K)
    if start == "cons":
        X[:, 0] = np.abs(X[:, 0])
    phis = []
    for g in range(1, GEN + 1):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1)[0]
            X[j] = q if rng.random() < alpha else relax(q, X, BETA)
        if g > GEN // 2:
            p = (X[:, 0] > 0).mean()
            phis.append(abs(2 * p - 1))
    return args, np.array(phis, np.float32)


def main():
    jobs = [(K, ai, st, s) for K in KS for ai in range(len(ALPHAS)) for st in STARTS
            for s in range(SEEDS[K])]
    jobs.sort(key=lambda j: -j[0])
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((K, ai, st, s), ph) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            out[f"K{K}_a{ai}_{st}_s{s}"] = ph
            if n % 40 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT, Ks=KS, alphas=ALPHAS, beta=BETA, **out)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
