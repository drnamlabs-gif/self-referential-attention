# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 3c: is the environmental spread sigma what breaks mean field at high beta?

At beta = 4 the consensus state is left at outside fractions far below the
mean-field edge even at K = 1280. The mean field puts every outside pattern
at |x| = 1. With spread sigma, a pattern's self-energy is beta |x|^2, so an
outside pattern of slightly larger norm holds a basin exponentially more
easily. Test: beta = 4, K = 320, consensus start, sigma in {0.02, 0.05, 0.1}.
If sigma is the cause, the small-sigma threshold moves into the mean-field
window (0.019 - 0.027).
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

BETA, K, GEN = 4.0, 320, 600
SIGMAS = [0.02, 0.05, 0.1]
ALPHAS = np.round(np.geomspace(0.001, 0.05, 8), 5)
SEEDS = 3
OUT = Path(__file__).parent / "data" / "sweep_sigma.npz"


def one(job):
    si, ai, st, seed = job
    sg, al = SIGMAS[si], float(ALPHAS[ai])
    rng = np.random.default_rng(seed * 7919 + si * 131 + ai * 17 + (st == "cons"))
    X = env(rng, K, s=sg)
    if st == "cons":
        X[:, 0] = np.abs(X[:, 0])
    ph = []
    for g in range(1, GEN + 1):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1, s=sg)[0]
            X[j] = q if rng.random() < al else relax(q, X, BETA)
        if g > GEN // 2:
            ph.append(abs(2 * (X[:, 0] > 0).mean() - 1))
    return job, np.array(ph, np.float32)


def main():
    jobs = [(si, ai, st, s) for si in range(len(SIGMAS)) for ai in range(len(ALPHAS))
            for st in ("sym", "cons") for s in range(SEEDS)]
    out, t0 = {}, time.time()
    with ProcessPoolExecutor(8) as ex:
        for n, ((si, ai, st, s), ph) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            out[f"s{si}_a{ai}_{st}_s{s}"] = ph
            if n % 12 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT, sigmas=SIGMAS, alphas=ALPHAS, beta=BETA, K=K, **out)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
