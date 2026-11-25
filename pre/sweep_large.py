# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 3a: two checks needed before the theory comparison is final.

(A) beta = 3, larger banks. In sweep_fss.py alpha_c(beta = 3) still drifted
    upward with K (0.03 -> 0.05). Is it converging into the mean-field
    coexistence window (0.095 - 0.14), or does it stay below?
    K in {160, 320, 640, 1280}, two starts (sym, cons), 10 alphas, 400 generations.
(B) Dwell times at beta = 2, alpha = 0.385 (coexistence point). For a
    first-order transition the mean time between switches of the order
    parameter must grow exponentially with K.
    K in {40, 80, 160, 320, 640}, one long run of 3000 generations per seed.
Both store the full phi = |2p - 1| series.
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

A_ALPHAS = np.round(np.geomspace(0.01, 0.16, 10), 4)
A_KS = [160, 320, 640, 1280]
A_SEEDS = 4
A_GEN = 400
B_ALPHA, B_KS, B_SEEDS, B_GEN = 0.385, [40, 80, 160, 320, 640], 6, 3000
MAX_WORKERS = 20
OUT = Path(__file__).parent / "data" / "sweep_large.npz"


def run(K, beta, alpha, start, seed, gen, keep_from):
    rng = np.random.default_rng(seed * 2654435761 % 2**32 + K * 7 + int(alpha * 1e4) * 101 + (start == "cons"))
    X = env(rng, K)
    if start == "cons":
        X[:, 0] = np.abs(X[:, 0])
    phis = []
    for g in range(1, gen + 1):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1)[0]
            X[j] = q if rng.random() < alpha else relax(q, X, beta)
        if g > keep_from:
            phis.append(abs(2 * (X[:, 0] > 0).mean() - 1))
    return np.array(phis, np.float32)


def one(job):
    kind, K, ai, start, seed = job
    if kind == "A":
        return job, run(K, 3.0, float(A_ALPHAS[ai]), start, seed, A_GEN, A_GEN // 2)
    return job, run(K, 2.0, B_ALPHA, start, seed, B_GEN, 0)


def main():
    jobs = [("A", K, ai, st, s) for K in A_KS for ai in range(len(A_ALPHAS)) for st in ("sym", "cons")
            for s in range(A_SEEDS)]
    jobs += [("B", K, 0, "sym", s) for K in B_KS for s in range(B_SEEDS)]
    cost = lambda j: j[1] ** 2 * (A_GEN if j[0] == "A" else B_GEN)
    jobs.sort(key=lambda j: -cost(j))
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((kind, K, ai, st, s), ph) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            out[f"{kind}_K{K}_a{ai}_{st}_s{s}"] = ph
            if n % 10 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
                np.savez_compressed(OUT, a_Ks=A_KS, a_alphas=A_ALPHAS, b_Ks=B_KS, b_alpha=B_ALPHA, **out)
    np.savez_compressed(OUT, a_Ks=A_KS, a_alphas=A_ALPHAS, b_Ks=B_KS, b_alpha=B_ALPHA, **out)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
