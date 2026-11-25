# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Test 1: does beta interpolate between the voter class and the majority class?

Hypothesis (stated before running, QUESTION.md log 2026-11-24):
  high beta, attention COPIES one stored pattern  -> noisy voter model,
             alpha_c ~ 1/K (alpha_c * K constant, vanishes for large banks)
  low beta,  attention AVERAGES                    -> majority rule / Ising,
             alpha_c finite and independent of K

Same system as sweep_K.py. Grid: beta in {1.5, 2, 3, 4, 6} x K in
{20, 40, 80, 160} x 16 alphas (log-spaced at small alpha) x 8 seeds, all
started alive (no hysteresis was found). Run length 4K generations or 300,
whichever is longer, so the voter consensus time (~K generations) fits; the
census is taken once per generation over the second half.

alpha_c(K, beta) = the alpha at which P_alive crosses 0.5 (linear
interpolation in log alpha).
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sweep_K import env, relax, census

BETAS = [1.5, 2.0, 3.0, 4.0, 6.0]
KS = [20, 40, 80, 160]
ALPHAS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3,
          0.4, 0.5, 0.6, 0.7, 0.8]
SEEDS = range(8)
MAX_WORKERS = 16
OUT = Path(__file__).parent / "data" / "sweep_alphac.npz"


def gens(K):
    return max(300, 4 * K)


def one(args):
    K, beta, alpha, seed = args
    rng = np.random.default_rng(seed * 104729 + K * 31 + int(beta * 10))
    X = env(rng, K)
    G = gens(K)
    alive, minority = [], []
    for g in range(G):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1)[0]
            X[j] = q if rng.random() < alpha else relax(q, X, beta)
        if g >= G // 2:
            alive.append(census(X, beta) >= 2)
            pos = (X[:, 0] > 0).mean()
            minority.append(min(pos, 1 - pos))
    return args, float(np.mean(alive)), float(np.mean(minority))


def main():
    jobs = [(K, b, a, s) for K in KS for b in BETAS for a in ALPHAS for s in SEEDS]
    jobs.sort(key=lambda j: -j[0])
    shape = (len(KS), len(BETAS), len(ALPHAS), len(SEEDS))
    P, M = np.full(shape, np.nan), np.full(shape, np.nan)
    t0 = time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((K, b, a, s), pa, mi) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            i = (KS.index(K), BETAS.index(b), ALPHAS.index(a), s)
            P[i], M[i] = pa, mi
            if n % 128 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT, Ks=KS, betas=BETAS, alphas=ALPHAS, p_alive=P, minority=M)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
