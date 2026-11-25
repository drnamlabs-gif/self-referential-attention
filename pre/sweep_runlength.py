# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 1 of the PRE plan: at high beta, is alpha_c a phase boundary or a clock?

If alpha_c(beta) is a true phase boundary, P_alive(alpha) measured over
windows ending at 300, 1000 and 3000 generations should agree. If it is only
a time scale (the paper's "beta sets the clock"), the curves should keep
moving right as the window moves later, and at alpha = 0 every bank should
eventually collapse.

One long run per (beta, alpha, seed), K = 40, 3000 generations. The census
is taken every 5 generations; P_alive(T) is the mean over the window
[T/2, T]. beta = 2 is the control: there alpha_c ~ 0.37 was K-independent
and should not move in time either.
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

BETAS = [2.0, 3.0, 4.0, 6.0]
ALPHAS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = range(8)
K, GEN, EVERY = 40, 3000, 5
MAX_WORKERS = 16
OUT = Path(__file__).parent / "data" / "sweep_runlength.npz"


def one(args):
    beta, alpha, seed = args
    rng = np.random.default_rng(seed * 7907 + int(beta * 100) + int(alpha * 1e5))
    X = env(rng, K)
    alive, minority, size = [], [], []
    for g in range(1, GEN + 1):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1)[0]
            X[j] = q if rng.random() < alpha else relax(q, X, beta)
        if g % EVERY == 0:
            alive.append(census(X, beta) >= 2)
            pos = (X[:, 0] > 0).mean()
            minority.append(min(pos, 1 - pos))
            size.append(np.abs(X[:, 0]).mean())
    return args, np.array(alive), np.array(minority, np.float32), np.array(size, np.float32)


def main():
    jobs = [(b, a, s) for b in BETAS for a in ALPHAS for s in SEEDS]
    n_t = GEN // EVERY
    shape = (len(BETAS), len(ALPHAS), len(SEEDS), n_t)
    A = np.zeros(shape, bool)
    M = np.zeros(shape, np.float32)
    S = np.zeros(shape, np.float32)
    t0 = time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((b, a, s), al, mi, sz) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            i = (BETAS.index(b), ALPHAS.index(a), s)
            A[i], M[i], S[i] = al, mi, sz
            if n % 32 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT, betas=BETAS, alphas=ALPHAS, K=K, every=EVERY,
                        gens=np.arange(1, n_t + 1) * EVERY, alive=A, minority=M, size=S)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
