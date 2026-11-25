# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 2 of the PRE plan: finite-size scaling of the consensus transition.

Continuous order parameter (instead of the discrete basin count):
    phi = |2 p - 1|,  p = fraction of stored points on the + side
phi ~ 0 in the symmetric (alive) phase and phi -> 1 in consensus.
Per run we keep the time series of phi over the second half and report
<phi>, <phi^2>, <phi^4>. From these:
    susceptibility  chi = K (<phi^2> - <phi>^2)
    Binder cumulant U  = 1 - <phi^4> / (3 <phi^2>^2)
Curves of U(alpha) for different K should cross at alpha_c, and a scaling
collapse in (alpha - alpha_c) K^{1/nu} gives the exponent nu.

Grid: beta in {1.5, 2, 3} (the range where step 1 showed a phase boundary)
x K in {20, 40, 80, 160, 320} x 11 alphas around each alpha_c x 8 seeds,
400 generations (step 1: alpha_c at beta <= 3 is stable from 300 on).
P_alive from a census every 10 generations is kept for comparison.
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

BETAS = [1.5, 2.0, 3.0]
ALPHA_GRID = {
    1.5: np.linspace(0.55, 0.90, 11),
    2.0: np.linspace(0.25, 0.50, 11),
    3.0: np.geomspace(0.004, 0.10, 11),
}
KS = [20, 40, 80, 160, 320]
SEEDS = range(8)
GEN, CENSUS_EVERY = 400, 10
MAX_WORKERS = 16
OUT = Path(__file__).parent / "data" / "sweep_fss.npz"


def one(args):
    K, beta, ai, seed = args
    alpha = float(ALPHA_GRID[beta][ai])
    rng = np.random.default_rng(seed * 15485863 + K * 97 + int(beta * 10) * 13 + ai)
    X = env(rng, K)
    phis, alive = [], []
    for g in range(1, GEN + 1):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1)[0]
            X[j] = q if rng.random() < alpha else relax(q, X, beta)
        if g > GEN // 2:
            p = (X[:, 0] > 0).mean()
            phis.append(abs(2 * p - 1))
            if g % CENSUS_EVERY == 0:
                alive.append(census(X, beta) >= 2)
    f = np.array(phis)
    return args, (f.mean(), (f ** 2).mean(), (f ** 4).mean(), float(np.mean(alive)))


def main():
    jobs = [(K, b, ai, s) for K in KS for b in BETAS for ai in range(11) for s in SEEDS]
    jobs.sort(key=lambda j: -j[0])
    shape = (len(KS), len(BETAS), 11, len(SEEDS))
    R = {m: np.full(shape, np.nan) for m in ("m1", "m2", "m4", "p_alive")}
    t0 = time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((K, b, ai, s), vals) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            i = (KS.index(K), BETAS.index(b), ai, s)
            for m, v in zip(("m1", "m2", "m4", "p_alive"), vals):
                R[m][i] = v
            if n % 88 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT, Ks=KS, betas=BETAS,
                        alphas=np.array([ALPHA_GRID[b] for b in BETAS]), **R)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
