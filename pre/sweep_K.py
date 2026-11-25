# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Is the alpha transition sharp? Finite-size scaling of the self-rewriting bank.

Same system as notebook 01, case 3: environment = two noisy clusters at
(+-1, 0), spread 0.1; bank of K points in 2-D; each write replaces a random
slot with a fresh environment sample (prob alpha) or with the attractor that
a fresh environment query relaxes to (dot-product attention, sharpness beta).

Grid: K in {20, 40, 80} x beta in {2, 4} x alpha (14 values) x start
{dead, alive} x 10 seeds. Time is counted in generations (K writes), 300 per
run. In the last 150 generations a basin census (30 probes) is taken once per
generation; P_alive is the fraction of censuses with K* >= 2.

A sharp transition predicts that the P_alive(alpha) curve steepens as K grows.
A crossover predicts curves that do not steepen.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

KS = [20, 40, 80]
BETAS = [2.0, 4.0]
ALPHAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STARTS = ["dead", "alive"]
SEEDS = range(10)
GEN, BURN = 300, 150
SPREAD, PROBE_SPREAD = 0.1, 0.3
MAX_WORKERS = 16
OUT = Path(__file__).parent / "data" / "sweep_K.npz"


def env(rng, n, s=SPREAD):
    sg = rng.choice([-1, 1], n)
    return np.c_[sg * 1.0, np.zeros(n)] + s * rng.standard_normal((n, 2))


PROBES = env(np.random.default_rng(9), 30, s=PROBE_SPREAD)


def relax(x, X, beta, it=300):
    for _ in range(it):
        z = beta * X @ x
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        xn = w @ X
        if np.abs(xn - x).max() < 1e-10:
            return xn
        x = xn
    return x


def relax_batch(P, X, beta, it=300):
    for _ in range(it):
        z = beta * P @ X.T
        z -= z.max(1, keepdims=True)
        w = np.exp(z)
        w /= w.sum(1, keepdims=True)
        Pn = w @ X
        if np.abs(Pn - P).max() < 1e-10:
            return Pn
        P = Pn
    return P


def census(X, beta):
    F = relax_batch(PROBES.copy(), X, beta)
    reps = []
    for x in F:
        if not any(np.linalg.norm(x - r) < 1e-3 for r in reps):
            reps.append(x)
    return len(reps)


def one(args):
    K, beta, alpha, start, seed = args
    rng = np.random.default_rng(seed * 7919 + K)
    if start == "dead":
        X = 0.01 * rng.standard_normal((K, 2))
    else:
        X = env(rng, K)
    alive, kstar, minority, size = [], [], [], []
    for g in range(GEN):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1)[0]
            X[j] = q if rng.random() < alpha else relax(q, X, beta)
        if g >= BURN:
            k = census(X, beta)
            kstar.append(k)
            alive.append(k >= 2)
            pos = (X[:, 0] > 0).mean()
            minority.append(min(pos, 1 - pos))
            size.append(np.abs(X[:, 0]).mean())
    return args, float(np.mean(alive)), float(np.mean(kstar)), float(np.mean(minority)), float(np.mean(size))


def main():
    jobs = [(K, b, a, s, sd) for K in KS for b in BETAS for a in ALPHAS
            for s in STARTS for sd in SEEDS]
    jobs.sort(key=lambda j: -j[0])          # big banks first
    shape = (len(KS), len(BETAS), len(ALPHAS), len(STARTS), len(SEEDS))
    res = {m: np.full(shape, np.nan) for m in ("p_alive", "kstar", "minority", "size")}
    t0 = time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((K, b, a, s, sd), pa, ks, mi, sz) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            i = (KS.index(K), BETAS.index(b), ALPHAS.index(a), STARTS.index(s), sd)
            res["p_alive"][i], res["kstar"][i], res["minority"][i], res["size"][i] = pa, ks, mi, sz
            if n % 84 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, Ks=KS, betas=BETAS, alphas=ALPHAS, starts=STARTS, **res)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
