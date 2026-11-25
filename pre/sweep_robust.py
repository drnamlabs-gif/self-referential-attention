# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 5: is the consensus transition a property of the two-mode toy, or of
self-rewriting attention memory in general?

E1 dimension : 2 antipodal modes embedded in d in {2, 8, 32, 128}; total noise
               norm fixed at the 2-D value (per-coordinate 0.1 sqrt(2/d)). beta = 2, K = 160.
E2 modes     : M in {2, 3, 4, 6} orthonormal modes in d = 8, beta = 5, K = 320.
               Mean-field (theory_modes.py) predicts the window moves to larger
               alpha roughly in proportion to M (fixed outside weight per mode).
E3 write rule: 2 antipodal modes in 2-D, beta = 2, K = 160, with
               fifo    : overwrite the oldest slot instead of a random one
               onestep : write a single attention step T(q), not the fixed point
Order parameter: each pattern is assigned to the mode with the largest overlap,
phi = (M p_max - 1) / (M - 1); consensus gives phi = 1 - alpha, diversity 0.
Stored per generation (second half of the run): phi and the median radial
position t of the patterns (0 = centroid, 1 = on a mode).
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from theory_modes import cons_edge, modes, sym_edge

NOISE = 0.1 * np.sqrt(2)
SEEDS, GEN = 4, 400
MAX_WORKERS = 20
OUT = Path(__file__).parent / "data" / "sweep_robust.npz"

E1_DS, E1_ALPHAS = [2, 8, 32, 128], np.round(np.linspace(0.25, 0.55, 11), 3)
E2_MS, E2_BETA = [2, 3, 4, 6], 5.0
E3_RULES = ["fifo", "onestep"]


def embed(E, d):
    out = np.zeros((len(E), d))
    out[:, :E.shape[1]] = E
    return out


def sample(rng, E, n, only=None):
    k = rng.integers(len(E), size=n) if only is None else np.full(n, only)
    return E[k] + NOISE / np.sqrt(E.shape[1]) * rng.standard_normal((n, E.shape[1]))


def retrieve(x, X, beta, onestep, it=300):
    for _ in range(1 if onestep else it):
        z = beta * X @ x
        z -= z.max()
        w = np.exp(z)
        xn = (w / w.sum()) @ X
        if np.abs(xn - x).max() < 1e-10:
            return xn
        x = xn
    return x


def run(E, K, beta, alpha, start, seed, rule="random", gen=GEN):
    M = len(E)
    rng = np.random.default_rng([seed, K, int(alpha * 1e5), M, E.shape[1], {"random": 0, "fifo": 1, "onestep": 2}[rule], start == "cons"])
    X = sample(rng, E, K, only=0 if start == "cons" else None)
    c = E.mean(0)
    u = E[0] - c
    s2 = u @ u
    ptr, out = 0, []
    for g in range(1, gen + 1):
        for _ in range(K):
            if rule == "fifo":
                j, ptr = ptr, (ptr + 1) % K
            else:
                j = rng.integers(K)
            q = sample(rng, E, 1)[0]
            X[j] = q if rng.random() < alpha else retrieve(q, X, beta, rule == "onestep")
        if g > gen // 2:
            lab = np.argmax(X @ E.T, 1)
            pm = np.bincount(lab, minlength=M).max() / K
            t = np.median(((X - c) * (E[lab] - c)).sum(1) / s2)
            out.append(((M * pm - 1) / (M - 1), t))
    return np.array(out, np.float32)


def e2_alphas(M):
    E = modes(M, "ortho")
    lo, hi = sym_edge(E2_BETA, E), cons_edge(E2_BETA, E)
    return np.round(np.geomspace(0.5 * lo, min(0.95, 1.4 * hi), 10), 4)


def one(job):
    kind, p, ai, start, seed = job
    if kind == "E1":
        E, K, beta, al, rule = embed(modes(2, "antipodal"), p), 160, 2.0, E1_ALPHAS[ai], "random"
    elif kind == "E2":
        E, K, beta, al, rule = embed(modes(p, "ortho"), 8), 320, E2_BETA, e2_alphas(p)[ai], "random"
    else:
        E, K, beta, al, rule = modes(2, "antipodal"), 160, 2.0, E1_ALPHAS[ai], p
    return job, run(E, K, beta, float(al), start, seed, rule)


def main():
    st = ("sym", "cons")
    jobs = [("E2", M, ai, s_, s) for M in E2_MS for ai in range(10) for s_ in st for s in range(SEEDS)]
    jobs += [("E1", d, ai, s_, s) for d in E1_DS for ai in range(len(E1_ALPHAS)) for s_ in st for s in range(SEEDS)]
    jobs += [("E3", r, ai, s_, s) for r in E3_RULES for ai in range(len(E1_ALPHAS)) for s_ in st for s in range(SEEDS)]
    e2a = {M: e2_alphas(M) for M in E2_MS}
    meta = dict(e1_ds=E1_DS, e1_alphas=E1_ALPHAS, e2_ms=E2_MS, e2_beta=E2_BETA,
                **{f"e2_alphas_M{M}": a for M, a in e2a.items()},
                **{f"e2_sym_M{M}": sym_edge(E2_BETA, modes(M, 'ortho')) for M in E2_MS},
                **{f"e2_cons_M{M}": cons_edge(E2_BETA, modes(M, 'ortho')) for M in E2_MS})
    out, t0 = {}, time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((k, p, ai, s_, s), r) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            out[f"{k}_{p}_a{ai}_{s_}_s{s}"] = r
            if n % 20 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
                np.savez_compressed(OUT, **meta, **out)
    np.savez_compressed(OUT, **meta, **out)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
