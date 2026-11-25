# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 3b: prediction test of the nucleation picture at beta = 4.

A consensus bank loses its consensus once enough outside points sit on the
minority side to hold their own basin. Mean field gives the critical
fraction w* = alpha_cons / 2, i.e. a critical nucleus of n* = K / K_x(beta)
points, K_x(4) ~ 74. For K < K_x a SINGLE outside point is a nucleus, so
small banks escape consensus at very small alpha and alpha_c is far below
the mean-field window (0.007 - 0.027). Prediction: alpha_c(beta = 4) must
RISE with K and enter that window once n* >> 1.
K in {160, 320, 640, 1280} (n* = 2 - 17), two starts, 800 generations.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sweep_large import run

ALPHAS = np.round(np.geomspace(0.0005, 0.05, 10), 5)
KS = [160, 320, 640, 1280]
SEEDS, GEN = 4, 800
MAX_WORKERS = 12
OUT = Path(__file__).parent / "data" / "sweep_beta4.npz"


def one(job):
    K, ai, st, s = job
    return job, run(K, 4.0, float(ALPHAS[ai]), st, s, GEN, GEN // 2)


def main():
    jobs = sorted([(K, ai, st, s) for K in KS for ai in range(len(ALPHAS)) for st in ("sym", "cons")
                   for s in range(SEEDS)], key=lambda j: -j[0])
    out, t0 = {}, time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((K, ai, st, s), ph) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            out[f"K{K}_a{ai}_{st}_s{s}"] = ph
            if n % 10 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
                np.savez_compressed(OUT, Ks=KS, alphas=ALPHAS, **out)
    np.savez_compressed(OUT, Ks=KS, alphas=ALPHAS, **out)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
