# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 5b: one-step writes never reach consensus at alpha >= 0.25 (sweep_robust E3).
What do they do at small alpha? 2 antipodal modes, 2-D, K = 160, sym start.
Stored per generation: phi and median radial position t."""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sweep_robust import modes, run

BETAS = [2.0, 4.0]
ALPHAS = np.r_[0.0, np.round(np.geomspace(0.001, 0.25, 9), 4)]
RULES = ["onestep", "random"]
OUT = Path(__file__).parent / "data" / "sweep_onestep.npz"


def one(job):
    b, ai, rule, s = job
    return job, run(modes(2, "antipodal"), 160, b, float(ALPHAS[ai]), "sym", s, rule, gen=600)


if __name__ == "__main__":
    jobs = [(b, ai, r, s) for b in BETAS for ai in range(len(ALPHAS)) for r in RULES for s in range(2)]
    with ProcessPoolExecutor(6) as ex:
        out = {f"b{b:g}_a{ai}_{r}_s{s}": v for (b, ai, r, s), v in ex.map(one, jobs)}
    np.savez_compressed(OUT, betas=BETAS, alphas=ALPHAS, **out)
    for b in BETAS:
        for r in RULES:
            print(f"beta={b} {r}")
            for ai, a in enumerate(ALPHAS):
                v = np.concatenate([out[f"b{b:g}_a{ai}_{r}_s{s}"] for s in range(2)])
                print(f"   a={a:.4f}  phi={v[:,0].mean():.2f}  t={v[:,1].mean():.2f}")
