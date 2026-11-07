# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Phase diagram of self-modeling on a modern Hopfield network: beta x delta.

Runs the self_train loop over a grid of (retrieval sharpness, writeback rate)
with several seeds per cell, in parallel. Saves every metric time series to
data/mhn_phase.npz; figures are built separately by make_mhn_figures.py.

Hard cap of 10 worker processes, one BLAS thread each, so the machine stays
usable for other jobs.
"""
from __future__ import annotations

import os

# one BLAS thread per worker -- must be set before numpy loads
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from selfmodel.mhn import MHNConfig, self_train

MAX_WORKERS = 10
BETAS = np.logspace(0.0, 3.0, 13)          # 1 .. 1000
DELTAS = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
SEEDS = np.arange(6)
STEPS = 4000
OUT = Path(__file__).parent / "data" / "mhn_phase.npz"

SERIES = ("n_distinct", "diversity", "retention", "novel_in_bank",
          "cum_novel", "denoise_unseen", "denoise_stored", "coherence")


def one_cell(args: tuple[int, int, int]) -> tuple[tuple[int, int, int], dict]:
    ib, id_, is_ = args
    cfg = MHNConfig(beta=float(BETAS[ib]), delta=float(DELTAS[id_]),
                    steps=STEPS, seed=int(SEEDS[is_]))
    out = self_train(cfg)
    slim = {k: out[k] for k in SERIES}
    slim["step"] = out["step"]
    slim["fixation_step"] = out["fixation_step"]
    return args, slim


def main() -> None:
    jobs = [(ib, id_, is_) for ib in range(len(BETAS))
            for id_ in range(len(DELTAS)) for is_ in range(len(SEEDS))]
    print(f"{len(jobs)} runs on {MAX_WORKERS} workers "
          f"(beta {BETAS[0]:g}..{BETAS[-1]:g}, delta {DELTAS[0]}..{DELTAS[-1]}, "
          f"{len(SEEDS)} seeds, {STEPS} steps)")

    n_rec = STEPS // MHNConfig.record_every + 1
    shape = (len(BETAS), len(DELTAS), len(SEEDS), n_rec)
    data = {k: np.full(shape, np.nan) for k in SERIES}
    fixation = np.full(shape[:3], -1, dtype=int)
    steps_axis = None

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for (ib, id_, is_), slim in ex.map(one_cell, jobs, chunksize=4):
            for k in SERIES:
                data[k][ib, id_, is_] = slim[k]
            fixation[ib, id_, is_] = slim["fixation_step"]
            steps_axis = slim["step"]
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)")

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, betas=BETAS, deltas=DELTAS, seeds=SEEDS,
                        steps=steps_axis, fixation=fixation, **data)
    print(f"saved {OUT}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
