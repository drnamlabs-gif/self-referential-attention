# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The Watson question on a modern Hopfield network: beta x gamma.

gamma = 0 is the pure-memory case (neutral drift, see run_mhn_phase.py).
gamma > 0 gives better compositions wider effective basins by biasing the
retrieval softmax with a frozen macro-scale problem energy -- the MHN
analogue of M1's correlation between attractor quality and basin size.

The claim under test: at intermediate beta, retrieval RECOMBINES memories,
and biased retrieval selects among the recombinants, so the bank should
discover compositions strictly better than the best one it ever stored --
prediction, not recall. At very high beta retrieval only copies, so the bank
can do no better than its best original. Per-seed yardsticks: E_best_original
(best stored composition) and E_opt (exact optimum by enumeration).

Hard cap of 10 worker processes, one BLAS thread each.
"""
from __future__ import annotations

import os

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
BETAS = np.array([10.0, 31.6, 100.0, 316.0, 1000.0])
GAMMAS = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
DELTA = 0.1
SEEDS = np.arange(10)
STEPS = 4000
OUT = Path(__file__).parent / "data" / "mhn_selection.npz"

SERIES = ("E_best", "E_mean", "best_is_novel", "n_distinct", "cum_novel",
          "denoise_unseen", "retention")


def one_cell(args: tuple[int, int, int]) -> tuple[tuple[int, int, int], dict]:
    ib, ig, is_ = args
    cfg = MHNConfig(beta=float(BETAS[ib]), gamma=float(GAMMAS[ig]),
                    delta=DELTA, steps=STEPS, seed=int(SEEDS[is_]))
    out = self_train(cfg)
    slim = {k: out[k] for k in SERIES}
    slim["step"] = out["step"]
    slim["fixation_step"] = out["fixation_step"]
    slim["E_opt"] = out["E_opt"]
    slim["E_best_original"] = out["E_best_original"]
    return args, slim


def main() -> None:
    jobs = [(ib, ig, is_) for ib in range(len(BETAS))
            for ig in range(len(GAMMAS)) for is_ in range(len(SEEDS))]
    print(f"{len(jobs)} runs on {MAX_WORKERS} workers "
          f"(beta {BETAS[0]:g}..{BETAS[-1]:g}, gamma {GAMMAS[0]}..{GAMMAS[-1]}, "
          f"delta={DELTA}, {len(SEEDS)} seeds)")

    n_rec = STEPS // MHNConfig.record_every + 1
    shape = (len(BETAS), len(GAMMAS), len(SEEDS), n_rec)
    data = {k: np.full(shape, np.nan) for k in SERIES}
    fixation = np.full(shape[:3], -1, dtype=int)
    e_opt = np.full(shape[:3], np.nan)
    e_orig = np.full(shape[:3], np.nan)
    steps_axis = None

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for (ib, ig, is_), slim in ex.map(one_cell, jobs, chunksize=2):
            for k in SERIES:
                data[k][ib, ig, is_] = slim[k]
            fixation[ib, ig, is_] = slim["fixation_step"]
            e_opt[ib, ig, is_] = slim["E_opt"]
            e_orig[ib, ig, is_] = slim["E_best_original"]
            steps_axis = slim["step"]

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, betas=BETAS, gammas=GAMMAS, delta=DELTA,
                        seeds=SEEDS, steps=steps_axis, fixation=fixation,
                        E_opt=e_opt, E_best_original=e_orig, **data)
    print(f"saved {OUT}  ({time.time()-t0:.0f}s total)")

    # headline table: did the bank beat its best original composition?
    E_min = np.nanmin(data["E_best"], axis=3)          # best ever held in bank
    beat = (E_min < e_orig - 1e-9).mean(axis=2)        # fraction of seeds
    gap0 = e_orig - e_opt                               # how far originals were
    gap1 = E_min - e_opt
    print("\nfraction of seeds where the bank DISCOVERED a better composition")
    print("        " + " ".join(f"g={g:<5g}" for g in GAMMAS))
    for i, b in enumerate(BETAS):
        print(f"b={b:6.0f} " + " ".join(f"{beat[i, j]:6.1f}" for j in range(len(GAMMAS))))
    print("\nmean gap to the exact optimum (originals -> after self-training)")
    for i, b in enumerate(BETAS):
        row = " ".join(f"{gap0[i, j].mean():4.2f}>{np.nanmean(gap1[i, j]):4.2f}"
                       for j in range(len(GAMMAS)))
        print(f"b={b:6.0f} {row}")


if __name__ == "__main__":
    main()
