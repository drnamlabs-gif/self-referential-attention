# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Basin census on real BERT heads: how many attractors does the retrieval
map of each self-fed head still have?

Re-simulates the self mode of run_llm_regime.py (seed 0) for each of the 12
pretrained heads x tau in {0.5, 1, 2, 4}; every 200 rounds the bank is frozen
and the head's retrieval map  xi <- softmax(tau * xi W_Q (X W_K)^T / 8) X  is
iterated to convergence from 64 fixed real probe states.  Distinct limit
points (cosine < 0.999 apart) = surviving basins.  Single torch job, GPU.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
DATA = HERE / "data" / "llm_bank"
OUT = HERE / "data" / "llm_basin_census.npz"
TAUS = [0.5, 1.0, 2.0, 4.0]
K, ROUNDS, CENSUS_EVERY, N_PROBE = 100, 4000, 200, 64
ITERS, TOL, DEDUP = 200, 1e-5, 0.999
TAGS = ["avg", "med", "sel"]


def census(B, WqT, bq, WkT, bk, probes, taus) -> list[int]:
    Kb = B @ WkT + bk                                    # (C, K, 64)
    X = probes[None].repeat(len(B), 1, 1).clone()        # (C, P, 768)
    for _ in range(ITERS):
        z = (X @ WqT + bq) @ Kb.transpose(1, 2) / 8.0    # (C, P, K)
        w = torch.softmax(z * taus[:, None, None], dim=-1)
        Xn = w @ B
        if (Xn - X).abs().max() < TOL:
            X = Xn
            break
        X = Xn
    out = []
    for c in range(len(B)):
        A = torch.nn.functional.normalize(X[c], dim=-1)
        sim = A @ A.T > DEDUP
        first = sim.float().argmax(dim=0)
        out.append(int((first == torch.arange(len(A),
                                              device=A.device)).sum()))
    return out


def main() -> None:
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool = np.load(DATA / "pool.npz")
    W = dict(np.load(DATA / "heads.npz").items())
    layers = W["layers"].tolist()
    head_sel = W["head_sel"]

    n_c = ROUNDS // CENSUS_EVERY + 1
    counts = np.zeros((len(layers), 3, len(TAUS), n_c), dtype=np.int32)
    t0 = time.time()
    for il, layer in enumerate(layers):
        P = torch.tensor(pool[f"pool_{layer}"], dtype=torch.float32,
                         device=dev)
        T = len(P)
        probes = P[torch.from_numpy(
            np.random.default_rng(123).choice(T, N_PROBE, replace=False))]
        for ig, hidx in enumerate(head_sel[il]):
            r = slice(64 * int(hidx), 64 * int(hidx) + 64)
            WqT = torch.tensor(W[f"Wq_{layer}"][r].T, device=dev)
            bq = torch.tensor(W[f"bq_{layer}"][r], device=dev)
            WkT = torch.tensor(W[f"Wk_{layer}"][r].T, device=dev)
            bk = torch.tensor(W[f"bk_{layer}"][r], device=dev)
            taus = torch.tensor(TAUS, device=dev)

            B = torch.stack([P[torch.from_numpy(
                np.random.default_rng(1000).choice(T, K, replace=False))]
                for _ in TAUS])                          # (4, K, 768)
            rng = np.random.default_rng(layer * 100 + int(hidx))
            ci = 0
            for t in range(ROUNDS + 1):
                if t % CENSUS_EVERY == 0:
                    counts[il, ig, :, ci] = census(B, WqT, bq, WkT, bk,
                                                   probes, taus)
                    ci += 1
                if t == ROUNDS:
                    break
                q = P[torch.from_numpy(rng.integers(T, size=len(TAUS)))]
                Kb = B @ WkT + bk
                z = (Kb @ (q @ WqT + bq)[:, :, None]).squeeze(2) / 8.0
                w = torch.softmax(z * taus[:, None], dim=-1)
                new = (w[:, None, :] @ B).squeeze(1)
                j = torch.from_numpy(rng.integers(K, size=len(TAUS))).to(dev)
                B[torch.arange(len(TAUS), device=dev), j] = new
            print(f"layer {layer} head {int(hidx)} ({TAGS[ig]}): "
                  f"tau=1 {counts[il, ig, 1, 0]} -> "
                  f"{counts[il, ig, 1, -1]} attractors "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(OUT, counts=counts, layers=np.array(layers),
                        tags=np.array(TAGS), taus=np.array(TAUS),
                        census_rounds=np.arange(n_c) * CENSUS_EVERY,
                        k90_init=np.load(HERE / "data" / "llm_regime.npz")
                        ["k90_init"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
