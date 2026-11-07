# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 3 of the pre-registered protocol (proposal_dam.tex): the MNIST grid.

Two retrieval rules, run as parallel arms:
  power    w ~ max(cos,0)^(n-1), n in {2,4,10,20}   -- Krotov DAM, the
           pre-registered arm; n is the feature->prototype dial.
  softmax  w ~ softmax(beta * cos), beta in {4,8,16,32} -- the log-sum-exp
           MHN of Ramsauer et al., i.e. VERBATIM the attention operation:
           one attention head whose key-value bank is overwritten with its
           own outputs.

Modes, run simultaneously as the protocol demands:
  self     replace a random memory with the network's own attractor
  real     C1: replace with a fresh real image  (must NOT drift)
  random   C3: replace with a random vector     (churn without self-reference)
  anchor   C4: self + pull toward the original bank (alpha = 0.99, 0.90)
  biased   EXPLORATORY, not pre-registered: self + basin bias -- memories
           closer to the frozen original bank are easier to retrieve
           (the stage-1 gamma ingredient, energy-based, no classifier).

Metrics per eval: effective rank, denoising acc/RMSE (external 1-NN judge,
never the model's own energy), distance to real data, mean PR (position on
the feature->prototype axis). Showcase runs also store bank snapshots for
the memory-evolution GIF. 10 workers, 1 BLAS thread each.
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
from mnist_c0 import load, eff_rank, N

MAX_WORKERS = 10
RULES = {"power": [2.0, 4.0, 10.0, 20.0], "softmax": [4.0, 8.0, 16.0, 32.0]}
MODES = [("self", 0.0), ("real", 0.0), ("random", 0.0),
         ("anchor", 0.99), ("anchor", 0.90), ("biased", 4.0)]
SEEDS = range(6)
K, ROUNDS, EVAL_EVERY, N_EVAL = 100, 6000, 150, 300
SHOWCASE = {("power", 2.0, "self", 0), ("power", 20.0, "self", 0),
            ("softmax", 4.0, "self", 0), ("softmax", 32.0, "self", 0),
            ("softmax", 32.0, "biased", 0)}
OUT = Path(__file__).parent / "data" / "mnist_grid.npz"
SNAP_DIR = Path(__file__).parent / "data" / "mnist_snaps"
METRICS = ("rank", "acc", "rmse", "dist_data", "pr")


def masked_weights(q: np.ndarray, Xo: np.ndarray, rule: str, sharp: float,
                   bias: np.ndarray | None) -> np.ndarray:
    qn = np.linalg.norm(q) + 1e-12
    xn = np.linalg.norm(Xo, axis=1) + 1e-12
    m = (Xo @ q) / (xn * qn)
    if rule == "power":
        w = np.maximum(m, 0.0) ** (sharp - 1.0)
        if bias is not None:
            w = w * np.exp(bias)
    else:
        z = sharp * m
        if bias is not None:
            z = z + bias
        z -= z.max()
        w = np.exp(z)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


def retrieve(Vq, mask, Xi, mu, rule, sharp, bias=None):
    Xc = Xi - mu
    out = np.empty_like(Vq)
    for b in range(Vq.shape[0]):
        ob = mask[b]
        w = masked_weights(Vq[b, ob] - mu[ob], Xc[:, ob], rule, sharp, bias)
        out[b] = mu + w @ Xc
    return out


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared distances via matmul -- no giant broadcasts."""
    return ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
            - 2.0 * (A @ B.T))


def fast_score(R, Vtrue, mask, ytrue, xr, yr):
    hid = ~mask
    err = np.array([np.sqrt(((R[i, hid[i]] - Vtrue[i, hid[i]]) ** 2).mean())
                    for i in range(len(R))])
    acc = (yr[sqdist(R, xr).argmin(1)] == ytrue).mean()
    return float(err.mean()), float(acc)


def pr_mean(Xi: np.ndarray, Rn: np.ndarray) -> float:
    """Mean participation ratio of the memories' correlation profiles."""
    Xc = Xi - Xi.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    C = np.maximum(Xn @ Rn.T, 0.0)
    return float((C.sum(1) ** 2 / ((C ** 2).sum(1) + 1e-12)).mean())


def one_run(args):
    rule, sharp, mode, param, seed = args
    xtr, ytr, xte, yte = load()
    mu = xtr.mean(0)
    rng = np.random.default_rng(seed)

    idx = np.concatenate([rng.choice(np.flatnonzero(ytr == c), K // 10,
                                     replace=False) for c in range(10)])
    Xi0 = xtr[idx].copy()
    Xi = Xi0.copy()
    Xi0c = Xi0 - mu
    Xi0n = Xi0c / (np.linalg.norm(Xi0c, axis=1, keepdims=True) + 1e-12)

    # frozen yardsticks, identical across every condition
    ev = np.random.default_rng(999).choice(len(xte), N_EVAL, replace=False)
    Vt, yt = xte[ev], yte[ev]
    mk = np.random.default_rng(998).random(Vt.shape) > 0.5
    Vq = np.where(mk, Vt, 0.0)
    rr = np.random.default_rng(7).choice(len(xtr), 3000, replace=False)
    xr, yr = xtr[rr], ytr[rr]
    Rc = xr - xr.mean(axis=1, keepdims=True)
    Rn = Rc / (np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-12)

    def data_bias() -> np.ndarray | None:
        if mode != "biased":
            return None
        Xc = Xi - mu
        Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
        s0 = (Xn @ Xi0n.T).max(axis=1)        # data-likeness of each memory
        return param * s0

    rows = {m: [] for m in METRICS}
    snaps = []
    for t in range(ROUNDS + 1):
        if t % EVAL_EVERY == 0:
            R = retrieve(Vq, mk, Xi, mu, rule, sharp, data_bias())
            e, a = fast_score(R, Vt, mk, yt, xr, yr)
            d = np.maximum(sqdist(Xi, xr), 0.0).min(1)
            rows["rank"].append(eff_rank(Xi - mu))
            rows["acc"].append(a)
            rows["rmse"].append(e)
            rows["dist_data"].append(float(np.sqrt(d).mean()))
            rows["pr"].append(pr_mean(Xi, Rn))
            if (rule, sharp, mode, seed) in SHOWCASE:
                snaps.append(Xi.astype(np.float32).copy())
        if t == ROUNDS:
            break

        j = rng.integers(K)
        if mode == "real":
            new = xtr[rng.integers(len(xtr))]
        elif mode == "random":
            new = rng.random(N)
        else:  # self, anchor, biased
            i = rng.integers(len(xtr))
            m1 = rng.random(N) > 0.5
            q = np.where(m1, xtr[i], 0.0)[None]
            new = retrieve(q, m1[None], Xi, mu, rule, sharp, data_bias())[0]
        Xi[j] = new
        if mode == "anchor":
            Xi = (1 - param) * Xi0 + param * Xi

    if snaps:
        SNAP_DIR.mkdir(exist_ok=True)
        np.savez_compressed(
            SNAP_DIR / f"{rule}_{sharp:g}_{mode}_s{seed}.npz",
            snaps=np.array(snaps), labels=ytr[idx])
    return args, {m: np.array(v) for m, v in rows.items()}


def main() -> None:
    jobs = [(rule, sharp, mode, param, seed)
            for rule, sharps in RULES.items() for sharp in sharps
            for mode, param in MODES for seed in SEEDS]
    print(f"{len(jobs)} runs on {MAX_WORKERS} workers, "
          f"{ROUNDS} rounds each, K={K}")

    n_ev = ROUNDS // EVAL_EVERY + 1
    rules = list(RULES)
    shape = (len(rules), 4, len(MODES), len(SEEDS), n_ev)
    data = {m: np.full(shape, np.nan) for m in METRICS}

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for (rule, sharp, mode, param, seed), rows in ex.map(
                one_run, jobs, chunksize=1):
            ir = rules.index(rule)
            isp = RULES[rule].index(sharp)
            im = MODES.index((mode, param))
            for m in METRICS:
                data[m][ir, isp, im, seed] = rows[m]
            done += 1
            if done % 48 == 0:
                print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)",
                      flush=True)

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT, rules=np.array(rules), sharps=np.array(list(RULES.values())),
        modes=np.array([f"{m}:{p:g}" for m, p in MODES]),
        seeds=np.array(list(SEEDS)),
        eval_rounds=np.arange(n_ev) * EVAL_EVERY, **data)
    print(f"saved {OUT}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
