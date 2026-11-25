# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 4: the alpha transition on real images (MNIST).

Bank of K = 100 digits. Retrieval: x <- mu + softmax(beta cos(x - mu, X - mu)) (X - mu),
iterated to a fixed point. Each write replaces a random slot with a fresh real
digit (prob alpha) or with the fixed point reached from a fresh real digit.
Starts: 'div' = 10 digits of each class, 'one' = 100 digits of class 1 (the
analogue of the consensus start). Observables, once per generation (K writes):
  neff : exp(entropy) of the class composition (labels by nearest real image)
  pmax : largest class share
  cmax : median cosine of a pattern to its nearest real image (1 = digit-like,
         low = blurred blend)
  rank : effective rank of the centred bank
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from mnist_c0 import eff_rank, load

K, GEN = 100, 300
BETAS = [4.0, 8.0, 16.0, 32.0, 64.0]
ALPHAS = np.round(np.geomspace(0.003, 0.5, 12), 4)
STARTS, SEEDS = ("div", "one"), 4
MAX_WORKERS = 18
OUT = Path(__file__).parent / "data" / "mnist_alpha.npz"

_D = {}


def _data():
    if not _D:
        xtr, ytr, _, _ = load()
        mu = xtr.mean(0)
        rr = np.random.default_rng(7).choice(len(xtr), 3000, replace=False)
        R = xtr[rr] - mu
        _D.update(xtr=xtr, ytr=ytr, mu=mu, R=R / np.linalg.norm(R, axis=1, keepdims=True), yr=ytr[rr])
    return _D


def fixed_point(q, Xc, mu, beta, it=200):
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    x = q - mu
    for _ in range(it):
        z = beta * (Xn @ x) / (np.linalg.norm(x) + 1e-12)
        z -= z.max()
        w = np.exp(z)
        xn = (w / w.sum()) @ Xc
        if np.abs(xn - x).max() < 1e-7:
            return xn
        x = xn
    return x


def measure(Xc, D):
    nrm = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    C = (Xc / nrm) @ D["R"].T
    lab = D["yr"][C.argmax(1)]
    p = np.bincount(lab, minlength=10) / len(lab)
    nz = p[p > 0]
    return np.exp(-(nz * np.log(nz)).sum()), p.max(), np.median(C.max(1)), eff_rank(Xc)


def run(beta, alpha, start, seed):
    D = _data()
    xtr, ytr, mu = D["xtr"], D["ytr"], D["mu"]
    rng = np.random.default_rng([seed, int(beta), int(alpha * 1e5), start == "one"])
    if start == "div":
        idx = np.concatenate([rng.choice(np.flatnonzero(ytr == c), K // 10, replace=False) for c in range(10)])
    else:
        idx = rng.choice(np.flatnonzero(ytr == 1), K, replace=False)
    Xc = xtr[idx] - mu
    out = [measure(Xc, D)]
    for g in range(GEN):
        for _ in range(K):
            j = rng.integers(K)
            q = xtr[rng.integers(len(xtr))]
            Xc[j] = q - mu if rng.random() < alpha else fixed_point(q, Xc, mu, beta)
        out.append(measure(Xc, D))
    return np.array(out, np.float32), Xc.astype(np.float32)


def one(job):
    b, ai, st, s = job
    return job, run(b, float(ALPHAS[ai]), st, s)


def main():
    jobs = [(b, ai, st, s) for b in BETAS for ai in range(len(ALPHAS)) for st in STARTS for s in range(SEEDS)]
    out, t0 = {}, time.time()
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        for n, ((b, ai, st, s), (r, Xf)) in enumerate(ex.map(one, jobs, chunksize=1), 1):
            key = f"b{b:g}_a{ai}_{st}_s{s}"
            out[key] = r
            if s == 0:
                out["bank_" + key] = Xf
            if n % 12 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
                np.savez_compressed(OUT, betas=BETAS, alphas=ALPHAS, **out)
    np.savez_compressed(OUT, betas=BETAS, alphas=ALPHAS, **out)
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
