# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Regime migration of real pretrained BERT attention heads under
self-reference.

Claim under test, stated in the field's own vocabulary (Ramsauer et al.,
"Hopfield Networks is All You Need"): an attention head is a modern Hopfield
network whose stored patterns are its KV bank.  If the bank is gradually
overwritten with the head's own retrievals (one attention step = one Hopfield
update, identity values so the retrieval lives in pattern space), the head
should migrate along Ramsauer's regime axis toward global averaging, with its
sharpness setting the SPEED of migration but not the destination -- the
prediction exported from the MNIST grid.

Setup per head (12 real heads, 4 layers x {most-averaging, median,
most-selective} by initial k90):
  bank   K=100 real token hidden states at that head's layer
  step   q ~ fresh real hidden state; w = softmax(tau * q W_Q (X W_K)^T / 8);
         self-modes overwrite a random slot with w @ X
  dial   tau in {0.5, 1, 2, 4} multiplies the pretrained logits
  modes  self | real (C1, fresh real state -- must stay put) | random (C3)
         | bias (self + gamma * data-likeness on the logits, stage-1 gamma)
  judge  frozen probe queries at the head's NATIVE scale (tau=1):
         k90 = #patterns covering 90% attention mass (their metastable size),
         exp(entropy), effective rank, distance to pool mean, NN distance to
         real states, #distinct slots (fixation vs blur).

All 64 conditions of a head run as one batched tensor; heads sequential.
Single torch job on the GPU, 2 CPU threads (house rule).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
DATA = HERE / "data" / "llm_bank"
OUT = HERE / "data" / "llm_regime.npz"
SNAP_OUT = HERE / "data" / "llm_regime_snaps.npz"

TAUS = [0.5, 1.0, 2.0, 4.0]
MODES = ["self", "real", "random", "bias"]
GAMMA = 4.0
SEEDS = 4
K, ROUNDS, EVAL, N_PROBE, NN_SUB = 100, 4000, 100, 256, 4096
METRICS = ("k90", "effH", "rank", "dist_mean", "nn_dist", "n_distinct")
TAGS = ["avg", "med", "sel"]
# (layer, tag) panels whose bank we film in PCA coordinates
SHOWCASE = {(1, "avg", 1.0, "self", 0), (1, "avg", 1.0, "real", 0),
            (11, "sel", 1.0, "self", 0), (11, "sel", 1.0, "real", 0),
            (1, "avg", 4.0, "self", 0), (11, "sel", 4.0, "self", 0)}


def k90_t(w: torch.Tensor) -> torch.Tensor:
    s, _ = torch.sort(w, dim=-1, descending=True)
    return (s.cumsum(-1) < 0.90).sum(-1).float() + 1.0


def run_head(layer: int, hidx: int, tag: str, P: torch.Tensor,
             W: dict, dev: str):
    r = slice(64 * hidx, 64 * hidx + 64)
    WqT = torch.tensor(W[f"Wq_{layer}"][r].T, device=dev)
    bq = torch.tensor(W[f"bq_{layer}"][r], device=dev)
    WkT = torch.tensor(W[f"Wk_{layer}"][r].T, device=dev)
    bk = torch.tensor(W[f"bk_{layer}"][r], device=dev)

    T = len(P)
    mu = P.mean(0)
    sd = P.std(0)
    probe = P[torch.from_numpy(
        np.random.default_rng(123).choice(T, N_PROBE, replace=False))]
    Qp = probe @ WqT + bq                                   # (P, 64)
    sub = P[torch.from_numpy(
        np.random.default_rng(7).choice(T, NN_SUB, replace=False))]
    sub2 = (sub ** 2).sum(1)
    # 2-PC basis of the pool for the snapshot film
    Pc = (P - mu)[torch.from_numpy(
        np.random.default_rng(5).choice(T, 2000, replace=False))]
    V2 = torch.linalg.svd(Pc, full_matrices=False).Vh[:2].T  # (768, 2)

    conds = [(it, im, s) for it in range(len(TAUS))
             for im in range(len(MODES)) for s in range(SEEDS)]
    C = len(conds)
    tau = torch.tensor([TAUS[it] for it, _, _ in conds], device=dev)[:, None]
    is_mode = {m: torch.tensor([MODES[im] == m for _, im, _ in conds],
                               device=dev) for m in MODES}
    retrieves = is_mode["self"] | is_mode["bias"]

    B = torch.stack([P[torch.from_numpy(
        np.random.default_rng(1000 + s).choice(T, K, replace=False))]
        for _, _, s in conds])                              # (C, K, 768)
    B0n = torch.nn.functional.normalize(B.clone(), dim=-1)
    rng = np.random.default_rng(layer * 100 + hidx)

    rows = {m: [] for m in METRICS}
    snaps = {}
    for t in range(ROUNDS + 1):
        if t % EVAL == 0:
            Kb = B @ WkT + bk                               # (C, K, 64)
            z = Qp @ Kb.transpose(1, 2) / 8.0               # (C, P, K)
            w = torch.softmax(z, dim=-1)
            rows["k90"].append(k90_t(w).mean(1))
            H = -(w * torch.log(w + 1e-12)).sum(-1)
            rows["effH"].append(H.exp().mean(1))
            s = torch.linalg.svdvals(B - B.mean(1, keepdim=True))
            p = s ** 2 / (s ** 2).sum(-1, keepdim=True)
            rows["rank"].append(1.0 / (p ** 2).sum(-1))
            rows["dist_mean"].append((B - mu).norm(dim=-1).mean(1))
            d2 = ((B ** 2).sum(-1)[..., None] + sub2
                  - 2.0 * (B @ sub.T)).clamp_min(0)
            rows["nn_dist"].append(d2.min(-1).values.sqrt().mean(1))
            g = torch.nn.functional.normalize(B, dim=-1)
            same = (g @ g.transpose(1, 2)) > 0.9999
            first = same.float().argmax(dim=1)              # (C, K)
            rows["n_distinct"].append(
                (first == torch.arange(K, device=dev)).sum(-1).float())
            for ci, (it, im, sd_) in enumerate(conds):
                key = (layer, tag, TAUS[it], MODES[im], sd_)
                if key in SHOWCASE:
                    snaps.setdefault(key, []).append(
                        ((B[ci] - mu) @ V2).cpu().numpy())
        if t == ROUNDS:
            break

        q = P[torch.from_numpy(rng.integers(T, size=C))]    # (C, 768)
        Kb = B @ WkT + bk
        z = (Kb @ (q @ WqT + bq)[:, :, None]).squeeze(2) / 8.0
        z = z * tau
        bi = is_mode["bias"]
        if bi.any():
            Bn = torch.nn.functional.normalize(B[bi], dim=-1)
            s0 = (Bn @ B0n[bi].transpose(1, 2)).amax(-1)    # data-likeness
            z[bi] = z[bi] + GAMMA * s0
        w = torch.softmax(z, dim=-1)
        new = (w[:, None, :] @ B).squeeze(1)                # (C, 768)
        new[is_mode["real"]] = P[torch.from_numpy(
            rng.integers(T, size=int(is_mode["real"].sum())))]
        nr = int(is_mode["random"].sum())
        new[is_mode["random"]] = mu + sd * torch.randn(nr, 768, device=dev)
        j = torch.from_numpy(rng.integers(K, size=C)).to(dev)
        B[torch.arange(C, device=dev), j] = torch.where(
            retrieves[:, None] | is_mode["real"][:, None]
            | is_mode["random"][:, None], new,
            B[torch.arange(C, device=dev), j])
        # (the where is a no-op guard: every mode writes; kept for clarity)

    out = {m: torch.stack(v, dim=-1).cpu().numpy() for m, v in rows.items()}
    return out, snaps


def main() -> None:
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool = np.load(DATA / "pool.npz")
    W = dict(np.load(DATA / "heads.npz").items())
    layers = W["layers"].tolist()
    head_sel = W["head_sel"]

    n_ev = ROUNDS // EVAL + 1
    shape = (len(layers), 3, len(TAUS), len(MODES), SEEDS, n_ev)
    data = {m: np.full(shape, np.nan, dtype=np.float32) for m in METRICS}
    all_snaps = {}

    t0 = time.time()
    for il, layer in enumerate(layers):
        P = torch.tensor(pool[f"pool_{layer}"], dtype=torch.float32,
                         device=dev)
        for ig, hidx in enumerate(head_sel[il]):
            rows, snaps = run_head(layer, int(hidx), TAGS[ig], P, W, dev)
            for m in METRICS:
                data[m][il, ig] = rows[m].reshape(
                    len(TAUS), len(MODES), SEEDS, n_ev)
            for key, frames in snaps.items():
                name = "snap_" + "_".join(str(x) for x in key)
                all_snaps[name] = np.array(frames, dtype=np.float32)
            print(f"layer {layer} head {hidx} ({TAGS[ig]}) done "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(
        OUT, layers=np.array(layers), head_sel=head_sel, tags=np.array(TAGS),
        taus=np.array(TAUS), modes=np.array(MODES), gamma=GAMMA,
        eval_rounds=np.arange(n_ev) * EVAL,
        k90_init=np.array([[data["k90"][il, ig, 1, 0, :, 0].mean()
                            for ig in range(3)]
                           for il in range(len(layers))]), **data)
    np.savez_compressed(SNAP_OUT, **all_snaps)
    print(f"saved {OUT} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
