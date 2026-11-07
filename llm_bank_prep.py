# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 0 of the LLM regime-migration experiment: cache the raw material.

Ramsauer et al. ("Hopfield Networks is All You Need") classified the attention
heads of pretrained BERT along the Hopfield regime axis -- from global
averaging (the retrieval covers the whole bank) to selective retrieval (a
single pattern) -- using the number of stored patterns needed to cover 90% of
the attention mass (k90, the size of the metastable state).  We take the SAME
model, bert-base-uncased, and prepare:

  1. a pool of real token hidden states at layers 1, 5, 8, 11 (the patterns a
     head at that depth actually eats), from two public-domain books;
  2. the pretrained W_Q / W_K of every head at those layers;
  3. each head's initial k90 against a K=100 bank, and a per-layer selection
     of three heads spanning the regime axis (most averaging / median / most
     selective).

One forward pass over the corpus, GPU, single job.  Everything downstream is
small matrix algebra on the cache.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
OUT = HERE / "data" / "llm_bank"
LAYERS = [1, 5, 8, 11]          # hidden_states index (0 = embeddings)
WINDOW, N_WINDOWS, BATCH = 128, 160, 16
K_BANK, N_PROBE = 100, 256
MODEL = "bert-base-uncased"


def corpus_text() -> str:
    parts = []
    for f in sorted((HERE / "data" / "gutenberg").glob("*.txt")):
        t = f.read_text(encoding="utf-8", errors="ignore")
        # strip the Gutenberg boilerplate head/tail crudely
        parts.append(t[2000:-20000])
    return "\n".join(parts)


def k90(w: np.ndarray) -> np.ndarray:
    """Per-row: number of top attention weights needed to reach 0.9 mass."""
    s = np.sort(w, axis=1)[:, ::-1]
    return (np.cumsum(s, axis=1) < 0.90).sum(axis=1) + 1


def main() -> None:
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    ids = tok(corpus_text(), return_tensors="pt",
              add_special_tokens=False).input_ids[0]
    print(f"corpus: {len(ids)} wordpieces")
    rng = np.random.default_rng(0)
    starts = rng.choice(len(ids) - WINDOW, N_WINDOWS, replace=False)
    windows = torch.stack([ids[s:s + WINDOW] for s in starts])

    pools = {l: [] for l in LAYERS}
    with torch.no_grad():
        for i in range(0, N_WINDOWS, BATCH):
            hs = model(windows[i:i + BATCH].to(dev),
                       output_hidden_states=True).hidden_states
            for l in LAYERS:
                pools[l].append(hs[l].reshape(-1, 768).cpu())
    pool = {f"pool_{l}": torch.cat(pools[l]).numpy().astype(np.float16)
            for l in LAYERS}
    n_tok = len(pool[f"pool_{LAYERS[0]}"])
    print(f"pooled {n_tok} hidden states per layer")

    # pretrained projections + initial k90 per head (bank of K_BANK patterns)
    heads, sel_rows = {}, []
    for l in LAYERS:
        att = model.encoder.layer[l - 1].attention.self
        Wq = att.query.weight.detach().cpu().numpy()
        bq = att.query.bias.detach().cpu().numpy()
        Wk = att.key.weight.detach().cpu().numpy()
        bk = att.key.bias.detach().cpu().numpy()
        heads |= {f"Wq_{l}": Wq, f"bq_{l}": bq, f"Wk_{l}": Wk, f"bk_{l}": bk}

        P = pool[f"pool_{l}"].astype(np.float32)
        bank = P[rng.choice(n_tok, K_BANK, replace=False)]
        probe = P[rng.choice(n_tok, N_PROBE, replace=False)]
        ks = []
        for h in range(12):
            r = slice(64 * h, 64 * h + 64)
            Q = probe @ Wq[r].T + bq[r]
            K = bank @ Wk[r].T + bk[r]
            z = (Q @ K.T) / 8.0
            z -= z.max(axis=1, keepdims=True)
            w = np.exp(z)
            w /= w.sum(axis=1, keepdims=True)
            ks.append(k90(w).mean())
        order = np.argsort(ks)
        chosen = [order[-1], order[len(order) // 2], order[0]]  # avg/med/sel
        sel_rows.append(chosen)
        print(f"layer {l:>2}: k90 per head "
              f"{np.array2string(np.array(ks), precision=1)}")
        print(f"          chosen heads {chosen} "
              f"(k90 {ks[chosen[0]]:.1f} / {ks[chosen[1]]:.1f} / "
              f"{ks[chosen[2]]:.1f})")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "pool.npz", **pool)
    np.savez_compressed(OUT / "heads.npz", layers=np.array(LAYERS),
                        head_sel=np.array(sel_rows), **heads)
    print("saved", OUT / "pool.npz", "and", OUT / "heads.npz")


if __name__ == "__main__":
    main()
