# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""One-iteration mechanism explainer for the self-modeling MHN experiment.

Walks a single loop iteration with the real code: random query -> softmax
parent weights -> per-module blend -> commit -> a novel recombinant word
written back into the bank. Small instance (5 memories x 6 modules) so every
number is visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
from selfmodel.mhn import (MHNConfig, make_alleles, compose, decode,
                           sample_sigmas, sigma_key, binarize)

FIG = Path(__file__).parent / "figures_mhn"
FIG.mkdir(exist_ok=True)

M, K, BETA = 6, 5, 14.0
CMAP = ListedColormap(["#e8a33d", "#2d6a9f"])  # module state -1 / +1
INK = "#1f2430"
plt.rcParams.update({"font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.facecolor": "white"})


def find_nice_example() -> tuple:
    """First seed where >= 3 parents matter and the offspring is novel."""
    cfg = MHNConfig(n_modules=M, k=8, K=K)
    for seed in range(500):
        rng = np.random.default_rng(seed)
        alleles = make_alleles(cfg, rng)
        sigmas = sample_sigmas(K, M, rng)
        bank = compose(sigmas, alleles)
        q = rng.choice([-1.0, 1.0], size=cfg.N) / np.sqrt(cfg.N)
        w = np.exp(BETA * (bank @ q))
        w /= w.sum()
        blend = w @ sigmas                      # per-module weighted vote
        child = np.where(blend >= 0, 1.0, -1.0)
        novel = sigma_key(child) not in {sigma_key(s) for s in sigmas}
        if novel and (w > 0.12).sum() >= 3 and np.min(np.abs(blend)) > 0.05:
            return sigmas, w, blend, child, seed
    raise RuntimeError("no example found")


def main() -> None:
    sigmas, w, blend, child, seed = find_nice_example()
    order = np.argsort(-w)  # parents sorted by weight, biggest on top

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6),
                             gridspec_kw={"width_ratios": [1.3, 0.8, 1.1, 1.3]})

    # (1) the bank
    ax = axes[0]
    ax.imshow(sigmas[order] > 0, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_title("1 · Memory bank\n(each row = a stored pattern,\n"
                 "each cell = one module's state)", fontsize=9.5)
    ax.set_xlabel("module")
    ax.set_xticks(range(M)); ax.set_xticklabels(range(1, M + 1), fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"memory {i+1}" for i in range(K)], fontsize=8)
    for y in range(1, K):
        ax.axhline(y - 0.5, color="white", lw=2)

    # (2) softmax parent weights
    ax = axes[1]
    ax.barh(range(K), w[order][::-1], color="#6b7280", height=0.6)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"{v:.2f}" for v in w[order][::-1]], fontsize=8)
    ax.set_title("2 · Random query:\nsoftmax picks PARENTS\n"
                 r"$w_\mu \propto e^{\beta\,\mathrm{overlap}}$", fontsize=9.5)
    ax.set_xlabel("retrieval weight")
    ax.set_xlim(0, w.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)

    # (3) per-module weighted vote
    ax = axes[2]
    cols = ["#2d6a9f" if b >= 0 else "#e8a33d" for b in blend]
    ax.bar(range(M), blend, color=cols, width=0.65)
    ax.axhline(0, color=INK, lw=1)
    ax.set_title("3 · Blend = weighted VOTE\nper module; commit = take\n"
                 "the sign (binarize)", fontsize=9.5)
    ax.set_xlabel("module")
    ax.set_xticks(range(M)); ax.set_xticklabels(range(1, M + 1), fontsize=8)
    ax.set_ylabel(r"$\sum_\mu w_\mu\,\sigma_{\mu m}$", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # (4) the offspring vs the bank
    ax = axes[3]
    stack = np.vstack([sigmas[order], child])
    ax.imshow(stack > 0, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_xticks(range(M)); ax.set_xticklabels(range(1, M + 1), fontsize=8)
    ax.set_xlabel("module")
    ax.set_yticks(range(K + 1))
    ax.set_yticklabels([f"memory {i+1}" for i in range(K)] + ["OFFSPRING"],
                       fontsize=8)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    for y in range(1, K + 1):
        ax.axhline(y - 0.5, color="white", lw=2)
    ax.axhline(K - 0.5, color=INK, lw=2.5)
    ax.set_title("4 · A NEW valid word --\nin no memory; it overwrites\n"
                 "a random slot. Repeat 4000x.", fontsize=9.5)

    fig.suptitle("One iteration of self-training a modern Hopfield network "
                 "= recombination in a population of memories "
                 f"(real run, seed {seed}, " + r"$\beta=$" + f"{BETA:g})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG / "mechanism_one_iteration.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
