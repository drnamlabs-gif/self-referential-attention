# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Verdict figures for the pre-registered MNIST self-reference experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).parent
FIG = HERE / "figures_mhn"
FIG.mkdir(exist_ok=True)
INK = "#1f2430"
plt.rcParams.update({"font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.facecolor": "white",
                     "axes.edgecolor": "#9aa1ad"})

# fixed identities across panels
MODE_STYLE = {0: ("self (own attractors)", "#b65b32", "-"),
              1: ("C1: real data", "#2d6a9f", "-"),
              2: ("C3: random vectors", "#6b7280", "-"),
              3: ("C4: anchored a=.99", "#2d9f6a", "-"),
              4: ("C4: anchored a=.90", "#2d9f6a", "--"),
              5: ("self + basin bias g=4", "#b65b32", ":")}


def main() -> None:
    d = np.load(HERE / "data" / "mnist_grid.npz", allow_pickle=True)
    sharps = d["sharps"]
    ev = d["eval_rounds"]
    pr, acc = d["pr"], d["acc"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    # (a) PR drift, attention rule beta=16: self vs controls
    ax = axes[0]
    ir, isp = 1, 2
    for im, (lab, c, ls) in MODE_STYLE.items():
        m = pr[ir, isp, im].mean(axis=0)
        se = pr[ir, isp, im].std(axis=0, ddof=1) / np.sqrt(pr.shape[3])
        ax.plot(ev, m, color=c, ls=ls, lw=1.8, label=lab)
        ax.fill_between(ev, m - se, m + se, color=c, alpha=0.15, lw=0)
    ax.axhline(2899, color=INK, lw=1, ls="--")
    ax.text(ev[-1], 2899, " global mean", ha="right", va="bottom", fontsize=8)
    ax.axhline(2470, color=INK, lw=1, ls=":")
    ax.text(ev[-1], 2470, " real images", ha="right", va="top", fontsize=8)
    ax.set_xlabel("replacement rounds")
    ax.set_ylabel("mean participation ratio of memories")
    ax.set_title("Attention rule ($\\beta=16$): self-reference\n"
                 "drifts the bank to the MEAN, controls do not")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)

    # (b) speed law: self mode, all sharpness values, both rules
    ax = axes[1]
    blues = ["#bcd4e8", "#7fa8cc", "#4a7fb0", "#2d6a9f"]
    oranges = ["#f2cdb3", "#e3a277", "#d07a45", "#b65b32"]
    for isp in range(4):
        ax.plot(ev, acc[0, isp, 0].mean(axis=0), color=oranges[isp], lw=1.8,
                label=f"DAM  $n={sharps[0][isp]:g}$")
        ax.plot(ev, acc[1, isp, 0].mean(axis=0), color=blues[isp], lw=1.8,
                label=f"attn $\\beta={sharps[1][isp]:g}$")
    ax.axhline(0.748, color=INK, lw=1, ls="--")
    ax.text(ev[-1], 0.748, " ceiling", ha="right", va="bottom", fontsize=8)
    ax.axhline(0.130, color=INK, lw=1, ls=":")
    ax.text(ev[-1], 0.130, " mean-image floor", ha="right", va="bottom",
            fontsize=8)
    ax.set_xlabel("replacement rounds")
    ax.set_ylabel("denoising accuracy (external judge)")
    ax.set_title("Capability only decays -- no window.\n"
                 "Sharpness sets the SPEED, never the destination")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) end state across all cells: destination independent of sharpness
    ax = axes[2]
    x = np.arange(4)
    for ir, (rule, c) in enumerate((("DAM $x^n$", "#b65b32"),
                                    ("attention softmax", "#2d6a9f"))):
        end_pr = pr[ir, :, 0, :, -1]
        m, se = end_pr.mean(axis=1), end_pr.std(axis=1, ddof=1) / np.sqrt(6)
        ax.errorbar(x + (0.06 if ir else -0.06), m, yerr=se, color=c, lw=1.8,
                    marker="o", ms=5, capsize=3, label=rule)
    ax.axhline(2899, color=INK, lw=1, ls="--")
    ax.text(3.4, 2899, " global mean", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["2 / 4", "4 / 8", "10 / 16", "20 / 32"], fontsize=9)
    ax.set_xlabel("sharpness ($n$ / $\\beta$)")
    ax.set_ylabel("final PR after 6000 rounds")
    ax.set_ylim(2400, 2950)
    ax.set_title("Same destination at every sharpness:\n"
                 "the global mean (self mode, mean $\\pm$ s.e., n=6)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("An attention / dense-associative memory retrained on its own retrievals "
                 "slides one-way to the mean -- pre-registered protocol, MNIST",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "mnist_verdict.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "mnist_verdict.png")


def make_gif() -> None:
    cases = [("DAM  $n=2$ (bluntest)", "power_2_self_s0.npz"),
             ("DAM  $n=20$ (sharpest)", "power_20_self_s0.npz"),
             ("attention  $\\beta=32$", "softmax_32_self_s0.npz")]
    snaps = []
    for _, f in cases:
        z = np.load(HERE / "data" / "mnist_snaps" / f)
        snaps.append(z["snaps"])  # (T, K, 784)
    T = min(s.shape[0] for s in snaps)

    def grid(S):  # (K,784) -> 10x10 tile image
        t = S[:100].reshape(10, 10, 28, 28)
        return t.transpose(0, 2, 1, 3).reshape(280, 280)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    ims = []
    for ax, (title, _), s in zip(axes, cases, snaps):
        im = ax.imshow(grid(s[0]), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        ims.append(im)
    counter = fig.text(0.5, 0.02, "round 0", ha="center", fontsize=10)
    fig.suptitle("The memory bank eating its own outputs (100 memories each)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    def update(f):
        for im, s in zip(ims, snaps):
            im.set_data(grid(s[f]))
        counter.set_text(f"round {f * 150}")
        return ims

    order = list(range(T)) + [T - 1] * 10
    FuncAnimation(fig, update, frames=order).save(
        FIG / "mnist_bank_evolution.gif", writer=PillowWriter(fps=10))
    plt.close(fig)
    print("wrote", FIG / "mnist_bank_evolution.gif")


if __name__ == "__main__":
    main()
    make_gif()
