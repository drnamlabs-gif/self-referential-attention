# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The flattening figure: attractor censuses of self-fed memory landscapes."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE / "figures_mhn"
INK, GRAY = "#1f2430", "#6b7280"
plt.rcParams.update({"font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.facecolor": "white",
                     "axes.edgecolor": "#9aa1ad"})


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) MNIST: the staircase of basin mergers
    m = np.load(HERE / "data" / "mnist_basin_census.npz")
    styles = {"power_2_self": ("DAM $n=2$", "#f2cdb3", "-"),
              "power_20_self": ("DAM $n=20$", "#b65b32", "-"),
              "softmax_4_self": ("attn $\\beta=4$", "#bcd4e8", "-"),
              "softmax_32_self": ("attn $\\beta=32$", "#2d6a9f", "-"),
              "softmax_32_biased": ("attn $\\beta=32$ + bias",
                                    "#2d9f6a", "--")}
    ax = axes[0]
    for lab, row in zip(m["labels"], m["counts"]):
        name, c, ls = styles[str(lab)]
        ax.plot(m["rounds"], row, color=c, ls=ls, lw=1.8, label=name,
                drawstyle="steps-post")
    ax.set_yscale("log")
    ax.axhline(1, color=INK, lw=1, ls=":")
    ax.text(m["rounds"][-1], 1, " single basin", ha="right", va="bottom",
            fontsize=8)
    ax.set_xlabel("replacement rounds")
    ax.set_ylabel("distinct attractors (100 probes)")
    ax.set_title("MNIST banks: basins merge in a cascade;\n"
                 "sharpness sets how many there are to lose")
    ax.legend(frameon=False, fontsize=8)

    # (b) BERT heads at tau=4: same cascade on real pretrained maps
    b = np.load(HERE / "data" / "llm_basin_census.npz")
    cnt, rounds = b["counts"], b["census_rounds"]
    ax = axes[1]
    cmap = plt.cm.plasma
    k0 = b["k90_init"]
    for il in range(cnt.shape[0]):
        for ig in range(cnt.shape[1]):
            c = cmap(0.1 + 0.8 * (1 - k0[il, ig] / k0.max()))
            ax.plot(rounds, cnt[il, ig, 3], color=c, lw=1.5, alpha=0.9,
                    drawstyle="steps-post")
    ax.set_yscale("log")
    ax.axhline(1, color=INK, lw=1, ls=":")
    ax.set_xlabel("replacement rounds")
    ax.set_ylabel("distinct attractors (64 probes)")
    ax.set_title("12 pretrained BERT heads ($\\tau=4$): 9 reach a\n"
                 "single basin, the rest keep $\\leq$4 of up to 26")

    # (c) the landscape dies before the bank does (tau=4, head means)
    ax = axes[2]
    d = np.load(HERE / "data" / "llm_regime.npz")
    ev = d["eval_rounds"]
    dist = d["dist_mean"][:, :, 3, 0].mean(axis=(0, 1, 2))   # tau=4 self
    dist0 = dist[0]
    mean_cnt = cnt[:, :, 3].reshape(-1, cnt.shape[-1]).astype(float)
    frac = (mean_cnt / mean_cnt[:, :1]).mean(axis=0)
    ax.plot(ev, dist / dist0, color="#b65b32", lw=2,
            label="bank distance to mean\n(fraction of start)")
    ax.plot(rounds, frac, color="#2d6a9f", lw=2,
            label="surviving basins\n(fraction of start)")
    ax.set_xlabel("replacement rounds")
    ax.set_ylabel("fraction remaining")
    ax.set_ylim(0, 1.05)
    ax.set_title("The landscape decays faster than the bank\n"
                 "travels ($\\tau=4$, means over 12 heads)")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Self-reference flattens the energy landscape: "
                 "a cascade of basin mergers ending in a single minimum",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "energy_flattening.png", dpi=180,
                bbox_inches="tight")
    print("wrote", FIG / "energy_flattening.png")


if __name__ == "__main__":
    main()
