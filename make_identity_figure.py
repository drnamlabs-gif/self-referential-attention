# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer figure: the modern Hopfield network and the transformer attention
head are the same equation (Ramsauer et al. 2020), and the self-reference
experiment is that head eating its own output."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

HERE = Path(__file__).parent
FIG = HERE / "figures_mhn"
FIG.mkdir(exist_ok=True)
INK, ORANGE, BLUE, GREEN = "#1f2430", "#b65b32", "#2d6a9f", "#2d9f6a"
GRAY, RED = "#6b7280", "#a33a2a"
plt.rcParams.update({"font.size": 10, "text.color": INK,
                     "figure.facecolor": "white"})


def panel_a(fig) -> None:
    ax = fig.add_axes([0.03, 0.67, 0.94, 0.30])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.text(0.1, 2.75, "(a)  One equation, two names",
            fontsize=12, fontweight="bold")
    ax.text(2.4, 2.0, "modern Hopfield retrieval\n(Ramsauer et al. 2020)",
            ha="center", fontsize=9.5, color=GRAY)
    ax.text(2.4, 1.25,
            r"$\xi^{\mathrm{new}} = X\,\mathrm{softmax}(\beta\,X^{\top}\xi)$",
            ha="center", fontsize=15)
    ax.text(7.6, 2.0, "transformer attention head\n(Vaswani et al. 2017)",
            ha="center", fontsize=9.5, color=GRAY)
    ax.text(7.6, 1.25,
            r"$z = V\,\mathrm{softmax}\!\left(\frac{1}{\sqrt{d}}K^{\top}q\right)$",
            ha="center", fontsize=15)
    ax.annotate("", xy=(5.95, 1.3), xytext=(4.05, 1.3),
                arrowprops=dict(arrowstyle="<->", lw=2, color=INK))
    ax.text(5.0, 1.52, "identical", ha="center", fontsize=10,
            style="italic")
    rows = [(r"memory bank $X$", r"keys & values $K,V$", ORANGE),
            (r"query pattern $\xi$", r"incoming token $q$", BLUE),
            (r"sharpness $\beta$", r"$1/\sqrt{d}$ (inverse temperature)",
             GREEN)]
    for i, (left, right, c) in enumerate(rows):
        y = 0.75 - i * 0.33
        ax.scatter([3.0], [y + 0.07], s=40, color=c, zorder=3)
        ax.text(3.25, y, f"{left}   =   {right}", fontsize=10, va="bottom")


def panel_b(fig) -> None:
    ax = fig.add_axes([0.03, 0.34, 0.94, 0.30])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.text(0.1, 2.75, "(b)  The experiment: an attention head that eats "
                       "its own output", fontsize=12, fontweight="bold")
    for i in range(6):
        x0 = 0.9 + i * 0.58
        ax.add_patch(FancyBboxPatch(
            (x0, 1.15), 0.44, 0.95, boxstyle="round,pad=0.02",
            fc=ORANGE if i == 3 else "#f2cdb3", ec=ORANGE, lw=1.2))
    ax.text(2.6, 0.78, "stored patterns = the KV bank\n"
                       "(100 MNIST images in our grid)",
            ha="center", fontsize=9, color=ORANGE)
    ax.add_patch(Circle((5.3, 1.62), 0.22, fc="#bcd4e8", ec=BLUE, lw=1.2))
    ax.text(5.3, 0.78, "query $q$\n(half-masked image)", ha="center",
            fontsize=9, color=BLUE)
    ax.add_patch(FancyBboxPatch((6.2, 1.28), 1.9, 0.68,
                                boxstyle="round,pad=0.03", fc="white",
                                ec=INK, lw=1.2))
    ax.text(7.15, 1.62, r"softmax$(\beta\,K^{\top}q)\,V$", ha="center",
            va="center", fontsize=10.5)
    ax.annotate("", xy=(6.18, 1.62), xytext=(5.55, 1.62),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=BLUE))
    ax.text(7.15, 1.05, "$K, V$ = the stored bank", ha="center",
            fontsize=9, color=ORANGE)
    ax.annotate("", xy=(9.0, 1.62), xytext=(8.15, 1.62),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=INK))
    ax.text(9.12, 1.62, "output\n(attractor)", fontsize=9, va="center")
    ax.annotate("", xy=(3.1, 2.18), xytext=(9.2, 1.82),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=RED,
                                connectionstyle="arc3,rad=0.12"))
    ax.text(6.9, 2.58, "the output OVERWRITES one stored pattern "
                       "(self-reference)", ha="center", fontsize=9.5,
            color=RED)


def panel_c(fig) -> None:
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.28])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.text(0.1, 2.7, "(c)  Ramsauer et al.'s own axis of attention-head "
                      "regimes", fontsize=12, fontweight="bold")
    ax.annotate("", xy=(9.2, 1.75), xytext=(0.8, 1.75),
                arrowprops=dict(arrowstyle="<->", lw=2, color=INK))
    ax.text(0.8, 2.0, "global averaging\noutput $\\approx$ mean of the "
                      "whole bank", fontsize=9.5, ha="left")
    ax.text(9.2, 2.0, "selective retrieval\noutput $\\approx$ one stored "
                      "memory", fontsize=9.5, ha="right")
    ax.text(5.0, 1.95, "size of the metastable state\n(they classified real "
                       "BERT heads along this axis)", ha="center",
            fontsize=9, color=GRAY)
    ax.annotate("", xy=(1.5, 1.25), xytext=(7.6, 1.25),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=RED))
    ax.text(4.55, 0.55, "our MNIST verdict, restated in their vocabulary: "
                        "feeding the bank its own retrievals drags EVERY "
                        "head toward global averaging —\n$\\beta$ (or $n$) "
                        "only sets the speed; the destination is always "
                        "the mean", ha="center", fontsize=10, color=RED)


def main() -> None:
    fig = plt.figure(figsize=(12.5, 8.4))
    panel_a(fig)
    panel_b(fig)
    panel_c(fig)
    fig.suptitle('"Hopfield transformer" = this identity '
                 "(Hopfield Networks is All You Need, ICLR 2021)",
                 y=0.995, fontsize=13)
    fig.savefig(FIG / "hopfield_is_attention.png", dpi=180,
                bbox_inches="tight")
    print("wrote", FIG / "hopfield_is_attention.png")


if __name__ == "__main__":
    main()
