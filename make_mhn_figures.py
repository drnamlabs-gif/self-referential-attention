# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figures for the self-modeling modern-Hopfield experiments.

Reads data/mhn_phase.npz and data/mhn_selection.npz, writes figures_mhn/.
Also renders an animated GIF of the memory bank evolving in the three
regimes (collapse / neutral drift / selection).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from selfmodel.mhn import MHNConfig, self_train

HERE = Path(__file__).parent
FIG = HERE / "figures_mhn"
FIG.mkdir(exist_ok=True)

INK = "#1f2430"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5,
                     "axes.labelsize": 10, "text.color": INK,
                     "axes.edgecolor": "#9aa1ad", "xtick.color": INK,
                     "ytick.color": INK, "axes.labelcolor": INK,
                     "figure.facecolor": "white"})


def heat(ax, Z, betas, xs, xlabel, title, cmap, norm=None, fmt="{:.0f}"):
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:.3g}" for x in xs], fontsize=8)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f"{b:.3g}" for b in betas], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"retrieval sharpness  $\beta$")
    ax.set_title(title, pad=8)
    plt.colorbar(im, ax=ax, shrink=0.85)
    return im


def figure_phase() -> None:
    d = np.load(HERE / "data" / "mhn_phase.npz")
    betas, deltas, steps = d["betas"], d["deltas"], d["steps"]
    fix = d["fixation"].astype(float)
    fix[fix < 0] = steps[-1] + 1  # survived the whole run

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    heat(axes[0, 0], fix.mean(axis=2), betas, deltas,
         r"writeback rate  $\delta$",
         "Time to fixation (steps; capped at 4001 = survived)", "Blues")
    heat(axes[0, 1], d["cum_novel"][:, :, :, -1].mean(axis=2), betas, deltas,
         r"writeback rate  $\delta$",
         "Novel valid compositions written (never stored)", "Greens")
    heat(axes[1, 0], d["denoise_unseen"][:, :, :, -1].mean(axis=2), betas,
         deltas, r"writeback rate  $\delta$",
         "Final denoising of unseen compositions (chance = 0.5)", "Purples")
    heat(axes[1, 1], d["retention"][:, :, :, -1].mean(axis=2), betas, deltas,
         r"writeback rate  $\delta$",
         "Final retention of original memories", "Oranges")
    fig.suptitle("Self-training a modern Hopfield network on its own retrievals\n"
                 "is neutral drift: it explores (top right) but holds nothing "
                 "(bottom row)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "phase_diagram_neutral.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "phase_diagram_neutral.png")


def figure_selection() -> None:
    d = np.load(HERE / "data" / "mhn_selection.npz")
    betas, gammas, steps = d["betas"], d["gammas"], d["steps"]
    Eb, Eo, Eopt = d["E_best"], d["E_best_original"], d["E_opt"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) improvement held at the end: polarity -> diverging, gray at zero
    delta_gap = (Eb[:, :, :, -1] - Eo).mean(axis=2)
    lim = np.max(np.abs(delta_gap))
    heat(axes[0], delta_gap, betas, gammas,
         r"basin bias  $\gamma$",
         "Final best-in-bank energy minus best original\n"
         "(blue = ends better than anything it stored)",
         "RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))

    # (b) trajectories at beta = 100
    ib = 2
    colors = {0: "#b65b32", 2: "#6b7280", 4: "#2d6a9f"}  # fixed identity order
    for ig, c in colors.items():
        m = Eb[ib, ig].mean(axis=0)
        se = Eb[ib, ig].std(axis=0, ddof=1) / np.sqrt(Eb.shape[2])
        axes[1].plot(steps, m, color=c, lw=2,
                     label=rf"$\gamma={gammas[ig]:g}$")
        axes[1].fill_between(steps, m - se, m + se, color=c, alpha=0.18,
                             lw=0)
    axes[1].axhline(Eo[ib].mean(), color=INK, ls="--", lw=1.2)
    axes[1].axhline(Eopt[ib].mean(), color=INK, ls=":", lw=1.2)
    axes[1].text(steps[-1], Eo[ib].mean(), " best stored", va="bottom",
                 ha="right", fontsize=8.5)
    axes[1].text(steps[-1], Eopt[ib].mean(), " exact optimum", va="bottom",
                 ha="right", fontsize=8.5)
    axes[1].set_xlabel("writeback steps")
    axes[1].set_ylabel(r"best macro energy in bank  $E_J$")
    axes[1].set_title(r"$\beta = 100$: recall loses, recombination + wide"
                      "\nbasins for good memories discovers (mean $\\pm$ s.e., n=10)")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)

    # (c) fraction of seeds that END strictly better than best original
    beat = (Eb[:, :, :, -1] < Eo - 1e-9).mean(axis=2)
    heat(axes[2], beat, betas, gammas, r"basin bias  $\gamma$",
         "Fraction of seeds ending with a composition\nbetter than any ever stored",
         "Blues", fmt="{:.1f}")

    fig.suptitle("The Watson test on a modern Hopfield network: self-training "
                 "predicts beyond its memories only when better compositions "
                 "get wider basins", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG / "watson_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "watson_selection.png")


def make_gif() -> None:
    """Bank rasters (slots x modules) evolving in the three regimes."""
    regimes = [
        ("Collapse\n" + r"$\beta=3,\ \gamma=0$",
         MHNConfig(beta=3.0, delta=0.1, steps=4000, seed=0,
                   snapshot_every=25)),
        ("Neutral drift\n" + r"$\beta=100,\ \gamma=0$",
         MHNConfig(beta=100.0, delta=0.1, steps=4000, seed=0,
                   snapshot_every=25)),
        ("Selection\n" + r"$\beta=100,\ \gamma=4$",
         MHNConfig(beta=100.0, delta=0.1, steps=4000, seed=0, gamma=4.0,
                   snapshot_every=25)),
    ]
    runs = [(title, self_train(cfg)) for title, cfg in regimes]
    n_frames = min(len(r["snapshots"]) for _, r in runs)

    cmap = ListedColormap(["#e8a33d", "#2d6a9f"])  # sigma = -1 / +1
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    images = []
    for ax, (title, r) in zip(axes, runs):
        im = ax.imshow(r["snapshots"][0] > 0, cmap=cmap, vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("module")
        ax.set_yticks([])
        images.append(im)
    axes[0].set_ylabel("memory slot")
    counter = fig.text(0.5, 0.015, "step 0", ha="center", fontsize=9)
    fig.suptitle("A modern Hopfield network retraining on its own retrievals "
                 "(each row = one stored pattern)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    def update(f):
        for im, (_, r) in zip(images, runs):
            im.set_data(r["snapshots"][f] > 0)
        counter.set_text(f"step {int(runs[0][1]['snapshot_steps'][f])}")
        return images

    # hold the final frame for a moment
    order = list(range(n_frames)) + [n_frames - 1] * 12
    anim = FuncAnimation(fig, update, frames=order, blit=False)
    out = FIG / "bank_evolution.gif"
    anim.save(out, writer=PillowWriter(fps=18))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    figure_phase()
    figure_selection()
    make_gif()
