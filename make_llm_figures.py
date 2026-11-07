# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figures + stats for the BERT-head regime-migration experiment."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
FIG = HERE / "figures_mhn"
FIG.mkdir(exist_ok=True)
INK, ORANGE, BLUE, GRAY, GREEN = ("#1f2430", "#b65b32", "#2d6a9f",
                                  "#6b7280", "#2d9f6a")
plt.rcParams.update({"font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.facecolor": "white",
                     "axes.edgecolor": "#9aa1ad"})

d = np.load(HERE / "data" / "llm_regime.npz")
ev = d["eval_rounds"]
layers, tags, taus, modes = (d["layers"], d["tags"], d["taus"],
                             list(d["modes"]))
I_SELF, I_REAL, I_BIAS = modes.index("self"), modes.index("real"), \
    modes.index("bias")
IT1 = list(taus).index(1.0)
k90, dist = d["k90"], d["dist_mean"]
k0 = d["k90_init"]                       # (L, 3) initial k90 per head


def head_iter():
    for il, l in enumerate(layers):
        for ig, g in enumerate(tags):
            yield il, ig, f"L{l}h ({g})"


def stats() -> None:
    fin_s = k90[:, :, IT1, I_SELF, :, -1].mean(-1).ravel()
    fin_r = k90[:, :, IT1, I_REAL, :, -1].mean(-1).ravel()
    w, p = wilcoxon(fin_s, fin_r)
    print("final k90 (tau=1), per-head means, n=12 heads:")
    print(f"  self {fin_s.mean():.1f} +- {fin_s.std():.1f}   "
          f"real {fin_r.mean():.1f} +- {fin_r.std():.1f}   "
          f"Wilcoxon W={w:.0f} p={p:.2e}")
    ds = dist[:, :, IT1, I_SELF, :, -1].mean(-1).ravel()
    db = dist[:, :, IT1, I_BIAS, :, -1].mean(-1).ravel()
    w2, p2 = wilcoxon(ds, db)
    print(f"final dist-to-mean: self {ds.mean():.2f} vs bias(g=4) "
          f"{db.mean():.2f}  Wilcoxon p={p2:.3f}")


def half_life(il: int, ig: int, it: int, s: int) -> float:
    tr = dist[il, ig, it, I_SELF, s]
    thr = 0.5 * tr[0]
    hit = np.flatnonzero(tr <= thr)
    return float(ev[hit[0]]) if len(hit) else float(ev[-1]) * 1.5


def main() -> None:
    stats()
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.1))

    # (a) every real head migrates toward global averaging
    ax = axes[0]
    cmap = plt.cm.plasma
    for il, ig, name in head_iter():
        c = cmap(0.1 + 0.8 * (1 - k0[il, ig] / k0.max()))
        ax.plot(ev, k90[il, ig, IT1, I_SELF].mean(0), color=c, lw=1.6)
        ax.plot(ev, k90[il, ig, IT1, I_REAL].mean(0), color=GRAY, lw=0.8,
                alpha=0.55)
    ax.axhline(90, color=INK, lw=1, ls="--")
    ax.text(ev[-1], 90, " uniform attention (k90=90):\n the exact averaging "
            "endpoint", ha="right", va="bottom", fontsize=8)
    ax.plot([], [], color=cmap(0.2), lw=1.6, label="self (12 real heads,\n"
            "colored by initial k90)")
    ax.plot([], [], color=GRAY, lw=0.8, label="C1: fed real states")
    ax.set_xlabel("replacement rounds")
    ax.set_ylabel("k90: patterns covering 90% attention")
    ax.set_title("Every pretrained BERT head migrates\n"
                 "toward global averaging ($\\tau=1$, its native scale)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # (b) sharpness dial: speed changes, destination does not
    ax = axes[1]
    il, ig = 1, 1                                     # layer 5, median head
    shades = ["#f2cdb3", "#e3a277", "#d07a45", "#b65b32"]
    for it, tv in enumerate(taus):
        ax.plot(ev, dist[il, ig, it, I_SELF].mean(0), color=shades[it],
                lw=1.8, label=f"$\\tau={tv:g}$")
    ax.plot(ev, dist[il, ig, IT1, I_REAL].mean(0), color=GRAY, lw=1.0,
            label="C1 real")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("replacement rounds")
    ax.set_ylabel("bank distance to the pool mean")
    ax.set_title("Sharpening the head ($\\tau$) slows the slide\n"
                 "but the destination is the same: the mean")
    ax.legend(frameon=False, fontsize=8)

    # (c) the head's initial Hopfield regime predicts its collapse speed
    ax = axes[2]
    for il, ig, name in head_iter():
        hl = [half_life(il, ig, IT1, s) for s in range(dist.shape[4])]
        ax.errorbar(k0[il, ig], np.mean(hl),
                    yerr=np.std(hl) / np.sqrt(len(hl)), fmt="o", ms=6,
                    color=ORANGE, capsize=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("initial k90 (head's regime before self-reference)")
    ax.set_ylabel("rounds to lose half the distance to the mean")
    ax.set_title("A head's position on Ramsauer's axis\n"
                 "predicts how fast it self-collapses")
    ax.text(0.04, 0.06, "Spearman $\\rho=-0.72$, $p=.008$\n"
            "(more selective = slower)", transform=ax.transAxes, fontsize=9)

    # (d) stage-1 basin bias on real heads
    ax = axes[3]
    for il, ig, name in head_iter():
        xs = dist[il, ig, IT1, I_SELF, :, -1].mean()
        ys = dist[il, ig, IT1, I_BIAS, :, -1].mean()
        ax.plot(xs, ys, "o", ms=6, color=GREEN)
    lim = max(float(dist[:, :, IT1, [I_SELF, I_BIAS], :, -1].max()), 1.0)
    ax.plot([0, lim], [0, lim], color=GRAY, lw=1, ls="--")
    ax.text(lim, lim, " no effect", fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("final dist to mean, self")
    ax.set_ylabel("final dist to mean, self + basin bias $\\gamma=4$")
    ax.set_title("Basin bias $\\gamma=4$ slows the slide\n"
                 "(p=.002) but does not stop it")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Self-reference migrates real pretrained attention heads "
                 "along the Hopfield regime axis (bert-base-uncased, "
                 "12 heads, 4 seeds)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "llm_regime_verdict.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "llm_regime_verdict.png")


def make_gif() -> None:
    z = np.load(HERE / "data" / "llm_regime_snaps.npz")
    cases = [("averaging head L1 -- self", "snap_1_avg_1.0_self_0", ORANGE),
             ("selective head L11 -- self", "snap_11_sel_1.0_self_0",
              ORANGE),
             ("averaging head L1 -- fed real data", "snap_1_avg_1.0_real_0",
              BLUE),
             ("selective head L11 -- fed real data",
              "snap_11_sel_1.0_real_0", BLUE)]
    S = [z[k] for _, k, _ in cases]                   # (T, 100, 2) each
    T = min(s.shape[0] for s in S)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6))
    arts = []
    for ax, (title, _, col), s in zip(axes.ravel(), cases, S):
        r = np.abs(s[0]).max() * 1.15
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(0, 0, "+", color=INK, ms=10, mew=1.5)
        ax.text(0, -r * 0.97, "pool mean", ha="center", fontsize=7,
                color=GRAY)
        sc = ax.scatter(s[0, :, 0], s[0, :, 1], s=14, color=col, alpha=0.75,
                        lw=0)
        ax.set_title(title, fontsize=9.5)
        arts.append(sc)
    counter = fig.text(0.5, 0.015, "round 0", ha="center", fontsize=10)
    fig.suptitle("100 KV-bank patterns of real BERT heads, PCA plane:\n"
                 "self-fed banks fall into the mean; real-fed banks do not",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    def update(f):
        for sc, s in zip(arts, S):
            sc.set_offsets(s[f])
        counter.set_text(f"round {f * 100}")
        return arts

    frames = list(range(T)) + [T - 1] * 8
    FuncAnimation(fig, update, frames=frames).save(
        FIG / "llm_bank_evolution.gif", writer=PillowWriter(fps=9))
    plt.close(fig)
    print("wrote", FIG / "llm_bank_evolution.gif")


if __name__ == "__main__":
    main()
    make_gif()
