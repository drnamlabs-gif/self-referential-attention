# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figures for the Physical Review E manuscript (report/pre/figures).

Reads the stored sweeps in data/ and the mean-field table data/theory.npz.
Prints every number quoted in the text.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sweep_K import env, relax
from theory import K_cross, clock

D = Path(__file__).parent / "data"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})
BLUE, RED, GREY = "#0e7490", "#c2410c", "#6b7280"
TH = np.load(D / "theory.npz")


def a_sym(b):
    return float(np.interp(b, TH["betas"], TH["a_sym"]))


def a_cons(b):
    return float(np.interp(b, TH["betas"], TH["a_cons"]))


def ac_phi(alphas, phi):
    """Threshold from the mean order parameter: <phi> crosses the midpoint (1 - alpha)/2."""
    al = np.asarray(alphas, float)
    g = np.asarray(phi) - (1 - al) / 2
    la = np.log(al)
    for i in range(len(al) - 1):
        if g[i] >= 0 > g[i + 1]:
            f = g[i] / (g[i] - g[i + 1])
            return float(np.exp(la[i] + f * (la[i + 1] - la[i])))
    return np.nan


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


def label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold")


# ------------------------------------------------------------------ fig 1
def bank_run(K, beta, alpha, start, gens, seed=1):
    rng = np.random.default_rng(seed)
    X = env(rng, K)
    if start == "cons":
        X[:, 0] = np.abs(X[:, 0])
    ph = []
    for _ in range(gens):
        for _ in range(K):
            j = rng.integers(K)
            q = env(rng, 1)[0]
            X[j] = q if rng.random() < alpha else relax(q, X, beta)
        ph.append(abs(2 * (X[:, 0] > 0).mean() - 1))
    return X, np.array(ph)


def landscape(ax, X, beta, title):
    g = np.linspace(-1.5, 1.5, 121)
    GX, GY = np.meshgrid(g, g)
    P = np.c_[GX.ravel(), GY.ravel()]
    Z = beta * P @ X.T
    zm = Z.max(1, keepdims=True)
    E = -(zm[:, 0] + np.log(np.exp(Z - zm).sum(1))) / beta + 0.5 * (P ** 2).sum(1)
    W = np.exp(Z - zm)
    W /= W.sum(1, keepdims=True)
    V = W @ X - P
    ax.contour(GX, GY, E.reshape(GX.shape), 14, colors=GREY, linewidths=0.4)
    s = slice(None, None, 10)
    ax.quiver(GX[s, s], GY[s, s], V[:, 0].reshape(GX.shape)[s, s], V[:, 1].reshape(GX.shape)[s, s],
              color=GREY, alpha=0.6, width=0.004, scale=12)
    ax.scatter(X[:, 0], X[:, 1], s=4, color=RED, zorder=3)
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", xticks=[-1, 0, 1], yticks=[-1, 0, 1], title=title)


def fig1():
    Xd, pd = bank_run(160, 2.0, 0.5, "sym", 80)
    Xc, pc = bank_run(160, 2.0, 0.3, "sym", 80)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    landscape(ax[0], Xd, 2.0, r"diverse, $\alpha=0.5$")
    landscape(ax[1], Xc, 2.0, r"consensus, $\alpha=0.3$")
    ax[2].plot(pd, color=BLUE, label=r"$\alpha=0.5$")
    ax[2].plot(pc, color=RED, label=r"$\alpha=0.3$")
    ax[2].axhline(0.7, color=RED, ls=":", lw=0.8)
    ax[2].set(xlabel="generation", ylabel=r"$\phi$", ylim=(0, 1), title=r"$K=160$, $\beta=2$")
    ax[2].legend()
    for a, s in zip(ax, "abc"):
        label(a, f"({s})")
    save(fig, "fig1_model")


# ------------------------------------------------------------------ fig 2
def fig2():
    z = np.load(D / "sweep_runlength.npz")
    g, S, M = z["gens"], z["size"][:, 0], z["minority"][:, 0]
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.7))
    cols = ["#5a6670", BLUE, "#7c3aed", RED]
    T_sim = []
    for bi, b in enumerate(z["betas"]):
        ax[0].plot(g, S[bi].mean(0), color=cols[bi], lw=1, label=rf"$\beta={b:g}$")
        # mean-field shrinkage dm/dt = -2 m exp(-2 beta m^2), from the initial norm
        m, t, tt, mm = S[bi, :, 0].mean(), 0.0, [], []
        dt = 0.05
        while t < g[-1] and m > 1 / np.sqrt(b):
            tt.append(t); mm.append(m)
            m -= dt * 2 * m * np.exp(-2 * b * m * m); t += dt
        ax[0].plot(np.maximum(tt, 1), mm, color=cols[bi], ls="--", lw=0.8)
        Ts = []
        for s in range(M.shape[1]):
            dead = M[bi, s] < 0.02
            i = next((i for i in range(len(g)) if dead[i:].all()), None)
            Ts.append(g[i] if i is not None else np.inf)
        T_sim.append(np.median(Ts))
        print(f"clock beta={b:g}: theory {clock(b):.1f}  sim median {np.median(Ts)}  ratio {np.median(Ts) / clock(b):.2f}")
    ax[0].set(xscale="log", xlabel="generation", ylabel=r"$\langle|x_1|\rangle$")
    ax[0].legend(fontsize=6)
    bb = np.linspace(1.5, 6.5, 50)
    ax[1].plot(bb, [clock(b) for b in bb], color="k", lw=1, label="theory")
    fin = np.isfinite(T_sim)
    ax[1].plot(z["betas"][fin], np.array(T_sim)[fin], "o", color=RED, ms=4, label="simulation")
    ax[1].plot(z["betas"][~fin], [3000] * (~fin).sum(), "^", color=RED, ms=4, mfc="none")
    ax[1].set(yscale="log", xlabel=r"$\beta$", ylabel=r"$T_{\rm coll}$ (generations)")
    ax[1].legend(fontsize=6)
    label(ax[0], "(a)"); label(ax[1], "(b)")
    fig.tight_layout()
    save(fig, "fig2_clock")


# ------------------------------------------------------------------ fig 3
def fig3():
    o = np.load(D / "sweep_order.npz")
    KO, AO = o["Ks"], o["alphas"]
    ser = lambda K, ai, st: [o[k] for k in o.files if k.startswith(f"K{K}_a{ai}_{st}_")]
    K = int(KO[-1])
    pick = [1, 4, 5, 6, 9]
    fig = plt.figure(figsize=(3.4, 3.0))
    gs = fig.add_gridspec(2, len(pick), height_ratios=[1, 1], hspace=0.6, wspace=0.15)
    bins = np.linspace(0, 1, 26)
    for c, ai in enumerate(pick):
        a = fig.add_subplot(gs[0, c])
        for st, col in (("sym", BLUE), ("cons", RED)):
            a.hist(np.concatenate(ser(K, ai, st)), bins=bins, color=col, alpha=0.55, density=True)
        a.axvline(1 - AO[ai], color="k", ls=":", lw=0.7)
        a.set(yticks=[], title=rf"$\alpha={AO[ai]:.2f}$", xlabel=r"$\phi$")
        a.title.set_fontsize(7)
        if c == 0:
            label(a, "(a)")
    for Kp in KO:
        win = [AO[ai] for ai in range(len(AO))
               if 0.1 < np.mean(np.concatenate(ser(int(Kp), ai, "sym") + ser(int(Kp), ai, "cons")) > 0.31) < 0.9]
        print(f"coexistence K={int(Kp)}: alphas with both states", np.round(win, 3))
    a = fig.add_subplot(gs[1, :])
    ai = int(np.argmin(np.abs(AO - 0.38)))
    for st, col in (("sym", BLUE), ("cons", RED)):
        v = ser(K, ai, st)[0]
        a.plot(np.arange(len(v)) + 201, v, color=col, lw=0.7)
    a.set(xlabel="generation", ylabel=r"$\phi$", title=rf"$K={K}$, $\alpha={AO[ai]:.2f}$")
    label(a, "(b)")
    save(fig, "fig3_order")


# ------------------------------------------------------------------ thresholds
def thresholds():
    """alpha_c(beta, K) from <phi>, pooled over every sweep that has it."""
    out = []  # (beta, K, alpha_c, source)
    z = np.load(D / "sweep_alphac.npz")
    phi = 1 - 2 * z["minority"].mean(-1)
    for ki, K in enumerate(z["Ks"]):
        for bi, b in enumerate(z["betas"]):
            al = z["alphas"][1:]
            out.append((float(b), int(K), ac_phi(al, phi[ki, bi, 1:]), "alphac"))
    f = np.load(D / "sweep_fss.npz")
    m1 = f["m1"].mean(-1)
    for ki, K in enumerate(f["Ks"]):
        for bi, b in enumerate(f["betas"]):
            out.append((float(b), int(K), ac_phi(f["alphas"][bi], m1[ki, bi]), "fss"))
    o = np.load(D / "sweep_order.npz")
    for K in o["Ks"]:
        m = [np.mean(np.concatenate([o[k] for k in o.files if k.startswith(f"K{K}_a{ai}_")]))
             for ai in range(len(o["alphas"]))]
        out.append((2.0, int(K), ac_phi(o["alphas"], m), "order"))
    for fn, b, ka, kk in (("sweep_large.npz", 3.0, "a_alphas", "a_Ks"), ("sweep_beta4.npz", 4.0, "alphas", "Ks")):
        if not (D / fn).exists():
            continue
        s = np.load(D / fn)
        pre = "A_" if fn == "sweep_large.npz" else ""
        for K in s[kk]:
            m = []
            for ai in range(len(s[ka])):
                v = [s[k] for k in s.files if k.startswith(f"{pre}K{K}_a{ai}_")]
                m.append(np.mean(np.concatenate(v)) if v else np.nan)
            if np.isnan(m).any():
                continue
            out.append((b, int(K), ac_phi(s[ka], m), fn.split(".")[0]))
    for r in out:
        print(f"alpha_c beta={r[0]:g} K={r[1]:5d}  {r[2]:.4f}  n*={r[1] / K_cross(r[0]):6.2f}  [{r[3]}]")
    return out


# ------------------------------------------------------------------ fig 4
def fig4(th):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    B = TH["betas"]
    ax.fill_between(B, TH["a_sym"], TH["a_cons"], color=GREY, alpha=0.25, lw=0)
    ax.plot(B, TH["a_sym"], color=BLUE, lw=1.2, label=r"$\alpha_{\rm sym}$ (diverse dies below)")
    ax.plot(B, TH["a_cons"], color=RED, lw=1.2, label=r"$\alpha_{\rm cons}$ (consensus dies above)")
    Ks = sorted({r[1] for r in th})
    cmap = plt.get_cmap("viridis")
    for i, K in enumerate(Ks):
        pts = [(r[0], r[2]) for r in th if r[1] == K and np.isfinite(r[2]) and r[2] > 0]
        if pts:
            p = np.array(pts)
            ax.plot(p[:, 0], p[:, 1], "o", ms=3, color=cmap(i / max(len(Ks) - 1, 1)), label=f"$K={K}$")
    ax.set(yscale="log", ylim=(2e-4, 1.2), xlim=(1.1, 6.1), xlabel=r"$\beta$", ylabel=r"$\alpha$")
    ax.text(1.5, 0.02, "consensus", color=RED)
    ax.text(4.6, 0.02, "diverse", color=BLUE)
    ax.legend(fontsize=5.5, ncol=2, loc="lower left")
    save(fig, "fig4_phase")


# ------------------------------------------------------------------ fig 5
def dwell(v, lo=0.2, hi=0.42):
    st, n = None, 0
    for x in v:
        s = 0 if x < lo else (1 if x > hi else st)
        if st is not None and s != st:
            n += 1
        st = s
    return len(v) / (n + 1), n


def fig5():
    f = np.load(D / "sweep_fss.npz")
    Ks, betas, AL = f["Ks"], f["betas"], f["alphas"]
    m1, m2, m4 = [f[k].mean(-1) for k in ("m1", "m2", "m4")]
    U = 1 - m4 / (3 * m2 ** 2)
    chi = Ks[:, None, None] * (m2 - m1 ** 2)
    fig, ax = plt.subplots(3, 4, figsize=(7.0, 4.6))
    cmap = plt.get_cmap("viridis")
    for bi, b in enumerate(betas):
        for ki, K in enumerate(Ks):
            kw = dict(color=cmap(ki / (len(Ks) - 1)), marker="o", ms=2, lw=0.8, label=f"$K={K}$")
            ax[0, bi].plot(AL[bi], m1[ki, bi], **kw)
            ax[1, bi].plot(AL[bi], chi[ki, bi], **kw)
            ax[2, bi].plot(AL[bi], U[ki, bi], **kw)
        ax[0, bi].set_title(rf"$\beta={b:g}$")
        for r in range(3):
            if b == 3.0:
                ax[r, bi].set_xscale("log")
            ax[r, bi].set_xlabel(r"$\alpha$")
    ax[0, 0].set_ylabel(r"$\langle\phi\rangle$"); ax[1, 0].set_ylabel(r"$\chi$"); ax[2, 0].set_ylabel(r"$U$")
    ax[0, 0].legend(fontsize=5)
    for bi, b in enumerate(betas):
        cm = chi[:, bi].max(1)
        s = np.polyfit(np.log(Ks), np.log(cm), 1)[0]
        print(f"chi_max beta={b:g}: {np.round(cm, 1)} slope {s:.2f}; U_min {np.round(U[:, bi].min(1), 2)}")
        ax[0, 3].plot(Ks, cm, "o-", ms=3, lw=0.8, label=rf"$\beta={b:g}$ ({s:.2f})")
    ax[0, 3].plot([20, 320], [1, 16], "k:", lw=0.8)
    ax[0, 3].set(xscale="log", yscale="log", xlabel="$K$", ylabel=r"$\chi_{\max}$")
    ax[0, 3].legend(fontsize=5.5)
    label(ax[0, 3], "(d)")
    s = np.load(D / "sweep_large.npz")
    ks, tau, err = [], [], []
    for K in s["b_Ks"]:
        v = [dwell(s[k]) for k in s.files if k.startswith(f"B_K{K}_")]
        if not v:
            continue
        T = np.array([x[0] for x in v])
        ks.append(int(K)); tau.append(T.mean()); err.append(T.std() / np.sqrt(len(T)))
        print(f"dwell K={K}: mean {T.mean():.1f}  per seed {np.round(T, 1)}  switches {[x[1] for x in v]}")
    ks, tau = np.array(ks), np.array(tau)
    ax[1, 3].errorbar(ks, tau, err, fmt="o-", ms=3, color=RED, lw=0.8)
    ok = tau < 3000 / 2
    if ok.sum() >= 2:
        c = np.polyfit(ks[ok], np.log(tau[ok]), 1)
        print(f"dwell: ln tau = {c[1]:.2f} + {c[0]:.4f} K  (exp fit);  power slope {np.polyfit(np.log(ks[ok]), np.log(tau[ok]), 1)[0]:.2f}")
        kk = np.linspace(ks.min(), ks.max(), 50)
        ax[1, 3].plot(kk, np.exp(np.polyval(c, kk)), "k:", lw=0.8)
    ax[1, 3].set(yscale="log", xlabel="$K$", ylabel="mean dwell (gen.)", title=r"$\beta=2$, $\alpha=0.385$")
    label(ax[1, 3], "(e)")
    v = [s[k] for k in s.files if k.startswith(f"B_K{int(s['b_Ks'][-1])}_")][0]
    ax[2, 3].plot(v, color=GREY, lw=0.4)
    ax[2, 3].set(xlabel="generation", ylabel=r"$\phi$", title=f"$K={int(s['b_Ks'][-1])}$")
    label(ax[2, 3], "(f)")
    for r, t in zip(range(3), "abc"):
        label(ax[r, 0], f"({t})")
    fig.tight_layout()
    save(fig, "fig5_fss")


# ------------------------------------------------------------------ fig 6
def phi_curve(s, pre, K, alphas):
    m = []
    for ai in range(len(alphas)):
        v = [s[k] for k in s.files if k.startswith(f"{pre}K{K}_a{ai}_")]
        m.append(np.mean(np.concatenate(v)) if v else np.nan)
    return np.array(m)


def fig6(th):
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    cmap = plt.get_cmap("viridis")
    # (a) beta = 3, growing K
    s = np.load(D / "sweep_large.npz")
    a = ax[0]
    a.axvspan(a_sym(3), a_cons(3), color=GREY, alpha=0.25, lw=0)
    for i, K in enumerate(s["a_Ks"]):
        a.plot(s["a_alphas"], phi_curve(s, "A_", K, s["a_alphas"]), "o-", ms=2.5, lw=0.8,
               color=cmap(i / (len(s["a_Ks"]) - 1)), label=f"$K={K}$")
    a.set(xscale="log", xlabel=r"$\alpha$", ylabel=r"$\langle\phi\rangle$", title=r"$\beta=3$, $\sigma=0.1$")
    a.legend(fontsize=5.5)
    # (b) beta = 4, K = 320, spread
    z = np.load(D / "sweep_sigma.npz")
    a = ax[1]
    a.axvspan(a_sym(4), a_cons(4), color=GREY, alpha=0.25, lw=0)
    for si, sg in enumerate(z["sigmas"]):
        m = [np.mean(np.concatenate([z[k] for k in z.files if k.startswith(f"s{si}_a{ai}_")]))
             for ai in range(len(z["alphas"]))]
        print(f"sigma={sg}: alpha_c(K=320, beta=4) = {ac_phi(z['alphas'], m):.4f}")
        a.plot(z["alphas"], m, "o-", ms=2.5, lw=0.8, color=plt.get_cmap("plasma")(si / 3),
               label=rf"$\sigma={sg:g}$ ($3\beta\sigma={3 * 4 * sg:.2f}$)")
    a.set(xscale="log", xlabel=r"$\alpha$", ylabel=r"$\langle\phi\rangle$", title=r"$\beta=4$, $K=320$")
    a.legend(fontsize=5.5)
    # (c) relaxation cycle
    a = ax[2]
    if (D / "diag_cycle.npz").exists():
        c = np.load(D / "diag_cycle.npz")
        r = c["cons_s0"]
        a.plot(r[:, 0], r[:, 1], color=RED, lw=0.9, label=r"$\phi$")
        a.plot(r[:, 0], r[:, 2], color=BLUE, lw=0.9, label=r"median $|x_1|$")
        a.plot(r[:, 0], r[:, 3] - 1, color=GREY, lw=0.6, ls="--", label=r"$K^*-1$")
        a.axhline(1 / np.sqrt(4), color=BLUE, ls=":", lw=0.6)
        for k in c.files:
            print("cycle", k, "phi jumps at generations", c[k][1:, 0][np.abs(np.diff(c[k][:, 1])) > 0.5])
    a.set(xlabel="generation", ylim=(-0.05, 1.25), title=r"$\beta=4$, $K=1280$, $\alpha=0.0005$")
    a.legend(fontsize=5.5, loc="lower right")
    for k, t in enumerate("abc"):
        label(ax[k], f"({t})")
    fig.tight_layout()
    save(fig, "fig6_nucleation")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    th = thresholds() if {"4", "6"} & set(which) else None
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": lambda: fig4(th), "5": fig5, "6": lambda: fig6(th), "7": __import__("fig_robust").fig7}[w]()
    print("figures in", FIG)
