# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Fig. 7: robustness (dimension, number of modes, write rule) and MNIST.

Reads data/sweep_robust.npz, data/sweep_onestep.npz, data/mnist_alpha.npz.
Prints every number quoted in the text.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from make_figs import BLUE, D, GREY, RED, a_cons, a_sym, ac_phi, label, save
from theory_modes import cons_edge as m_cons
from theory_modes import modes
from theory_modes import sym_edge as m_sym


def a_onestep(al, b):
    """One-step writes: the self-written radius a solves a = T_1(1; a), a single attention step."""
    a = 1.0
    for _ in range(200000):
        an = (al * np.sinh(b) + (1 - al) * a * np.sinh(b * a)) / (al * np.cosh(b) + (1 - al) * np.cosh(b * a))
        if abs(an - a) < 1e-13:
            break
        a = an
    return a


def fig7():
    z = np.load(D / "sweep_robust.npz")

    def ac(kind, p, al):
        res = []
        for st in ("sym", "cons"):
            m = [np.mean([z[f"{kind}_{p}_a{ai}_{st}_s{s}"][:, 0].mean() for s in range(4)]) for ai in range(len(al))]
            res.append(ac_phi(al, m))
        return res

    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.0))
    # (a) dimension
    a = ax[0]
    ds = [int(d) for d in z["e1_ds"]]
    A = np.array([ac("E1", d, z["e1_alphas"]) for d in ds])
    a.axhspan(a_sym(2), a_cons(2), color=GREY, alpha=0.15, lw=0)
    a.axhline(a_sym(2), color=GREY, lw=0.7)
    a.plot(ds, A[:, 0], "o", ms=3.5, color=BLUE, label="diverse start")
    a.plot(ds, A[:, 1], "s", ms=3.5, mfc="none", color=RED, label="consensus start")
    fifo = ac("E3", "fifo", z["e1_alphas"])
    print("E1 ac", dict(zip(ds, A.round(3).tolist())), "fifo", np.round(fifo, 3))
    a.set(xscale="log", xlabel="dimension $d$", ylabel=r"$\alpha_c$", ylim=(0.3, 0.55),
          title=r"$M=2$, $\beta=2$")
    a.legend(fontsize=5.5, loc="upper left")
    # (b) number of modes
    a = ax[1]
    Ms = [int(M) for M in z["e2_ms"]]
    B = np.array([ac("E2", M, z[f"e2_alphas_M{M}"]) for M in Ms])
    Mg = np.arange(2, 9)
    ls = [m_sym(5.0, modes(M, "ortho"), tol=2e-4) for M in Mg]
    lc = [m_cons(5.0, modes(M, "ortho"), tol=2e-4) for M in Mg]
    print("E2 ac", dict(zip(Ms, B.round(3).tolist())), "MF sym", np.round(ls, 3), "cons", np.round(lc, 3))
    a.fill_between(Mg, ls, lc, color=GREY, alpha=0.15, lw=0)
    a.plot(Mg, ls, color=GREY, lw=0.8, label=r"mean field $\alpha_{\rm sym}(M)$")
    a.plot(Ms, B[:, 0], "o", ms=3.5, color=BLUE)
    a.plot(Ms, B[:, 1], "s", ms=3.5, mfc="none", color=RED)
    a.set(xlabel="number of modes $M$", ylabel=r"$\alpha_c$", title=r"$M$ modes, $\beta=5$")
    a.legend(fontsize=5.5, loc="upper left")
    # (c) one-step writes
    a = ax[2]
    o = np.load(D / "sweep_onestep.npz")
    al = o["alphas"]
    ag = np.geomspace(1e-3, 0.3, 200)
    for b, col in ((2.0, BLUE), (4.0, RED)):
        t = [np.mean([o[f"b{b:g}_a{ai}_onestep_s{s}"][:, 1].mean() for s in range(2)]) for ai in range(len(al))]
        a.plot(al[1:], t[1:], "o", ms=3, color=col, label=rf"$\beta={b:g}$")
        a.plot(ag, [a_onestep(x, b) for x in ag], color=col, lw=0.8)
        print(f"onestep beta={b} t", np.round(t, 2), "MF", np.round([a_onestep(x, b) for x in al], 2))
    a.set(xscale="log", xlabel=r"$\alpha$", ylabel=r"radius $t$", title="one-step writes")
    a.legend(fontsize=5.5, loc="upper left")
    # (d) MNIST
    a = ax[3]
    m = np.load(D / "mnist_alpha.npz")
    cm = plt.get_cmap("viridis")
    for bi, b in enumerate(m["betas"]):
        for st in ("div", "one"):
            v = [np.mean([m[f"b{b:g}_a{ai}_{st}_s{s}"][150:, 0].mean() for s in range(4)])
                 for ai in range(len(m["alphas"]))]
            if st == "div":
                a.plot(m["alphas"], v, "o-", ms=2.5, lw=0.8, color=cm(bi / 4.5), label=rf"$\beta={b:g}$")
            else:
                a.plot(m["alphas"], v, "s", ms=3.5, mfc="none", mew=0.6, color=cm(bi / 4.5))
    a.set(xscale="log", xlabel=r"$\alpha$", ylabel=r"classes in the bank, $N_{\rm eff}$", ylim=(0.5, 10),
          title=r"MNIST, $K=100$")
    a.legend(fontsize=5, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.2, labelspacing=0.2)
    for x in ax:
        x.title.set_fontsize(7)
    for k, t in enumerate("abcd"):
        label(ax[k], f"({t})")
    fig.tight_layout()
    save(fig, "fig7_robust")


if __name__ == "__main__":
    fig7()
