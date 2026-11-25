# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Thresholds from sweep_robust.npz: alpha_c (from each start) per condition."""
import numpy as np
from make_figs import ac_phi

z = np.load("data/sweep_robust.npz")


def curve(kind, p, n, start):
    m, t = [], []
    for ai in range(n):
        r = [z[f"{kind}_{p}_a{ai}_{start}_s{s}"] for s in range(4) if f"{kind}_{p}_a{ai}_{start}_s{s}" in z]
        m.append(np.mean([x[:, 0].mean() for x in r]) if r else np.nan)
        t.append(np.mean([x[:, 1].mean() for x in r]) if r else np.nan)
    return np.array(m), np.array(t)


def table(kind, ps, alist):
    for p in ps:
        al = alist(p)
        ms, ts = curve(kind, p, len(al), "sym")
        mc, tc = curve(kind, p, len(al), "cons")
        print(f"{kind} {p}: ac(sym)={ac_phi(al, ms):.3f} ac(cons)={ac_phi(al, mc):.3f}")
        print("   a   " + " ".join(f"{a:6.3f}" for a in al))
        print("  sym  " + " ".join(f"{a:6.2f}" for a in ms))
        print("  cons " + " ".join(f"{a:6.2f}" for a in mc))
        print("  t_s  " + " ".join(f"{a:6.2f}" for a in ts))


table("E1", list(z["e1_ds"]), lambda p: z["e1_alphas"])
table("E3", ["fifo", "onestep"], lambda p: z["e1_alphas"])
for M in z["e2_ms"]:
    print(f"M={M}: MF sym={float(z[f'e2_sym_M{M}']):.3f} cons={float(z[f'e2_cons_M{M}']):.3f}")
table("E2", list(z["e2_ms"]), lambda p: z[f"e2_alphas_M{p}"])
