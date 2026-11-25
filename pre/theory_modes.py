# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Mean-field spinodals for M modes of general geometry (noise-free).

Modes are unit vectors E[0..M-1]. Fresh patterns sit on the modes with
weight alpha/M each. Self-written patterns sit at c + a (E[k] - c), c the centroid.
  diverse state : self weight (1-alpha)/M at a*E[k] for every k; stationarity
                  needs a query at E[0] to relax to a*E[0] with a > 0.
  consensus     : all self weight at E[0] (|x| = 1); stable while a query at
                  E[1] is captured by the E[0] basin.
For M = 2 antipodal modes these reduce to theory.sym_edge_fast / cons_edge.
"""
import numpy as np


def modes(M, kind):
    if kind == "antipodal":
        assert M == 2
        return np.array([[1.0, 0.0], [-1.0, 0.0]])
    return np.eye(M)          # orthonormal


def _relax(x, P, w, b, it=6000):
    lw = np.log(w)
    for _ in range(it):
        z = b * P @ x + lw
        z -= z.max()
        p = np.exp(z)
        xn = (p / p.sum()) @ P
        if np.abs(xn - x).max() < 1e-13:
            break
        x = xn
    return x


def diverse_exists(al, b, E, A=np.linspace(2e-3, 1, 500)):
    M = len(E)
    c = E.mean(0)
    u = E[0] - c
    s2 = u @ u
    G, X = [], []
    for a in A:
        P = np.vstack([E, c + a * (E - c)])
        w = np.r_[np.full(M, al / M), np.full(M, (1 - al) / M)]
        x = _relax(E[0].copy(), P, w, b)
        t = (x - c) @ u / s2          # position along the mode axis, 0 = centroid, 1 = mode
        X.append(t)
        G.append(t - a)
    G, X = np.array(G), np.array(X)
    ok = (G[:-1] >= 0) & (G[1:] < 0) & (X[1:] > 2e-3) & (np.abs(np.diff(X)) < 0.02)
    return bool(ok.any())


def sym_edge(b, E, tol=1e-4):
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        m = (lo + hi) / 2
        lo, hi = (lo, m) if diverse_exists(m, b, E) else (m, hi)
    return hi


def cons_stable(al, b, E):
    M = len(E)
    w = np.full(M, al / M)
    w[0] += 1 - al
    x = _relax(E[1].copy(), E, w, b)
    return x @ E[0] > x @ E[1]


def cons_edge(b, E, tol=1e-5):
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        m = (lo + hi) / 2
        lo, hi = (m, hi) if cons_stable(m, b, E) else (lo, m)
    return lo


if __name__ == "__main__":
    print("check antipodal b=2:", sym_edge(2, modes(2, "antipodal")), cons_edge(2, modes(2, "antipodal")))
    for b in (3, 4, 5, 6):
        for M in (2, 3, 4, 6):
            E = modes(M, "ortho")
            print(f"b={b} M={M}  sym={sym_edge(b, E):.4f}  cons={cons_edge(b, E):.4f}", flush=True)
