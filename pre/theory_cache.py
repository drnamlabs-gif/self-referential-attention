# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Tabulate the mean-field spinodals and the alpha = 0 clock on a beta grid."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from theory import sym_edge_fast, cons_edge, K_cross, clock

BETAS = np.round(np.r_[np.linspace(1.1, 3, 12), np.linspace(3.25, 6, 12)], 3)

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        sym = list(ex.map(sym_edge_fast, BETAS))
    np.savez("data/theory.npz", betas=BETAS, a_sym=sym, a_cons=[cons_edge(b) for b in BETAS],
             K_x=[K_cross(b) for b in BETAS], clock=[clock(b) for b in BETAS])
    for b, s in zip(BETAS, sym):
        print(b, round(s, 5), round(cons_edge(b), 5))
