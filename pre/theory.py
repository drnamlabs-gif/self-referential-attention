# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Mean-field theory of the consensus transition (two populations: fresh points at +-1, self-written points at +-a)."""
import numpy as np
def attract(a, al, b, x=1.0, it=4000):
    for _ in range(it):
        xn = (al*np.sinh(b*x) + (1-al)*a*np.sinh(b*a*x)) / (al*np.cosh(b*x) + (1-al)*np.cosh(b*a*x))
        if abs(xn-x) < 1e-12: break
        x = xn
    return xn
def stat(al,b,a0=1.0):
    a=a0
    for _ in range(1500): a=0.9*a+0.1*attract(a,al,b)
    return a
def sym_edge(b):   # lowest alpha at which the symmetric (alive) state survives
    lo,hi=0.0,1.0
    for _ in range(25):
        m=(lo+hi)/2
        if stat(m,b)>1e-3: hi=m
        else: lo=m
    return hi
def cons_edge(b):  # highest alpha at which consensus survives: minority basin from fresh mass alpha/2 alone
    if b<=1: return 1.0
    x0=np.sqrt(1-1/b); hc=b*x0-np.arctanh(x0)
    return 2/(1+np.exp(2*hc))


def h_c(b):
    """Curie-Weiss spinodal field: a fresh minority of weight w keeps its own basin iff 0.5 ln((1-w)/w) < h_c."""
    x0 = np.sqrt(1 - 1 / b)
    return b * x0 - np.arctanh(x0)


def K_cross(b):
    """Bank size at which ONE outside point is already a critical nucleus (n* = 1)."""
    return 1 + np.exp(2 * h_c(b))


def clock(b, m0=1.0):
    """alpha = 0 collapse time in generations. Each self-write retrieves
    m tanh(beta m x) ~ m (1 - 2 exp(-2 beta m^2)), so dm/dt = -2 m exp(-2 beta m^2);
    integrate from m0 down to the pitchfork m_c = 1/sqrt(beta)."""
    m = np.linspace(1 / np.sqrt(b), m0, 4001)
    f = np.exp(2 * b * m ** 2) / (2 * m)
    return float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(m)))


def _diverse_exists(al, b, A=np.linspace(1e-3, 1, 2000)):
    """True if a = F_a^inf(1) has a solution a > 0: the relaxed position x*(a)
    of a query at +1 reaches the self-pattern position a for some a."""
    x = np.ones_like(A)
    for _ in range(5000):
        x = (al * np.sinh(b * x) + (1 - al) * A * np.sinh(b * A * x)) /             (al * np.cosh(b * x) + (1 - al) * np.cosh(b * A * x))
    G = x - A
    # a genuine crossing of the diagonal, not a jump of x*(a) to the central basin
    ok = (G[:-1] >= 0) & (G[1:] < 0) & (x[1:] > 1e-3) & (np.abs(np.diff(x)) < 0.02)
    return bool(ok.any())


def sym_edge_fast(b, tol=1e-5):
    """Diverse-state spinodal alpha_sym(beta) by bisection on existence."""
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        m = (lo + hi) / 2
        if _diverse_exists(m, b):
            hi = m
        else:
            lo = m
    return hi


def minority_basin(r, n, K, b, u=1.0):
    """Does a group of n outside patterns at -r (rest of the bank, K - n, at +u)
    hold its own fixed point? 1-D projection of the retrieval map."""
    x = -r
    for _ in range(3000):
        a = n * np.exp(-b * r * x)
        c = (K - n) * np.exp(b * u * x)
        x = (-r * a + u * c) / (a + c)
    return x < 0


def r_star(b, K):
    """Smallest norm at which ONE outside pattern founds a minority basin."""
    lo, hi = 1.0, 3.0
    if minority_basin(lo, 1, K, b):
        return lo
    for _ in range(40):
        m = (lo + hi) / 2
        lo, hi = (lo, m) if minority_basin(m, 1, K, b) else (m, hi)
    return hi
