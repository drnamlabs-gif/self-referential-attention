# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Self-modeling on a modern (log-sum-exp) Hopfield network.

Stage 1 of the Watson->Krotov bridge: apply the M1 self-optimization loop --
relax from a random state, then imprint the state the system itself settled
into -- to a modern Hopfield network (Ramsauer et al. 2020), where "the
weights" ARE the stored patterns. Writing the network's own retrievals back
into its own memory bank is therefore the minimal model of retraining a
model on its own generations.

The pattern set is compositional, following the random-features setting of
Negri et al. (2023) and the modular problems of Watson et al. (2011): N units
split into M modules; module m carries a frozen "allele" a_m in {-1,+1}^k and
a valid pattern commits to sigma_m * a_m with sigma_m = +-1. The composition
space holds 2^M valid patterns; the bank stores K of them. Because every unit
of a module shares its allele up to a global sign, ANY mixture of valid
patterns binarises back to a valid composition, so the self-modeling loop is
exactly a selection-recombination process on the macro-variables sigma:

    random query          ->  softmax retrieval   = parent weighting
    relax to fixed point  ->  mixture state       = recombination of parents
    binarise (commit)     ->  a valid offspring composition
    overwrite random slot ->  death-birth replacement, population size K

beta sets how sharply retrieval selects parents; delta sets the replacement
rate. Expected regimes: high beta copies single parents (memorisation, slow
neutral drift), intermediate beta recombines (novel compositions the bank
never stored -- the Watson generalisation), low beta writes the majority
pattern everywhere (immediate collapse). Fixation of a finite population
under resampling is the associative-memory face of model collapse.

Measurement discipline (same as selfmodel.core): every metric is scored
against FROZEN structures -- the allele matrix, the original K compositions,
and a held-out set of unseen compositions. Learning never touches the
yardstick.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MHNConfig:
    n_modules: int = 12      # M macro-variables
    k: int = 10              # units per module; N = n_modules * k
    K: int = 24              # bank size (population)
    beta: float = 30.0       # inverse temperature of the softmax
    delta: float = 0.25      # writeback mixing rate into the chosen slot
    steps: int = 4000        # writeback events
    retrieval_iters: int = 100
    tol: float = 1e-9
    record_every: int = 50
    n_probe: int = 64        # held-out unseen compositions for metrics
    probe_flip: float = 0.25 # fraction of units flipped before denoise probes
    seed: int = 0
    # --- the Watson ingredient: a frozen macro-scale problem ------------
    # gamma > 0 biases retrieval toward memories whose composition scores a
    # lower problem energy: logits = beta*overlap - gamma*E_J(pattern).
    # This is the MHN analogue of M1's unequal basin sizes -- better
    # attractors are reached more often -- NOT an explicit accept/reject.
    gamma: float = 0.0
    j_seed: int = 0          # the problem is frozen and shared across seeds
    snapshot_every: int = 0  # if > 0, store the decoded bank every so many steps

    @property
    def N(self) -> int:
        return self.n_modules * self.k


# --------------------------------------------------------------------------
# compositional patterns
# --------------------------------------------------------------------------


def make_alleles(cfg: MHNConfig, rng: np.random.Generator) -> np.ndarray:
    """Frozen allele matrix, (M, N). Row m is +-1/sqrt(N) on module m's units,
    zero elsewhere; rows are orthonormal up to the factor k/N."""
    A = np.zeros((cfg.n_modules, cfg.N))
    for m in range(cfg.n_modules):
        block = rng.choice([-1.0, 1.0], size=cfg.k)
        A[m, m * cfg.k : (m + 1) * cfg.k] = block / np.sqrt(cfg.N)
    return A


def compose(sigma: np.ndarray, alleles: np.ndarray) -> np.ndarray:
    """sigma (..., M) in {-1,+1} -> unit-norm pattern(s) (..., N)."""
    return sigma @ alleles


def decode(x: np.ndarray, alleles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the alleles. Returns (sigma_hat, coherence): sigma_hat is
    the per-module sign, coherence the per-module magnitude (1.0 for a valid
    composition, less for mixtures or garbage)."""
    M = alleles.shape[0]
    c = M * (x @ alleles.T)          # for valid patterns: exactly sigma
    sigma_hat = np.where(c >= 0, 1.0, -1.0)
    return sigma_hat, np.abs(c)


def sample_sigmas(n: int, M: int, rng: np.random.Generator,
                  exclude: set[bytes] | None = None) -> np.ndarray:
    """n distinct compositions, optionally disjoint from `exclude` (packed keys)."""
    seen: set[bytes] = set() if exclude is None else set(exclude)
    out = []
    while len(out) < n:
        s = rng.choice([-1.0, 1.0], size=M)
        key = sigma_key(s)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return np.array(out)


def sigma_key(sigma: np.ndarray) -> bytes:
    return np.packbits(sigma > 0).tobytes()


# --------------------------------------------------------------------------
# the macro-scale problem (optional, the Watson ingredient)
# --------------------------------------------------------------------------


def make_problem(M: int, rng: np.random.Generator) -> np.ndarray:
    """Frozen frustrated couplings over the macro-variables: a small spin
    glass, symmetric, zero diagonal (cf. frustrated_conventions in m3.py)."""
    J = rng.standard_normal((M, M))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J


def macro_energy(sigma: np.ndarray, J: np.ndarray) -> np.ndarray:
    """E_J(sigma) = -sigma J sigma / M, normalised to O(1) per module."""
    s = np.atleast_2d(sigma)
    return -np.einsum("bi,ij,bj->b", s, J, s) / J.shape[0]


def ground_truth(J: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact optimum by enumeration (M <= ~20). The frozen yardstick."""
    M = J.shape[0]
    bits = ((np.arange(2 ** M)[:, None] >> np.arange(M)) & 1)
    sigmas = np.where(bits > 0, 1.0, -1.0)
    E = macro_energy(sigmas, J)
    i = int(np.argmin(E))
    return float(E[i]), sigmas[i]


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------


def retrieve(X: np.ndarray, bank: np.ndarray, beta: float,
             iters: int = 100, tol: float = 1e-9,
             bias: np.ndarray | None = None) -> np.ndarray:
    """Batched fixed-point retrieval: X <- softmax(beta X bank^T + bias) bank.
    The CCCP update of Ramsauer et al.; energy is monotone non-increasing.
    `bias` (K,) widens the effective basin of favoured memories."""
    X = np.atleast_2d(X).copy()
    for _ in range(iters):
        logits = beta * (X @ bank.T)
        if bias is not None:
            logits = logits + bias
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        X_new = P @ bank
        if np.max(np.abs(X_new - X)) < tol:
            X = X_new
            break
        X = X_new
    return X


def energy(x: np.ndarray, bank: np.ndarray, beta: float) -> float:
    """Log-sum-exp energy of a single state (Ramsauer et al., Eq. 3,
    dropping constants that do not depend on x)."""
    logits = beta * (bank @ x)
    m = logits.max()
    lse = m + np.log(np.exp(logits - m).sum())
    return float(-lse / beta + 0.5 * x @ x)


def binarize(x: np.ndarray) -> np.ndarray:
    """The commitment step: per-unit sign, back to unit norm. For any mixture
    of valid compositions this lands exactly on a valid composition."""
    b = np.where(x >= 0, 1.0, -1.0)
    return b / np.sqrt(x.shape[-1])


# --------------------------------------------------------------------------
# the self-modeling loop
# --------------------------------------------------------------------------


def self_train(cfg: MHNConfig) -> dict:
    """Run the loop; return metric time series plus final state."""
    rng = np.random.default_rng(cfg.seed)
    alleles = make_alleles(cfg, rng)

    bank_sigma = sample_sigmas(cfg.K, cfg.n_modules, rng)
    original_keys = {sigma_key(s) for s in bank_sigma}
    probe_sigma = sample_sigmas(cfg.n_probe, cfg.n_modules, rng,
                                exclude=original_keys)

    originals = compose(bank_sigma, alleles)
    bank = originals.copy()

    # the frozen problem, if any; shared across seeds via its own generator
    J = make_problem(cfg.n_modules, np.random.default_rng(cfg.j_seed))
    E_opt, _ = ground_truth(J)
    E_best_original = float(macro_energy(bank_sigma, J).min())

    def bias() -> np.ndarray | None:
        if cfg.gamma == 0.0:
            return None
        sig, _ = decode(bank, alleles)
        return -cfg.gamma * macro_energy(sig, J)

    ever_written: set[bytes] = set(original_keys)
    rec: dict[str, list] = {k: [] for k in (
        "step", "n_distinct", "diversity", "retention", "novel_in_bank",
        "cum_novel", "denoise_unseen", "denoise_stored", "coherence",
        "E_best", "E_mean", "best_is_novel")}

    def record(step: int) -> None:
        sig, coh = decode(bank, alleles)
        keys = [sigma_key(s) for s in sig]
        uniq = set(keys)
        # mean pairwise macro-Hamming distance, normalised to [0, 1]
        G = (sig @ sig.T) / cfg.n_modules
        ham = (1.0 - G[np.triu_indices(cfg.K, 1)]) / 2.0
        rec["step"].append(step)
        rec["n_distinct"].append(len(uniq))
        rec["diversity"].append(float(ham.mean()))
        rec["retention"].append(
            sum(k in uniq for k in original_keys) / len(original_keys))
        rec["novel_in_bank"].append(
            sum(k not in original_keys for k in uniq) / cfg.K)
        rec["cum_novel"].append(len(ever_written) - len(original_keys))
        rec["coherence"].append(float(coh.mean()))
        rec["denoise_unseen"].append(denoise_accuracy(probe_sigma))
        rec["denoise_stored"].append(denoise_accuracy(bank_sigma))
        E = macro_energy(sig, J)
        i_best = int(np.argmin(E))
        rec["E_best"].append(float(E[i_best]))
        rec["E_mean"].append(float(E.mean()))
        rec["best_is_novel"].append(float(keys[i_best] not in original_keys))

    def denoise_accuracy(sigmas: np.ndarray) -> float:
        """Corrupt a composition, retrieve, decode: fraction of module signs
        recovered. Memorisation pulls unseen probes onto stored patterns
        (low); generalisation denoises them in place (high); collapse pulls
        everything onto one pattern (low)."""
        X = compose(sigmas, alleles)
        n_flip = int(round(cfg.probe_flip * cfg.N))
        Xc = X.copy()
        for row in Xc:
            idx = rng.choice(cfg.N, size=n_flip, replace=False)
            row[idx] *= -1.0
        Xr = retrieve(Xc, bank, cfg.beta, cfg.retrieval_iters, cfg.tol, bias())
        sig, _ = decode(Xr, alleles)
        return float((sig == sigmas).mean())

    snapshots: list[np.ndarray] = []
    snapshot_steps: list[int] = []

    def snap(step: int) -> None:
        if cfg.snapshot_every and step % cfg.snapshot_every == 0:
            snapshots.append(decode(bank, alleles)[0].copy())
            snapshot_steps.append(step)

    record(0)
    snap(0)
    for t in range(1, cfg.steps + 1):
        q = rng.choice([-1.0, 1.0], size=cfg.N) / np.sqrt(cfg.N)
        s = retrieve(q, bank, cfg.beta, cfg.retrieval_iters, cfg.tol, bias())[0]
        b = binarize(s)
        ever_written.add(sigma_key(decode(b, alleles)[0]))
        slot = rng.integers(cfg.K)
        mixed = (1.0 - cfg.delta) * bank[slot] + cfg.delta * b
        bank[slot] = mixed / np.linalg.norm(mixed)
        if t % cfg.record_every == 0:
            record(t)
        snap(t)

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["bank_final"] = bank
    out["bank_sigma_final"] = decode(bank, alleles)[0]
    out["original_sigma"] = bank_sigma
    out["fixation_step"] = next(
        (int(s) for s, n in zip(out["step"], out["n_distinct"]) if n == 1), -1)
    out["E_opt"] = E_opt
    out["E_best_original"] = E_best_original
    out["J"] = J
    if snapshots:
        out["snapshots"] = np.array(snapshots)
        out["snapshot_steps"] = np.array(snapshot_steps)
    return out
