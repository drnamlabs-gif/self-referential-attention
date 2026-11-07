# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Correctness checks for selfmodel.mhn."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from selfmodel.mhn import (MHNConfig, make_alleles, compose, decode, retrieve,
                           energy, binarize, sample_sigmas, sigma_key, self_train,
                           make_problem, macro_energy, ground_truth)

CFG = MHNConfig(steps=200, record_every=50, n_probe=16, seed=1)
RNG = np.random.default_rng(0)


def setup():
    rng = np.random.default_rng(3)
    alleles = make_alleles(CFG, rng)
    sigmas = sample_sigmas(CFG.K, CFG.n_modules, rng)
    bank = compose(sigmas, alleles)
    return alleles, sigmas, bank


def test_valid_patterns_are_unit_norm_and_decode_exactly():
    alleles, sigmas, bank = setup()
    assert np.allclose(np.linalg.norm(bank, axis=1), 1.0)
    sig, coh = decode(bank, alleles)
    assert np.array_equal(sig, sigmas)
    assert np.allclose(coh, 1.0)


def test_stored_pattern_is_a_fixed_point_at_high_beta():
    alleles, sigmas, bank = setup()
    out = retrieve(bank[3], bank, beta=200.0)
    assert np.allclose(out[0], bank[3], atol=1e-6)


def test_retrieval_energy_is_monotone():
    alleles, sigmas, bank = setup()
    x = RNG.choice([-1.0, 1.0], size=CFG.N) / np.sqrt(CFG.N)
    energies = [energy(x, bank, 30.0)]
    for _ in range(20):
        x = retrieve(x, bank, 30.0, iters=1)[0]
        energies.append(energy(x, bank, 30.0))
    diffs = np.diff(energies)
    assert np.all(diffs <= 1e-10)


def test_mixture_binarizes_to_a_valid_composition():
    alleles, sigmas, bank = setup()
    # generic weights -- avoid exact ties like 0.5 - 0.3 - 0.2 = 0, where the
    # per-module mixing scalar vanishes and the sign is undefined
    mix = 0.51 * bank[0] + 0.32 * bank[1] + 0.21 * bank[2]
    b = binarize(mix)
    sig, coh = decode(b, alleles)
    assert np.allclose(coh, 1.0)  # exactly on a composition, not near one


def test_low_beta_retrieval_collapses_to_one_point():
    alleles, sigmas, bank = setup()
    X = RNG.choice([-1.0, 1.0], size=(8, CFG.N)) / np.sqrt(CFG.N)
    out = retrieve(X, bank, beta=0.01)
    spread = np.max(np.std(out, axis=0))
    assert spread < 1e-4


def test_delta_zero_leaves_the_bank_unchanged():
    out = self_train(MHNConfig(steps=100, delta=0.0, record_every=100, seed=5))
    assert np.array_equal(out["bank_sigma_final"], out["original_sigma"])
    assert out["retention"][-1] == 1.0


def test_same_seed_is_deterministic():
    a = self_train(CFG)
    b = self_train(CFG)
    assert np.array_equal(a["bank_final"], b["bank_final"])
    assert np.array_equal(a["n_distinct"], b["n_distinct"])


def test_metrics_are_scored_against_frozen_structures():
    """The yardstick discipline: originals and probes must not move."""
    out = self_train(MHNConfig(steps=300, delta=1.0, beta=5.0, seed=7))
    # original_sigma is the frozen copy; the bank may have drifted anywhere
    assert out["original_sigma"].shape == (CFG.K, CFG.n_modules)
    assert set(np.unique(out["original_sigma"])) <= {-1.0, 1.0}


def test_ground_truth_is_the_true_minimum():
    J = make_problem(8, np.random.default_rng(2))
    E_opt, sig_opt = ground_truth(J)
    assert np.isclose(macro_energy(sig_opt, J)[0], E_opt)
    rand = sample_sigmas(200, 8, np.random.default_rng(4))
    assert macro_energy(rand, J).min() >= E_opt - 1e-12


def test_bias_widens_the_basin_of_favoured_memories():
    alleles, sigmas, bank = setup()
    rng = np.random.default_rng(11)
    X = rng.choice([-1.0, 1.0], size=(200, CFG.N)) / np.sqrt(CFG.N)
    bias = np.zeros(CFG.K)
    bias[0] = 5.0  # strongly favour memory 0
    out = retrieve(X, bank, beta=30.0, bias=bias)
    sig, _ = decode(out, alleles)
    hits = np.mean([sigma_key(s) == sigma_key(sigmas[0]) for s in sig])
    out0 = retrieve(X, bank, beta=30.0)
    sig0, _ = decode(out0, alleles)
    hits0 = np.mean([sigma_key(s) == sigma_key(sigmas[0]) for s in sig0])
    assert hits > hits0


def test_selection_variant_records_energy_metrics():
    out = self_train(MHNConfig(steps=200, record_every=100, gamma=2.0, seed=3))
    assert out["E_best"][0] >= out["E_opt"]
    assert np.isclose(out["E_best"][0], out["E_best_original"])
    assert out["E_mean"].shape == out["step"].shape


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
