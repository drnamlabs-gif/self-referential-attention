# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Certify the grid's softmax rule against the field's own implementation.

hflayers.Hopfield (ml-jku/hopfield-layers, the official code of "Hopfield
Networks is All You Need", Ramsauer et al. 2020) with identity projections
computes exactly

    Z = softmax(scaling * Q K^T) V .

The softmax arm of run_mnist_grid.py computes

    out = mu + softmax(beta * cos(q - mu, X - mu)) @ (X - mu) ,

i.e. the same operation with K = unit-normalised centred memories,
Q = unit-normalised centred query, V = centred memories, scaling = beta.
This script feeds identical inputs to both and reports the max abs
difference.  Machine-precision agreement means the MNIST verdict was
obtained ON the field's attention operation, not an approximation of it.
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

import numpy as np
import torch
from hflayers import Hopfield

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_mnist_grid import masked_weights

rng = np.random.default_rng(0)
K, N, B, beta = 100, 784, 7, 16.0
X = rng.random((K, N))
Q = rng.random((B, N))
mu = X.mean(0)

# --- our rule, exactly as the grid ran it (full mask) ---
mask = np.ones(N, dtype=bool)
ours = np.stack([
    mu + masked_weights(q - mu, X - mu, "softmax", beta, None) @ (X - mu)
    for q in Q])

# --- their layer, identity projections, scaling = beta ---
Xc = X - mu
Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
Qc = Q - mu
Qn = Qc / np.linalg.norm(Qc, axis=1, keepdims=True)

hop = Hopfield(
    scaling=beta,
    state_pattern_as_static=True,
    stored_pattern_as_static=True,
    pattern_projection_as_static=True,
    normalize_stored_pattern=False,
    normalize_stored_pattern_affine=False,
    normalize_state_pattern=False,
    normalize_state_pattern_affine=False,
    normalize_pattern_projection=False,
    normalize_pattern_projection_affine=False,
    disable_out_projection=True)

with torch.no_grad():
    theirs = hop((torch.from_numpy(Xn[None]),      # stored patterns  (K)
                  torch.from_numpy(Qn[None]),      # state patterns   (Q)
                  torch.from_numpy(Xc[None])))     # pattern proj.    (V)
theirs = mu + theirs.squeeze(0).numpy()

diff = np.abs(ours - theirs).max()
print(f"hflayers.Hopfield vs run_mnist_grid softmax rule "
      f"(K={K}, d={N}, {B} queries, beta={beta}):")
print(f"  max |difference| = {diff:.3e}")
print("  VERDICT:", "IDENTICAL (machine precision)" if diff < 1e-10
      else "MISMATCH -- investigate")
