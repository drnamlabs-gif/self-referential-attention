# How much outside information keeps a self-rewriting memory diverse?

Code, stored results and figures for

> Nam H. Le. *How much outside information keeps a self-rewriting associative
> memory diverse? A first-order consensus transition in modern Hopfield
> networks.* Manuscript, 2026.

A dense associative memory overwrites a random slot either with a fresh
environmental sample (probability `alpha`) or with its own retrieval of an
environmental query, iterated to a fixed point. Below a threshold
`alpha_c(beta)` the memory breaks the symmetry between environmental modes and
ends in consensus; the transition is first order.

## Layout

| Path | What it is | Figure |
|---|---|---|
| `theory.py`, `theory_cache.py` | Two-population mean field: spinodals `alpha_sym`, `alpha_cons`, collapse clock, rare-pattern norm `r*` | 2, 4, 6 |
| `theory_modes.py` | Mean field for `M` modes of any geometry | 7b |
| `sweep_K.py` | Model (environment, retrieval, basin census) and first `K` sweep | 1 |
| `sweep_order.py`, `sweep_runlength.py`, `sweep_alphac.py` | Order parameter, run-length stability, thresholds | 3, 4 |
| `sweep_fss.py`, `sweep_large.py` | Finite-size scaling, larger banks, dwell times | 5, 6a |
| `sweep_beta4.py`, `sweep_sigma.py`, `diag_cycle.py` | Sharp retrieval: rare outside patterns, spread test, relaxation cycle | 6 |
| `sweep_robust.py`, `sweep_onestep.py` | Dimension, number of modes, write rule (FIFO, one attention step) | 7a-c |
| `mnist_alpha.py` | MNIST with fixed-point writes and outside fraction `alpha` | 7d |
| `make_figs.py`, `fig_robust.py` | All figures, written to `figures/`; print every number quoted in the text | |
| `analyse_robust.py` | Threshold tables for the robustness sweep | |
| `01`-`06_*.ipynb` | Step-by-step notebooks, from the simplest cases to theory against simulation | |
| `data/` | Stored results behind every figure (`*.npz`) | |

## Reproducing

```bash
pip install -r ../requirements.txt
cd pre
python make_figs.py        # Figs. 1-7 from the stored results in data/
python fig_robust.py       # Fig. 7 only
```

Each `sweep_*.py` regenerates one file in `data/` (multiprocess; set
`MAX_WORKERS` for your machine). `mnist_alpha.py` needs the standard Keras
`mnist.npz` in `../data/` or `~/.keras/datasets/` (see the top-level README).
