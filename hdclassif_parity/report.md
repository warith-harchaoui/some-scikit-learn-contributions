# HDDC parity report — `HighDimensionalGaussianMixture` vs `HDclassif::hddc()`

This file is **generated** by `04_compare.py` from the CSV dumps in
`r_out/` and `py_out/`. Re-run the full pipeline (`01_prepare_data.py`
→ `02_run_r_hdclassif.R` → `03_run_python_hddc.py` → `04_compare.py`)
to refresh.

## What this checks

For every (dataset, sub-model) pair, both implementations are given
**identical** input: the same observations, the same KMeans-derived
initial hard partition, the same Cattell threshold, the same model
code, the same maximum iterations and EM tolerance. They then run a
single EM pass from that shared starting point. Anything they
disagree on afterwards is attributable to the estimator
implementation, not to the data, the init, or the random seed.

The comparison Hungarian-matches clusters on means proximity before
diffing per-cluster quantities, so a row that differs only in cluster
ordering still aligns to zero.

## What the columns mean

| Column | Definition | Tolerance |
| --- | --- | ---: |
| `ΔPSNC_BIC %` | `100 · (BIC_R_sklearn − BIC_Py) / (2·n·log K)` — see `docs/INFORMATION_CRITERIA.md` §3 | `0.1%` |
| `ΔPSNC_LL %` | `100 · (loglik_Py − loglik_R) / (n·log K)` (same sign convention as BIC) | `0.1%` |
| `Δn_par` | integer parameter-count difference | `0` |
| `max Δπ` | worst per-cluster mixing-proportion difference (after Hungarian match) | `1e-2` |
| `max Δμ` | worst per-cluster mean L2 difference | (logged) |
| `max Δb` | worst per-cluster noise-variance difference | `1e-2` |
| `Σ Δd_k` | sum of absolute signal-dim differences across clusters | `0` |
| `maxθ°(Q)` | largest principal angle (deg) between R and Py per-cluster signal subspaces | `5°` |
| `NMI`, `ARI` | hard-label agreement between R and Py assignments (permutation-invariant) | `NMI > 0.95` |

PSNC (Per-Sample Nats Criterion) normalises the cost by `n · log K`
so that two datasets with very different `(n, K)` become directly
comparable. Reported as a **percentage of the uniform-random
baseline**: **0% = perfect prediction**, **100% = the model is no
better than uniformly guessing among K classes** (i.e. rolling a
fair K-sided die). A `ΔPSNC` of `0.001%` means the two
implementations differ by one part in a hundred-thousand of the
uniform-random cost — well below any practical threshold for
distinguishing model fits.

A row passes ✓ when **every** metric clears its tolerance. ⚠ flags
any deviation. Some ⚠ rows are documented divergences in HDclassif
conventions rather than implementation bugs — see the per-dataset
discussion below.

## Conventions reconciled in this pipeline

The Python and R sides do not literally agree on their public outputs;
the comparison script normalises three known conventions before
diffing:

1. **BIC sign.** HDclassif uses `BIC = 2·loglik − ν·log n` (higher is
   better); sklearn uses `BIC = ν·log n − 2·loglik` (lower is
   better). `04_compare.py` flips R's BIC sign before diffing.
2. **Tied noise `b` is mixing-proportion weighted.** HDclassif's
   `n="E"` collapse is `b = Σ π_k (trace_k − sig_k) / (p − Σ π_k d_k)`,
   not the unweighted average of per-cluster `b_k`. The PR's HDDC
   implements the weighted form (`pr_hddc/_hddc.py::_apply_model_constraints`).
3. **`b_k` denominator is `p − d_k`, not `rank_eff − d_k`.** When
   `n < p`, the empirical scatter has rank at most `n − 1`, but
   HDclassif averages noise mass over the full `(p − d_k)` model
   noise subspace (treating null-space directions as zero-variance
   contributors). The PR's HDDC matches this.

## Summary

**12 / 12** sub-model fits pass the strict parity tolerance defined above. The remaining rows are discussed below; none indicates an EM-math bug in the PR's HDDC implementation.

## `olivetti`  (n=100, p=4096, K=10)

_4/4 sub-models pass strict tolerance._

|      model |  ΔPSNC_BIC % |   ΔPSNC_LL % |  Δn_par |     max Δπ |     max Δμ |     max Δb |   Σ Δd_k |   maxθ°(Q) |    NMI |    ARI |
|------------|--------------|--------------|---------|------------|------------|------------|----------|------------|--------|--------|
|        AVV |       0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0002 |  1.000 |  1.000 |
|        AEE |      -0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0002 |  1.000 |  1.000 |
|     AEE_d1 |       0.0000 |      -0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0001 |  1.000 |  1.000 |
|     AEE_d2 |      -0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0002 |  1.000 |  1.000 |

## `synth_highdim`  (n=200, p=30, K=4)

_4/4 sub-models pass strict tolerance._

|      model |  ΔPSNC_BIC % |   ΔPSNC_LL % |  Δn_par |     max Δπ |     max Δμ |     max Δb |   Σ Δd_k |   maxθ°(Q) |    NMI |    ARI |
|------------|--------------|--------------|---------|------------|------------|------------|----------|------------|--------|--------|
|        AVV |       0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0006 |  1.000 |  1.000 |
|        AEE |       0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0006 |  1.000 |  1.000 |
|     AEE_d1 |      -0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0001 |  1.000 |  1.000 |
|     AEE_d2 |      -0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0006 |  1.000 |  1.000 |

## `synth_lowdim`  (n=600, p=5, K=3)

_4/4 sub-models pass strict tolerance._

|      model |  ΔPSNC_BIC % |   ΔPSNC_LL % |  Δn_par |     max Δπ |     max Δμ |     max Δb |   Σ Δd_k |   maxθ°(Q) |    NMI |    ARI |
|------------|--------------|--------------|---------|------------|------------|------------|----------|------------|--------|--------|
|        AVV |       0.0000 |      -0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0006 |  1.000 |  1.000 |
|        AEE |      -0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0006 |  1.000 |  1.000 |
|     AEE_d1 |       0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0003 |  1.000 |  1.000 |
|     AEE_d2 |      -0.0000 |       0.0000 |       0 |     0.0000 |     0.0000 |     0.0000 |        0 |     0.0006 |  1.000 |  1.000 |

## Scope

The parity set covers two seeded synthetic mixtures (with and
without `n << p`) and raw Olivetti faces (`n=100, p=4096`, no
PCA). Each is a setting where bit-equivalent agreement with
HDclassif is the right pass/fail bar — the EM trajectory is
short enough and the data clean enough that NumPy/LAPACK and
Rcpp/Eigen produce numerically identical fits.
