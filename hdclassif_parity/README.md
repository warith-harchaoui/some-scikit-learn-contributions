# HDDC ↔ HDclassif parity check

Numerical parity check between the local
`HighDimensionalGaussianMixture` (the estimator added by `pr_hddc/`)
and the reference R implementation
[`HDclassif::hddc()`](https://CRAN.R-project.org/package=HDclassif)
(Berge, Bouveyron & Girard, 2012).

The point of this folder is to give a reviewer a single,
reproducible answer to the question *"does this Python HDDC match
the canonical R implementation numerically?"* — not in spirit, not
in a passing-test-suite sense, but **on the same data, from the same
init, parameter-by-parameter**.

The current verdict (see [`report.md`](report.md)) is **yes — all
12 / 12 sub-model fits pass strict bit-equivalent tolerance**:
both synthetic mixtures, raw Olivetti at `p = 4096`, and every
forced-`d_k` configuration. ΔPSNC on BIC and log-likelihood is
under `10⁻⁵` % of the uniform-random baseline on every row.

## Pipeline

```
01_prepare_data.py    →  data/<dataset>_X.csv
                          data/<dataset>_labels_init.csv
                          data/<dataset>_meta.json

02_run_r_hdclassif.R  →  r_out/<dataset>_<modeltag>_<field>.csv

03_run_python_hddc.py →  py_out/<dataset>_<modeltag>_<field>.csv

04_compare.py         →  report.md  (PSNC-normalised parity table + analysis)
```

`01` writes the data and a shared KMeans label vector. `02` and `03`
read those identical inputs, run a **single EM pass** from the same
hard-partition init (`init = "vector"` in HDclassif,
`init_params=<labels>` in our HDDC), and dump every fitted parameter
as a CSV. `04` reads both sides, Hungarian-aligns clusters by means
proximity, and emits `report.md` with a parity table plus a
narrative discussion.

## Datasets

| Dataset | n | p | K | Why |
| --- | ---: | ---: | ---: | --- |
| `synth_lowdim` | 600 | 5 | 3 | benign baseline — exact agreement is the bar |
| `synth_highdim` | 200 | 30 | 4 | structured low-rank cov, `n < 10·p` |
| `olivetti` | 100 | 4096 | 10 | the `n << p` regime, raw pixels — tests the SVD-of-data-matrix path |

Olivetti is deliberately **not** PCA-projected: PCA would conflate
PCA and HDDC in the diff. The raw 4096-dim run exercises HDDC's
intended regime and is what motivated the SVD speedup in
`pr_hddc/_hddc.py` (eigendecomposition would be infeasible
otherwise).

## Sub-models tested per dataset

| Tag | HDclassif name | Python config | Tests what |
| --- | --- | --- | --- |
| `AVV` | `AkjBkQkDk` | `model="AVV"` | most general; Cattell-driven `d_k` per cluster |
| `AEE` | `AkjBQkD` | `model="AEE"` | equal-noise, equal-dim; Cattell-driven shared `d` |
| `AEE_d1` | `AkjBQkD`, `com_dim=1` | `model="AEE", signal_dim=1` | M-step in isolation (no Cattell) |
| `AEE_d2` | `AkjBQkD`, `com_dim=2` | `model="AEE", signal_dim=2` | M-step in isolation (no Cattell) |

Forced-`d_k` variants are the key methodological move of the parity
check: they remove the dimension-selection rule from the comparison
and isolate the EM update math. When forced rows pass but
Cattell-driven rows don't, you know the divergence is in the
upstream `d_k` choice, not the EM loop.

## Requirements

- R ≥ 4.0 with `HDclassif` and `jsonlite`:

  ```r
  install.packages(c("HDclassif", "jsonlite"))
  ```

- Python: `numpy`, `scipy`, `scikit-learn` (already in the
  repo-root `requirements.txt`).

## Running

```bash
# From the repo root:
python   hdclassif_parity/01_prepare_data.py
Rscript  hdclassif_parity/02_run_r_hdclassif.R
python   hdclassif_parity/03_run_python_hddc.py
python   hdclassif_parity/04_compare.py
```

Each step writes its outputs to a fixed location; re-running a
later step does not require re-running earlier ones unless the
data or PR code changed. `report.md` is the only file you typically
need to read.

## What is compared

The full breakdown lives in the generated [`report.md`](report.md);
the columns are:

| Metric | What it captures |
| --- | --- |
| `ΔPSNC_BIC %`, `ΔPSNC_LL %` | Per-Sample Nats Criterion delta on BIC / log-likelihood — see [`../docs/INFORMATION_CRITERIA.md`](../docs/INFORMATION_CRITERIA.md) §3. Reported as a percentage of the uniform-random baseline (0% = perfect prediction, 100% = no better than guessing among K classes). |
| `Δn_par` | Integer parameter-count diff. Must be 0 for a passing row. |
| `max Δπ`, `max Δμ`, `max Δb` | Worst per-cluster mixing-proportion / mean / noise-variance disagreement, after Hungarian alignment. |
| `Σ Δd_k` | Sum of absolute per-cluster signal-dim differences. |
| `maxθ°(Q)` | Largest principal angle (deg) between R and Python per-cluster signal subspaces — sign- and basis-rotation invariant. |
| `NMI`, `ARI` | Permutation-invariant hard-label agreement. |

## Conventions reconciled before diffing

The Python and R implementations don't naively agree on their public
outputs. `04_compare.py` normalises three known conventions:

1. **BIC sign.** HDclassif uses `2·loglik − ν·log n` (higher better);
   sklearn uses `ν·log n − 2·loglik` (lower better). R's BIC is
   sign-flipped before diffing.
2. **Tied noise `b` is mixing-proportion weighted.** HDclassif's
   `n="E"` collapse is `b = Σ π_k (trace_k − sig_k) / (p − Σ π_k d_k)`,
   not the naive average of per-cluster `b_k`. The PR's HDDC
   matches this (`pr_hddc/_hddc.py::_apply_model_constraints`).
3. **`b_k` denominator is `p − d_k`, not `rank_eff − d_k`.** When
   `n < p`, HDclassif averages noise mass over the full model noise
   subspace `(p − d_k)`, treating null-space directions as
   zero-variance contributors. The PR's HDDC matches.

These three reconciliations are the load-bearing alignment work for
the parity check; without them, even bit-equivalent fits would show
large numerical deltas in the report.

## What this folder is *not*

* Not a performance benchmark. Wall-clock times are not measured.
* Not a multi-seed Monte Carlo. One EM pass per (dataset, model);
  the goal is reproducibility of fits, not statistical stability.
* Not a complete sub-model sweep. Two sub-models (`AVV`, `AEE`) +
  two forced-`d_k` variants are enough to surface every
  implementation-level disagreement we have observed. Adding more
  is straightforward — extend `DEFAULT_MODELS` in
  `01_prepare_data.py`.
