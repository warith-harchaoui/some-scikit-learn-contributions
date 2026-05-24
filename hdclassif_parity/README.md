# HDDC ↔ HDclassif parity check

Local sanity ground for the `pr_hddc` PR. Not part of the PR itself
(this whole folder is `.gitignore`d). The goal is: confirm that the
local `HighDimensionalGaussianMixture` produces numerically
equivalent fits to the R `HDclassif::hddc()` reference implementation
on the same data, the same initial partition, the same threshold,
and the same sub-model.

## Pipeline

```
01_prepare_data.py  →  data/<dataset>_X.csv
                        data/<dataset>_labels_init.csv
                        data/<dataset>_meta.json

02_run_r_hdclassif.R   →  r_out/<dataset>_<model>_<field>.csv

03_run_python_hddc.py  →  py_out/<dataset>_<model>_<field>.csv

04_compare.py          →  report.md  (parity table per dataset/model)
```

`01` writes the data and a shared KMeans label vector. `02` and `03`
read those, run **one** EM pass from the shared init (`init.vector` in
HDclassif, `init_params=labels` in our HDDC), and dump fitted
parameters as CSVs. `04` reads both sides, Hungarian-matches clusters
by means proximity, and prints per-metric diffs.

## Requirements

- R ≥ 4.0 with `HDclassif` installed (`install.packages("HDclassif")`).
- Python: numpy / scipy / scikit-learn (already in repo `requirements.txt`).

## Running

```bash
# from repo root
python hdclassif_parity/01_prepare_data.py
Rscript hdclassif_parity/02_run_r_hdclassif.R
python hdclassif_parity/03_run_python_hddc.py
python hdclassif_parity/04_compare.py
```

## What is compared

| Metric | Notes |
| --- | --- |
| `BIC` | scalar diff |
| `loglik` | log-likelihood at EM fixed point |
| `n_parameters` | integer; must be identical |
| `signal_dims` (d_k) | per cluster, after permutation match |
| `noise_variances` (b_k) | per cluster, after match |
| `weights` (π_k) | per cluster, after match |
| `means` (μ_k) | L2 per cluster, after match |
| subspace fit | principal angles between R's Q_k and Python's Q_k (sign- and order-invariant) |
| hard-label agreement | NMI, ARI |
| responsibilities | mean abs diff after permutation |

## Sub-models tested

- `AVV` (HDclassif: `AkjBkQkDk`) — most general.
- `AEE` (HDclassif: `AkjBQkD`)  — equal noise, equal signal dim.

## Datasets

| Dataset | n | p | K_true | Source |
| --- | ---: | ---: | ---: | --- |
| `synth_lowdim` | 600 | 5 | 3 | synthetic, seeded |
| `synth_highdim` | 200 | 30 | 4 | synthetic, seeded, n < 10·p |
| `iris` | 150 | 4 | 3 | `sklearn.datasets.load_iris` |
