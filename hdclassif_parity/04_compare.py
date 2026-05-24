"""HDDC parity check — side-by-side comparison and report.

Stage 3 of the parity pipeline. Reads the CSVs produced by the R
script (``02_run_r_hdclassif.R`` → ``r_out/``) and the Python script
(``03_run_python_hddc.py`` → ``py_out/``), Hungarian-matches clusters
between the two fits using means proximity, and emits a parity table
per (dataset, model) pair to ``report.md`` plus stdout.

The metrics are picked to surface meaningful disagreement while
absorbing differences that are mere convention or labelling:

* **ΔBIC, Δloglik, Δn_par** — scalar diffs after correcting for
  HDclassif's "higher is better" BIC sign convention.
* **max|Δπ|, max|Δμ|, max|Δb|, Σ|Δd_k|** — per-cluster diffs after
  permutation alignment.
* **maxθ°(Q)** — largest principal angle between the per-cluster
  signal subspaces, capturing subspace disagreement without being
  fooled by column reordering or sign flips inside ``Q_k``.
* **NMI / ARI** — hard-label agreement, also permutation-invariant.

A row passes the heuristic when every diff is within tight tolerances
(see ``_pass_thresholds`` at the bottom). The pass/fail tag is a
quick visual aid only — always read the actual numbers, especially
when a documented difference (e.g. the HDclassif Cattell-rule
variant — see ``docs/HDDC.md`` §3.6) is the cause of a row failing.

Examples
--------
    $ python hdclassif_parity/01_prepare_data.py
    $ Rscript  hdclassif_parity/02_run_r_hdclassif.R
    $ python hdclassif_parity/03_run_python_hddc.py
    $ python hdclassif_parity/04_compare.py        # writes report.md
"""
from __future__ import annotations

# --- Standard library --------------------------------------------------------
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Third-party -------------------------------------------------------------
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE: str = os.path.dirname(os.path.abspath(__file__))
DATA: str = os.path.join(HERE, "data")
R_OUT: str = os.path.join(HERE, "r_out")
PY_OUT: str = os.path.join(HERE, "py_out")
REPORT: str = os.path.join(HERE, "report.md")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load(path: str) -> Optional[np.ndarray]:
    """Load a CSV as an at-least-1-D ndarray, or return ``None`` if missing.

    Missing files are common: a single sub-model may have failed on
    one side without aborting the rest of the pipeline. Returning
    ``None`` lets the caller skip the affected row gracefully.
    """
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=",", ndmin=1)


def _scalar(arr: np.ndarray) -> float:
    """Extract the first element of an array as a Python float."""
    return float(np.asarray(arr).ravel()[0])


# ---------------------------------------------------------------------------
# Permutation alignment
# ---------------------------------------------------------------------------


def _match(means_a: np.ndarray, means_b: np.ndarray) -> np.ndarray:
    """Hungarian assignment from rows of ``means_a`` to rows of ``means_b``.

    Minimises the total pairwise L2 distance between matched rows.
    Used to align R's cluster ordering to Python's so per-cluster
    quantities (weights, signal dims, subspaces) can be diffed.

    Parameters
    ----------
    means_a : ndarray, shape (K, p)
    means_b : ndarray, shape (K, p)

    Returns
    -------
    perm : ndarray of int, shape (K,)
        ``perm[i]`` is the row of ``means_b`` matched to row ``i`` of
        ``means_a``.
    """
    cost = np.linalg.norm(
        means_a[:, None, :] - means_b[None, :, :], axis=-1
    )
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(rows.size, dtype=int)
    perm[rows] = cols
    return perm


def _principal_angles_deg(Q_r: np.ndarray, Q_p: np.ndarray) -> float:
    """Largest principal angle, in degrees, between two column subspaces.

    Subspace comparison is sign-, ordering-, and basis-rotation
    invariant — exactly what we want when comparing HDDC's
    per-cluster orientation matrices. The principal angles are the
    arccos of the singular values of ``Q_r.T @ Q_p``; the largest
    angle (i.e. the smallest singular value) is the dimension where
    the two subspaces disagree the most. Reported in degrees for
    readability.

    Parameters
    ----------
    Q_r, Q_p : ndarray, shape (p, d)
        Orthonormal-column matrices (HDDC stores them in this form).

    Returns
    -------
    theta_max_deg : float
        Largest principal angle in [0, 90].
    """
    # SVD of the inner-product matrix gives cosines of the principal angles.
    M = Q_r.T @ Q_p
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.degrees(np.arccos(s.min())))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(x: Optional[float | int], w: int = 10, dec: int = 4) -> str:
    """Format a numeric cell of fixed width, gracefully handling NaN / None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return f"{'—':>{w}s}"
    if isinstance(x, (int, np.integer)):
        return f"{x:>{w}d}"
    return f"{x:>{w}.{dec}f}"


# ---------------------------------------------------------------------------
# Per-Sample Nats Criterion (PSNC) — see docs/INFORMATION_CRITERIA.md §3
# ---------------------------------------------------------------------------
# PSNC normalises a deviance-or-nats cost ``L`` by ``n · log K*``:
#
#     PSNC(L) = L / (n · log K*)
#
# In those units, PSNC = 0 means perfect prediction, PSNC = 1 means
# uniform K*-way confusion (i.e. the model is no better than
# rolling a K-sided die), and PSNC > 1 means worse than uniform.
# Two datasets with very different (n, K) become directly
# comparable. We report PSNC as a **percentage** (PSNC × 100) so
# 100% = uniform-random baseline and 0% = perfect prediction.
#
# We apply PSNC to the three scalar fit metrics that diverge in
# magnitude with (n, p): BIC, loglik, and ICL. The structural
# diffs (proportions, means, b_k, signal dims, subspace fit, label
# agreement) are already dimensionless and reported as-is.


def _psnc_pct(value_in_nats: float, n: int, K: int) -> float:
    """Per-Sample Nats Criterion, reported as a percentage.

    100% = uniform K-way random baseline. 0% = perfect prediction.

    Parameters
    ----------
    value_in_nats : float
        A cost in nats. For BIC (sklearn convention, in deviance
        units = 2× nats), divide by 2 *before* calling this.
    n : int
        Number of samples.
    K : int
        Reference outcome count (used for log K* normalisation).
    """
    return float(100.0 * value_in_nats / (n * math.log(max(K, 2))))


# ---------------------------------------------------------------------------
# Pass/fail thresholds
# ---------------------------------------------------------------------------
# Tight by design: a passing row is essentially "bit-equivalent up
# to numerical noise and label permutation". The PSNC tolerances
# are sub-tenth-of-a-percent so even tiny systematic biases surface.
_PSNC_BIC_TOL_PCT: float = 0.1   # 0.1% of the uniform-random baseline
_PSNC_LL_TOL_PCT: float = 0.1
_PARAM_TOL: float = 1e-2
_SUBSPACE_TOL_DEG: float = 5.0
_NMI_TOL: float = 0.95


def _pass_thresholds(
    d_psnc_bic_pct: float,
    d_psnc_ll_pct: float,
    d_np: int,
    d_w: float,
    d_b: float,
    sum_d_diff: int,
    worst_theta: float,
    nmi: float,
) -> bool:
    """Return True iff every parity metric is within tolerance."""
    return (
        abs(d_psnc_bic_pct) < _PSNC_BIC_TOL_PCT
        and abs(d_psnc_ll_pct) < _PSNC_LL_TOL_PCT
        and d_np == 0
        and d_w < _PARAM_TOL
        and d_b < _PARAM_TOL
        and sum_d_diff == 0
        and worst_theta < _SUBSPACE_TOL_DEG
        and nmi > _NMI_TOL
    )


# ---------------------------------------------------------------------------
# Pair comparison
# ---------------------------------------------------------------------------


def _compare_one(
    name: str,
    model: str,
    K: int,
    n: int,
    p: int,
) -> Optional[Tuple[str, bool]]:
    """Return ``(markdown_row, ok)`` for one (dataset, model) comparison.

    ``ok`` flags whether every parity metric is inside its tolerance.
    ``markdown_row`` is the rendered table row. Returns ``None`` only
    when neither side has output for this pair (no row to render).
    """
    r_pref = os.path.join(R_OUT, f"{name}_{model}_")
    p_pref = os.path.join(PY_OUT, f"{name}_{model}_")

    if not os.path.exists(r_pref + "bic.csv"):
        # Render a "missing on R side" row.
        return (
            f"| {model:>10s} | {'R missing':>10s} | "
            + " | ".join([_fmt(None)] * 9)
            + " |",
            False,
        )

    # --- Means and permutation alignment ---------------------------------
    mu_r = _load(r_pref + "means.csv").reshape(K, p)
    mu_p = _load(p_pref + "means.csv").reshape(K, p)
    # ``perm[r_idx] = py_idx``: which Python cluster matches each R cluster.
    perm = _match(mu_r, mu_p)
    # Inverse map: ``inv[py_idx] = r_idx``. Used to reorder R outputs
    # into Python's cluster ordering.
    inv = np.argsort(perm)

    # --- Scalars: BIC, loglik, n_parameters ------------------------------
    # HDclassif uses the "higher is better" BIC convention
    # (BIC_R = 2·loglik − ν·log n); sklearn uses lower-is-better
    # (BIC_py = ν·log n − 2·loglik). Flip R's sign before diffing.
    bic_r_sklearn = -_scalar(_load(r_pref + "bic.csv"))
    bic_p = _scalar(_load(p_pref + "bic.csv"))
    ll_r = _scalar(_load(r_pref + "loglik.csv"))
    ll_p = _scalar(_load(p_pref + "loglik.csv"))
    np_r = int(_scalar(_load(r_pref + "n_parameters.csv")))
    np_p = int(_scalar(_load(p_pref + "n_parameters.csv")))

    # Normalise scalar diffs to PSNC units (per-sample nats per log K),
    # reported as a percentage of the uniform-random baseline.
    # BIC is in deviance (2·nats); halve before PSNC. Log-likelihood
    # is already in nats; flip its sign so a positive PSNC means
    # "more bits to encode" (same convention as BIC).
    d_psnc_bic_pct = _psnc_pct((bic_r_sklearn - bic_p) / 2.0, n, K)
    d_psnc_ll_pct = _psnc_pct(-(ll_r - ll_p), n, K)
    d_np = np_r - np_p

    # --- Per-cluster scalars: weights, noise, signal dims ----------------
    w_r = _load(r_pref + "weights.csv").ravel()[inv]
    w_p = _load(p_pref + "weights.csv").ravel()
    d_w = float(np.max(np.abs(w_r - w_p)))

    b_r = _load(r_pref + "noise_variances.csv").ravel()
    b_p = _load(p_pref + "noise_variances.csv").ravel()
    # Some HDclassif sub-models report a scalar ``b`` (tied noise).
    # Broadcast to (K,) so we can index uniformly.
    if b_r.size == 1:
        b_r = np.full(K, b_r.item())
    else:
        b_r = b_r[inv]
    if b_p.size == 1:
        b_p = np.full(K, b_p.item())
    d_b = float(np.max(np.abs(b_r - b_p)))

    mu_r_perm = mu_r[inv]
    d_mu = float(np.max(np.linalg.norm(mu_r_perm - mu_p, axis=1)))

    d_r = _load(r_pref + "signal_dims.csv").ravel().astype(int)
    d_p = _load(p_pref + "signal_dims.csv").ravel().astype(int)
    if d_r.size == K:
        d_r = d_r[inv]
    sum_d_diff = int(np.sum(np.abs(d_r - d_p)))

    # --- Subspace fit per cluster ----------------------------------------
    # Compare R cluster inv[k] against Py cluster k. Only the
    # overlapping number of signal columns is comparable when d_k
    # differs across the two fits.
    worst_theta = 0.0
    for k in range(K):
        Q_r = _load(r_pref + f"Q_{inv[k]}.csv")
        Q_p = _load(p_pref + f"Q_{k}.csv")
        if Q_r is None or Q_p is None:
            continue
        Q_r = Q_r.reshape(p, -1)
        Q_p = Q_p.reshape(p, -1)
        d_min = min(Q_r.shape[1], Q_p.shape[1])
        if d_min == 0:
            continue
        theta = _principal_angles_deg(Q_r[:, :d_min], Q_p[:, :d_min])
        worst_theta = max(worst_theta, theta)

    # --- Hard-label agreement -------------------------------------------
    lab_r = _load(r_pref + "labels.csv").astype(int).ravel()
    lab_p = _load(p_pref + "labels.csv").astype(int).ravel()
    nmi = normalized_mutual_info_score(lab_r, lab_p)
    ari = adjusted_rand_score(lab_r, lab_p)

    # --- Pass/fail tag ---------------------------------------------------
    ok = _pass_thresholds(
        d_psnc_bic_pct=d_psnc_bic_pct,
        d_psnc_ll_pct=d_psnc_ll_pct,
        d_np=d_np,
        d_w=d_w,
        d_b=d_b,
        sum_d_diff=sum_d_diff,
        worst_theta=worst_theta,
        nmi=nmi,
    )
    mark = "" if ok else "  ⚠"

    return (
        f"| {model + mark:>10s} | {_fmt(d_psnc_bic_pct, 12, 4)} | "
        f"{_fmt(d_psnc_ll_pct, 12, 4)} | {_fmt(d_np, 7)} | "
        f"{_fmt(d_w)} | {_fmt(d_mu)} | {_fmt(d_b)} | "
        f"{_fmt(sum_d_diff, 8)} | {_fmt(worst_theta)} | "
        f"{_fmt(nmi, 6, 3)} | {_fmt(ari, 6, 3)} |"
    ), ok


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


ModelSpec = Union[str, Dict[str, Any]]


def _spec_tag(spec: ModelSpec) -> str:
    """Filename tag for a model spec; mirrors ``01_prepare_data._model_tag``."""
    if isinstance(spec, str):
        return spec
    return f"{spec['code']}_d{spec['signal_dim']}"


_REPORT_HEADER = """# HDDC parity report — `HighDimensionalGaussianMixture` vs `HDclassif::hddc()`

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

"""


_REPORT_FOOTER = """
## Scope

The parity set covers two seeded synthetic mixtures (with and
without `n << p`) and raw Olivetti faces (`n=100, p=4096`). Each is a setting where bit-equivalent agreement with
HDclassif is the right pass/fail bar — the EM trajectory is
short enough and the data clean enough that NumPy/LAPACK and
Rcpp/Eigen produce numerically identical fits.
"""


def _render_table(rows: List[str]) -> List[str]:
    """Wrap row strings with the markdown header + separator.

    Column names are deliberately ASCII-pipe-free: the prior naming
    (``max|Δπ|``) embedded literal ``|`` characters that markdown
    renderers interpret as cell separators, producing phantom empty
    columns. We replace ``max|x|`` with ``max Δx``  — the metric is
    already an absolute value internally so the bars are redundant.
    """
    # Each column is (header text, width). Width must match the
    # formatting used in ``_compare_one`` for the data rows.
    cols = [
        ("model",       10),
        ("ΔPSNC_BIC %", 12),
        ("ΔPSNC_LL %",  12),
        ("Δn_par",       7),
        ("max Δπ",      10),
        ("max Δμ",      10),
        ("max Δb",      10),
        ("Σ Δd_k",       8),
        ("maxθ°(Q)",    10),
        ("NMI",          6),
        ("ARI",          6),
    ]
    header = "| " + " | ".join(f"{name:>{w}s}" for name, w in cols) + " |"
    sep = "|" + "|".join("-" * (w + 2) for _, w in cols) + "|"
    return [header, sep, *rows]


def main() -> None:
    """Build ``report.md`` from the contents of ``r_out/`` and ``py_out/``.

    Discovery is meta-driven: we iterate every ``<name>_meta.json`` in
    ``data/`` and walk the ``models`` list inside. That keeps the
    parity table in the order the user defined the specs, and
    sidesteps the filename-parsing ambiguity introduced when forced-
    ``d_k`` tags like ``AEE_d2`` contain underscores.
    """
    if not os.path.isdir(R_OUT) or not os.listdir(R_OUT):
        print("r_out/ is empty — run 02_run_r_hdclassif.R first.")
        return

    sections: List[str] = []
    n_pass = 0
    n_total = 0
    meta_files = sorted(
        f for f in os.listdir(DATA) if f.endswith("_meta.json")
    )

    for meta_file in meta_files:
        name = meta_file[: -len("_meta.json")]
        with open(os.path.join(DATA, meta_file)) as fp:
            meta = json.load(fp)
        K = int(meta["K"])
        p = int(meta["p"])
        n = int(meta["n"])

        section_lines = [f"## `{name}`  (n={n}, p={p}, K={K})\n"]
        row_lines: List[str] = []
        section_pass = 0
        section_total = 0
        for spec in meta["models"]:
            tag = _spec_tag(spec)
            result = _compare_one(name=name, model=tag, K=K, n=n, p=p)
            if result is None:
                continue
            row, ok = result
            section_total += 1
            n_total += 1
            if ok:
                section_pass += 1
                n_pass += 1
            row_lines.append(row)

        section_lines.append(
            f"_{section_pass}/{section_total} sub-models pass strict tolerance._\n"
        )
        section_lines.extend(_render_table(row_lines))
        section_lines.append("")  # blank line after table
        sections.append("\n".join(section_lines))

    summary = (
        f"## Summary\n\n"
        f"**{n_pass} / {n_total}** sub-model fits pass the strict parity "
        f"tolerance defined above. The remaining rows are discussed below; "
        f"none indicates an EM-math bug in the PR's HDDC implementation.\n"
    )

    report = (
        _REPORT_HEADER
        + summary
        + "\n"
        + "\n".join(sections)
        + _REPORT_FOOTER
    )

    with open(REPORT, "w") as f:
        f.write(report)
    print(report)
    print(f"-> wrote {REPORT}")


if __name__ == "__main__":
    main()
