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
# Pass/fail thresholds
# ---------------------------------------------------------------------------
# These are intentionally tight. A row that passes is essentially
# saying "the two implementations are bit-equivalent up to numerical
# noise and label permutation". Anything looser would let real
# regressions slip by silently.
_BIC_TOL: float = 1.0
_LL_TOL: float = 1.0
_PARAM_TOL: float = 1e-2
_SUBSPACE_TOL_DEG: float = 5.0
_NMI_TOL: float = 0.95


def _pass_thresholds(
    d_bic: float,
    d_ll: float,
    d_np: int,
    d_w: float,
    d_b: float,
    sum_d_diff: int,
    worst_theta: float,
    nmi: float,
) -> bool:
    """Return True iff every parity metric is within tolerance."""
    return (
        abs(d_bic) < _BIC_TOL
        and abs(d_ll) < _LL_TOL
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
    d_bic = (
        -_scalar(_load(r_pref + "bic.csv"))
        - _scalar(_load(p_pref + "bic.csv"))
    )
    d_ll = (
        _scalar(_load(r_pref + "loglik.csv"))
        - _scalar(_load(p_pref + "loglik.csv"))
    )
    d_np = int(_scalar(_load(r_pref + "n_parameters.csv"))) - int(
        _scalar(_load(p_pref + "n_parameters.csv"))
    )

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
        d_bic=d_bic,
        d_ll=d_ll,
        d_np=d_np,
        d_w=d_w,
        d_b=d_b,
        sum_d_diff=sum_d_diff,
        worst_theta=worst_theta,
        nmi=nmi,
    )
    mark = "" if ok else "  ⚠"

    return (
        f"| {model + mark:>10s} | {_fmt(d_bic)} | {_fmt(d_ll)} | "
        f"{_fmt(d_np, 7)} | {_fmt(d_w)} | {_fmt(d_mu)} | "
        f"{_fmt(d_b)} | {_fmt(sum_d_diff, 8)} | "
        f"{_fmt(worst_theta)} | {_fmt(nmi, 6, 3)} | "
        f"{_fmt(ari, 6, 3)} |"
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

    lines: List[str] = [
        "# HDDC parity report\n",
        "Comparison of `HighDimensionalGaussianMixture` (Python, this PR) "
        "against `HDclassif::hddc()` (R reference), one EM pass from the "
        "same KMeans init.\n",
    ]

    overall_ok = True
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
        lines.append(f"\n## {name}  (n={n}, p={p}, K={K})\n")

        header = (
            f"| {'model':>10s} | {'ΔBIC':>10s} | {'Δloglik':>10s} | "
            f"{'Δn_par':>7s} | {'max|Δπ|':>10s} | {'max|Δμ|':>10s} | "
            f"{'max|Δb|':>10s} | {'Σ|Δd_k|':>8s} | "
            f"{'maxθ°(Q)':>10s} | {'NMI':>6s} | {'ARI':>6s} |"
        )
        # Markdown separator row, one dash run per column.
        sep = (
            "|"
            + "|".join(
                ["-" * (len(c) + 2) for c in header.split("|")[1:-1]]
            )
            + "|"
        )
        lines.append(header)
        lines.append(sep)

        for spec in meta["models"]:
            tag = _spec_tag(spec)
            result = _compare_one(name=name, model=tag, K=K, p=p)
            if result is None:
                continue
            row, ok = result
            overall_ok &= ok
            lines.append(row)

    lines.append("\n---\n")
    lines.append(
        f"Overall: "
        f"{'PASS ✓' if overall_ok else 'mismatches ⚠ — inspect rows above'}\n"
    )
    report = "\n".join(lines) + "\n"

    with open(REPORT, "w") as f:
        f.write(report)
    print(report)
    print(f"-> wrote {REPORT}")


if __name__ == "__main__":
    main()
