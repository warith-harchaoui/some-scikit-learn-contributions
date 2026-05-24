"""HDDC parity check — Python side.

Stage 2 of the parity pipeline. For every (dataset, model) pair
discovered under ``data/`` (those side-cars are written by
``01_prepare_data.py``), fits the local draft
``HighDimensionalGaussianMixture`` once from the *same* shared
KMeans init that the R script also consumes, then dumps the fitted
parameters to ``py_out/`` as CSVs in the same per-field layout the R
script produces. ``04_compare.py`` then diffs the two folders
field-by-field.

Output files per (dataset, model) pair under ``py_out/`` (every R
filename has a matching Python file here):

* ``<name>_<model>_bic.csv``               — scalar, sklearn convention.
* ``<name>_<model>_icl.csv``               — scalar (Python-only, no R counterpart).
* ``<name>_<model>_loglik.csv``            — scalar at EM fixed point.
* ``<name>_<model>_n_parameters.csv``      — integer free-parameter count.
* ``<name>_<model>_weights.csv``           — (K,) mixing proportions.
* ``<name>_<model>_signal_dims.csv``       — (K,) per-cluster d_k.
* ``<name>_<model>_noise_variances.csv``   — (K,) noise variance per cluster.
* ``<name>_<model>_means.csv``             — (K, p) cluster means.
* ``<name>_<model>_eigenvalues.csv``       — (K, p) eigenvalues, zero-padded.
* ``<name>_<model>_Q_<k>.csv``             — (p, d_k) orientation per cluster.
* ``<name>_<model>_labels.csv``            — (n,) 0-indexed hard labels.
* ``<name>_<model>_responsibilities.csv``  — (n, K) posterior responsibilities.

Examples
--------
After running ``01_prepare_data.py``:

    $ python hdclassif_parity/03_run_python_hddc.py
    PY : synth_lowdim    AVV
    PY : synth_lowdim    AEE
    ...
    Python parity dump complete -> .../hdclassif_parity/py_out

Then run the R counterpart and ``04_compare.py`` to produce
``report.md``.

Notes
-----
The local HDDC implementation is loaded via ``importlib.util`` from
``pr_hddc/_hddc.py`` rather than via ``sklearn.mixture`` because the
PR has not landed yet and there is no installed sklearn version of
the class.
"""
from __future__ import annotations

# --- Standard library --------------------------------------------------------
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# --- Third-party -------------------------------------------------------------
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE: str = os.path.dirname(os.path.abspath(__file__))
CONTRIB: str = os.path.dirname(HERE)
DATA: str = os.path.join(HERE, "data")
OUT: str = os.path.join(HERE, "py_out")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Load the draft HDDC implementation from pr_hddc/_hddc.py.
# ---------------------------------------------------------------------------
# The PR hasn't been merged upstream, so we can't ``from sklearn.mixture
# import HighDimensionalGaussianMixture``. We side-load the file directly
# and also register it in ``sys.modules`` so any downstream pickling
# inside the estimator works (e.g., ``check_estimator`` round-trip).
_spec = importlib.util.spec_from_file_location(
    "_pr_hddc_draft", os.path.join(CONTRIB, "pr_hddc", "_hddc.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_pr_hddc_draft"] = _mod
_spec.loader.exec_module(_mod)
HighDimensionalGaussianMixture = _mod.HighDimensionalGaussianMixture


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _icl_of(fit: Any, X: np.ndarray) -> float:
    """Best-effort ICL accessor: returns ``fit.icl(X)`` or ``nan``.

    The Python-side HDDC always exposes ``icl()`` so this is mostly
    a safety net for forks that strip the method.
    """
    return float(fit.icl(X)) if hasattr(fit, "icl") else float("nan")


def _save(path: str, arr: Any, fmt: str = "%.10g") -> None:
    """Write ``arr`` to ``path`` as CSV with one row per record.

    Wraps ``np.savetxt`` to absorb scalars and 1-D arrays into a
    consistent 2-D layout so the comparison script always knows what
    to expect.

    Parameters
    ----------
    path : str
        Output file path.
    arr : array-like
        Scalar, 1-D, or 2-D values to persist.
    fmt : str, default="%.10g"
        ``np.savetxt`` format string. Use ``"%d"`` for integer fields.
    """
    arr = np.atleast_1d(np.asarray(arr))
    if arr.ndim == 1:
        # Scalars get a single 1x1 row; vectors become a column for
        # readability when opened in a spreadsheet.
        arr = arr.reshape(-1, 1) if arr.size > 1 else arr.reshape(1, 1)
    np.savetxt(path, np.atleast_2d(arr), delimiter=",", fmt=fmt)


# ---------------------------------------------------------------------------
# Per-(dataset, model) fit + dump
# ---------------------------------------------------------------------------


ModelSpec = Union[str, Dict[str, Any]]


def _resolve_model_spec(spec: ModelSpec) -> Tuple[str, Optional[int], str]:
    """Parse one element of ``meta.json["models"]``.

    Parameters
    ----------
    spec : str or dict
        Either a plain geometric code (``"AVV"``) for a Cattell-driven
        fit, or a dict ``{"code": "AEE", "signal_dim": 2}`` forcing
        ``d_k = signal_dim`` on every cluster.

    Returns
    -------
    code : str
        Three-letter geometric model code.
    signal_dim : int or None
        Forced ``d_k`` value, or ``None`` to let Cattell pick.
    tag : str
        Filename tag — bare code or ``<code>_d<n>``.
    """
    if isinstance(spec, str):
        return spec, None, spec
    code = spec["code"]
    sd = int(spec["signal_dim"])
    return code, sd, f"{code}_d{sd}"


def fit_one(
    name: str,
    spec: ModelSpec,
    X: np.ndarray,
    init: np.ndarray,
    K: int,
    cattell_threshold: float,
    random_state: int,
    out_dir: str,
) -> None:
    """Fit the local HDDC for one (dataset, model) pair and persist outputs.

    Parameters
    ----------
    name : str
        Dataset name, for logging only.
    spec : str or dict
        Plain geometric code (Cattell-driven) or
        ``{"code": ..., "signal_dim": ...}`` (forced d_k).
    X : ndarray, shape (n, p)
        Observations.
    init : ndarray of int, shape (n,)
        0-indexed initial cluster labels (the shared KMeans output).
    K : int
        Number of mixture components.
    cattell_threshold : float
        Sensitivity for the Cattell scree rule that picks ``d_k``
        when ``signal_dim`` is not forced.
    random_state : int
        Seed used for any internal RNG; here mostly cosmetic since
        ``init_params`` is a hard partition that fixes the start.
    out_dir : str
        Directory under which CSVs are written. Filenames are
        ``<name>_<tag>_<field>.csv`` where ``<tag>`` reflects any
        forced ``d_k`` (e.g. ``synth_lowdim_AEE_d2_bic.csv``).
    """
    code, signal_dim, tag = _resolve_model_spec(spec)
    out_prefix = os.path.join(out_dir, f"{name}_{tag}_")

    if signal_dim is None:
        print(f"PY : {name:14s}  {tag}")
    else:
        print(f"PY : {name:14s}  {tag}  (signal_dim={signal_dim})")

    # A single EM pass from the shared init — exactly what the R side
    # also does — so the two implementations are placed on equal
    # footing for the parity diff. ``signal_dim`` is honoured only on
    # E-suffix sub-models (codes ending in E); for V-suffix codes the
    # local HDDC silently ignores it, so we never reach this branch
    # for an AVV/IVV/UVV spec.
    kwargs = dict(
        n_components=K,
        model=code,
        cattell_threshold=cattell_threshold,
        init_params=init,
        n_init=1,
        max_iter=200,
        tol=1e-3,
        random_state=random_state,
    )
    if signal_dim is not None:
        kwargs["signal_dim"] = signal_dim

    try:
        fit = HighDimensionalGaussianMixture(**kwargs).fit(X)
    except Exception as exc:  # noqa: BLE001
        # We log+skip rather than raise so a single ill-conditioned
        # sub-model doesn't block the whole sweep.
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return

    # --- Scalars ---------------------------------------------------------
    _save(out_prefix + "bic.csv", [fit.bic(X)])
    _save(out_prefix + "icl.csv", [_icl_of(fit, X)])
    _save(out_prefix + "loglik.csv", [fit.lower_bound_])
    _save(out_prefix + "n_parameters.csv", [fit._n_parameters()], fmt="%d")

    # --- Per-cluster scalars --------------------------------------------
    _save(out_prefix + "weights.csv", fit.weights_)
    _save(
        out_prefix + "signal_dims.csv",
        np.asarray(fit.signal_dims_, dtype=int),
        fmt="%d",
    )
    _save(out_prefix + "noise_variances.csv", fit.noise_variances_)

    # --- Means -----------------------------------------------------------
    _save(out_prefix + "means.csv", fit.means_)

    # --- Eigenvalues padded to (K, p) -----------------------------------
    # ``eigenvalues_`` is a list of per-cluster arrays of length ``p``,
    # but to mirror the R-side fixed shape we always write a K x p
    # matrix.
    p = X.shape[1]
    eigvals = np.zeros((K, p))
    for k, ev in enumerate(fit.eigenvalues_):
        eigvals[k, : len(ev)] = ev
    _save(out_prefix + "eigenvalues.csv", eigvals)

    # --- Per-cluster orientation (signal columns only) ------------------
    # We slice the first ``d_k`` columns so each Q_<k>.csv exactly
    # matches the R-side ``fit$Q[[k]]`` shape.
    for k, Q in enumerate(fit.eigenvectors_):
        dk = int(fit.signal_dims_[k])
        _save(out_prefix + f"Q_{k}.csv", Q[:, :dk])

    # --- Labels and responsibilities ------------------------------------
    _save(out_prefix + "labels.csv", fit.labels_.astype(int), fmt="%d")
    # Recompute responsibilities via predict_proba: HDDC follows
    # GaussianMixture's convention of not storing them on the fit.
    _save(out_prefix + "responsibilities.csv", fit.predict_proba(X))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Discover datasets in ``data/`` and fit every (dataset, model) pair.

    Dataset discovery is filename-driven (any ``<name>_meta.json`` under
    ``data/``), so adding a dataset to ``01_prepare_data.py``
    auto-wires it here with no edits.
    """
    datasets = sorted(
        f[: -len("_meta.json")]
        for f in os.listdir(DATA)
        if f.endswith("_meta.json")
    )

    for name in datasets:
        with open(os.path.join(DATA, f"{name}_meta.json")) as fp:
            meta = json.load(fp)

        X = np.loadtxt(os.path.join(DATA, f"{name}_X.csv"), delimiter=",")
        init = np.loadtxt(
            os.path.join(DATA, f"{name}_labels_init.csv"),
            delimiter=",",
            dtype=int,
        )

        K: int = int(meta["K"])
        thr: float = float(meta["cattell_threshold"])
        seed: int = int(meta["seed"])
        # `models` can mix plain strings and dicts; both are JSON-valid.
        model_specs: Sequence[ModelSpec] = meta["models"]

        for spec in model_specs:
            fit_one(
                name=name,
                spec=spec,
                X=X,
                init=init,
                K=K,
                cattell_threshold=thr,
                random_state=seed,
                out_dir=OUT,
            )

    print("Python parity dump complete ->", OUT)


if __name__ == "__main__":
    main()
