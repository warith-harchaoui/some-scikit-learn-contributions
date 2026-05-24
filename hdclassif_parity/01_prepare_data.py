"""Prepare datasets and shared KMeans initial labels for the HDDC parity check.

This is the first stage of the three-stage HDDC vs HDclassif parity
pipeline. Its job is to write **identical** data and **identical**
initial cluster assignments to disk in plain CSV so that both the R
side (``02_run_r_hdclassif.R``) and the Python side
(``03_run_python_hddc.py``) start from the exact same numerical
inputs. Anything those scripts do differently afterwards is then
attributable to the estimator implementation, not to the data.

Per-dataset artefacts under ``data/``:

* ``<name>_X.csv``           — the observations, shape (n, p), no header.
* ``<name>_labels_init.csv`` — 0-indexed KMeans cluster labels, length n.
                               Each script will adjust to its language's
                               indexing convention (R uses 1-indexing).
* ``<name>_meta.json``       — small JSON with ``n``, ``p``, ``K``,
                               ``cattell_threshold``, list of HDDC
                               sub-model codes to fit, and the RNG seed
                               so the run is reproducible.

The script writes five datasets:

* ``synth_lowdim``  — small synthetic mixture, easy regime.
* ``synth_highdim`` — synthetic ``n << p`` mixture, the regime HDDC
                       was designed for.
* ``iris``          — the classic 150x4 sanity check.
* ``digits``        — ``load_digits``: 1797 samples, 64 features.
* ``olivetti``      — first 10 people from ``fetch_olivetti_faces``,
                       **no PCA**: n=100, p=4096. Tests HDDC on its
                       intended ``n << p`` regime with real input.
                       PCA would test PCA+HDDC parity, not HDDC
                       parity, so we deliberately keep raw pixels.

Examples
--------
Run from the repo root:

    $ python hdclassif_parity/01_prepare_data.py
    Writing parity-check datasets to: .../hdclassif_parity/data
      synth_lowdim   n=600  p=5   K=3  ...
      ...
      olivetti       n=100  p=99  K=10 ...
    done.

Then run ``Rscript 02_run_r_hdclassif.R`` and the Python equivalent.

Notes
-----
The synthetic mixtures use independent RNG seeds per dataset so that
adding a dataset never perturbs another dataset's output. KMeans is
seeded so the shared init labels are deterministic across runs.
"""
from __future__ import annotations

# --- Standard library --------------------------------------------------------
import json
import os
from typing import Iterable, Optional, Sequence, Tuple

# --- Third-party -------------------------------------------------------------
import numpy as np
from numpy.random import Generator
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces, load_digits, load_iris


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# All paths are resolved relative to this file so the script works no
# matter what the caller's current working directory is.
HERE: str = os.path.dirname(os.path.abspath(__file__))
DATA: str = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

# Shared RNG seed for KMeans init across datasets. Each synthetic
# dataset additionally gets its own data-generation seed (see calls
# to ``_synth_mixture`` below) so that two datasets never share a
# random state by accident.
SEED: int = 0

# HDDC sub-model specs to fit per dataset. Each spec is either:
#
# * a plain three-letter geometric code (e.g. ``"AVV"``), which lets
#   the Cattell scree rule pick each ``d_k``;
# * a dict ``{"code": "AEE", "signal_dim": 2}`` for E-suffix models
#   that supports a forced common ``d``. On the R side this maps to
#   HDclassif's ``com_dim=`` argument; on the Python side it maps
#   to ``HighDimensionalGaussianMixture(signal_dim=...)``.
#
# Forcing ``d_k`` is the cleanest way to isolate the EM math from
# the scree-rule step — when forced-``d_k`` rows pass parity but
# Cattell-driven rows do not, the divergence sits in the
# dimension-selection rule, not the EM update.
DEFAULT_MODELS: Tuple = (
    "AVV",                                  # Cattell-driven, most general
    "AEE",                                  # Cattell-driven, equal noise+dim
    {"code": "AEE", "signal_dim": 1},       # forced d_k = 1
    {"code": "AEE", "signal_dim": 2},       # forced d_k = 2
)


def _model_tag(spec) -> str:
    """Canonical filename tag for a model spec.

    Plain string specs use the code as-is (``"AVV"`` -> ``"AVV"``).
    Dict specs append ``_d<n>`` (``{"code":"AEE","signal_dim":2}`` ->
    ``"AEE_d2"``).
    """
    if isinstance(spec, str):
        return spec
    return f"{spec['code']}_d{spec['signal_dim']}"

# ---------------------------------------------------------------------------
# Synthetic mixture generator
# ---------------------------------------------------------------------------


def _synth_mixture(
    n_per: int,
    p: int,
    K: int,
    signal_dim: int,
    snr: float = 4.0,
    rng: Optional[int | Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a low-rank-plus-noise Gaussian mixture in ``p`` dimensions.

    Each component has its own random orthonormal basis ``Q_k`` and
    eigenvalue spectrum: ``signal_dim`` "signal" eigenvalues sampled in
    [2, 5], and ``p - signal_dim`` "noise" eigenvalues fixed at 0.3.
    This is exactly the kind of structured covariance HDDC's
    ``[a_kj b_k Q_k d_k]`` family is designed to recover, so synthetic
    fits should be nearly identical between the R and Python
    implementations.

    Parameters
    ----------
    n_per : int
        Number of samples per cluster. Total dataset size is
        ``n_per * K``.
    p : int
        Ambient feature dimension.
    K : int
        Number of mixture components.
    signal_dim : int
        Intrinsic dimensionality (rank of the signal part of each
        component covariance). Must satisfy ``1 <= signal_dim < p``.
    snr : float, default=4.0
        Scale of the random means. Larger values make the mixture
        components more separable.
    rng : int, numpy.random.Generator, or None
        Seed or Generator. ``None`` uses seed 0 — be sure to vary it
        per dataset so two synthetic mixtures never share their RNG.

    Returns
    -------
    X : ndarray, shape (n_per * K, p)
        Shuffled observations.
    y : ndarray of int, shape (n_per * K,)
        Ground-truth cluster index per row. Useful for debugging,
        not used by the parity check itself.

    Examples
    --------
    >>> X, y = _synth_mixture(n_per=10, p=4, K=2, signal_dim=1, rng=42)
    >>> X.shape, y.shape
    ((20, 4), (20,))
    """
    rng = np.random.default_rng(0 if rng is None else rng)

    # Random component means; ``snr`` controls how far apart they are.
    means = rng.normal(scale=snr, size=(K, p))

    X = np.empty((n_per * K, p))
    y = np.empty(n_per * K, dtype=int)

    for k in range(K):
        # Random orthonormal basis for this component's covariance.
        Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
        # Signal block (top `signal_dim` eigenvalues) + isotropic noise tail.
        sig = np.diag(
            np.r_[
                rng.uniform(2.0, 5.0, size=signal_dim),
                np.full(p - signal_dim, 0.3),
            ]
        )
        cov = Q @ sig @ Q.T
        # Cholesky with a tiny jitter to stay PSD numerically.
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(p))
        X[k * n_per:(k + 1) * n_per] = (
            means[k] + rng.normal(size=(n_per, p)) @ L.T
        )
        y[k * n_per:(k + 1) * n_per] = k

    # Shuffle so KMeans cannot exploit the contiguous-class layout.
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------


def _write_dataset(
    name: str,
    X: np.ndarray,
    K: int,
    model_codes: Sequence[str] = DEFAULT_MODELS,
    cattell_threshold: float = 0.5,
    kmeans_n_init: int = 10,
) -> None:
    """Write ``X``, KMeans init labels, and meta.json for one dataset.

    Parameters
    ----------
    name : str
        Short identifier; becomes the file-name prefix.
    X : ndarray, shape (n, p)
        Observations.
    K : int
        Number of mixture components to fit.
    model_codes : sequence of str
        HDDC sub-model codes to evaluate on this dataset.
    cattell_threshold : float, default=0.5
        Threshold the Cattell scree rule uses to pick each cluster's
        signal dimension ``d_k``. Both the R and Python sides receive
        this same value; see ``docs/HDDC.md`` §3 for the rationale
        behind the default.
    kmeans_n_init : int, default=10
        Number of KMeans++ restarts used to produce the shared init
        labels. Best (by inertia) is kept.

    Notes
    -----
    The KMeans run here is *only* the initialiser. Both HDclassif's
    ``init = "vector"`` and our local HDDC's ``init_params=<labels>``
    accept a hard partition, then refine it with a single EM run
    starting from that partition.
    """
    # Observations: one row per sample, no header (R reads with header=FALSE).
    np.savetxt(
        os.path.join(DATA, f"{name}_X.csv"),
        X,
        delimiter=",",
        fmt="%.10g",
    )

    # Shared KMeans init. We export 0-indexed labels; the R side adds
    # 1 to match its convention.
    km = KMeans(n_clusters=K, n_init=kmeans_n_init, random_state=SEED).fit(X)
    np.savetxt(
        os.path.join(DATA, f"{name}_labels_init.csv"),
        km.labels_.astype(int),
        delimiter=",",
        fmt="%d",
    )

    # Small machine-readable side-car so both scripts see the same
    # K, threshold, sub-model list, and seed without parsing CSVs.
    meta = {
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "K": int(K),
        "cattell_threshold": float(cattell_threshold),
        "models": list(model_codes),
        "seed": SEED,
    }
    with open(os.path.join(DATA, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  {name:14s}  n={X.shape[0]:>4d}  p={X.shape[1]:>3d}  "
        f"K={K:>2d}  models={list(model_codes)}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _load_olivetti_raw(n_people: int = 10) -> np.ndarray:
    """Return the raw 4096-dim Olivetti faces for the first ``n_people``.

    No PCA, no standardisation — just the 64×64 grayscale images
    flattened to ``p = 4096`` features. Each person contributes 10
    images, so ``n = 10 * n_people``.

    This is the regime HDDC was built for (``n << p``); the parity
    check intentionally tests HDDC on raw input, since any
    pre-projection would conflate PCA and HDDC in the diff.

    Parameters
    ----------
    n_people : int, default=10
        Number of distinct individuals to retain.

    Returns
    -------
    X : ndarray, shape (n_people * 10, 4096)
        Raw face pixels in [0, 1].
    """
    faces = fetch_olivetti_faces()
    X_all, y_all = faces.data, faces.target
    mask = y_all < n_people
    return X_all[mask].astype(float)


def main() -> None:
    """Write all five parity datasets to ``data/``.

    Two synthetic mixtures (easy + ``n << p``), iris, digits, and
    raw Olivetti faces (no PCA). Together they cover the regimes
    the PR's empirical arguments lean on, with no preprocessing
    that would conflate two algorithms in the parity diff.
    """
    print("Writing parity-check datasets to:", DATA)

    # 1. Easy synthetic: low-dim, well-separated, few clusters.
    X, _ = _synth_mixture(n_per=200, p=5, K=3, signal_dim=2, rng=1)
    _write_dataset("synth_lowdim", X, K=3)

    # 2. High-dim synthetic: n < 10·p, structured low-rank covariance.
    #    This is the regime HDDC was designed for.
    X, _ = _synth_mixture(n_per=50, p=30, K=4, signal_dim=4, rng=2)
    _write_dataset("synth_highdim", X, K=4)

    # 3. Iris — small real benchmark, exhaustively studied.
    iris = load_iris()
    _write_dataset("iris", iris.data.astype(float), K=3)

    # 4. Digits — sklearn standard 1797x64 dataset, 10 classes.
    digits = load_digits()
    _write_dataset("digits", digits.data.astype(float), K=10)

    # 5. Raw Olivetti faces, 10 people, no PCA: n=100, p=4096.
    #    Tests HDDC in its intended n << p regime on real input.
    X_oliv = _load_olivetti_raw(n_people=10)
    _write_dataset("olivetti", X_oliv, K=10)

    print("done.")


if __name__ == "__main__":
    main()
