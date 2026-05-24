"""Auto-select the best mixture (family + K) by ICL.

The ICL of any mixture model (Gaussian or HDDC) is
``ICL = BIC + 2H`` in sklearn's lower-is-better convention. Both
terms are well-defined for any mixture:

* ``BIC = -2 log L + ν(K, family) · log n`` uses the model's
  family-specific parameter count.
* ``H = -Σ τ log τ`` uses the posterior responsibilities and is
  family-agnostic.

So **argmin_{K, family} ICL(K, family)** is a principled head-to-head
selector across model structures *and* number of clusters in one
go. This module exposes that exhaustive search.

Usage
-----
>>> from auto_mixture import auto_select_mixture
>>> result = auto_select_mixture(X, K_grid=range(2, 16))
>>> print(result.family, result.K, result.icl)
>>> labels = result.fit.predict(X)

By default, the search covers all four ``GaussianMixture``
covariance types (``spherical``, ``diag``, ``tied``, ``full``) and
all 14 HDDC sub-models (``AVV, AEV, IVV, IEV, UVV, UEV, AVE, CVE,
AEE, CEE, IVE, IEE, UVE, UEE``), so 18 families × |K_grid| fits.

The candidate families can be restricted via the ``gmm_families``
and ``hddc_models`` parameters — e.g. only HDDC, or only diagonal
GMM, or only the four ``A**`` HDDC variants.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import logging

log = logging.getLogger("auto_mixture")



# Load the local HDDC class without going through ``sklearn.mixture``
# (which would require the PR to have landed). The figures and demos
# do this same trick.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB = os.path.dirname(_HERE)
_spec = importlib.util.spec_from_file_location(
    "_auto_mixture_hddc",
    os.path.join(_CONTRIB, "pr_hddc", "_hddc.py"),
)
_hddc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hddc_mod)
HighDimensionalGaussianMixture = _hddc_mod.HighDimensionalGaussianMixture


GMM_COVARIANCE_TYPES = ("spherical", "diag", "tied", "full")
HDDC_MODELS = (
    "AVV", "AEV", "IVV", "IEV", "UVV", "UEV",
    "AVE", "CVE", "AEE", "CEE", "IVE", "IEE", "UVE", "UEE",
)


# Shared compat shim lives next to the figures helpers; same six
# lines of code, one source of truth. Delete the import + alias once
# the ``pr_gmm_icl`` PR lands and ``gmm.icl(X)`` is upstream.
sys.path.insert(0, os.path.join(_CONTRIB, "figures"))
from _icl_compat import icl_gmm as _icl_gmm  # noqa: E402


def _icl_of(fit, X: np.ndarray) -> float:
    return float(fit.icl(X)) if hasattr(fit, "icl") else _icl_gmm(fit, X)


SUPPORTED_CRITERIA = ("icl", "bic")


def _score_of(fit, X: np.ndarray, criterion: str) -> float:
    """Compute the selection score for any GMM- or HDDC-shaped fit."""
    if criterion == "icl":
        return _icl_of(fit, X)
    if criterion == "bic":
        return float(fit.bic(X))
    raise ValueError(
        f"unknown criterion {criterion!r}; expected one of "
        f"{SUPPORTED_CRITERIA}"
    )


@dataclass
class MixtureSelectionResult:
    """Result of an exhaustive criterion-driven mixture search."""
    family: str
    K: int
    score: float                           # value of the winning criterion
    criterion: str                         # "icl" or "bic"
    fit: object
    n_fits: int                            # how many (K, family) pairs tried
    elapsed: float                         # wall clock in seconds
    score_grid: dict = field(default_factory=dict)
    """``score_grid[(family, K)] = float`` for every successful fit."""

    @property
    def icl(self) -> float:
        """Back-compat alias for ``score`` when criterion=='icl'."""
        return self.score

    @property
    def icl_grid(self) -> dict:
        """Back-compat alias for ``score_grid``."""
        return self.score_grid


def auto_select_mixture(
    X: np.ndarray,
    K_grid: Iterable[int] | None = None,
    *,
    criterion: str = "icl",
    n_init: int = 5,
    max_iter: int = 200,
    cattell_threshold: float = 0.5,
    reg_covar_gmm: float | dict[str, float] | None = None,
    gmm_families: Sequence[str] | None = GMM_COVARIANCE_TYPES,
    hddc_models: Sequence[str] | None = HDDC_MODELS,
    random_state: int = 0,
    verbose: bool = False,
) -> MixtureSelectionResult:
    """Pick the (family, K) with the lowest ICL on ``X``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    K_grid : iterable of int or None, default=None
        Candidate numbers of components. If ``None``, the grid is
        estimated automatically via
        :func:`estimate_k_range.estimate_k_range` (k-means++ elbow
        rule) — typically a fast sub-second call.
    n_init : int, default=5
        Passed to every mixture estimator. For both GMM and HDDC,
        this is the kmeans++ exploration budget on the
        cluster-assignment space (see `pr_hddc/_hddc.py`).
    max_iter : int, default=200
        EM iteration cap.
    cattell_threshold : float, default=0.5
        ``cattell_threshold`` for every HDDC fit.
    reg_covar_gmm : float, dict, or None, default=None
        Regularisation for the GMM family. ``None`` uses sane
        per-covariance-type defaults (``1e-1`` for ``full``,
        ``1e-3`` otherwise). A scalar applies to all GMM types.
        A dict allows per-type overrides.
    gmm_families : sequence of str or None
        Subset of ``GMM_COVARIANCE_TYPES`` to include. ``None``
        disables GMM entirely.
    hddc_models : sequence of str or None
        Subset of ``HDDC_MODELS`` to include. ``None`` disables HDDC
        entirely.
    random_state : int, default=0
    verbose : bool, default=False
        If True, log every fit as it finishes.

    Returns
    -------
    MixtureSelectionResult
    """
    if K_grid is None:
        # ``estimate_k_range`` lives next to this module in ``tools/``.
        # Make sure it is importable even when ``auto_mixture`` is
        # used as a library from outside ``tools/``.
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from estimate_k_range import estimate_k_range  # noqa: E402
        est = estimate_k_range(X, random_state=random_state)
        K_grid = list(range(est.K_min, est.K_max + 1))
    else:
        K_grid = list(K_grid)
    if not K_grid:
        raise ValueError("K_grid is empty.")
    if not gmm_families and not hddc_models:
        raise ValueError("Need at least one of gmm_families / hddc_models.")
    if criterion not in SUPPORTED_CRITERIA:
        raise ValueError(
            f"unknown criterion {criterion!r}; expected one of "
            f"{SUPPORTED_CRITERIA}"
        )

    def _gmm_reg_for(cov: str) -> float:
        if reg_covar_gmm is None:
            return 1e-1 if cov == "full" else 1e-3
        if isinstance(reg_covar_gmm, dict):
            return float(reg_covar_gmm.get(
                cov, 1e-1 if cov == "full" else 1e-3
            ))
        return float(reg_covar_gmm)

    # Build the per-family "fit given a shared KMeans init" closures.
    # We pass the kmeans centroids to GMM via ``means_init`` (which
    # short-circuits sklearn's internal kmeans pass) and the kmeans
    # labels to HDDC via ``init_params=labels`` (which skips HDDC's
    # internal kmeans pass). That way KMeans runs once per K, not
    # once per (K, family).
    families: list[tuple[str, Callable[[int, np.ndarray, np.ndarray], object]]] = []

    if gmm_families:
        for cov in gmm_families:
            reg = _gmm_reg_for(cov)
            def _make_gmm(K, labels, centroids, cov=cov, reg=reg):
                return GaussianMixture(
                    n_components=K, covariance_type=cov,
                    reg_covar=reg, random_state=random_state,
                    n_init=1,
                    means_init=centroids,
                    max_iter=max_iter,
                ).fit(X)
            families.append((f"GMM ({cov})", _make_gmm))

    if hddc_models:
        for model in hddc_models:
            def _make_hddc(K, labels, centroids, model=model):
                return HighDimensionalGaussianMixture(
                    n_components=K, model=model,
                    cattell_threshold=cattell_threshold,
                    init_params=labels,
                    n_init=1,
                    max_iter=max_iter,
                    random_state=random_state,
                ).fit(X)
            families.append((f"HDDC ({model})", _make_hddc))

    best_family, best_K, best_score, best_fit = None, None, np.inf, None
    score_grid: dict[tuple[str, int], float] = {}
    n_fits = 0
    t0 = time.time()

    # Outer loop: KMeans once per K, shared across all 18 families.
    for K in K_grid:
        try:
            km = KMeans(
                n_clusters=K, init="k-means++",
                n_init=max(n_init, 1), random_state=random_state,
            ).fit(X)
        except Exception as exc:                    # noqa: BLE001
            if verbose:
                log.info(f"  KMeans K={K} FAILED: "
                      f"{type(exc).__name__}: {exc}")
            continue
        labels = km.labels_.astype(np.int64)
        centroids = km.cluster_centers_

        # Inner loop: every (family) fits *from this shared init*.
        for family, make in families:
            try:
                fit = make(K, labels, centroids)
            except Exception as exc:                # noqa: BLE001
                if verbose:
                    log.info(f"  {family} K={K} FAILED: "
                          f"{type(exc).__name__}: {exc}")
                continue
            score = _score_of(fit, X, criterion)
            score_grid[(family, K)] = score
            n_fits += 1
            if verbose:
                marker = " *" if score < best_score else ""
                log.info(f"  {family:18s} K={K:3d}  "
                      f"{criterion.upper()}={score:.2f}{marker}")
            if score < best_score:
                best_family, best_K, best_score, best_fit = (
                    family, K, score, fit
                )

    if best_fit is None:
        raise RuntimeError(
            "Every (family, K) attempt failed. Inspect data / increase "
            "regularisation."
        )

    return MixtureSelectionResult(
        family=best_family,
        K=best_K,
        score=best_score,
        criterion=criterion,
        fit=best_fit,
        n_fits=n_fits,
        elapsed=time.time() - t0,
        score_grid=score_grid,
    )


# ---------------------------------------------------------------------------
# Standalone CLI for quick experimentation.
# ---------------------------------------------------------------------------
def _cli(argv: Sequence[str] | None = None) -> int:
    """``python tools/auto_mixture.py [dataset|--input X.npz]``.

    Two modes:

    - **Train**: search for the best (family, K) on a dataset and
      optionally pickle the result.
      ``python auto_mixture.py digits --output best.pkl``
    - **Predict**: load a previously-pickled result and emit
      ``labels`` / ``proba`` on new data.
      ``python auto_mixture.py --load best.pkl --input new_X.npz
      --output preds.npz``
    """
    import argparse
    import pickle
    parser = argparse.ArgumentParser(
        prog="auto_mixture",
        description=(
            "Argmin-{ICL,BIC} mixture (family, K) over all "
            "GaussianMixture covariance types and HDDC sub-models. "
            "Also supports inference from a pickled result."
        ),
    )
    parser.add_argument(
        "dataset",
        choices=("digits", "iris", "wine", "olivetti"),
        nargs="?",
        help="Built-in sklearn dataset to search over (mutually "
             "exclusive with --input).",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a .npz file containing an 'X' array (and "
             "optionally a 'y' array). Overrides 'dataset' if given.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Train mode: path to write the full "
             "MixtureSelectionResult as a .pkl. "
             "Predict mode: path to write labels + proba as a .npz.",
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Inference mode. Path to a .pkl produced by an earlier "
             "training run (or any object with a fitted .predict / "
             ".predict_proba). Skips the search and emits predictions.",
    )
    parser.add_argument(
        "--criterion", choices=SUPPORTED_CRITERIA, default="icl",
        help="Selection criterion. Default: icl.",
    )
    parser.add_argument(
        "--k-min", type=int, default=None,
        help="Lower bound of K_grid (inclusive). If neither --k-min "
             "nor --k-max is given, the grid is estimated from X via "
             "the k-means++ elbow rule in estimate_k_range.py.",
    )
    parser.add_argument(
        "--k-max", type=int, default=None,
        help="Upper bound of K_grid (inclusive).",
    )
    parser.add_argument(
        "--n-init", type=int, default=5,
        help="kmeans++ exploration budget per K.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=200,
        help="EM iteration cap per fit.",
    )
    parser.add_argument(
        "--top", type=int, default=8,
        help="How many top-scoring configurations to print.",
    )
    parser.add_argument(
        "--gmm-only", action="store_true",
        help="Restrict the search to the four GMM covariance types.",
    )
    parser.add_argument(
        "--hddc-only", action="store_true",
        help="Restrict the search to the 14 HDDC sub-models.",
    )
    args = parser.parse_args(argv)

    if args.input:
        npz = np.load(args.input, allow_pickle=True)
        if "X" in npz.files:
            X = np.asarray(npz["X"], dtype=float)
        elif len(npz.files) == 1:
            X = np.asarray(npz[npz.files[0]], dtype=float)
        else:
            raise SystemExit(
                f"--input {args.input}: expected an 'X' key, got "
                f"{list(npz.files)}"
            )
        y = (np.asarray(npz["y"])
             if "y" in npz.files else None)
        K_true = (len(np.unique(y)) if y is not None else None)
        label = f"npz:{os.path.basename(args.input)}"
    else:
        ds_name = args.dataset or "digits"
        from sklearn import datasets as _ds
        loaders = {
            "digits":   lambda: _ds.load_digits(return_X_y=True),
            "iris":     lambda: _ds.load_iris(return_X_y=True),
            "wine":     lambda: _ds.load_wine(return_X_y=True),
            "olivetti": lambda: (
                (lambda d: (d.data, d.target))(_ds.fetch_olivetti_faces())
            ),
        }
        X, y = loaders[ds_name]()
        K_true = len(np.unique(y))
        label = ds_name

    k_true_str = f"K_true={K_true}" if K_true is not None else "K_true=?"
    log.info(f"dataset={label}  n={X.shape[0]}  p={X.shape[1]}  {k_true_str}")

    # -------- Inference mode -------------------------------------------------
    if args.load:
        with open(args.load, "rb") as fh:
            loaded = pickle.load(fh)
        fit = loaded.fit if hasattr(loaded, "fit") else loaded
        family = getattr(loaded, "family", "(unknown)")
        K = getattr(loaded, "K", "(unknown)")
        log.info(f"\nloaded {args.load}: family={family}  K={K}")

        labels = fit.predict(X)
        proba = (fit.predict_proba(X)
                 if hasattr(fit, "predict_proba") else None)

        # Quick report.
        from collections import Counter
        sizes = Counter(int(c) for c in labels)
        log.info(f"predicted clusters: K_used={len(sizes)}  "
              f"sizes={sorted(sizes.values(), reverse=True)}")

        if args.output:
            out = {"labels": labels}
            if proba is not None:
                out["proba"] = proba
            np.savez_compressed(args.output, **out)
            log.info(f"wrote {args.output}  "
                  f"({os.path.getsize(args.output) / 1e6:.2f} MB)")
        return 0

    # -------- Train mode -----------------------------------------------------
    if args.k_min is None and args.k_max is None:
        # Auto K_grid via k-means++ elbow. Pass None to the helper;
        # it will estimate and log the chosen [K_min, K_max].
        K_grid = None
        log.info(f"criterion={args.criterion.upper()}  "
                 f"K_grid=auto  n_init={args.n_init}")
    else:
        k_min = 2 if args.k_min is None else args.k_min
        k_max = 30 if args.k_max is None else args.k_max
        K_grid = range(k_min, k_max + 1)
        log.info(f"criterion={args.criterion.upper()}  "
                 f"K_grid=[{k_min}..{k_max}]  n_init={args.n_init}")

    gmm_families = None if args.hddc_only else GMM_COVARIANCE_TYPES
    hddc_models = None if args.gmm_only else HDDC_MODELS

    result = auto_select_mixture(
        X,
        K_grid=K_grid,
        criterion=args.criterion,
        n_init=args.n_init,
        max_iter=args.max_iter,
        gmm_families=gmm_families,
        hddc_models=hddc_models,
    )
    crit = args.criterion.upper()
    # Surface the K range that was actually searched (estimate_k_range
    # already logs its inference line, but echoing the final span makes
    # the train-run output self-contained).
    Ks_searched = sorted({K for (_, K) in result.score_grid})
    log.info(f"\nbest: {result.family}  K={result.K}  "
          f"{crit}={result.score:.2f}")
    log.info(f"K_grid searched: [{Ks_searched[0]}..{Ks_searched[-1]}]  "
             f"({len(Ks_searched)} values)")
    log.info(f"n_fits={result.n_fits}  elapsed={result.elapsed:.1f}s")
    ranked = sorted(result.score_grid.items(), key=lambda kv: kv[1])
    log.info(f"\ntop {args.top} by {crit}:")
    for (family, K), s in ranked[: args.top]:
        log.info(f"  {family:18s} K={K:3d}  {crit}={s:.2f}")

    if args.output:
        with open(args.output, "wb") as fh:
            pickle.dump(result, fh)
        log.info(f"\nwrote {args.output}  "
              f"({os.path.getsize(args.output) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(_cli())
