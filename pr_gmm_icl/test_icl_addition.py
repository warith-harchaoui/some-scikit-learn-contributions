"""Tests to add to ``sklearn/mixture/tests/test_gaussian_mixture.py``.

Five lightweight, deterministic tests that pin the public contract
of the new ``icl`` method on ``GaussianMixture``. Heavier scientific
demonstrations (Student-t mixture sweep, held-out vs IC, galaxies)
live outside CI under ``examples/`` — the CI suite only checks
implementation invariants.

The tests below:

1. ``test_gaussian_mixture_icl`` — ICL = BIC + 2H identity and
   ICL ≥ BIC, across all four covariance types.
2. ``test_gaussian_mixture_icl_not_fitted_raises`` — calling icl()
   before fit raises NotFittedError, matching bic().
3. ``test_gaussian_mixture_icl_equals_bic_on_hard_partition`` —
   entropy term vanishes on a well-separated mixture.
4. ``test_gaussian_mixture_icl_equals_bic_at_K1`` — K=1 corner case:
   responsibilities are identically 1, entropy is exactly 0.
5. ``test_gaussian_mixture_icl_no_runtime_warning_on_hard_partition``
   — ``xlogy`` handles 0·log(0) cleanly with no warning.
"""

import warnings

import numpy as np
import pytest

from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture

COVARIANCE_TYPE = ("full", "tied", "diag", "spherical")


def test_gaussian_mixture_icl():
    """ICL satisfies its defining identities across all covariance types."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    for cv_type in COVARIANCE_TYPE:
        gmm = GaussianMixture(
            n_components=2, covariance_type=cv_type, random_state=0,
        ).fit(X)
        bic = gmm.bic(X)
        icl = gmm.icl(X)
        # ICL = BIC + 2H with H >= 0 -> ICL >= BIC.
        assert icl >= bic - 1e-8
        # Reconstruct H and check the identity directly.
        _, log_resp = gmm._estimate_log_prob_resp(X)
        resp = np.exp(log_resp)
        entropy = -np.nansum(resp * log_resp)
        assert np.isclose(icl, bic + 2.0 * entropy)


def test_gaussian_mixture_icl_not_fitted_raises():
    """Calling icl() before fit raises NotFittedError (matches bic)."""
    gmm = GaussianMixture(n_components=2)
    X = np.random.RandomState(0).randn(10, 2)
    with pytest.raises(NotFittedError):
        gmm.icl(X)


def test_gaussian_mixture_icl_equals_bic_on_hard_partition():
    """When responsibilities collapse to a hard partition, ICL == BIC."""
    rng = np.random.RandomState(0)
    centers = np.array([[-50.0, 0.0], [0.0, 50.0], [50.0, -50.0]])
    X = np.vstack([c + rng.randn(60, 2) for c in centers])
    gmm = GaussianMixture(
        n_components=3, covariance_type="full", random_state=0, tol=1e-7,
    ).fit(X)
    assert np.isclose(gmm.icl(X), gmm.bic(X), rtol=0, atol=1e-3)


def test_gaussian_mixture_icl_equals_bic_at_K1():
    """At K=1, responsibilities are identically 1 -> entropy = 0 -> ICL = BIC."""
    rng = np.random.RandomState(0)
    X = rng.randn(80, 3)
    gmm = GaussianMixture(
        n_components=1, covariance_type="full", random_state=0,
    ).fit(X)
    # Bit-equivalent: entropy is mathematically zero, not just small.
    assert gmm.icl(X) == gmm.bic(X)


def test_gaussian_mixture_icl_no_runtime_warning_on_hard_partition():
    """``icl`` on a near-hard partition must not emit RuntimeWarning.

    Pins the ``xlogy`` design choice — switching to
    ``resp * np.log(resp)`` with an ``eps`` clip would silently
    re-introduce the warning.
    """
    rng = np.random.RandomState(0)
    centers = np.array([[-50.0, 0.0], [0.0, 50.0], [50.0, -50.0]])
    X = np.vstack([c + rng.randn(60, 2) for c in centers])
    gmm = GaussianMixture(
        n_components=3, covariance_type="full", random_state=0, tol=1e-7,
    ).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        gmm.icl(X)
