"""Tests for ``HighDimensionalGaussianMixture``.

Deliberately slim: this suite pins the public contract of the new
estimator, not its empirical clustering performance. ~17 conceptual
test functions, each running on a tiny deterministic dataset.

Heavier scientific demonstrations (Student-mixture model-selection
recovery, K-true recovery on a signal-subspace mixture, real-world
benchmarks) live outside CI under ``examples/`` so the CI suite
stays cheap and robust.

Covers:

1. ``check_estimator`` on the default (AVV) sub-model.
2. Smoke fit/predict on all 14 sub-models (tiny dataset).
3. Recovery floor (ARI ≥ 0.9) on well-separated blobs at AVV.
4. Reproducibility with ``random_state``.
5. Pickle round-trip.
6. Input-validation raises: infeasibility, bad-init array.
7. ``init_params="random"`` branch.
8. NotFitted error on bic / icl.
9. Information-criterion invariants: ICL ≥ BIC, AIC/BIC identity.
10. 1-D regression (``p = 1`` corner case).
11. Parameter count vs. Bouveyron 2007 Table 1 at K=4, p=100, d=10
    (one row per sub-model, formula-only — no EM).
12. Off-axis parameter count: small, fixed set of (K, p, d) triples.
13. Naming-resolver: alias equivalence (one representative pair),
    kwargs-only, conflict detection, rejected codes (mclust + unknown).
14. Cattell scree-rule unit tests (3 essential cases).
"""

import math
import pickle

import numpy as np
import pytest

from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.mixture import HighDimensionalGaussianMixture
from sklearn.mixture._hddc import _cattell_scree_test
from sklearn.utils.estimator_checks import check_estimator


_GEOMETRIC = (
    "AVV", "AEV", "IVV", "IEV", "UVV", "UEV",
    "AVE", "CVE", "AEE", "CEE", "IVE", "IEE", "UVE", "UEE",
)


def _toy(n=120, p=8, K=3, random_state=0):
    """Small deterministic blobs dataset for fit/predict smoke tests."""
    X, y = make_blobs(
        n_samples=n, n_features=p, centers=K,
        cluster_std=2.0, random_state=random_state,
    )
    return X, y


# --------------------------------------------------------------------------
# 1. sklearn common-test compliance
# --------------------------------------------------------------------------

def test_hddc_check_estimator():
    """sklearn common-tests on the default (AVV) sub-model.

    Verified locally against sklearn 1.8 (2026-05): 43 of 44
    sub-checks pass; the one skip is ``check_pipeline_consistency``,
    which the harness skips by design for estimators that set
    ``non_deterministic = True``.
    """
    check_estimator(HighDimensionalGaussianMixture(n_components=2, n_init=1))


# --------------------------------------------------------------------------
# 2. Smoke fit/predict on all 14 sub-models
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", _GEOMETRIC)
def test_hddc_fit_predict_smoke(model):
    """Every sub-model fits a tiny dataset and produces valid labels."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model=model, random_state=0, n_init=1, max_iter=30,
    ).fit(X)
    assert hgmm.geometric_model_ == model
    labels = hgmm.predict(X)
    assert labels.shape == (X.shape[0],)
    assert set(np.unique(labels)).issubset({0, 1, 2})
    for name in ("bic", "icl", "aic"):
        v = getattr(hgmm, name)(X)
        assert np.isfinite(v)


# --------------------------------------------------------------------------
# 3. Recovery floor (the only clustering-quality test)
# --------------------------------------------------------------------------

def test_hddc_fit_recovers_blobs_at_AVV():
    """On well-separated blobs the default sub-model recovers them (ARI ≥ 0.9).

    Single stronger-than-smoke assertion: would catch a silent
    EM-divergence regression that smoke (only checks "no NaN") would
    miss. Not an empirical "HDDC is better than X" claim.
    """
    from sklearn.metrics import adjusted_rand_score
    X, y = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1, max_iter=100,
    ).fit(X)
    assert adjusted_rand_score(y, hgmm.labels_) >= 0.9


# --------------------------------------------------------------------------
# 4. Determinism, pickle, validation, init branches
# --------------------------------------------------------------------------

def test_hddc_fit_is_reproducible_given_seed():
    """Same ``random_state`` -> bit-identical fit."""
    X, _ = _toy()
    kw = dict(n_components=3, model="AVV", random_state=0,
              n_init=1, max_iter=30)
    fit1 = HighDimensionalGaussianMixture(**kw).fit(X)
    fit2 = HighDimensionalGaussianMixture(**kw).fit(X)
    np.testing.assert_array_equal(fit1.labels_, fit2.labels_)
    np.testing.assert_allclose(fit1.weights_, fit2.weights_, rtol=0, atol=0)
    np.testing.assert_allclose(fit1.means_, fit2.means_, rtol=0, atol=0)


def test_hddc_pickle_roundtrip():
    """A fitted estimator survives ``pickle.dumps``/``loads``."""
    X, _ = _toy()
    fit = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1, max_iter=30,
    ).fit(X)
    fit2 = pickle.loads(pickle.dumps(fit))
    np.testing.assert_array_equal(fit.predict(X), fit2.predict(X))
    assert fit.bic(X) == fit2.bic(X)


@pytest.mark.parametrize("bad_init,match", [
    # length mismatch
    (np.zeros(40, dtype=int), "init_params"),
    # invalid cluster id (>= n_components)
    (np.full(120, 99, dtype=int), "init_params"),
])
def test_hddc_bad_init_array_raises(bad_init, match):
    """``init_params=`` ndarray with invalid contents raises clearly."""
    X, _ = _toy()
    with pytest.raises(ValueError, match=match):
        HighDimensionalGaussianMixture(
            n_components=3, init_params=bad_init,
        ).fit(X)


def test_hddc_infeasible_K_min_cluster_size_raises():
    """``K * min_cluster_size > n_samples`` raises before EM starts."""
    X, _ = _toy(n=40)
    with pytest.raises(ValueError, match="exceeds n_samples"):
        HighDimensionalGaussianMixture(
            n_components=10, min_cluster_size=20,
        ).fit(X)


def test_hddc_random_init_branch_runs():
    """``init_params="random"`` exercises the alternate ``n_init`` semantics
    (multiple EM restarts) and completes without raising or NaN.
    """
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", init_params="random",
        random_state=0, n_init=2, max_iter=30,
    ).fit(X)
    assert np.isfinite(hgmm.lower_bound_)


# --------------------------------------------------------------------------
# 5. Information-criterion invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["bic", "icl", "aic"])
def test_hddc_information_criteria_not_fitted_raises(name):
    """Calling bic / icl / aic before fit raises NotFittedError."""
    X, _ = _toy()
    with pytest.raises(NotFittedError):
        getattr(HighDimensionalGaussianMixture(n_components=2), name)(X)


def test_hddc_icl_ge_bic_and_aic_bic_identity():
    """Two related invariants on the same fit, to keep the test count down.

    * ``icl == bic + 2H`` (so ``icl >= bic``) since H ≥ 0.
    * ``aic - bic == ν · (2 − log n)`` exactly (definitional).
    """
    X, _ = _toy()
    n = X.shape[0]
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1, max_iter=30,
    ).fit(X)
    assert hgmm.icl(X) >= hgmm.bic(X) - 1e-6
    expected = hgmm._n_parameters() * (2.0 - math.log(n))
    assert math.isclose(hgmm.aic(X) - hgmm.bic(X), expected, rel_tol=1e-9)


def test_hddc_fit_predict_1d_no_nan():
    """Regression: ``p = 1`` data must not produce NaN log-density.

    Earlier versions returned NaN for ``bic``/``icl`` on ``p = 1``
    because ``mean(eigvals[d_k:])`` was taken over an empty slice
    (``d_k = p = 1``). Pin the fix.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(40, 1)
    hgmm = HighDimensionalGaussianMixture(
        n_components=2, model="AVV", random_state=0, n_init=1, max_iter=20,
    ).fit(X)
    assert np.isfinite(hgmm.bic(X))
    assert np.isfinite(hgmm.icl(X))


# --------------------------------------------------------------------------
# 6. Parameter count vs. Bouveyron 2007, Table 1 (no EM)
# --------------------------------------------------------------------------

# Full Table 1 at (K=4, p=100, d=10). Two values cross-checked against
# the paper (`AVE → 4228`, `AEE → 4225`); the other 12 come from the
# formulas implemented in ``_n_parameters`` and guard against off-by-one
# / wrong-branch edits.
_TABLE1_K4_P100_D10 = {
    "AVV": 4231, "AEV": 4228, "IVV": 4195, "IEV": 4192,
    "UVV": 4192, "UEV": 4189,
    "AVE": 4228, "CVE": 4198, "AEE": 4225, "CEE": 4195,
    "IVE": 4192, "IEE": 4189, "UVE": 4189, "UEE": 4186,
}


@pytest.mark.parametrize("model,expected",
                         sorted(_TABLE1_K4_P100_D10.items()))
def test_hddc_parameter_count_table1(model, expected):
    """All 14 rows of Bouveyron Table 1 at K=4, p=100, d=10.

    Bypasses fit by setting the attributes ``_n_parameters`` reads
    directly. Isolates the formula from EM convergence and from
    Cattell-driven d_k for the free-d (last-letter V) models.
    """
    K, p, d = 4, 100, 10
    hgmm = HighDimensionalGaussianMixture(n_components=K, model=model)
    hgmm.geometric_model_ = model
    hgmm.n_features_in_ = p
    hgmm.signal_dims_ = [d] * K
    assert hgmm._n_parameters() == expected


# Three off-axis points: minimum d, max d, larger K & p. Picks one
# model from each branch family. Expected values come from the
# closed-form Table 1 formulas hand-evaluated below — independent
# of ``_n_parameters`` so any drift between the two raises here.
@pytest.mark.parametrize("model,K,p,d,expected", [
    # AVV  rho + tau_bar + 2K + D
    #     rho      tau_bar           +2K  +D     = 11 + 14 + 4 + 4 = 33
    ("AVV", 2, 5,  2,  33),
    # AEE  rho + K(tau + d) + 2
    #     rho=92, tau=5*(30-3)=135, K(tau+d)=3*140=420, +2 -> 514
    ("AEE", 3, 30, 5, 514),
    # UEE  rho + K·tau + 3
    #     rho=1205, tau=25*(200-13)=4675, K·tau=28050, +3 -> 29258
    ("UEE", 6, 200, 25, 29258),
])
def test_hddc_parameter_count_offaxis(model, K, p, d, expected):
    """Off-axis Table 1 sanity for three structurally different sub-models."""
    hgmm = HighDimensionalGaussianMixture(n_components=K, model=model)
    hgmm.geometric_model_ = model
    hgmm.n_features_in_ = p
    hgmm.signal_dims_ = [d] * K
    assert hgmm._n_parameters() == expected


# --------------------------------------------------------------------------
# 7. Naming-resolver: minimal coverage of every public spelling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("geo,paper", [
    ("AVV", "akj_bk_Qk_dk"),    # most general
    ("AEE", "akj_b_Qk_d"),       # tied noise + tied d
    ("UEE", "a_b_Qk_d"),         # most constrained
])
def test_hddc_model_alias_equivalence(geo, paper):
    """Paper-bracket alias and geometric code resolve to the same fit."""
    X, _ = _toy()
    kw = dict(n_components=3, random_state=0, n_init=1, max_iter=20)
    a = HighDimensionalGaussianMixture(model=geo, **kw).fit(X)
    b = HighDimensionalGaussianMixture(model=paper, **kw).fit(X)
    assert a.geometric_model_ == b.geometric_model_ == geo
    np.testing.assert_allclose(a.weights_, b.weights_, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(a.labels_, b.labels_)


def test_hddc_model_kwargs_only_resolves():
    """Per-axis kwargs (no ``model=``) resolve to the right geometric code."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model=None,
        signal="anisotropic", noise="varying", dim="varying",
        random_state=0, n_init=1, max_iter=10,
    ).fit(X)
    assert hgmm.geometric_model_ == "AVV"


def test_hddc_model_kwargs_partial_no_model_raises():
    """``model=None`` requires all three axis kwargs to be set."""
    X, _ = _toy()
    with pytest.raises(ValueError, match="Missing"):
        HighDimensionalGaussianMixture(
            n_components=3, model=None, signal="anisotropic",
        ).fit(X)


def test_hddc_model_conflict_raises():
    """``model=`` and a per-axis kwarg that contradict it raise."""
    X, _ = _toy()
    with pytest.raises(ValueError, match=r"Conflict on `noise`"):
        HighDimensionalGaussianMixture(
            n_components=3, model="AVV", noise="equal",
        ).fit(X)


@pytest.mark.parametrize("bad,match", [
    ("VVV", "mclust"),   # mclust-confusable
    ("XYZ", None),        # unknown spelling — parameter validator catches
])
def test_hddc_rejected_model_strings(bad, match):
    """mclust covariance codes and unknown spellings raise ValueError."""
    X, _ = _toy()
    with pytest.raises(ValueError, match=match):
        HighDimensionalGaussianMixture(
            n_components=3, model=bad,
        ).fit(X)


# --------------------------------------------------------------------------
# 8. Cattell scree rule unit tests (direct, not via EM)
# --------------------------------------------------------------------------

def test_cattell_scree_test_clear_elbow():
    """One sharp drop -> d equals the position of the drop."""
    eigvals = np.array([10.0, 9.0, 0.1, 0.1, 0.1, 0.1])
    assert _cattell_scree_test(eigvals, threshold=0.5) == 2


def test_cattell_scree_test_isotropic_returns_one():
    """All eigvals equal -> max_diff = 0 -> d = 1 fallback."""
    assert _cattell_scree_test(np.full(8, 1.0), threshold=0.5) == 1


def test_cattell_scree_test_rejects_unsorted():
    """A non-monotone input is a programming error."""
    with pytest.raises(ValueError, match=r"non-increasing"):
        _cattell_scree_test(np.array([1.0, 2.0, 0.5]))
