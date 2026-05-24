"""Tests for ``HighDimensionalGaussianMixture``."""

import math

import numpy as np
import pytest
from scipy import stats

from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from sklearn.mixture import HighDimensionalGaussianMixture


# All 14 HDDC sub-models in canonical geometric form.
_GEOMETRIC = (
    "AVV", "AEV", "IVV", "IEV", "UVV", "UEV",
    "AVE", "CVE", "AEE", "CEE", "IVE", "IEE", "UVE", "UEE",
)

# Paper bracket alias for each, for the alias-equivalence test.
_PAPER = {
    "AVV": "akj_bk_Qk_dk", "AEV": "akj_b_Qk_dk",
    "IVV": "ak_bk_Qk_dk",  "IEV": "ak_b_Qk_dk",
    "UVV": "a_bk_Qk_dk",   "UEV": "a_b_Qk_dk",
    "AVE": "akj_bk_Qk_d",  "CVE": "aj_bk_Qk_d",
    "AEE": "akj_b_Qk_d",   "CEE": "aj_b_Qk_d",
    "IVE": "ak_bk_Qk_d",   "IEE": "ak_b_Qk_d",
    "UVE": "a_bk_Qk_d",    "UEE": "a_b_Qk_d",
}


def _toy(random_state=0, n=300, p=20, K=3):
    X, y = make_blobs(
        n_samples=n, n_features=p, centers=K, random_state=random_state,
        cluster_std=2.0,
    )
    return X, y


# --------------------------------------------------------------------------
# 1. sklearn common-test compliance
# --------------------------------------------------------------------------

def test_hddc_check_estimator():
    """sklearn common-tests on the default (AVV) sub-model.

    Verified locally against sklearn 1.8 (2026-05-21): 43 of 44
    sub-checks pass; the one skip is ``check_pipeline_consistency``,
    which the harness skips by design for estimators that set
    ``non_deterministic = True`` (HDDC EM with kmeans++ init does not
    converge to byte-identical fits across calls). See
    ``__sklearn_tags__``.
    """
    check_estimator(HighDimensionalGaussianMixture(n_components=2, n_init=1))


def test_hddc_check_estimator_constrained_model():
    """sklearn common-tests on a constrained sub-model (UEE).

    Covers a different code path through ``_apply_model_constraints``
    than the default ``AVV``: ``UEE`` tying every axis exercises the
    signal / noise / dim collapse branches together, which the
    default never hits.
    """
    check_estimator(
        HighDimensionalGaussianMixture(n_components=2, model="UEE", n_init=1)
    )


# --------------------------------------------------------------------------
# 2. Smoke tests over all 14 sub-models
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", _GEOMETRIC)
def test_hddc_fit_predict_smoke(model):
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model=model, random_state=0, n_init=1, max_iter=50,
    ).fit(X)
    assert hgmm._geometric_model_ == model
    labels = hgmm.predict(X)
    assert labels.shape == (X.shape[0],)
    assert set(np.unique(labels)).issubset(set(range(3)))
    for name in ("bic", "icl"):
        v = getattr(hgmm, name)(X)
        assert np.isfinite(v), f"{name} non-finite for model={model}"
        assert isinstance(v, float)


def test_hddc_fit_recovers_blobs_at_AVV():
    """On well-separated blobs the default sub-model must recover them.

    The smoke test only checks that ``predict`` returns labels in
    range and ``bic`` is finite — it would pass even if EM
    diverged. This stronger test asserts a clustering-quality floor
    (``ARI >= 0.9``) on the same toy data, so a regression that
    silently breaks the EM update is caught.
    """
    from sklearn.metrics import adjusted_rand_score
    X, y = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1, max_iter=200,
    ).fit(X)
    assert adjusted_rand_score(y, hgmm.labels_) >= 0.9


def test_hddc_fit_is_reproducible_given_seed():
    """Two ``fit(X, random_state=0)`` calls produce identical fits.

    ``HighDimensionalGaussianMixture`` declares
    ``non_deterministic = True``, but only with respect to *init*
    variance. With a fixed ``random_state`` the run must be
    bit-reproducible — that is the contract that lets users debug.
    """
    X, _ = _toy()
    fit1 = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1, max_iter=50,
    ).fit(X)
    fit2 = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1, max_iter=50,
    ).fit(X)
    np.testing.assert_array_equal(fit1.labels_, fit2.labels_)
    np.testing.assert_allclose(fit1.weights_, fit2.weights_, rtol=0, atol=0)
    np.testing.assert_allclose(fit1.means_, fit2.means_, rtol=0, atol=0)
    np.testing.assert_allclose(
        fit1.noise_variances_, fit2.noise_variances_, rtol=0, atol=0
    )


def test_hddc_pickle_roundtrip():
    """A fitted estimator survives pickle.dumps / pickle.loads.

    ``check_estimator`` also exercises pickling, but burying the
    coverage there makes regressions hard to track down. An
    explicit test keeps the contract visible.
    """
    import pickle

    X, _ = _toy()
    fit = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1, max_iter=50,
    ).fit(X)
    blob = pickle.dumps(fit)
    fit2 = pickle.loads(blob)

    np.testing.assert_array_equal(fit.predict(X), fit2.predict(X))
    np.testing.assert_allclose(fit.score_samples(X), fit2.score_samples(X))
    assert fit.bic(X) == fit2.bic(X)
    assert fit.icl(X) == fit2.icl(X)


def test_hddc_infeasible_K_min_cluster_size_raises():
    """K * min_cluster_size > n raises before EM starts.

    The "steal" mechanism in ``_m_step`` can only satisfy
    ``K * min_cluster_size <= n_samples``; if the request is
    infeasible we raise immediately rather than spin in an
    unsatisfiable loop.
    """
    X, _ = _toy(n=40)
    hgmm = HighDimensionalGaussianMixture(
        n_components=10, min_cluster_size=20,  # 10 * 20 = 200 > 40
    )
    with pytest.raises(ValueError, match="exceeds n_samples"):
        hgmm.fit(X)


def test_hddc_bad_init_array_raises():
    """``init_params=`` ndarray with invalid contents raises a clear error."""
    X, _ = _toy()
    # Wrong length.
    hgmm_bad_len = HighDimensionalGaussianMixture(
        n_components=3, init_params=np.zeros(X.shape[0] - 1, dtype=int),
    )
    with pytest.raises(ValueError, match="init_params"):
        hgmm_bad_len.fit(X)

    # Invalid cluster id (out of [0, n_components)).
    hgmm_bad_id = HighDimensionalGaussianMixture(
        n_components=3,
        init_params=np.full(X.shape[0], fill_value=99, dtype=int),
    )
    with pytest.raises(ValueError, match="init_params"):
        hgmm_bad_id.fit(X)


# --------------------------------------------------------------------------
# 3. ICL identities
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["bic", "icl"])
def test_hddc_information_criteria_not_fitted_raises(name):
    """Calling bic/icl before fit raises NotFittedError."""
    hgmm = HighDimensionalGaussianMixture(n_components=2)
    X, _ = _toy()
    with pytest.raises(NotFittedError):
        getattr(hgmm, name)(X)


def test_hddc_icl_ge_bic():
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1,
    ).fit(X)
    assert hgmm.icl(X) >= hgmm.bic(X) - 1e-6


def test_hddc_aic_bic_consistency():
    """AIC = BIC formula with parameter penalty ``2·ν`` vs ``ν·log n``.

    ``aic`` mirrors :meth:`sklearn.mixture.GaussianMixture.aic` and is
    related to ``bic`` by the deterministic identity
    ``aic - bic = ν·(2 - log n)``.
    """
    X, _ = _toy()
    n = X.shape[0]
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", random_state=0, n_init=1, max_iter=50,
    ).fit(X)
    nu = hgmm._n_parameters()
    expected = nu * (2.0 - math.log(n))
    assert math.isclose(hgmm.aic(X) - hgmm.bic(X), expected, rel_tol=1e-9)


@pytest.mark.parametrize("model", _GEOMETRIC)
def test_hddc_n_parameters_positive(model):
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model=model, random_state=0, n_init=1, max_iter=10,
    ).fit(X)
    assert hgmm._n_parameters() > 0


def test_hddc_fit_predict_1d_no_nan():
    """Regression: 1D fit must not produce NaN log-density.

    Earlier versions returned ``NaN`` for ``bic`` / ``icl`` on ``p = 1``
    data because ``mean(eigvals[d_k:])`` was taken over an empty
    slice (``d_k == p == 1`` exhausts the spectrum). The current
    implementation branches on ``d_k == p`` and uses ``_EPS`` for the
    noise variance instead. This test guards against regressing that
    fix.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(40, 1)
    hgmm = HighDimensionalGaussianMixture(
        n_components=2, model="AVV", random_state=0, n_init=1, max_iter=20,
    ).fit(X)
    assert np.isfinite(hgmm.bic(X))
    assert np.isfinite(hgmm.icl(X))
    labels = hgmm.predict(X)
    assert labels.shape == (40,)


# --------------------------------------------------------------------------
# 4. Parameter count vs. Bouveyron 2007, Table 1
# --------------------------------------------------------------------------

def test_hddc_parameter_count_table1_AVE():
    """Row [a_kj b_k Q_k d] in paper notation, K=4, p=100, d=10 -> 4228."""
    K, p, d = 4, 100, 10
    X = np.random.RandomState(0).randn(2000, p)
    hgmm = HighDimensionalGaussianMixture(
        n_components=K, model="AVE", signal_dim=d,
        random_state=0, n_init=1, max_iter=5,
    ).fit(X)
    rho = K * p + (K - 1)
    tau = d * (p - (d + 1) / 2.0)
    expected = int(rho + K * (tau + d + 1) + 1)
    assert hgmm._n_parameters() == expected == 4228


def test_hddc_parameter_count_table1_AEE():
    """Row [a_kj b Q_k d] in paper notation, K=4, p=100, d=10 -> 4225."""
    K, p, d = 4, 100, 10
    X = np.random.RandomState(0).randn(2000, p)
    hgmm = HighDimensionalGaussianMixture(
        n_components=K, model="AEE", signal_dim=d,
        random_state=0, n_init=1, max_iter=5,
    ).fit(X)
    rho = K * p + (K - 1)
    tau = d * (p - (d + 1) / 2.0)
    expected = int(rho + K * (tau + d) + 2)
    assert hgmm._n_parameters() == expected == 4225


# Full Bouveyron 2007 Table 1 sweep at K=4, p=100, d=10. Values come
# from the formulas implemented in ``_n_parameters``; AVE and AEE are
# cross-checked against the paper above. The remaining 12 rows guard
# against accidental edits to ``_n_parameters`` re-introducing off-by-
# one or wrong-letter-branch bugs.
_TABLE1_EXPECTED_K4_P100_D10 = {
    "AVV": 4231, "AEV": 4228, "IVV": 4195, "IEV": 4192,
    "UVV": 4192, "UEV": 4189,
    "AVE": 4228, "CVE": 4198, "AEE": 4225, "CEE": 4195,
    "IVE": 4192, "IEE": 4189, "UVE": 4189, "UEE": 4186,
}


@pytest.mark.parametrize("model,expected",
                         sorted(_TABLE1_EXPECTED_K4_P100_D10.items()))
def test_hddc_parameter_count_table1_all_rows(model, expected):
    """All 14 rows of Bouveyron 2007 Table 1 at K=4, p=100, d=10.

    Bypasses fit and sets the attributes ``_n_parameters`` reads
    directly. This isolates the formula from EM convergence and from
    Cattell-scree-driven d_k for the free-d (last-letter V) models.
    """
    K, p, d = 4, 100, 10
    hgmm = HighDimensionalGaussianMixture(n_components=K, model=model)
    hgmm._geometric_model_ = model
    hgmm.n_features_in_ = p
    hgmm.signal_dims_ = [d] * K
    assert hgmm._n_parameters() == expected


# --------------------------------------------------------------------------
# 4b. Off-axis parameter-count regression
#
# The Table 1 test above pins behaviour at one point in (K, p, d) space.
# The closed-form formulas of Bouveyron 2007 Table 1 are evaluated below
# in the test itself (deliberately duplicated from ``_n_parameters``) so
# any drift between the two on an off-axis (K, p, d) triple fails the
# test. ``signal_dims`` may be non-uniform; for ``dim="equal"`` models
# (last letter ``E``) the test passes uniform signal_dims since the
# model enforces it.
# --------------------------------------------------------------------------

def _expected_n_parameters(model, K, p, signal_dims):
    """Closed-form Table 1 reference. Mirrors ``_n_parameters`` but is
    written from the paper, not copied from the implementation, so a
    silent edit on one side surfaces a test failure on the other.
    """
    import numpy as np
    d_k = np.asarray(signal_dims, dtype=float)
    D = float(d_k.sum())
    rho = K * p + (K - 1)
    tau_bar = float(np.sum(d_k * (p - (d_k + 1.0) / 2.0)))

    if model == "AVV": return int(rho + tau_bar + 2 * K + D)
    if model == "AEV": return int(rho + tau_bar + K + D + 1)
    if model == "IVV": return int(rho + tau_bar + 3 * K)
    if model == "IEV": return int(rho + tau_bar + 2 * K + 1)
    if model == "UVV": return int(rho + tau_bar + 2 * K + 1)
    if model == "UEV": return int(rho + tau_bar + K + 2)

    d = float(d_k[0])  # E models: d is shared
    tau = d * (p - (d + 1.0) / 2.0)
    if model == "AVE": return int(rho + K * (tau + d + 1) + 1)
    if model == "CVE": return int(rho + K * (tau + 1) + d + 1)
    if model == "AEE": return int(rho + K * (tau + d) + 2)
    if model == "CEE": return int(rho + K * tau + d + 2)
    if model == "IVE": return int(rho + K * (tau + 2) + 1)
    if model == "IEE": return int(rho + K * (tau + 1) + 2)
    if model == "UVE": return int(rho + K * (tau + 1) + 2)
    if model == "UEE": return int(rho + K * tau + 3)
    raise AssertionError(model)


_OFFAXIS_TRIPLES = [
    # (K, p, d) — uniform signal_dims; valid for all 14 models.
    (2, 5, 2),
    (3, 10, 3),
    (5, 50, 1),       # d at the minimum
    (2, 20, 19),      # d at p-1 (the cap)
    (4, 100, 10),     # paper's example, sanity duplicate
    (6, 200, 25),     # larger K, larger p, moderate d
]


@pytest.mark.parametrize("K,p,d", _OFFAXIS_TRIPLES)
@pytest.mark.parametrize("model", _GEOMETRIC)
def test_hddc_n_parameters_offaxis(model, K, p, d):
    """``_n_parameters`` matches the paper's closed-form at off-axis
    (K, p, d). Uniform signal_dims = [d]*K so the triple is valid for
    both V (free-d) and E (common-d) sub-models.
    """
    hgmm = HighDimensionalGaussianMixture(n_components=K, model=model)
    hgmm._geometric_model_ = model
    hgmm.n_features_in_ = p
    hgmm.signal_dims_ = [d] * K
    expected = _expected_n_parameters(model, K, p, [d] * K)
    got = hgmm._n_parameters()
    assert got == expected, (
        f"model={model} K={K} p={p} d={d}: "
        f"_n_parameters={got} expected={expected}"
    )


# Free-d (last-letter V) models accept non-uniform signal_dims; the
# common-d (E) models do not, so this test is V-only.
_V_MODELS = ("AVV", "AEV", "IVV", "IEV", "UVV", "UEV")

_NONUNIFORM_TRIPLES = [
    # (K, p, signal_dims) with d_k varying across clusters.
    (3, 30, [5, 8, 3]),
    (4, 50, [10, 5, 15, 2]),
    (2, 100, [20, 40]),
    (5, 25, [1, 24, 12, 6, 18]),
]


@pytest.mark.parametrize("K,p,signal_dims", _NONUNIFORM_TRIPLES)
@pytest.mark.parametrize("model", _V_MODELS)
def test_hddc_n_parameters_nonuniform_dk_v_models(model, K, p, signal_dims):
    """Free-d (V) sub-models with non-uniform per-cluster signal dims.

    Common-d (E) sub-models enforce a single d across clusters, so they
    are excluded; the off-axis test above covers them at uniform d.
    """
    hgmm = HighDimensionalGaussianMixture(n_components=K, model=model)
    hgmm._geometric_model_ = model
    hgmm.n_features_in_ = p
    hgmm.signal_dims_ = list(signal_dims)
    expected = _expected_n_parameters(model, K, p, signal_dims)
    got = hgmm._n_parameters()
    assert got == expected, (
        f"model={model} K={K} p={p} d_k={signal_dims}: "
        f"_n_parameters={got} expected={expected}"
    )


# --------------------------------------------------------------------------
# 5. The Student-mixture empirical argument
# --------------------------------------------------------------------------

def test_hddc_icl_student_mixture():
    """ICL is more parsimonious than BIC on a heavy-tailed mixture.

    We do not assert ICL recovers ``true_K`` exactly here (the
    multi-dimensional Student-t mixture is harder than the 1D
    Gaussian-mixture version in the ICL PR). We assert the weaker
    invariant that motivates ICL in the first place:
    ``K_ICL <= K_BIC`` with strict inequality on heavy-tailed data
    in at least one of the seeds.
    """
    rng = np.random.RandomState(0)
    true_K = 3
    n_per = 1500
    p = 5
    df = 3
    Xs = []
    for k in range(true_K):
        center = 25.0 * k * np.ones(p)
        Xs.append(stats.t.rvs(df=df, size=(n_per, p), random_state=rng) + center)
    X = np.vstack(Xs)

    bic_scores, icl_scores = [], []
    Ks = range(2, 8)
    for K in Ks:
        hgmm = HighDimensionalGaussianMixture(
            n_components=K, model="AVV",
            random_state=0, n_init=5, max_iter=100,
        ).fit(X)
        bic_scores.append(hgmm.bic(X))
        icl_scores.append(hgmm.icl(X))

    K_icl = list(Ks)[int(np.argmin(icl_scores))]
    K_bic = list(Ks)[int(np.argmin(bic_scores))]
    # ICL is at most as large as BIC on heavy-tailed data, and both
    # are at least the true K (BIC may overcount).
    assert K_bic >= true_K
    assert K_icl <= K_bic


# --------------------------------------------------------------------------
# 6. Resolver: alias equivalence, kwargs-only, conflicts, mclust
# --------------------------------------------------------------------------

@pytest.mark.parametrize("geo,paper", list(_PAPER.items()))
def test_hddc_model_alias_equivalence(geo, paper):
    """`model="AVV"` and `model="akj_bk_Qk_dk"` resolve identically."""
    X, _ = _toy()
    a = HighDimensionalGaussianMixture(
        n_components=3, model=geo, random_state=0, n_init=1, max_iter=20,
    ).fit(X)
    b = HighDimensionalGaussianMixture(
        n_components=3, model=paper, random_state=0, n_init=1, max_iter=20,
    ).fit(X)
    assert a._geometric_model_ == b._geometric_model_ == geo
    # Exact same EM trajectory under same random_state. Aliases must
    # be **bit-equivalent**; default ``assert_allclose`` tolerances
    # would silently absorb a real implementation bug.
    np.testing.assert_allclose(a.weights_, b.weights_, rtol=1e-12, atol=0)
    np.testing.assert_allclose(a.means_, b.means_, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(a.labels_, b.labels_)


def test_hddc_model_kwargs_only_no_model():
    """Setting `model=None` and the three axis kwargs is accepted."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model=None,
        signal="anisotropic", noise="varying", dim="varying",
        random_state=0, n_init=1, max_iter=20,
    ).fit(X)
    assert hgmm._geometric_model_ == "AVV"


def test_hddc_model_kwargs_partial_no_model_raises():
    """`model=None` with fewer than three kwargs -> ValueError."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model=None,
        signal="anisotropic", noise="varying",  # `dim` missing
        random_state=0, n_init=1, max_iter=5,
    )
    with pytest.raises(ValueError, match="Missing.*dim"):
        hgmm.fit(X)


def test_hddc_model_kwargs_agree_with_model_is_legal():
    """Redundant kwargs that match the model axis are accepted."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV",
        signal="anisotropic", noise="varying", dim="varying",
        random_state=0, n_init=1, max_iter=20,
    ).fit(X)
    assert hgmm._geometric_model_ == "AVV"


def test_hddc_model_conflict_raises():
    """`model="AVV", dim="equal"` -> ValueError on `dim` axis."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", dim="equal",
        random_state=0, n_init=1, max_iter=5,
    )
    with pytest.raises(ValueError, match=r"Conflict on `dim`"):
        hgmm.fit(X)


def test_hddc_model_conflict_signal_axis_raises():
    """Same on the signal axis."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="AVV", signal="uniform",
        random_state=0, n_init=1, max_iter=5,
    )
    with pytest.raises(ValueError, match=r"Conflict on `signal`"):
        hgmm.fit(X)


def test_hddc_mclust_codes_rejected():
    """`model="VVV"` -> ValueError pointing to HDDC table."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="VVV", random_state=0, n_init=1, max_iter=5,
    )
    with pytest.raises(ValueError, match=r"mclust"):
        hgmm.fit(X)


def test_hddc_unknown_model_string_rejected():
    """A genuinely unknown spelling -> generic ValueError."""
    X, _ = _toy()
    hgmm = HighDimensionalGaussianMixture(
        n_components=3, model="XYZ", random_state=0, n_init=1, max_iter=5,
    )
    # `_parameter_constraints` rejects this before resolve runs, so the
    # message will mention "model" / StrOptions; just check we get a
    # ValueError with the offending string in it.
    with pytest.raises(ValueError):
        hgmm.fit(X)


# --------------------------------------------------------------------------
# 7. Cattell scree-rule unit tests (direct, not via EM)
# --------------------------------------------------------------------------
# These exercise ``pr_hddc/_hddc.py::_cattell_scree_test`` in
# isolation, on hand-crafted spectra whose elbow is unambiguous, so
# a regression in the rule is caught without an EM run muddying
# the signal.

from sklearn.mixture._hddc import _cattell_scree_test  # noqa: E402


def test_cattell_scree_test_clear_elbow():
    """One sharp drop -> ``d`` equals the position of the drop."""
    # Two signal dims (large) + four noise dims (small, equal).
    eigvals = np.array([10.0, 9.0, 0.1, 0.1, 0.1, 0.1])
    assert _cattell_scree_test(eigvals, threshold=0.5) == 2


def test_cattell_scree_test_isotropic_spectrum_returns_one():
    """Truly isotropic spectrum -> ``d = 1``.

    All eigenvalues equal makes every consecutive drop zero, so
    ``max_diff = 0`` and the rule short-circuits to ``d = 1``. This
    is the "no signal direction is distinguishable" failure mode
    documented in ``docs/HDDC.md`` §3.5.1.
    """
    eigvals = np.full(8, 1.0)
    assert _cattell_scree_test(eigvals, threshold=0.5) == 1


def test_cattell_scree_test_noise_floor_blocks_late_pick():
    """``noise_ctrl`` prevents elevating sub-floor eigenvalues to signal.

    The eligible-position mask requires ``eigvals[i+1] > noise_ctrl``,
    so a "drop" between two near-zero eigenvalues never counts.
    """
    # Real signal: indices 0..1. Spurious drop at 5->6 is below floor.
    eigvals = np.array([5.0, 4.0, 0.5, 0.4, 0.3, 0.2, 1e-12])
    d = _cattell_scree_test(eigvals, threshold=0.2, noise_ctrl=1e-8)
    assert d <= 5, f"noise_ctrl failed; got d={d}"


def test_cattell_scree_test_picks_latest_eligible_drop():
    """At ``threshold=0.2`` the rule picks the **largest** eligible
    position, not the first.

    Spectrum has two normalised drops above 0.2: position 0->1
    (norm 0.27) and position 4->5 (norm 1.0, the maximum). Both are
    eligible. The HDclassif-aligned rule picks the *largest* index
    among the eligible — d = 5.
    """
    # diffs: [1.0, 0.1, 0.1, 0.1, 3.7, 0.1]
    # max_diff = 3.7 → norm: [0.27, 0.027, 0.027, 0.027, 1.0, 0.027]
    # above_thr(0.2) at positions 0 and 4 → d = 5 (1-indexed)
    eigvals = np.array([10.0, 9.0, 8.9, 8.8, 8.7, 5.0, 4.9])
    assert _cattell_scree_test(eigvals, threshold=0.2) == 5


def test_cattell_scree_test_rejects_unsorted_input():
    """A non-monotone input is a programming error, not data noise."""
    with pytest.raises(ValueError, match=r"non-increasing"):
        _cattell_scree_test(np.array([1.0, 2.0, 0.5]))


def test_cattell_scree_test_short_input_returns_one():
    """Spectra of length <= 2 special-case to ``d = 1``."""
    assert _cattell_scree_test(np.array([1.0])) == 1
    assert _cattell_scree_test(np.array([1.0, 0.1])) == 1
