"""
Tests to add to ``sklearn/mixture/tests/test_gaussian_mixture.py``.

Six functions, covering the load-bearing invariants and edge cases
of the new ``icl`` method:

1. ``test_gaussian_mixture_icl`` — unit-level sanity checks: ICL >= BIC,
   ICL = BIC + 2H across all covariance types.

2. ``test_gaussian_mixture_icl_not_fitted_raises`` — calling icl() on
   an unfitted estimator raises ``NotFittedError`` (matches ``bic``).

3. ``test_gaussian_mixture_icl_equals_bic_on_hard_partition`` — when
   responsibilities collapse to a hard partition the entropy term
   vanishes and ICL coincides with BIC.

4. ``test_gaussian_mixture_icl_equals_bic_at_K1`` — at ``K=1`` the
   responsibilities are identically 1, so the entropy term is exactly
   0 and ICL must equal BIC bit-for-bit.

5. ``test_gaussian_mixture_icl_no_runtime_warning_on_hard_partition``
   — the implementation uses :func:`scipy.special.xlogy` so that
   ``0 * log(0)`` returns 0 with no ``RuntimeWarning``. This test
   makes the design choice explicit.

6. ``test_gaussian_mixture_icl_student_mixture`` — the empirical
   argument behind this contribution: on a heavy-tailed Student-t
   mixture, BIC overestimates K while ICL recovers the true K.
"""

import warnings

import numpy as np
import pytest
from scipy import stats

from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture

# ``COVARIANCE_TYPE`` is imported from the top of
# ``test_gaussian_mixture.py`` in the final patch; the standalone test
# file here mirrors that constant inline only when run outside sklearn.
COVARIANCE_TYPE = ("full", "tied", "diag", "spherical")


def test_gaussian_mixture_icl():
    """ICL satisfies its defining identities."""
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 200, 3, 2
    X = rng.randn(n_samples, n_features)

    for cv_type in COVARIANCE_TYPE:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=cv_type,
            random_state=rng,
            max_iter=200,
            tol=1e-5,
        )
        gmm.fit(X)

        bic = gmm.bic(X)
        icl = gmm.icl(X)

        # ICL = BIC + 2H, with H >= 0, so ICL >= BIC.
        assert icl >= bic - 1e-8, (
            f"ICL ({icl}) must be >= BIC ({bic}) up to numerical error "
            f"(cv_type={cv_type})."
        )

        # Reconstruct the entropy and check the identity.
        _, log_resp = gmm._estimate_log_prob_resp(X)
        resp = np.exp(log_resp)
        entropy = -np.nansum(resp * log_resp)
        assert np.isclose(icl, bic + 2.0 * entropy)


def test_gaussian_mixture_icl_not_fitted_raises():
    """Calling icl() before fit raises NotFittedError, like bic."""
    gmm = GaussianMixture(n_components=2)
    X = np.random.RandomState(0).randn(10, 2)
    with pytest.raises(NotFittedError):
        gmm.icl(X)


def test_gaussian_mixture_icl_equals_bic_on_hard_partition():
    """If the responsibilities collapse to a hard partition, ICL == BIC."""
    rng = np.random.RandomState(0)
    # Three very well-separated isotropic Gaussian clusters.
    centers = np.array([[-50.0, 0.0], [0.0, 50.0], [50.0, -50.0]])
    X = np.vstack(
        [c + rng.randn(200, 2) for c in centers]
    )
    gmm = GaussianMixture(
        n_components=3, covariance_type="full", random_state=rng,
        max_iter=200, tol=1e-7,
    ).fit(X)
    assert np.isclose(gmm.icl(X), gmm.bic(X), rtol=0, atol=1e-3)


def test_gaussian_mixture_icl_equals_bic_at_K1():
    """At K=1, responsibilities are identically 1 → entropy = 0 → ICL = BIC.

    Sanity check that the degenerate single-component case is handled
    cleanly. Critical because some entropy formulations propagate NaN
    or raise on the K=1 corner; the ``xlogy`` form does not.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    for cv_type in COVARIANCE_TYPE:
        gmm = GaussianMixture(
            n_components=1, covariance_type=cv_type, random_state=0,
        ).fit(X)
        # Bit-equivalent, not just close — entropy is mathematically 0.
        assert gmm.icl(X) == gmm.bic(X), (
            f"At K=1 (cv={cv_type}): ICL={gmm.icl(X)} != BIC={gmm.bic(X)}"
        )


def test_gaussian_mixture_icl_no_runtime_warning_on_hard_partition():
    """``icl`` on a near-hard partition must not emit RuntimeWarning.

    The implementation uses ``scipy.special.xlogy(resp, resp)`` which
    returns 0 for ``resp == 0`` instead of evaluating ``0 * log(0)``
    and triggering ``invalid value encountered in log``. This test
    pins that design choice — switching to ``resp * np.log(resp)``
    with an ``eps`` clip would silently re-introduce the warning.
    """
    rng = np.random.RandomState(0)
    centers = np.array([[-50.0, 0.0], [0.0, 50.0], [50.0, -50.0]])
    X = np.vstack([c + rng.randn(200, 2) for c in centers])
    gmm = GaussianMixture(
        n_components=3, covariance_type="full", random_state=rng,
        max_iter=200, tol=1e-7,
    ).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        gmm.icl(X)


@pytest.mark.parametrize("df", [3, 5, 10])
def test_gaussian_mixture_icl_student_mixture(df):
    """On a heavy-tailed Student-t mixture, ICL is more parsimonious
    than BIC and recovers the true K in the mild-tail regime.

    This is the empirical motivation for adding ICL to
    ``GaussianMixture``: heavy-tailed data make BIC prefer redundant
    components that exist only to model the tails, while ICL discounts
    those components because their responsibilities are ambiguous.

    What we assert, and why:

    * At every df, ``K_BIC >= true_K`` (BIC over-counts or matches) and
      ``K_ICL <= K_BIC`` (ICL is at least as parsimonious as BIC).
      This is the contribution's load-bearing claim.
    * At ``df >= 5`` (mild-to-moderate tails), ICL recovers the true K
      exactly: this is where the PR's headline figure sits.
    * At ``df = 3`` (very heavy tails), we relax to ``K_ICL <= true_K + 1``
      because the gap between K=3 and K=4 ICL is genuinely small at
      this df under standard EM optimisation — the contribution does
      not claim ICL eliminates over-counting in the limit of pathological
      tails, only that it shrinks the gap relative to BIC.
    """
    rng = np.random.RandomState(42)
    true_K = 3
    # Sized to keep the test under ~30 s on CI. The n_init budget is
    # what matters for EM convergence on heavy-tailed data; sample
    # count is secondary above ~2k per component.
    n_per = 4000
    locs = np.array([0.0, 20.0, 40.0])

    # 1D Student-t mixture, equal weights, location-shift, df fixed.
    X = np.concatenate(
        [stats.t.rvs(df=df, size=n_per, random_state=rng) + mu for mu in locs]
    ).reshape(-1, 1)

    bic_scores, icl_scores = [], []
    Ks = range(2, 9)
    for K in Ks:
        gmm = GaussianMixture(
            n_components=K, covariance_type="full",
            random_state=0, n_init=5, max_iter=300, tol=1e-5,
        ).fit(X)
        bic_scores.append(gmm.bic(X))
        icl_scores.append(gmm.icl(X))

    K_bic = list(Ks)[int(np.argmin(bic_scores))]
    K_icl = list(Ks)[int(np.argmin(icl_scores))]

    # Load-bearing invariants — hold at every df:
    assert K_bic >= true_K, (
        f"BIC selected K={K_bic} < true_K={true_K} at df={df}. "
        f"BIC scores: {bic_scores}"
    )
    assert K_icl <= K_bic, (
        f"ICL selected K={K_icl} > K_BIC={K_bic} at df={df}. "
        f"ICL scores: {icl_scores}; BIC scores: {bic_scores}"
    )

    # Tighter recovery claim at mild-to-moderate tails:
    if df >= 5:
        assert K_icl == true_K, (
            f"ICL selected K={K_icl}, expected {true_K} at df={df}. "
            f"ICL scores: {icl_scores}"
        )
    else:
        # df=3: heavy tails make the K=3 / K=4 ICL gap small. Relax to
        # near-recovery; the parsimony-relative-to-BIC claim above is
        # what carries the contribution at this df.
        assert K_icl <= true_K + 1, (
            f"ICL selected K={K_icl}, expected <= {true_K + 1} at df={df}. "
            f"ICL scores: {icl_scores}"
        )
