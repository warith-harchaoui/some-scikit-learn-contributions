"""Scientific demonstration: ICL vs BIC on a Student-t mixture.

**Not part of the CI test suite.** This is the heavier-than-CI
version of the empirical claim that motivates adding ``icl()`` to
``GaussianMixture``:

* On a heavy-tailed Student-t mixture with location-shifted equal-
  weight components and ``df ∈ {3, 5, 10}``, BIC adds spurious
  components past the true K to absorb tail mass while ICL discounts
  them via the entropy penalty.

The repo CI keeps only the lightweight invariant tests
(``pr_gmm_icl/test_icl_addition.py``: identity, hard-partition, K=1,
no-warning, not-fitted). The fragile "ICL recovers true K, BIC does
not" claim is **not** a CI assertion — fitting GMM on heavy-tailed
1-D data is stochastic enough that pinning K-selection to a single
integer in CI is brittle. The demo below runs the full sweep and
prints (rather than asserts) the K-selection summary.

Run:

    python examples/icl_student_t_demo.py

Companion narrative figure: ``figures/figures_icl.py::make_fig_icl_student``.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture


def main() -> None:
    """Sweep ``df`` and ``K``, report BIC- vs ICL-selected K."""
    rng = np.random.RandomState(42)
    true_K = 3
    n_per = 4000
    locs = np.array([0.0, 20.0, 40.0])

    print(
        f"\nStudent-t mixture: true_K={true_K}, n_per={n_per}, "
        f"locs={list(locs)}\n"
    )
    print(f"{'df':>4s}  {'K_BIC':>5s}  {'K_ICL':>5s}  recovery")
    print("-" * 40)
    for df in (3, 5, 10):
        X = np.concatenate(
            [stats.t.rvs(df=df, size=n_per, random_state=rng) + mu
             for mu in locs]
        ).reshape(-1, 1)

        bic_scores, icl_scores = [], []
        Ks = list(range(2, 9))
        for K in Ks:
            gmm = GaussianMixture(
                n_components=K, covariance_type="full",
                random_state=0, n_init=10, max_iter=300, tol=1e-5,
            ).fit(X)
            bic_scores.append(gmm.bic(X))
            icl_scores.append(gmm.icl(X))
        K_bic = Ks[int(np.argmin(bic_scores))]
        K_icl = Ks[int(np.argmin(icl_scores))]
        recovery = (
            "ICL recovers K"
            if K_icl == true_K else
            f"ICL picks K={K_icl} (true {true_K})"
        )
        print(f"{df:>4d}  {K_bic:>5d}  {K_icl:>5d}  {recovery}")


if __name__ == "__main__":
    main()
