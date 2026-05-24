"""Scientific demonstration: ICL vs BIC on a 5-D Student-t mixture under HDDC.

**Not part of the CI test suite.** The HDDC PR's empirical claim is
that ICL is more parsimonious than BIC under the AVV sub-model on
heavy-tailed multivariate data; pinning a specific (K_BIC, K_ICL)
pair in CI would be brittle. This script runs the sweep and prints
the result instead.

Run:

    python examples/hddc_student_mixture_demo.py
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.mixture import HighDimensionalGaussianMixture


def main() -> None:
    rng = np.random.RandomState(42)
    true_K, n_per, p, df = 3, 1500, 5, 3
    centers = [25.0 * k * np.ones(p) for k in range(true_K)]
    X = np.vstack(
        [stats.t.rvs(df=df, size=(n_per, p), random_state=rng) + c
         for c in centers]
    )

    print(
        f"\n5-D Student-t mixture: true_K={true_K}, p={p}, df={df}, "
        f"n_per={n_per}\n"
    )
    print(f"{'K':>3s}  {'BIC':>10s}  {'ICL':>10s}")
    print("-" * 30)
    bic_scores, icl_scores = [], []
    Ks = list(range(2, 8))
    for K in Ks:
        hgmm = HighDimensionalGaussianMixture(
            n_components=K, model="AVV",
            random_state=0, n_init=5, max_iter=100,
        ).fit(X)
        b, i = hgmm.bic(X), hgmm.icl(X)
        bic_scores.append(b); icl_scores.append(i)
        print(f"{K:>3d}  {b:>10.1f}  {i:>10.1f}")

    K_bic = Ks[int(np.argmin(bic_scores))]
    K_icl = Ks[int(np.argmin(icl_scores))]
    print(f"\nK_BIC = {K_bic}, K_ICL = {K_icl}  (true K = {true_K})")
    print("Expected pattern: K_BIC >= true_K, K_ICL <= K_BIC.")


if __name__ == "__main__":
    main()
