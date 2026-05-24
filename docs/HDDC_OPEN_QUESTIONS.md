# HDDC — open questions for scikit-learn maintainers

This file frames the **HDDC PR as a design discussion**, not as a
ready-to-merge change. The implementation, tests, parity check, and
figures are deliberately complete enough to make the trade-offs
concrete, but the scope and the public API surface are open.

scikit-learn is explicitly selective on new algorithms — they ask
for a well-established, useful, widely-implemented method that
beats or complements what is already in the library. HDDC meets
the first three (Bouveyron, Girard & Schmid 2007; the `HDclassif`
R package on CRAN since 2012; a clean parsimonious-Gaussian-mixture
family for `n << p`), but the **shape** of the PR — one estimator
exposing 14 sub-models plus three naming schemes — is a lot for a
first review pass.

The five questions below are the ones we want a maintainer's
verdict on before broadening or narrowing the proposal.

---

## Q1. Belongs in `sklearn.mixture`, or in `scikit-learn-contrib`?

HDDC is a parsimonious extension of `GaussianMixture` for the
`n << p` regime — exactly the regime where the existing
`covariance_type` choices fail (full overfits, diagonal can't
represent cluster-specific correlation). It sits naturally next to
`GaussianMixture` in user mental model.

But: scikit-learn's
[FAQ on new algorithm inclusion](https://scikit-learn.org/stable/faq.html#new-algorithms-inclusion-criteria)
prioritizes algorithms that are (a) widely used, (b) outperform
existing methods in many settings, and (c) don't introduce too
many new hyperparameters. HDDC does (a) and (b), but (c) is open
for discussion given the 14-sub-model surface.

**The question is whether `sklearn.mixture` is the right home, or
whether `scikit-learn-contrib/hddc` would be a better staging
ground first.**

## Q2. Start with a single HDDC sub-model before all 14?

The 14 sub-models in Bouveyron's Table 1 differ by tying choices on
three axes (signal eigenvalues / noise variance / signal dimension).
The most general one is `AVV` (= `[a_kj b_k Q_k d_k]`); the most
constrained is `UEE` (= `[a b Q_k d]`). The parsimony / fit
trade-off across the 14 is the empirical content of Bouveyron 2007.

We could:

- **Ship all 14** (current PR shape): full coverage, but a large
  validation surface and a non-trivial public API right away.
- **Ship `AVV` only** (most general; reduces gracefully): minimum
  viable HDDC, with the other 13 added in follow-up PRs.
- **Ship `AVV` + `UEE`** (most general + most constrained): a
  meaningful range without exploding the surface.

**The question is whether the 14-sub-model surface should be in
the first PR or staged over follow-ups.**

## Q3. API for `model=`: paper notation, geometric code, or per-axis kwargs?

Three spellings are currently accepted (`docs/HDDC.md` §1):

- **Paper bracket** — `model="akj_bk_Qk_dk"` (faithful to Bouveyron 2007).
- **Geometric code** — `model="AVV"` (3-letter `<Signal><Noise><Dim>`).
- **Per-axis kwargs** — `signal="anisotropic", noise="varying", dim="varying"`.

The README recommends `model="AVV"` as the canonical form; the
other two are accepted as aliases under strict no-conflict
semantics. But three spellings is a lot of API surface for the
first PR.

**The question is whether we should ship one canonical spelling
only (probably the geometric code), keep the paper notation as a
documentation alias, and drop per-axis kwargs for v1.**

## Q4. Dense-only contract, or include sparse?

The current implementation assumes dense `float64` input. The
per-cluster eigendecomposition (or SVD-of-data-matrix when
`p > n`) is intrinsically dense; sparse input would need to be
densified, which defeats the memory benefit users would expect.
`GaussianMixture` itself is dense-only for the same reason.

**The question is whether the PR should explicitly document "dense
input only" and bail with a clear error on sparse `X`, rather than
silently densify in `validate_data`.**

## Q5. Minimum baseline comparison expected for the user guide?

The PR's empirical argument lives in `figures/figures_hddc.py`,
which compares:

- `HDDC(AVV)` vs `GaussianMixture(covariance_type="diag")` on raw
  Olivetti faces (`n=100, p=4096`).
- `HDDC(AVV)` vs `GaussianMixture(covariance_type="diag")` on
  `load_digits` (`n=1797, p=64`).
- The 14-sub-model parameter budget vs the 4 `GaussianMixture`
  covariance types.

For inclusion in the sklearn user guide, the maintainer-recommended
baselines for a new clustering estimator are typically
`KMeans`, `GaussianMixture` (relevant covariance types), and the
most common "dimensionality reduction + cluster" pipeline
(`PCA + GaussianMixture`). We have the first two but not the third
as a head-to-head.

**The question is the minimum baseline set that a sklearn user
guide section needs: is the current `GaussianMixture(diag)`
comparison sufficient, or do we need to add `KMeans` and
`PCA + GaussianMixture` head-to-head plots?**

---

## How this PR is staged

* **Code, tests, docs, figures, parity check**: in this repo (the
  companion repo). The CI test suite is deliberately lightweight
  (~17 conceptual test functions, ~60 collected tests; heavier
  scientific demonstrations live under `examples/`).
* **What lands in scikit-learn first**: open to the maintainers'
  verdict on Q1–Q5.

The HDDC PR is intentionally framed as **"here is a complete
working draft that supports a small set of scope decisions"**
rather than **"please merge as-is."**
