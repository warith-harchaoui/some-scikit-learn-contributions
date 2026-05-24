# For data scientists: what changes for your code

**Author:** [Warith Harchaoui](https://www.linkedin.com/in/warith-harchaoui/)
**Special thanks to:** [Pierre-Alexandre Mattei](https://pamattei.github.io/) 

![ICL vs BIC on a Student-t mixture: BIC overcounts K, ICL recovers the truth.](../figures/fig_icl_student.png)

This document targets people who use scikit-learn day to day and want
to know what these two PRs let them write.

## Today

```python
from sklearn.mixture import GaussianMixture

best_K, best_bic = None, float("inf")
for K in range(2, 21):
    gmm = GaussianMixture(n_components=K, random_state=0).fit(X)
    if gmm.bic(X) < best_bic:
        best_bic, best_K = gmm.bic(X), K
```

This works, but tends to *overestimate* `best_K` whenever the data is
heavier-tailed than a Gaussian. Try it on `df=3` 1D Student data and
you will see BIC pick eight or nine components for a true `K=3`.

## After the ICL PR: ICL in `GaussianMixture`

```python
from sklearn.mixture import GaussianMixture

best_K, best_icl = None, float("inf")
for K in range(2, 21):
    gmm = GaussianMixture(n_components=K, random_state=0).fit(X)
    if gmm.icl(X) < best_icl:
        best_icl, best_K = gmm.icl(X), K
```

A one-line drop-in. `icl()` follows the same lower-is-better
convention as `bic()`. On hard, well-separated clusters,
`icl(X) == bic(X)`; otherwise `icl(X) >= bic(X)` and the gap is `2 H`
where `H` is the entropy of the soft cluster assignments.

**When to prefer ICL over BIC.**

| Situation | Recommended |
| --- | --- |
| Data is approximately Gaussian, you care about density estimation | `bic` |
| Data has tails (Student, log-normal, finite samples of anything real) | `icl` |
| Components are well-separated | both agree |
| You want the *cleanest* clustering, not the *best* density | `icl` |

## After the HDDC PR: HDDC

When you have *many features* relative to the number of samples
(text vectors, gene expression, image features), `GaussianMixture`
already struggles because each component carries a full `p x p`
covariance. HDDC fits a parsimonious GMM where each cluster's
covariance is parameterized in its own low-dimensional subspace:

```python
from sklearn.mixture import HighDimensionalGaussianMixture

hgmm = HighDimensionalGaussianMixture(
    n_components=10,
    model="akj_bk_Qk_dk",     # the most general HDDC sub-model
    random_state=0,
).fit(X)

labels = hgmm.predict(X)
proba  = hgmm.predict_proba(X)
hgmm.bic(X), hgmm.icl(X)
```

The `model=...` argument is HDDC's analogue of `covariance_type`:
14 sub-models, ordered by parsimony. For most use cases, leaving it at
the default and tuning `n_components` with `icl` is the right move.

### Picking K and model jointly

```python
import itertools

MODELS = ("akj_bk_Qk_dk", "akj_b_Qk_dk", "ak_bk_Qk_dk",
          "ak_b_Qk_dk", "a_bk_Qk_dk", "a_b_Qk_dk")

best = None
for K, m in itertools.product(range(2, 11), MODELS):
    hgmm = HighDimensionalGaussianMixture(
        n_components=K, model=m, random_state=0, n_init=2,
    ).fit(X)
    score = hgmm.icl(X)
    if best is None or score < best[0]:
        best = (score, K, m)

icl, K, m = best
print(f"Pick K={K} with model={m}  (ICL={icl:.1f})")
```

This is the "two-axis" selection that ICL makes well-behaved.

## What does not change

- `predict`, `predict_proba`, `score_samples`, `score` work the same
  way.
- The existing `bic` is untouched. If your codebase already uses
  BIC, nothing breaks.
- `BayesianGaussianMixture` is not modified.

## How to play with the argument right now

```bash
pip install -r requirements.txt
python figures/figures_icl.py    # ICL-PR figures: Student mixture + galaxies
python figures/figures_hddc.py   # HDDC-PR figures: synthetic + digits + Olivetti
# (or: python figures/make_all.py runs both)
```

The figures in `figures/` are deterministic — same seeds, same output —
and demonstrate ICL beating BIC on a heavy-tailed Student-$t$ mixture
as well as HDDC beating plain GMM in the high-dim regime.
