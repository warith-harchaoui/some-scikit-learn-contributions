# Downstream tools — `auto_select_mixture` + `estimate_k_range`

These two modules are **downstream personal work**, not part of the
two scikit-learn PRs (`pr_gmm_icl`, `pr_hddc`). They are reference
implementations that show how the PRs compose end-to-end and that
serve as candidates for a future sklearn helper.

- [Part 1. `auto_select_mixture` — ICL-best across families and K](#part-1-auto_select_mixture--icl-best-across-families-and-k-in-one-call)
- [Part 2. `estimate_k_range` — pick `K_grid` automatically](#part-2-estimate_k_range--pick-k_grid-automatically)

## Part 1. `auto_select_mixture` — ICL-best across families and K in one call: GMM + HDDC

Selecting both the **model structure** (Gaussian-mixture covariance
type, HDDC sub-model, …) and the **number of clusters $K$** is usually
done as two separate decisions: practitioners pick a covariance
flavour first ("I'll use `GaussianMixture` with
`covariance_type='diag'`") and then sweep $K$ with BIC or ICL. That
split is artificial.

`auto_select_mixture` does both at once: it sweeps every
$(\text{family}, K)$ pair on the grid and returns the **global ICL
minimum**.

### 1.1 Why ICL is the right common scoreboard

ICL on sklearn's lower-is-better convention is

$$
\mathrm{ICL}(K, \text{family})
\;=\; \mathrm{BIC}(K, \text{family}) \;+\; 2H,
\qquad
H \;=\; -\sum_i \sum_k \tau_{ik}\,\log \tau_{ik}.
$$

Both terms are well-defined for any mixture:

- **BIC** uses the family-specific parameter count
  $\nu(K, \text{family}) \cdot \log n$ — fewer parameters means a
  smaller penalty, exactly as it should.
- **Entropy** uses the posterior responsibilities $\tau$, which every
  mixture estimator exposes regardless of how $\Sigma_k$ is
  parameterised.

So $\arg\min_{K,\,\text{family}} \mathrm{ICL}$ is **principled across
families**, not just within one. That is what makes a unified search
possible.

### 1.2 What's searched by default

| Group | Variants |
| --- | --- |
| `GaussianMixture` | `spherical`, `diag`, `tied`, `full` (4 covariance types) |
| `HighDimensionalGaussianMixture` | `AVV, AEV, IVV, IEV, UVV, UEV, AVE, CVE, AEE, CEE, IVE, IEE, UVE, UEE` (14 sub-models from Bouveyron 2007 Table 1) |

That's **18 families × |K_grid|** total fits. With a typical
`K_grid = range(2, 16)` that's ~250 mixtures. With the shared-KMeans
optimisation below, this runs in **tens of seconds** on moderate-sized
datasets on a laptop.

### 1.3 Shared KMeans across families at fixed K

Naive nested loop:

```
for family in 18 families:
    for K in K_grid:
        KMeans(K, n_init=N)        # ← 18 × |K_grid| KMeans calls
        EM_from_kmeans(family, K)
```

Smarter (what `auto_select_mixture` does):

```
for K in K_grid:
    KMeans(K, n_init=N)            # ← 1 × |K_grid| KMeans calls
    for family in 18 families:
        EM_from_shared_init(family, K)
```

For each $K$ in `K_grid`, KMeans runs **once** with `n_init`
kmeans++ restarts (kept best by inertia). The resulting labels seed
every HDDC variant via `init_params=labels`; the resulting centroids
seed every GMM variant via `means_init=centroids`. Each family then
does one EM refinement.

In practice the shared-KMeans implementation finishes in tens of
seconds on a laptop where the naive nested loop would take minutes
(KMeans dominates the wall clock when `K_grid` is non-trivial and
`p` is large, so factoring it out is a sizeable speedup).

### 1.4 Quickstart (CLI)

The module ships a tiny CLI for sanity-checking on built-in sklearn
datasets:

```bash
$ python tools/auto_mixture.py digits --k-min 6 --k-max 14 --n-init 5
dataset=digits  n=1797  p=64  K_true=10

best: GMM (diag)  K=12  ICL=281826.27
n_fits=90  elapsed=66.4s

top 8 by ICL:
  GMM (diag)         K= 12  ICL=281826.27
  GMM (diag)         K= 14  ICL=302510.58
  GMM (diag)         K=  6  ICL=324667.91
  GMM (diag)         K= 10  ICL=334514.47
  GMM (diag)         K=  8  ICL=343575.52
  GMM (tied)         K= 14  ICL=393228.37
  GMM (tied)         K= 12  ICL=396006.75
  GMM (tied)         K= 10  ICL=397056.04
```

Use ``--hddc-only`` or ``--gmm-only`` to restrict the search;
``iris``, ``wine``, ``olivetti`` are also available out of the box.

The CLI has a **predict mode** that loads a previously-trained result
and emits labels / probabilities for new data:

```bash
# Train and pickle the winning fit:
python tools/auto_mixture.py digits --output best.pkl

# Predict on new data using that fit:
python tools/auto_mixture.py --load best.pkl --input new_X.npz \
                             --output preds.npz
```

### 1.5 API

```python
from auto_mixture import (
    auto_select_mixture,
    GMM_COVARIANCE_TYPES,
    HDDC_MODELS,
)

result = auto_select_mixture(
    X,
    K_grid=None,             # None → estimate_k_range(X) picks the grid
    criterion="icl",         # "icl" (default) or "bic"
    n_init=5,                # kmeans++ budget per K
    max_iter=200,            # EM iteration cap per family
    cattell_threshold=0.5,   # HDDC signal-dim selector (see docs/HDDC.md §3)
    reg_covar_gmm=None,      # float, {cov: float}, or None (per-type defaults:
                             #   1e-1 for "full", 1e-3 otherwise)
    # Optional restrictions:
    # gmm_families=("diag", "full"),
    # hddc_models=("AVV", "AEE"),
)

print(result.family)         # e.g. "HDDC (AVV)"
print(result.K)              # e.g. 10
print(result.score)          # winning criterion value (lower is better)
print(result.criterion)      # "icl" or "bic"
print(result.icl)            # back-compat alias for `score`
labels = result.fit.predict(X)

# All criterion values that were tested:
for (family, K), s in sorted(result.score_grid.items(),
                              key=lambda kv: kv[1])[:5]:
    print(f"{family:20s} K={K:3d}  {result.criterion.upper()}={s:.2f}")
```

`result.score_grid` is keyed by `(family, K)`; `result.icl_grid` is a
back-compat alias kept for callers from before the `criterion` knob
was added.

The returned object also reports `n_fits` (how many configurations
actually succeeded — some HDDC sub-models can fail on small clusters,
and the loop logs+skips rather than aborts) and `elapsed` (wall clock,
in seconds).

The `cattell_threshold=0.5` default is the HDDC scree-rule sensitivity
and is **not** a noise-knob — it changes `d_k`, which changes
`_n_parameters`, which changes BIC and ICL. The rationale and
HDclassif comparison live in [`../docs/HDDC.md`](../docs/HDDC.md) §3.

### 1.6 When does what win?

The selector empirically rewards model parsimony in proportion to
the data's effective dimensionality:

| Dataset | $n$, $p$, true $K$ | Typical winner |
| --- | --- | --- |
| 1D synthetic Student mixture | $n=24{,}000$, $p=1$, $K^{\star}=3$ | `GMM (full)` $K \approx K^{\star}$ |
| `load_digits` | $n=1797$, $p=64$, $K^{\star}=10$ | `GMM (diag)` or `HDDC (AVV)` near $K^{\star}$ |
| Olivetti faces (PCA→99) | $n=100$, $p=99$, $K^{\star}=10$ | `HDDC (AVV)` near $K^{\star}$ |

Roughly: as $p / n$ grows, the winning family migrates from
full-$\Sigma$ GMM to diagonal GMM to one of the HDDC sub-models.
That's the curse-of-dimensionality story written out as a
model-selection result.

### 1.7 Relationship to the two core PRs

`auto_select_mixture` is **not** itself one of the two PRs to
scikit-learn (`pr_gmm_icl`, `pr_hddc`). It is a **downstream
consumer** of both: it depends on the ICL method added by
`pr_gmm_icl` and on the `HighDimensionalGaussianMixture` estimator
added by `pr_hddc`.

The companion code in `tools/auto_mixture.py` is therefore
positioned as:

- a **reference implementation** that demonstrates how the two PRs
  compose, and
- a **candidate for a future sklearn helper** (e.g.
  `sklearn.mixture.select_by_icl(X, K_grid=...)`) if and when the
  two core PRs land.

## Part 2. `estimate_k_range` — pick `K_grid` automatically

A 100-line companion module that picks a sensible upper bound `K_max`
for the search grid of `auto_select_mixture` in well under a second,
without fitting a single mixture model.

### 2.1 Algorithm (q75-of-relative-gains elbow rule on k-means++ seeding)

The recipe (exact constants taken from the user's spec):

```
min_cluster_size = 10
K_hard_max       = min(100, n // min_cluster_size)
epsilon          = 0.02
patience         = 3
R                = 20
```

1. **R independent runs of kmeans++ seeding.** For each run
   `r = 1..R`:
   - Pick the first centre uniformly at random.
   - Add centres one at a time using the kmeans++ rule (sample
     proportional to squared distance to the nearest existing
     centre).
   - Record the seeding inertia
     $\Phi_K = \sum_i \min_{k \le K} \| x_i - c_k \|^2$
     after each new centre.
   - **No Lloyd's iterations.** Cost per run is
     $O(K_{\text{hard\,max}} \cdot n \cdot p)$.

2. **Relative gain per run.** For each step $K \to K+1$:

   $$
   g_K^{(r)} \;=\; \frac{\Phi_K^{(r)} - \Phi_{K+1}^{(r)}}{\Phi_K^{(r)}}.
   $$

3. **Aggregate across runs with the 75th percentile.**

   $$
   \widehat{g}_K \;=\; q_{0.75}\bigl(\{g_K^{(r)}\}_{r=1}^R\bigr).
   $$

   Using `q75` rather than the mean is robust to a few lucky
   first-picks (small gain at K=2) or unlucky ones (artificially
   small gain at a particular K). It says "even on three out of four
   runs, gain is below ε".

4. **Stop at first $K$ with `patience` consecutive flat gains.**

   $$
   K_{\text{stop}} \;=\; \min \bigl\{ K \;:\;
   \widehat{g}_{K-i} < \epsilon \text{ for } i = 0, \dots, \text{patience}-1\bigr\}.
   $$

5. **Return** `(K_min, K_max) = (2, K_stop)`. If the rule never fires
   within `K_hard_max`, the result hits the ceiling and the module
   logs a hint to bump `k_hard_max`.

### 2.2 Empirical sanity check

| Dataset | $n$ | $p$ | `K_true` | Returned `K_max` | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| iris    |  150 |   4 |  3 | 15 (cap) | small n, no flat region within 15 |
| digits  | 1797 |  64 | 10 | 24       | comfortably above $K_{\text{true}}$ |

Pattern: for "real" datasets with $n$ in the thousands, the rule
picks an upper bound a bit above $K_{\text{true}}$, which is the
right shape for an exhaustive search grid in `auto_select_mixture`.
On the small iris case it hits the `K_hard_max=15` ceiling — expected,
since the relative gains never drop below 2% on a clean 4-D dataset;
relax `epsilon` if that bothers you.

### 2.3 API

```python
from estimate_k_range import estimate_k_range

est = estimate_k_range(
    X,
    min_cluster_size=10,
    k_hard_max=100,
    epsilon=0.02,
    patience=3,
    n_runs=20,
    random_state=0,
)
print(est.K_min, est.K_max)            # (2, K_max)
print(est.gain_summary[:10])           # first 10 q75 gains
print(est.K_hard_max)                  # actual cap used
```

The returned `KRangeEstimate` dataclass also keeps the full
`(n_runs, K_hard_max)` inertia and gain tables, in case you want to
visualise the elbow.

### 2.4 CLI

```bash
python tools/estimate_k_range.py digits
# → K_grid = [2, 24]   (K_hard_max=100, epsilon=0.020, patience=3)
```

### 2.5 Integration with `auto_select_mixture`

`auto_select_mixture(X)` accepts `K_grid=None` (the default), which
triggers a single `estimate_k_range(X)` call to choose the grid:

```python
from auto_mixture import auto_select_mixture

# Old explicit grid:
res = auto_select_mixture(X, K_grid=range(2, 16))

# New default — estimates K_grid from X via kmeans++ elbow:
res = auto_select_mixture(X)
```

Same flag exists in the CLI: omit `--k-min` and `--k-max` and the
grid is estimated.

### 2.6 Why this is principled (and what the gains *mean*)

A kmeans++ seeding inertia drop $g_K$ is a **lower bound** on the
inertia drop of a fully-EM-trained K-means at the same K (kmeans++
seeding gives a $O(\log K)$-competitive initial state; Lloyd's
iterations only improve it). So if even the cheap seeding stops
gaining ≥ 2% per cluster for several K in a row, no honest
clustering at higher K is going to fit *meaningfully* better.

Calling out three knobs for the curious:

- **`epsilon = 0.02`**: the threshold for "negligible". 2% is the
  practical conservative pick; tighter (1%) lets the grid grow
  further; looser (5%) cuts the grid faster.
- **`patience = 3`**: how many consecutive flat steps trigger the
  stop. 3 prevents a single low-gain K from cutting the grid in the
  middle of a multi-step plateau-then-recovery pattern.
- **`R = 20`**: trades runtime for stability of the q75 estimator.
  Below ~10 the q75 is jittery; above ~30 the marginal gain is small.
  Stable K relative to seed is already a sign of good clustering.

## See also

- [`../docs/INFORMATION_CRITERIA.md`](../docs/INFORMATION_CRITERIA.md) §2 — the
  one-page derivation of $\mathrm{ICL} = \mathrm{BIC} + 2H$.
- [`../docs/INFORMATION_CRITERIA.md`](../docs/INFORMATION_CRITERIA.md) §3 — the
  PSNC y-axis convention used to render ICL curves on a shared axis.
- [`../docs/HDDC.md`](../docs/HDDC.md) — naming-scheme tradeoffs for the
  14 HDDC sub-models swept by `auto_select_mixture`, the per-row
  parameter-count audit (which BIC and ICL inherit), and the
  Cattell scree rule behind the `cattell_threshold` knob.
- [`../docs/FOR_REVIEWERS.md`](../docs/FOR_REVIEWERS.md) — the broader
  scientific argument for why ICL beats BIC on heavy-tailed data.
