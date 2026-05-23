# Information criteria — why, how, and how to plot them

**Author:** [Warith Harchaoui](https://www.linkedin.com/in/warith-harchaoui/)
**Special thanks to:** [Pierre-Alexandre Mattei](https://pamattei.github.io/) for fruitful discussions

This document covers everything around BIC and ICL that supports the
two PRs: the conceptual case for ICs over held-out likelihood, the
one-page ICL sign-convention derivation that aligns with sklearn's
`bic` method, and the Per-Sample Nats Criterion (PSNC) — the
directly interpretable y-axis used across all `K`-sweep figures in
this repository.

- [Part 1. Why BIC and ICL exist](#part-1-why-bic-and-icl-exist--and-why-you-should-not-compute-them-on-a-validation-set)
- [Part 2. The ICL sign convention](#part-2-the-icl-sign-convention)
- [Part 3. PSNC — Per-Sample Nats Criterion](#part-3-psnc--per-sample-nats-criterion)

## Part 1. Why BIC and ICL exist — and why you should *not* compute them on a validation set

A natural reviewer question: "I have data, I can split it, I can
compute the log-likelihood on the held-out part — why do I need BIC or
ICL at all?" This part gives the pedagogical answer.

![Held-out log-likelihood and BIC keep overshooting K=3 on heavy-tailed data; only ICL recovers the truth.](../figures/fig_holdout_vs_ic.png)

### 1.1 The two approaches, side by side

We are trying to pick a number of components `K` (and optionally a
sub-model) for a Gaussian mixture. Two recipes:

#### Recipe V — Validation log-likelihood

```python
X_tr, X_val = train_test_split(X)
scores = {}
for K in candidates:
    g = GaussianMixture(n_components=K).fit(X_tr)
    scores[K] = g.score(X_val)        # higher = better
best_K = max(scores, key=scores.get)
```

#### Recipe I — Information criterion

```python
scores = {}
for K in candidates:
    g = GaussianMixture(n_components=K).fit(X)   # all of X
    scores[K] = g.bic(X)               # or .icl(X), lower = better
best_K = min(scores, key=scores.get)
```

Both are honest model-selection recipes. They differ in **what
quantity they estimate** and **how they pay for the estimate**.

| | Recipe V (validation) | Recipe I (information criterion) |
| --- | --- | --- |
| Estimates | Generalization density `E_{x ~ f*}[log f_K(x)]` | Penalized in-sample fit (BIC: marginal evidence; ICL: completed evidence) |
| Uses for fitting | Training fold only | All data |
| Uses for scoring | Validation fold only | All data |
| Cost of selection | Lose `~30%` of data to validation, plus inflated variance | One parameter count per candidate; closed form |
| Cluster-aware? | No (only density-aware) | BIC no, ICL yes |
| Asymptotic guarantee | Converges to true generalization risk | BIC: true model; ICL: true clustering |
| Hand-tuning | Choose split ratio, choose seed, possibly k-fold | Choose K candidates only |
| Heavy-tail regime | Still overpicks K (see below) | BIC overpicks K; ICL does not |

So both are useful tools, but they are answering *different
questions*. BIC was not designed as a "cheap surrogate for validation
likelihood"; it was designed because validation likelihood is **not**
the right target for the question "how many clusters."

### 1.2 Why information criteria exist

Three intellectual reasons.

#### 1.2.1 They use all the data

Mixture models, and especially HDDC-style models in the `n << p`
regime, have *small effective sample size per component*. Splitting
30% off as validation can mean losing most of the data for the
smallest components. Information criteria fit on all of `X` and
compute the penalty in closed form, so no data is wasted.

#### 1.2.2 They target different optimums

These are not opinions, they are theorems:

- **BIC** is a Laplace approximation to `log p(X | K) =
  log integral p(X | theta, K) p(theta | K) d theta`. If your goal is
  to *identify the true model* in a nested family, BIC is consistent
  under regularity conditions (Schwarz, 1978 [Schwarz78]_).
- **ICL** is the same Laplace approximation but applied to the
  *completed* likelihood `p(X, Z | K)`, with `Z` the latent partition.
  If your goal is to *recover the true partition*, ICL is consistent
  (Biernacki, Celeux & Govaert, 2000 [BCG2000]_; Baudry et al.,
  2010 [Baudry2010]_).

Held-out log-likelihood estimates *predictive density*. It is a poor
target for the BIC / ICL questions. In particular: held-out
log-likelihood tends to **overestimate `K` on heavy-tailed data**, for
the exact same reason BIC does — extra components improve predictive
density even when they correspond to no real cluster.

#### 1.2.3 They are computationally cheap

Each criterion is one EM run plus an `O(K)` parameter count and an
`O(n K)` entropy. K-fold validation requires `k` EM runs per
candidate. For an HDDC grid over `(K, model)` with `K in {2..15}` and
14 sub-models, the difference is the difference between 210 EM runs
and `5 * 210 = 1050` EM runs.

### 1.3 Why you should *not* compute BIC / ICL on a validation set

This is the misconception worth heading off in the doc. "Validation
BIC" — meaning, fit on training data, then compute `bic` on
validation data — is **not a thing**, and using it produces incoherent
selection. Three reasons.

#### 1.3.1 The penalty assumes in-sample log-likelihood

`BIC` is

```
BIC = -2 log L(theta_hat) + nu log n
```

where `L(theta_hat)` is the *maximized* training likelihood and `n` is
the training sample size. The `-2 log L` term overstates fit because
`theta_hat` overfits; the `nu log n` term is calibrated to subtract
that overstatement. On a held-out set, `theta_hat` is no longer the
maximizer, so the bias `nu log n` is supposed to correct is **not
there to correct**. You end up over-penalizing.

The Laplace derivation (Kass & Raftery, 1995 [KR95]_) of BIC is
explicit about this: the integral is around the MLE on the *training*
data; subbing in a held-out log-likelihood breaks the approximation.

There is a second, deeper failure mode that matters specifically for
mixture model selection: the Laplace step also assumes the model is
**regular** (non-singular Fisher information). A finite Gaussian
mixture is non-regular as soon as a redundant component is present —
the parameter directions that merge two components or send one
weight to zero make the Fisher information singular. Under
non-regularity, the `nu log n` penalty is no longer the right BIC
correction; the literature has developed singular-BIC variants
(Keribin, 2000, *Consistent estimation of the order of mixture
models*; Drton & Plummer, 2017, *A Bayesian information criterion
for singular models*; Watanabe, 2013, WBIC). The practical
consequence on heavy-tailed data is that BIC keeps adding components
to model the tails — each spurious cluster reduces the deviance more
than `log n` can absorb. ICL's entropy term is what blocks that
drift.

#### 1.3.2 ICL inherits the BIC issue plus a third

ICL inherits the BIC issue (`nu log n` penalty assumes training
likelihood) and adds another: the entropy `H = -sum tau log tau` is
computed at the EM fixed point on the *fitted* data. On a validation
set the responsibilities `tau_ik` are computed from a model `theta_hat`
that was not fitted to those points; the entropy is no longer the
"completion uncertainty" the ICL approximation rests on.

#### 1.3.3 What you *should* do if you want held-out evaluation

Use the held-out log-likelihood **directly**, without an information-
criterion penalty:

```python
X_tr, X_val = train_test_split(X)
for K in candidates:
    g = GaussianMixture(n_components=K).fit(X_tr)
    print(K, g.score(X_val))         # this is fine
```

Aware that this answers the predictive-density question, not the
"how many clusters" question. For the latter, fit on all of `X` and
use `.icl(X)`.

For a more advanced treatment (Bayesian model evaluation, WAIC, PSIS-
LOO), see Vehtari, Gelman & Gabry (2017) [VGG17]_.

### 1.4 TL;DR

| You want to ... | Use |
| --- | --- |
| Predict future samples well | Held-out `score(X_val)` |
| Identify the true number of components, density-wise | BIC |
| Identify the true partition (clustering) | ICL |
| Combine in-sample fit + held-out evaluation | **Don't** compute information criteria on held-out data; report `score(X_val)` directly |

## Part 2. The ICL sign convention

Two conventions are common in the literature; we ship the one aligned
with `GaussianMixture.bic`. This part records the derivation.

![ICL = BIC + 2H decomposition: the entropy stack grows with K, ICL picks K=3](../figures/fig_icl_decomposition.png)

### 2.1 Setup

Mixture density `f(x; theta) = sum_k pi_k f_k(x; theta_k)`, with latent
assignment `z_i in {1, ..., K}`. The completed-data log-likelihood is

```
L_c(theta, z) = sum_{i=1}^n sum_{k=1}^K z_ik [log pi_k + log f_k(x_i; theta_k)]
```

Soft responsibilities at the EM fixed point: `tau_ik = P(z_i = k | x_i; theta_hat)`.

### 2.2 Two BIC conventions in the literature

- **Lower-is-better** (used by sklearn and most software):

  `BIC_LO(K) = -2 log L(theta_hat) + nu(K) log n`

- **Higher-is-better** (used by `mclust`, original Schwarz convention):

  `BIC_HI(K) = 2 log L(theta_hat) - nu(K) log n  =  -BIC_LO(K)`

These are mirror images via `BIC_HI = -BIC_LO`.

### 2.3 ICL, higher-is-better

Following Biernacki, Celeux & Govaert (2000) using the Laplace
approximation to `log p(x, z | K)` at the MAP allocation, *and then*
replacing `z_ik` by the soft `tau_ik`, one gets

```
ICL_HI(K) = log L(theta_hat) - (nu(K) / 2) log n  -  H
         = (1/2) BIC_HI(K)  -  H
```

with `H = - sum_i sum_k tau_ik log tau_ik >= 0`.

Some references absorb the `1/2` into the BIC: `ICL_HI = BIC_HI / 2 - H`.
Others write `ICL_HI = BIC_HI - 2 H` directly, treating "BIC" as
shorthand for `2 log L - nu log n`. Either way, the entropy enters
**negatively** when "higher is better."

### 2.4 ICL, lower-is-better (sklearn convention)

Multiply through by `-2`:

```
ICL_LO(K) = -2 ICL_HI(K)
         = -2 log L(theta_hat) + nu(K) log n  +  2 H
         = BIC_LO(K) + 2 H
```

This is the formula we ship in `GaussianMixture.icl`:

```python
def icl(self, X):
    _, log_resp = self._estimate_log_prob_resp(X)
    resp = np.exp(log_resp)
    entropy = -xlogy(resp, resp).sum()
    return self.bic(X) + 2.0 * entropy
```

### 2.5 Invariants

- `H >= 0`, with equality iff the partition is hard (`tau_ik in {0,1}`).
- Hence `ICL >= BIC`, equality iff the partition is hard.
- Adding a redundant component that splits the responsibilities of an
  existing component strictly increases `H` and hence `ICL`.
- The criterion is invariant under permutations of components, since
  `H` is.

### 2.6 Cross-check against `flexmix` (R) and `mclust` (R)

- `flexmix::ICL` returns a lower-is-better quantity equal to
  `BIC + 2 * sum_i sum_k tau_ik log(tau_ik) * (-1)`, i.e. `BIC + 2 H`.
- `mclust::icl` returns a higher-is-better quantity equal to
  `2 log L - nu log n - 2 H`, i.e. `BIC_HI - 2 H`.

Multiplying `mclust`'s formula by `-1` gives `-2 log L + nu log n + 2 H`,
which is `flexmix`'s formula. The two R packages agree up to sign,
and sklearn's `icl` matches `flexmix` (same convention as `bic`).

### 2.7 TL;DR

```
ICL(K) = BIC(K) + 2 H,    H = - sum tau log tau >= 0,    lower is better.
```

## Part 3. PSNC — Per-Sample Nats Criterion

*A directly interpretable scale for probabilistic models, training
dashboards, and model-selection plots.*

![Four curves (train ll, held-out ll, BIC, ICL) on one Per-Sample-Nats-Criterion y-axis, with the green Perfection floor at y=0 and the red Confusion ceiling at y=1.](../figures/fig_holdout_vs_ic.png)

### 3.1 One-sentence summary

**PSNC is a directly interpretable nats scale: it measures
probabilistic cost or uncertainty as a fraction of full uniform
confusion per sample.**

### 3.2 Introduction — why PSNC?

Probabilistic models do not only answer *which label?* or *which
cluster?* They answer with probabilities.

That is useful because probabilities tell us how confident, uncertain,
calibrated, or confused the model is. But the usual probabilistic
criteria are hard to read in their raw form:

- negative log-likelihood,
- log-loss,
- cross-entropy,
- entropy,
- BIC,
- ICL.

A validation negative log-likelihood of `5,500` is mathematically
valid, but it is not directly interpretable. Is it good? Is it close
to random? Is it comparable to another dataset? What if one task has
2 classes and another has 50?

PSNC is meant to make these quantities readable.

It expresses probabilistic cost in units of:

> one full confusion event per sample.

That makes it useful not only for final reports, but also for
**prototyping**, **debugging**, **training dashboards**, **model
monitoring**, and **quick comparisons during experimentation**.

Instead of asking:

> what does this raw log-loss mean?

PSNC lets us ask:

> how many confusion units per sample is the model paying?

### 3.3 What is a nat?

A **nat** is a unit of information.

It is the natural-log equivalent of a bit.

- bits use $\log_2$,
- nats use the natural logarithm $\log$, also written $\ln$.

In machine learning, negative log-likelihood and cross-entropy are
usually computed with natural logarithms. That means they are measured
in **nats**.

For one event with probability $p$, the information cost is:

$$
I(p) = -\log p.
$$

Examples:

| Probability assigned to the true event | Cost in nats | Interpretation |
|---:|---:|---|
| $1$ | $0$ | perfect certainty on the correct answer |
| $1/2$ | $\log 2$ | binary uniform uncertainty |
| $1/K$ | $\log K$ | uniform uncertainty over $K$ possibilities |
| close to $0$ | very large | confidently wrong |

This is the core intuition behind PSNC.

If a model assigns probability $1$ to the correct answer, it pays $0$
nats. If it assigns probability $1/K$, like a uniform random guess
among $K$ possibilities, it pays $\log K$ nats. So $\log K$ is the
natural unit of full $K$-way confusion.

### 3.4 Definition

The **Per-Sample Nats Criterion** is:

$$
\mathrm{PSNC}=\frac{\mathcal{L}}{n\log R}.
$$

where:

- $\mathcal{L}$ is a non-negative probabilistic cost expressed in nats,
- $n$ is the number of samples,
- $R$ is the number of reference outcomes.

For supervised classification:

$$
R = C,
$$

where $C$ is the number of classes.

For clustering entropy:

$$
R = K^{\star},
$$

where $K^{\star}$ is a meaningful reference number of clusters.

The interpretation is direct:

| PSNC | Meaning |
|---:|---|
| $0$ | perfect probabilistic prediction, or zero uncertainty |
| $0 < \mathrm{PSNC} < 1$ | better than uniform confusion |
| $1$ | uniform confusion over the reference outcomes |
| $\mathrm{PSNC} > 1$ | worse than uniform confusion |

This is the whole point.

PSNC is not a new criterion. It is a **directly interpretable
normalization** of criteria already used in probabilistic modeling.

### 3.5 Supervised classification

In supervised classification, the cleanest criterion is negative
log-likelihood, also called log-loss or cross-entropy.

For each sample $i$, let $p_i$ be the probability assigned by the
model to the true label.

The negative log-likelihood is:

$$
\mathrm{NLL}=\sum_{i=1}^{n} -\log p_i.
$$

If there are $C$ classes, the uniform random classifier assigns:

$$
p_i = \frac{1}{C}
$$

to the true class. Its per-sample cost is:

$$
-\log\left(\frac{1}{C}\right) = \log C.
$$

Therefore supervised PSNC is:

$$
\mathrm{PSNC}_{\mathrm{classif}}=\frac{\mathrm{NLL}}{n\log C}.
$$

Equivalently, if $\overline{\mathrm{NLL}}$ is the average log-loss per
sample:

$$
\mathrm{PSNC}_{\mathrm{classif}}=\frac{\overline{\mathrm{NLL}}}{\log C}.
$$

It measures the model's log-loss as a fraction of the
uniform-classifier log-loss.

#### Example 1 — one-hot prediction

Suppose a classifier predicts the true class with probability $1$ for
every sample. Then each sample costs $-\log(1)=0$, so
$\mathrm{PSNC}=0$. This is the ideal case. The model is not confused
at all.

#### Example 2 — uniform prediction: $1/C$

Suppose there are $C=10$ classes. A completely uniform classifier
assigns probability $1/10$ to every class. For the true class, the
cost per sample is $-\log(1/10)=\log 10$. So

$$
\mathrm{PSNC}=\frac{\log 10}{\log 10} = 1.
$$

That is why PSNC = 1 means full uniform confusion. It is not
arbitrary. It is exactly the cost of random guessing with uniform
probabilities.

#### Example 3 — better than uniform

Again take $C=10$ classes. Suppose the model assigns probability $0.5$
to the true class on average. The per-sample cost is $-\log(0.5)=\log 2$,
so

$$
\mathrm{PSNC}=\frac{\log 2}{\log 10} \approx 0.30.
$$

The model pays about $0.30$ confusion units per sample. That is much
more readable than saying only "average log-loss = 0.693."

#### Example 4 — worse than uniform

Suppose there are $C=10$ classes but the model assigns probability
$0.01$ to the true class on average. Then $-\log(0.01)=\log 100$ and

$$
\mathrm{PSNC}=\frac{\log 100}{\log 10} = 2.
$$

The model pays $2$ confusion units per sample. That is worse than
uniform guessing — a useful dashboard signal that the model is
systematically assigning too little probability to the truth.

#### Example 5 — why calibration matters

Accuracy only checks whether the top label is correct. Log-loss checks
whether the probability assigned to the true label is reasonable.

Consider two classifiers on the same correctly classified sample:

| Model | Predicted probability for true class | Cost |
|---|---:|---:|
| A | $0.51$ | $-\log(0.51) \approx 0.67$ |
| B | $0.99$ | $-\log(0.99) \approx 0.01$ |

Both models are correct. Accuracy treats them the same. PSNC does
not — it rewards the calibrated, confident, correct prediction.

Now consider two wrong predictions:

| Model | Probability assigned to true class | Cost |
|---|---:|---:|
| C | $0.40$ | $-\log(0.40) \approx 0.92$ |
| D | $0.001$ | $-\log(0.001) \approx 6.91$ |

Both models are wrong. Accuracy treats them the same. PSNC does
not — it strongly penalizes confident mistakes.

That is why PSNC is useful for calibration, monitoring, and training
dashboards.

#### Practical dashboard interpretation

For supervised classification, PSNC gives a simple reading:

| PSNC range | Dashboard interpretation |
|---:|---|
| near $0$ | excellent probabilistic predictions |
| $0.2$ to $0.5$ | strong model, far better than uniform |
| around $1$ | close to uniform guessing |
| above $1$ | worse than uniform; investigate immediately |
| increasing during training | overfitting, drift, instability, or miscalibration may be happening |
| decreasing during training | probabilistic predictions are improving |

This makes PSNC useful while building models, not only after the
final evaluation.

### 3.6 Unsupervised clustering

The same idea applies to clustering when the model produces soft
assignments. Suppose each sample has assignment probabilities across
clusters:

$$
\tau_{i1}, \tau_{i2}, \ldots, \tau_{iK}.
$$

The assignment entropy is:

$$
H=-\sum_{i=1}^{n}\sum_{k=1}^{K}\tau_{ik}\log \tau_{ik}.
$$

By convention, terms with $\tau_{ik}=0$ contribute $0$ to the entropy.

If a sample is assigned to one cluster with probability $1$, the
entropy is $0$. If a sample is uniformly spread over $K$ clusters,
each cluster has probability $1/K$ and the entropy is $\log K$.

For a reference number of clusters $K^{\star}$:

$$
\mathrm{PSNC}_{H}=\frac{H}{n\log K^{\star}}.
$$

#### Example 6 — hard clustering: one-hot assignments

If every sample is assigned with probability $1$ to one cluster and
$0$ to all others, the assignments are one-hot. Then $H=0$ and
$\mathrm{PSNC}_{H}=0$. The clustering has no assignment uncertainty.

#### Example 7 — uniform cluster uncertainty: $1/K^{\star}$

Suppose the reference is $K^{\star}=5$ clusters. If every sample is
uniformly uncertain over the 5 reference clusters, then each
assignment probability is $1/5$. The per-sample entropy is $\log 5$
and

$$
\mathrm{PSNC}_{H}=\frac{\log 5}{\log 5} = 1.
$$

That is full assignment confusion.

#### Example 8 — soft but useful clustering

Suppose $K^{\star}=5$ and $\mathrm{PSNC}_{H}=0.25$. This means the
clustering is not perfectly hard, but its assignment uncertainty is
only one quarter of full 5-way uniform confusion. That is a readable
statement — much clearer than saying $H = 348.2$, which depends on
the number of samples and the number of reference clusters.

### 3.7 BIC, ICL, and entropy

For mixture models, BIC and ICL are often reported on the deviance
scale:

$$
\mathrm{BIC}=-2\log L + \mathrm{penalty},
$$

$$
\mathrm{ICL}=\mathrm{BIC}+2H.
$$

To express them in nats, divide by 2:

$$
\mathrm{BIC}^{\mathrm{nat}}=\frac{\mathrm{BIC}}{2},
$$

$$
\mathrm{ICL}^{\mathrm{nat}}=\frac{\mathrm{ICL}}{2}.
$$

Then:

$$
\mathrm{ICL}^{\mathrm{nat}}=\mathrm{BIC}^{\mathrm{nat}} + H.
$$

This identity is useful because the difference between ICL and BIC is
exactly the assignment entropy in nats:

$$
\mathrm{ICL}^{\mathrm{nat}} - \mathrm{BIC}^{\mathrm{nat}} = H.
$$

Therefore:

$$
\frac{\mathrm{ICL}^{\mathrm{nat}} - \mathrm{BIC}^{\mathrm{nat}}}{n\log K^{\star}}=\frac{H}{n\log K^{\star}}=\mathrm{PSNC}_{H}.
$$

This is the clean PSNC interpretation in the BIC/ICL plot.

The BIC and ICL curves themselves can still be divided by
$n\log K^{\star}$ for visualization, but their absolute values should
not be interpreted as pure confusion. The directly interpretable
quantity is the entropy component, or equivalently the normalized
ICL-BIC gap. That was the original motivation for using PSNC in
clustering plots.

### 3.8 Why PSNC is useful

#### 1. It is directly interpretable

PSNC = 1 has a concrete meaning:

> uniform confusion over the reference outcomes.

This is much more meaningful than an arbitrary raw NLL or entropy
value.

#### 2. It compares across sample sizes

Raw NLL and entropy grow with the number of samples. PSNC divides by
$n$. That makes curves easier to compare across validation sets,
experiments, and dashboards.

#### 3. It compares across numbers of classes or clusters

A 2-class problem and a 20-class problem do not have the same
uncertainty scale. PSNC divides by $\log R$. That makes the
interpretation stable across tasks.

#### 4. It is useful during training

PSNC can be plotted during training as a live metric. It tells you
whether the model is moving toward or away from uniform confusion. It
is useful for early prototyping, training dashboards, validation
monitoring, calibration checks, data drift detection, comparing model
versions, and explaining progress to non-specialists.

#### 5. It highlights calibration

Two models can have the same accuracy but very different log-loss.
PSNC makes that visible in a normalized unit. It rewards putting high
probability on the true label and penalizes confident mistakes.

#### 6. It is simple

There is no new statistical machinery. Use the probabilistic criterion
you already trust. Put it in nats. Divide by $n\log R$. That is PSNC.

### 3.9 Minimal implementation idea

For classification:

$$
\mathrm{PSNC}_{\mathrm{classif}}=\frac{\mathrm{NLL}}{n\log C}.
$$

For clustering entropy:

$$
\mathrm{PSNC}_{H}=\frac{H}{n\log K^{\star}}.
$$

For the ICL-BIC entropy gap:

$$
\mathrm{PSNC}_{H}=\frac{\mathrm{ICL}^{\mathrm{nat}} - \mathrm{BIC}^{\mathrm{nat}}}{n\log K^{\star}}.
$$

If BIC or ICL is reported on the deviance scale, first convert to
nats:

$$
\mathrm{criterion}^{\mathrm{nat}}=\frac{\mathrm{criterion}^{\mathrm{deviance}}}{2}.
$$

## References

.. [Baudry2010] Baudry, J.-P., Raftery, A. E., Celeux, G., Lo, K., &
   Gottardo, R. (2010). Combining mixture components for clustering.
   *Journal of Computational and Graphical Statistics*, 19(2), 332-353.

.. [BCG2000] Biernacki, C., Celeux, G., & Govaert, G. (2000).
   Assessing a mixture model for clustering with the integrated
   completed likelihood. *IEEE TPAMI*, 22(7), 719-725.

.. [KR95] Kass, R. E., & Raftery, A. E. (1995). Bayes Factors.
   *Journal of the American Statistical Association*, 90(430),
   773-795.

.. [Schwarz78] Schwarz, G. (1978). Estimating the dimension of a
   model. *Annals of Statistics*, 6(2), 461-464.

.. [VGG17] Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical
   Bayesian model evaluation using leave-one-out cross-validation
   and WAIC. *Statistics and Computing*, 27(5), 1413-1432.

.. [Arlot10] Arlot, S., & Celisse, A. (2010). A survey of
   cross-validation procedures for model selection. *Statistics
   Surveys*, 4, 40-79.
