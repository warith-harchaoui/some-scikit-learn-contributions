# Some contributions to scikit-learn

**Author:** [Warith Harchaoui](https://www.linkedin.com/in/warith-harchaoui/) · warith@deraison.ai

**Special thanks to:** [Pierre-Alexandre Mattei](https://pamattei.github.io/) for fruitful discussions

Companion package for a set of coordinated pull requests to
[scikit-learn](https://scikit-learn.org): draft code, tests, doc
snippets, real-world demos, and technical material.

![ICL vs BIC on a Student-t mixture](figures/fig_icl_student.png)

> BIC and held-out likelihood over-count clusters on heavy-tailed
> data; ICL doesn't. These PRs make ICL a first-class option for
> `GaussianMixture` and introduce a new high-dimensional clustering
> estimator that ships with ICL out of the box.

## The PRs

| Order | Folder | What it adds | Status |
| --- | --- | --- | --- |
| 1 | [`pr_gmm_icl/`](pr_gmm_icl/) | **ICL PR** — `icl()` method on `GaussianMixture`, alongside `bic()` | ready |
| 2 | [`pr_hddc/`](pr_hddc/)       | **HDDC PR** — new estimator `HighDimensionalGaussianMixture` (14 sub-models) | API review pending |

Each `pr_*/` subfolder contains:

- the **code** to insert in the scikit-learn source tree,
- the **tests** to append,
- a **doc snippet** for `doc/modules/`,
- a **changelog entry** stub, and
- the **GitHub PR description** copy-pasta.

The PRs are independent **in code** — neither imports from the
other — but the opening order **`pr_gmm_icl` → `pr_hddc`** matters
for review hygiene: the ICL PR is the smaller, more reviewable
change and sets the `ICL = BIC + 2·H` sign convention that HDDC
reuses, so landing ICL first lets the HDDC PR cite it instead of
re-arguing the convention. HDDC ships next because it builds on
the same information-criterion story (`bic` and `icl` together).

## Repository layout

```
.
├── README.md
├── LICENSE                            <- BSD-3-Clause (scikit-learn compatible)
├── requirements.txt
├── .gitignore
│
├── pr_gmm_icl/                        <- "add ICL to GaussianMixture"
├── pr_hddc/                           <- "add HighDimensionalGaussianMixture"
│
├── figures/                           <- PR-supporting figures + their scripts (sklearn contrib scope)
│   ├── _style.py                      <- shared house style
│   ├── make_figures.py                <- synthetic figures
│   ├── real_world_examples.py         <- real-world demos (galaxies, digits, Olivetti)
│   └── *.png
│
├── docs/                              <- PR-related docs (sklearn reviewer scope)
│   ├── DATA_SCIENTIST.md              <- "what changes for my code"
│   ├── FOR_REVIEWERS.md               <- scientific argument & references
│   ├── INFORMATION_CRITERIA.md        <- why BIC and ICL (not held-out) + ICL sign convention + PSNC y-axis
│   └── HDDC.md                        <- naming schemes + Bouveyron Table 1 audit + Cattell scree rule for d_k
│
└── tools/                             <- downstream personal code (NOT in PR)
    ├── README.md                      <- auto_select_mixture + estimate_k_range docs
    ├── auto_mixture.py                <- auto_select_mixture: argmin-criterion across 4 GMM + 14 HDDC × K_grid (+ CLI)
    └── estimate_k_range.py            <- kmeans++ elbow rule that picks K_grid for auto_mixture (+ CLI)
```

**Repo organisation rule.** Everything in `pr_*/`, `figures/`, and
`docs/` is in scope for the two scikit-learn PRs. Everything in
`tools/` is **downstream personal work** — a reference
implementation of ideas (auto-select across mixture families,
kmeans++ elbow rule) that builds on top of the PRs but is not
proposed as an addition to sklearn itself.

## Quickstart (reproduce the figures)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Synthetic figures used throughout the docs:
python figures/make_figures.py

# Real-world demos (galaxies, digits, Olivetti faces):
python figures/real_world_examples.py
```

Both scripts are deterministic — same random seeds, byte-identical
output. Figures land alongside the scripts as PNGs.

## Reading order

- **Skim**: this README + [`docs/DATA_SCIENTIST.md`](docs/DATA_SCIENTIST.md)
  (~5 minutes).
- **Per-PR pitch**: [`pr_gmm_icl/README.md`](pr_gmm_icl/README.md),
  [`pr_hddc/README.md`](pr_hddc/README.md). Each is self-contained
  and is also the body pasted into the GitHub PR.
- **Reviewer / maintainer**: [`docs/FOR_REVIEWERS.md`](docs/FOR_REVIEWERS.md),
  then the per-PR `pr_*/README.md` files. The appendices
  ([`docs/INFORMATION_CRITERIA.md`](docs/INFORMATION_CRITERIA.md),
  [`docs/HDDC.md`](docs/HDDC.md))
  are pulled in by cross-references where the math needs to be exact.
- **Downstream tooling** (independent of the sklearn PRs):
  [`tools/README.md`](tools/README.md).

## Scientific argument

### The ICL PR — heavy-tail-robust model selection for `GaussianMixture`

[ICL PR](pr_gmm_icl/README.md)

scikit-learn ships BIC for selecting the number of components `K`
in a finite Gaussian mixture. BIC is consistent only when the data
really is a Gaussian mixture; on heavy-tailed data — which covers
most real-world distributions — BIC over-counts components, because
the extra "components" are absorbing the tails. The under-used fix
(known in the Bayesian and R communities, less so in Python) is
**ICL** (Biernacki, Celeux & Govaert 2000), which adds an entropy
penalty on the soft cluster assignments. The first PR
(`pr_gmm_icl`) makes ICL available next to BIC under the
scikit-learn convention `ICL = BIC + 2·H`, lower is better.

### The HDDC PR — `GaussianMixture` for `n << p`

[HDDC PR](pr_hddc/README.md)

The second PR (`pr_hddc`) generalises `GaussianMixture` to the
high-dimensional regime via Bouveyron-Girard-Schmid 2007's HDDC
family: 14 parsimonious sub-models that flatten the per-component
covariance into a low-rank signal subspace plus an isotropic noise
residual, instead of estimating `K · p (p+1) / 2` covariance
parameters from a smaller-than-`p` sample. Heavy-tail-robust model
selection matters even more in the Small-Data regime
(`n << p`); HDDC exposes `bic` and `icl` out of the box so both the
family **and** the number of clusters can be picked with the right
criterion.

## License

BSD-3-Clause, matching scikit-learn. See [`LICENSE`](LICENSE).
