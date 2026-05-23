"""
================================================================
HDDC vs Gaussian Mixture on handwritten digits (parsimony at p=64)
================================================================

This example compares
:class:`~sklearn.mixture.HighDimensionalGaussianMixture` (HDDC,
sub-model "AVV") against
:class:`~sklearn.mixture.GaussianMixture` (covariance_type="diag")
on the 8x8 handwritten digits dataset (``sklearn.datasets.load_digits``).

Two regimes are compared:

1. **K known** — both estimators are given the true number of clusters
   (K = 10, one per digit class).
2. **K unknown** — both estimators sweep K over a grid and pick the
   best by ICL (lower is better).

Clustering quality is measured by

- NMI: normalized mutual information between cluster labels and true
  classes,
- ARI: adjusted Rand index,
- ACC: clustering accuracy via the Hungarian assignment between
  cluster labels and class labels.

The figure is a single bar chart with the six metrics (NMI / ARI /
ACC × {K known, K = ICL-selected}) for both estimators.

To run this example, you need scikit-learn with both ICL on
``GaussianMixture`` and the ``HighDimensionalGaussianMixture``
estimator installed.
"""
# %%
# Imports
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from sklearn.datasets import load_digits
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture, HighDimensionalGaussianMixture

log = logging.getLogger("plot_hddc_digits")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# %%
# Data
X, y = load_digits(return_X_y=True)
log.info(f"n_samples = {X.shape[0]}, n_features = {X.shape[1]}, "
      f"true K = {len(np.unique(y))}")


# %%
# Hungarian clustering accuracy
def clustering_accuracy(y_true, y_pred):
    K = max(int(y_true.max()), int(y_pred.max())) + 1
    M = np.zeros((K, K), dtype=int)
    for c, t in zip(y_pred, y_true):
        M[int(c), int(t)] += 1
    row_ind, col_ind = linear_sum_assignment(-M)
    return float(M[row_ind, col_ind].sum() / len(y_true))


def evaluate(fit, X, y):
    labels = fit.predict(X)
    return (
        normalized_mutual_info_score(y, labels),
        adjusted_rand_score(y, labels),
        clustering_accuracy(y, labels),
    )


# %%
# K known (oracle)
K_true = 10
gmm_known = GaussianMixture(
    n_components=K_true, covariance_type="diag",
    reg_covar=1e-3, random_state=0, n_init=5, max_iter=200,
).fit(X)

hgmm_known = HighDimensionalGaussianMixture(
    n_components=K_true, model="AVV",
    cattell_threshold=0.5, random_state=0, n_init=5, max_iter=100,
).fit(X)

# %%
# K unknown — pick by ICL
K_grid = [4, 6, 8, 10, 12, 14, 16, 18, 20]

best_gmm = min(
    (
        GaussianMixture(
            n_components=K, covariance_type="diag",
            reg_covar=1e-3, random_state=0, n_init=5, max_iter=200,
        ).fit(X)
        for K in K_grid
    ),
    key=lambda g: g.icl(X),
)
best_hgmm = min(
    (
        HighDimensionalGaussianMixture(
            n_components=K, model="AVV",
            cattell_threshold=0.5, random_state=0, n_init=5, max_iter=100,
        ).fit(X)
        for K in K_grid
    ),
    key=lambda g: g.icl(X),
)

# %%
# Scores
known = {
    "GMM (diag)": evaluate(gmm_known, X, y),
    "HDDC (AVV)": evaluate(hgmm_known, X, y),
}
unknown = {
    "GMM (diag)": evaluate(best_gmm, X, y),
    "HDDC (AVV)": evaluate(best_hgmm, X, y),
}
log.info(f"GMM   K-known  NMI/ARI/ACC = {known['GMM (diag)']}")
log.info(f"HDDC  K-known  NMI/ARI/ACC = {known['HDDC (AVV)']}")
log.info(f"GMM   K*={best_gmm.n_components} ICL NMI/ARI/ACC = "
      f"{unknown['GMM (diag)']}")
log.info(f"HDDC  K*={best_hgmm.n_components} ICL NMI/ARI/ACC = "
      f"{unknown['HDDC (AVV)']}")

# %%
# Plot
fig, ax = plt.subplots(figsize=(10, 5.5))
labels = ["GMM (diag)", "HDDC (AVV)"]
x = np.arange(len(labels))
w = 0.13
metrics = ("NMI", "ARI", "ACC")
colors = {"NMI": "#FFCC00", "ARI": "#007AFF", "ACC": "#FF9500"}
offs_k = [-2.5 * w, -1.5 * w, -0.5 * w]
offs_u = [+0.5 * w, +1.5 * w, +2.5 * w]
for offs, store, scenario, hatch in [
    (offs_k, known,   "K known", None),
    (offs_u, unknown, "K=ICL",   "//"),
]:
    for i, m in enumerate(metrics):
        vals = [store[lab][i] for lab in labels]
        ax.bar(x + offs[i], vals, w, color=colors[m],
               alpha=0.85 if scenario == "K known" else 0.45,
               hatch=hatch, edgecolor=colors[m],
               label=f"{m} · {scenario}")
        for xi, v in zip(x, vals):
            ax.text(xi + offs[i], v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Agreement with ground truth (higher is better)")
ax.set_title("GMM vs HDDC on digits — Clustering Recovery")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
          frameon=False, fontsize=9)
fig.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.show()
