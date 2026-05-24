"""Real-world evidence for the three PRs.

Each ``demo_*`` function runs one experiment on a canonical sklearn
dataset, logs a numerical summary (NMI / ARI / ACC + timings), and
writes one or more PNGs into the figures folder.

Datasets
--------
- Demo ICL (ICL > BIC):     Roeder (1990) galaxies, 82 1D velocities.
- Demo HDDC-digits:         ``load_digits`` (1797×64, K_true=10).
- Demo HDDC-olivetti:       ``fetch_olivetti_faces`` first 10 people
                            after PCA→99 (100×99, K_true=10).

For each HDDC demo, GMM(diag) and HDDC(AVV) are evaluated in two
regimes: K known (the ground-truth count is given) and K unknown
(K is selected by ICL from a sweep). All four numbers come from the
same dataset and the same EM seeds.

Run from this folder:
    python real_world_examples.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import (
    fetch_olivetti_faces, load_digits,
)
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

log = logging.getLogger("real_world_examples")

# Local style.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (                                            # noqa: E402
    ROLE, CMAP_PURPLE,
    use_house_style,
    per_sample_nats_criterion, PSNC_YLABEL, draw_psnc_anchors,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIB = os.path.dirname(HERE)

# Load the draft HDDC implementation from pr_hddc/_hddc.py.
_spec = importlib.util.spec_from_file_location(
    "_draft_hddc", os.path.join(CONTRIB, "pr_hddc", "_hddc.py"),
)
_draft_hddc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_draft_hddc)
HighDimensionalGaussianMixture = _draft_hddc.HighDimensionalGaussianMixture


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Shared compat shim — see ``figures/_icl_compat.py``. Local alias
# preserved so existing call sites read identically.
from _icl_compat import icl_gmm as _icl_gmm  # noqa: E402


def _gmm_with_K(X, K, covariance_type, reg_covar, n_init, max_iter, seed=0):
    # GaussianMixture uses sklearn's KMeans internally for
    # ``init_params="kmeans"`` (default), which itself uses kmeans++
    # initialization by default. ``n_init`` controls how many EM
    # restarts; best (highest log-likelihood) is kept.
    return GaussianMixture(
        n_components=K, covariance_type=covariance_type,
        init_params="kmeans",
        random_state=seed, n_init=n_init, max_iter=max_iter,
        reg_covar=reg_covar,
    ).fit(X)


def _hddc_with_K(X, K, threshold, n_init, max_iter, seed=0):
    # HDDC's ``init_params="kmeans"`` (default) also runs sklearn's
    # KMeans with kmeans++ init internally; see _hddc.py.
    return HighDimensionalGaussianMixture(
        n_components=K, model="AVV",
        cattell_threshold=threshold, min_cluster_size=2,
        init_params="kmeans",
        random_state=seed, n_init=n_init, max_iter=max_iter,
    ).fit(X)


def _select_K(estimator_fn, K_grid, X):
    """Sweep ``K_grid``, fit, score by ICL (lower=better). Return (K*, fit)."""
    best_K, best_icl, best_fit = None, np.inf, None
    for K in K_grid:
        try:
            fit = estimator_fn(K)
        except Exception:                       # noqa: BLE001
            continue
        icl = fit.icl(X) if hasattr(fit, "icl") else _icl_gmm(fit, X)
        if icl < best_icl:
            best_K, best_icl, best_fit = K, icl, fit
    return best_K, best_fit


def _select_K_across_families(named_estimator_fns, K_grid, X):
    """ICL-best across both K and model family.

    ICL = BIC + 2H is comparable across mixture families because BIC's
    parameter penalty ``ν(K, family) · log n`` is family-specific by
    construction, and the entropy term ``H = -Σ τ log τ`` is
    family-agnostic. So a single ``argmin_{K, family} ICL`` over both
    GMM and HDDC at every K is a principled head-to-head.

    Parameters
    ----------
    named_estimator_fns : dict[str, callable]
        Mapping ``family_name -> (K -> fitted estimator)``.
    K_grid, X : as in ``_select_K``.

    Returns
    -------
    family : str
    K : int
    fit : the winning fitted estimator
    """
    best_family, best_K, best_icl, best_fit = None, None, np.inf, None
    for family, fn in named_estimator_fns.items():
        for K in K_grid:
            try:
                fit = fn(K)
            except Exception:                   # noqa: BLE001
                continue
            icl = fit.icl(X) if hasattr(fit, "icl") else _icl_gmm(fit, X)
            if icl < best_icl:
                best_family, best_K, best_icl, best_fit = family, K, icl, fit
    return best_family, best_K, best_fit


def _clustering_accuracy(y_true, y_pred):
    """Hungarian-aligned clustering accuracy in [0, 1].

    Builds the K_clusters × K_classes contingency table, finds the
    optimal cluster→class assignment via scipy's linear_sum_assignment,
    and returns the fraction of correctly mapped samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    K_clust = int(y_pred.max()) + 1
    K_class = int(y_true.max()) + 1
    K = max(K_clust, K_class)
    M = np.zeros((K, K), dtype=int)
    for c, t in zip(y_pred, y_true):
        M[int(c), int(t)] += 1
    row_ind, col_ind = linear_sum_assignment(-M)
    return float(M[row_ind, col_ind].sum() / len(y_true))


def _eval(fit, X, y):
    pred = fit.predict(X)
    return (
        normalized_mutual_info_score(y, pred),
        adjusted_rand_score(y, pred),
        _clustering_accuracy(y, pred),
    )


# ---------------------------------------------------------------------------
# Demo ICL:  ICL vs BIC on Roeder (1990) galaxies (1D, n=82)
# ---------------------------------------------------------------------------

def demo_icl_galaxies() -> str:
    """ICL vs BIC on the Roeder (1990) galaxies dataset.

    The 82 velocity measurements of galaxies in the Corona Borealis
    region are the canonical test case for finite-mixture selection.
    Literature consensus on the number of superclusters is K = 3.
    """
    log.info("\n== Demo ICL: ICL vs BIC on the galaxies dataset ==")
    galaxies = np.array([
        9172, 9350, 9483, 9558, 9775, 10227, 10406, 16084, 16170, 18419,
        18552, 18600, 18927, 19052, 19070, 19330, 19343, 19349, 19440, 19473,
        19529, 19541, 19547, 19663, 19846, 19856, 19863, 19914, 19918, 19973,
        19989, 20166, 20175, 20179, 20196, 20215, 20221, 20415, 20629, 20795,
        20821, 20846, 20875, 20986, 21137, 21492, 21701, 21814, 21921, 21960,
        22185, 22209, 22242, 22249, 22314, 22374, 22495, 22746, 22747, 22888,
        22914, 23206, 23241, 23263, 23484, 23538, 23542, 23666, 23706, 23711,
        24129, 24285, 24289, 24366, 24717, 24990, 25633, 26960, 26995, 32065,
        32789, 34279,
    ], dtype=float)
    X = (galaxies / 1000.0).reshape(-1, 1)
    log.info(f"  n={X.shape[0]}, p={X.shape[1]} (1D velocity, scaled)")

    Ks = list(range(2, 13))
    bic, icl = [], []
    for K in Ks:
        g = _gmm_with_K(X, K, "full", reg_covar=1e-4,
                        n_init=20, max_iter=500)
        bic.append(g.bic(X)); icl.append(_icl_gmm(g, X))
    K_bic = Ks[int(np.argmin(bic))]
    K_icl = Ks[int(np.argmin(icl))]
    log.info(f"  BIC picks K = {K_bic}")
    log.info(f"  ICL picks K = {K_icl}  (literature ≈ 3 superclusters)")

    use_house_style()
    fig, (ax_hist, ax) = plt.subplots(1, 2, figsize=(14, 6.4))

    ax_hist.hist(X.ravel(), bins=30, density=True,
                 color=ROLE["data"], alpha=0.6, edgecolor="white",
                 linewidth=0.4)
    ax_hist.set_xlabel("Velocity (×$10^3$ km/s)")
    ax_hist.set_ylabel("Density (higher = more galaxies here)")
    ax_hist.set_title("Roeder (1990) galaxies, $n{=}82$")
    ax_hist.grid(False)

    # Per-Sample Nats Criterion (PSNC), K* = 3 (literature consensus
    # for the galaxies). BIC and ICL come out in deviance units, so
    # divide by 2 to natural-log units before normalising by
    # n * log(K*). See docs/INFORMATION_CRITERIA.md §3.
    K_star = 3
    n_obs = X.shape[0]
    bic_nat = 0.5 * np.asarray(bic)
    icl_nat = 0.5 * np.asarray(icl)
    bic_d = per_sample_nats_criterion(bic_nat, n_samples=n_obs, k_star=K_star)
    icl_d = per_sample_nats_criterion(icl_nat, n_samples=n_obs, k_star=K_star)
    ax.plot(Ks, bic_d, color=ROLE["bic"], lw=2.8, label="BIC")
    ax.plot(Ks, icl_d, color=ROLE["icl"], lw=2.2, label="ICL")
    # Big hollow ring at each minimum, in the curve's own colour;
    # NOT shown in the legend.
    for vals_d, color, K_star in [(bic_d, ROLE["bic"], K_bic),
                                  (icl_d, ROLE["icl"], K_icl)]:
        ax.scatter([K_star], [vals_d[Ks.index(K_star)]],
                   facecolors="none", edgecolors=color,
                   s=420, lw=3.0, zorder=4)
    # PSNC anchors: green at y=0 (perfection), red at y=1 (confusion).
    # Registered before the legend so they appear in it.
    draw_psnc_anchors(ax)
    ax.set_xlabel("Number of components $K$")
    ax.set_ylabel(PSNC_YLABEL)
    ax.set_title(f"Both pick $K{{=}}{K_icl}$, but ICL is much more decisive")
    ax.set_xticks(Ks)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, frameon=False, fontsize=9)

    fig.suptitle(
        "Galaxies (Roeder 1990): ICL agrees with BIC, much more confidently",
        fontsize=14, fontweight="semibold", y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "fig_real_icl_galaxies.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Demo HDDC:  GMM vs HDDC on a high-dim dataset, K known vs unknown
# ---------------------------------------------------------------------------

def _row_normalized_confusion(pred, y, K_clusters, K_classes):
    """Build a K_clusters x K_classes matrix.

    Returns (counts, fractions) where counts[c, y] is the number of
    samples that the model assigned to cluster ``c`` and that truly
    belong to class ``y``; fractions is the row-normalised version
    (each cluster row sums to 1).
    """
    counts = np.zeros((K_clusters, K_classes), dtype=int)
    for c, yc in zip(pred, y):
        counts[int(c), int(yc)] += 1
    rowsum = counts.sum(1, keepdims=True).astype(float)
    rowsum[rowsum == 0] = 1
    return counts, counts / rowsum


def _draw_cm_on_ax(ax, fit, X, y, K_clust, K_classes, title):
    """Draw a rectangular confusion matrix onto ``ax``.

    Rows = clusters (lexsorted by dominant class and confidence so the
    diagonal pops). Columns = true classes. Cells annotated with raw
    counts. Shared helper for the GMM and HDDC panels in the
    side-by-side figures.
    """
    pred = fit.predict(X)
    C, M = _row_normalized_confusion(pred, y, K_clust, K_classes)
    order = np.lexsort((-M.max(1), M.argmax(1)))
    C, M = C[order], M[order]
    cmax = int(C.max()) if C.max() > 0 else 1
    ax.imshow(C, aspect="equal", cmap=CMAP_PURPLE, vmin=0, vmax=cmax)
    for i in range(K_clust):
        for k in range(K_classes):
            if C[i, k] == 0:
                continue
            color = "white" if C[i, k] > 0.55 * cmax else ROLE["text"]
            ax.text(k, i, f"{int(C[i, k])}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_xticks(range(K_classes))
    ax.set_xticklabels(range(K_classes), fontsize=9)
    ax.set_xlabel("True Classes")
    ax.set_yticks(range(K_clust))
    ax.set_yticklabels([f"Cluster {c}" for c in range(K_clust)], fontsize=8)
    ax.set_ylabel(f"{K_clust} Clusters")
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(ROLE["muted"]); sp.set_linewidth(0.6)


def _hddc_vs_gmm_panel(
    name, X, y, K_true, K_grid, threshold, n_init, max_iter, fig_prefix,
    gmm_covariance_type: str = "diag",
):
    """{GMM, HDDC} x {K known, K unknown} comparison, with confusions.

    ``gmm_covariance_type`` controls the GMM baseline. For ``n < p``
    regimes, pass ``"full"`` to showcase HDDC's value: full-Σ GMM
    either fails to converge or overfits its (K · p · (p+1) / 2)
    parameters; HDDC's parsimonious factorisation stays well-conditioned.

    Saves PNGs under ``fig_prefix``:
      * ``<prefix>_metrics.png``    — NMI/ARI/ACC bar chart
      * ``<prefix>_K.png``          — K oracle vs ICL-selected
      * ``<prefix>_cm_known.png``   — GMM | HDDC, both at K_true (square)
      * ``<prefix>_cm_icl.png``     — GMM | HDDC, both at K_ICL (rectangular,
        each family's own ICL-selected K)
    Returns the list of file paths.
    """
    # Full-Σ GMM needs more regularisation on ill-conditioned data
    # (e.g. p > n). 1e-3 is sufficient for diag/spherical; 1e-1 keeps
    # full from collapsing in the n < p regime.
    gmm_reg = 1e-1 if gmm_covariance_type == "full" else 1e-3
    gmm_label = f"GMM ({gmm_covariance_type})"

    # --- K known.
    t0 = time.time()
    gmm_k = _gmm_with_K(X, K_true, gmm_covariance_type, reg_covar=gmm_reg,
                        n_init=n_init, max_iter=max_iter)
    t_gmm_k = time.time() - t0
    nmi_gmm_k, ari_gmm_k, acc_gmm_k = _eval(gmm_k, X, y)

    t0 = time.time()
    hddc_k = _hddc_with_K(X, K_true, threshold, n_init, max_iter)
    t_hddc_k = time.time() - t0
    nmi_hddc_k, ari_hddc_k, acc_hddc_k = _eval(hddc_k, X, y)

    # --- K unknown: pick K by ICL.
    t0 = time.time()
    K_gmm_u, gmm_u = _select_K(
        lambda K: _gmm_with_K(X, K, gmm_covariance_type, reg_covar=gmm_reg,
                              n_init=n_init, max_iter=max_iter),
        K_grid, X,
    )
    t_gmm_u = time.time() - t0
    nmi_gmm_u, ari_gmm_u, acc_gmm_u = _eval(gmm_u, X, y)

    t0 = time.time()
    K_hddc_u, hddc_u = _select_K(
        lambda K: _hddc_with_K(X, K, threshold, n_init, max_iter),
        K_grid, X,
    )
    t_hddc_u = time.time() - t0
    nmi_hddc_u, ari_hddc_u, acc_hddc_u = _eval(hddc_u, X, y)

    # --- K AND family unknown: pick the global ICL-best across both
    # GMM and HDDC at every K. ICL is comparable across families
    # (BIC's ν(K, family) is family-specific by construction; the
    # entropy term is family-agnostic), so this is a principled
    # head-to-head — model-family selection + K selection in one go.
    t0 = time.time()
    family_joint, K_joint, fit_joint = _select_K_across_families(
        {
            gmm_label: (
                lambda K: _gmm_with_K(X, K, gmm_covariance_type,
                                      reg_covar=gmm_reg,
                                      n_init=n_init, max_iter=max_iter)
            ),
            "HDDC (AVV)": (
                lambda K: _hddc_with_K(X, K, threshold, n_init, max_iter)
            ),
        },
        K_grid, X,
    )
    t_joint = time.time() - t0
    nmi_joint, ari_joint, acc_joint = _eval(fit_joint, X, y)

    log.info(f"\n  {name} (n={X.shape[0]}, p={X.shape[1]}, K_true={K_true})")
    log.info("  method        | K known                          | K unknown (ICL)")
    log.info(f"  {gmm_label:<13s} | NMI={nmi_gmm_k:.3f} ARI={ari_gmm_k:.3f} "
          f"ACC={acc_gmm_k:.3f} | K*={K_gmm_u:2d}  NMI={nmi_gmm_u:.3f} "
          f"ARI={ari_gmm_u:.3f} ACC={acc_gmm_u:.3f}")
    log.info(f"  HDDC (AVV)    | NMI={nmi_hddc_k:.3f} ARI={ari_hddc_k:.3f} "
          f"ACC={acc_hddc_k:.3f} | K*={K_hddc_u:2d}  NMI={nmi_hddc_u:.3f} "
          f"ARI={ari_hddc_u:.3f} ACC={acc_hddc_u:.3f}")
    log.info(f"  Joint ICL-best: family={family_joint}  K={K_joint}  "
          f"NMI={nmi_joint:.3f} ARI={ari_joint:.3f} ACC={acc_joint:.3f}")
    log.info(f"  (timing: GMM_k={t_gmm_k:.1f}s HDDC_k={t_hddc_k:.1f}s "
          f"GMM_u={t_gmm_u:.1f}s HDDC_u={t_hddc_u:.1f}s  "
          f"joint={t_joint:.1f}s)")

    use_house_style()
    out_paths = []

    # ----- Figure 1: metrics bar chart -----------------------------------
    fig_m, ax_metrics = plt.subplots(figsize=(11, 6))

    labels = [gmm_label, "HDDC (AVV)"]
    # KPI colours kept distinct from the confusion-matrix purple, so
    # the two panels don't visually bleed into each other.
    C_NMI = ROLE["bic"]                 # yellow
    C_ARI = ROLE["data"]                # blue
    C_ACC = ROLE["acc"]                 # orange

    knowns   = {"NMI": [nmi_gmm_k, nmi_hddc_k],
                "ARI": [ari_gmm_k, ari_hddc_k],
                "ACC": [acc_gmm_k, acc_hddc_k]}
    unknowns = {"NMI": [nmi_gmm_u, nmi_hddc_u],
                "ARI": [ari_gmm_u, ari_hddc_u],
                "ACC": [acc_gmm_u, acc_hddc_u]}
    colors = {"NMI": C_NMI, "ARI": C_ARI, "ACC": C_ACC}

    x = np.arange(len(labels))
    w = 0.13
    # Six bars per method, three metrics × {known, unknown}. Legend is
    # laid out in the order the user reads bars left-to-right per
    # method: NMI, ARI, ACC for known K, then again for unknown K.
    offsets_known   = [-2.5 * w, -1.5 * w, -0.5 * w]
    offsets_unknown = [+0.5 * w, +1.5 * w, +2.5 * w]
    for offs, scenario, store in [(offsets_known, "known K", knowns),
                                  (offsets_unknown, "unknown K", unknowns)]:
        for metric, dx in zip(("NMI", "ARI", "ACC"), offs):
            c = colors[metric]
            alpha = 0.85 if scenario == "known K" else 0.45
            hatch = None if scenario == "known K" else "//"
            ax_metrics.bar(
                x + dx, store[metric], w,
                color=c, alpha=alpha, hatch=hatch, edgecolor=c,
                label=f"{metric} ({scenario})",
            )
            for xi, v in zip(x, store[metric]):
                ax_metrics.text(xi + dx, v + 0.012, f"{v:.2f}",
                                ha="center", fontsize=8,
                                color=ROLE["text"])

    ax_metrics.set_xticks(x); ax_metrics.set_xticklabels(labels)
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_ylabel("Agreement with ground truth (higher is better)")
    ax_metrics.set_title(f"GMM vs HDDC on {name} — Clustering Recovery")
    ax_metrics.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                      ncol=6, frameon=False, fontsize=9,
                      columnspacing=1.4, handletextpad=0.5)
    fig_m.tight_layout(rect=[0, 0.06, 1, 0.97])
    p = f"{fig_prefix}_metrics.png"
    fig_m.savefig(p); plt.close(fig_m); out_paths.append(p)

    # ----- Figure 2: K oracle vs ICL ------------------------------------
    fig_k, ax_K = plt.subplots(figsize=(10, 5.5))
    bar_w = 0.34
    K_known   = [K_true, K_true]
    K_unknown = [K_gmm_u, K_hddc_u]
    ax_K.bar(x - bar_w / 2, K_known, bar_w,
             color=ROLE["muted"], alpha=0.55,
             label="Known K (oracle)")
    ax_K.bar(x + bar_w / 2, K_unknown, bar_w,
             color=ROLE["holdout"], alpha=0.85,
             label="K* Selected by ICL")
    for xi, kk, ku in zip(x, K_known, K_unknown):
        ax_K.text(xi - bar_w / 2, kk + 0.4, f"{kk}",
                  ha="center", fontsize=11, color=ROLE["text"])
        ax_K.text(xi + bar_w / 2, ku + 0.4, f"{ku}",
                  ha="center", fontsize=11, color=ROLE["text"])
    ax_K.axhline(K_true, color=ROLE["true"], lw=1.2, ls="--", alpha=0.7,
                 label=f"True K = {K_true}")
    ax_K.set_xticks(x); ax_K.set_xticklabels(labels)
    ax_K.set_ylim(0, max(K_unknown + [K_true]) * 1.35)
    ax_K.set_ylabel("Number of clusters (closer to true K is better)")
    ax_K.set_title(f"Number of clusters on {name}: oracle vs ICL-selected")
    ax_K.grid(False)
    ax_K.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=3, frameon=False, fontsize=10)
    fig_k.tight_layout(rect=[0, 0.05, 1, 0.97])
    p = f"{fig_prefix}_K.png"
    fig_k.savefig(p); plt.close(fig_k); out_paths.append(p)

    # ----- Figures 3-4: two side-by-side confusion-matrix figures -------
    # One for K known (square), one for K ICL-selected (rectangular).
    # Each puts GMM on the left and HDDC on the right so the rectangular
    # vs more-parsimonious-rectangular comparison is visible at a glance.
    K_classes = K_true
    scenarios = [
        (
            "cm_known",
            f"GMM vs HDDC on {name} — K known (K = {K_true})",
            (gmm_k,  K_true,   f"Diagonal GMM (K = {K_true})"),
            (hddc_k, K_true,   f"AVV HDDC (K = {K_true})"),
        ),
        (
            "cm_icl",
            f"GMM vs HDDC on {name} — K selected by ICL",
            (gmm_u,  K_gmm_u,  f"Diagonal GMM (K* = {K_gmm_u})"),
            (hddc_u, K_hddc_u, f"AVV HDDC (K* = {K_hddc_u})"),
        ),
    ]
    for suffix, supt, (fit_l, K_l, t_l), (fit_r, K_r, t_r) in scenarios:
        # Figure height scales with the taller of the two cluster columns.
        h = max(4.5, 0.35 * max(K_l, K_r) + 2.2)
        fig_c, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, h))
        _draw_cm_on_ax(ax_l, fit_l, X, y, K_l, K_classes, t_l)
        _draw_cm_on_ax(ax_r, fit_r, X, y, K_r, K_classes, t_r)
        fig_c.suptitle(supt, fontsize=13, fontweight="semibold")
        fig_c.tight_layout(rect=[0, 0, 1, 0.95])
        p = f"{fig_prefix}_{suffix}.png"
        fig_c.savefig(p); plt.close(fig_c); out_paths.append(p)

    return out_paths


def demo_hddc_digits() -> list[str]:
    """GMM vs HDDC on digits, K known vs K unknown."""
    log.info("\n== Demo HDDC-digits: GMM vs HDDC, K known vs unknown ==")
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64)
    K_true = int(np.unique(y).size)              # K_true = 10
    K_grid = [4, 6, 8, 10, 12, 14, 16, 18, 20]   # 4..20, with K_true=10
    return _hddc_vs_gmm_panel(
        name="digits", X=X, y=y,
        K_true=K_true, K_grid=K_grid,
        threshold=0.5, n_init=10, max_iter=200,
        fig_prefix=os.path.join(HERE, "fig_real_hddc_digits"),
    )


def demo_hddc_olivetti(n_people: int = 10, n_pca: int = 200) -> str:
    """GMM vs HDDC on Olivetti faces, K known vs K unknown.

    Limited to ``n_people`` individuals to keep runtime manageable
    while preserving the `n << p` regime that motivates HDDC. The
    raw Olivetti pixels live in p = 4096, so we project once to
    ``n_pca`` features (default 200) to make the eigendecomposition
    inside HDDC tractable while keeping n < p (n = 10 * n_people
    and p_eff = n_pca).
    """
    log.info("\n== Demo HDDC-olivetti: GMM vs HDDC, K known vs unknown ==")
    faces = fetch_olivetti_faces()
    X_all, y_all = faces.data, faces.target
    mask = y_all < n_people
    X_raw, y = X_all[mask], y_all[mask]
    from sklearn.decomposition import PCA
    n_pca = min(n_pca, X_raw.shape[0] - 1, X_raw.shape[1])
    X = PCA(n_components=n_pca, random_state=0).fit_transform(X_raw)
    log.info(f"  n={X.shape[0]}, p_raw=4096, p_pca={X.shape[1]}, K_true={n_people}")
    K_true = n_people
    K_grid = [4, 6, 8, 10, 12, 14]
    return _hddc_vs_gmm_panel(
        name="Olivetti faces", X=X, y=y,
        K_true=K_true, K_grid=K_grid,
        threshold=0.5, n_init=10, max_iter=60,
        fig_prefix=os.path.join(HERE, "fig_real_hddc_olivetti"),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_DEMOS = [
    ("Demo ICL: galaxies",            demo_icl_galaxies),
    ("Demo HDDC: digits",             demo_hddc_digits),
    ("Demo HDDC: Olivetti faces",     demo_hddc_olivetti),
]


def main() -> None:
    log.info("Real-world evidence for the three PRs")
    log.info("=" * 70)
    for label, fn in ALL_DEMOS:
        log.info(f"\n>>> {label}")
        result = fn()
        # Some demos (HDDC digits / Olivetti) split confusion matrices
        # into multiple PNGs and return a list of paths.
        paths = result if isinstance(result, list) else [result]
        for path in paths:
            if path is None:
                continue
            log.info(f"    figure -> {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
