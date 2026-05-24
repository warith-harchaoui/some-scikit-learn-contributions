"""ICL-PR figures: synthetic Student-mixture demonstrations + Roeder galaxies.

Each ``make_*`` and ``demo_*`` function writes one PNG into this
folder. Figures are deterministic — same random seed, same matplotlib
configuration — so re-runs produce byte-identical output.

Run from the repo root or this folder:

    python figures/figures_icl.py
"""
from __future__ import annotations

import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

log = logging.getLogger("figures_icl")

# Local style module + shared ICL-on-GMM compat helper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (                                            # noqa: E402
    PALETTE, ROLE, CMAP_PURPLE,
    use_house_style,
    per_sample_nats_criterion, PSNC_YLABEL, draw_psnc_anchors,
)
from _icl_compat import icl_gmm as _icl_for_gmm                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Make ``_icl_for_gmm`` reachable under the alias the real_world
# demo block uses.
_icl_gmm = _icl_for_gmm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gmm_with_K(X, K, covariance_type, reg_covar, n_init, max_iter, seed=0):
    """Convenience builder: fit ``GaussianMixture`` with the given knobs.

    Used by the galaxies demo to keep the per-K fit a single
    readable line.
    """
    return GaussianMixture(
        n_components=K, covariance_type=covariance_type,
        init_params="kmeans",
        random_state=seed, n_init=n_init, max_iter=max_iter,
        reg_covar=reg_covar,
    ).fit(X)


def _student_mixture_1d(true_K=3, n_per=4000, df=3, seed=42, sep=18.0):
    """Draw a 1-D mixture of ``true_K`` location-shifted Student-``t``
    components with ``df`` degrees of freedom and ``n_per`` samples per
    component. Returns ``X`` of shape ``(true_K * n_per, 1)``."""
    rng = np.random.RandomState(seed)
    centers = sep * np.arange(true_K)
    samples = [
        stats.t.rvs(df=df, size=n_per, random_state=rng) + c for c in centers
    ]
    X = np.concatenate(samples).reshape(-1, 1)
    return X, centers


# ---------------------------------------------------------------------------
# Figure 1: ICL vs BIC on a Student-t mixture (hero figure for the ICL PR)
# ---------------------------------------------------------------------------

def make_fig_icl_student() -> str:
    """ICL vs BIC on a heavy-tailed Student-``t`` mixture.

    Hero figure for the ICL PR: shows BIC over-counting components
    on heavy-tailed data while ICL recovers the true ``K``. Writes
    ``fig_icl_student.png`` and returns its path.
    """
    use_house_style()
    X, centers = _student_mixture_1d(true_K=3, n_per=4000, df=3, sep=18.0)

    Ks = list(range(2, 16))
    bic, icl = [], []
    for K in Ks:
        gmm = GaussianMixture(
            n_components=K, covariance_type="full",
            random_state=0, n_init=5, max_iter=300,
        ).fit(X)
        bic.append(gmm.bic(X))
        icl.append(_icl_for_gmm(gmm, X))
    K_bic = Ks[int(np.argmin(bic))]
    K_icl = Ks[int(np.argmin(icl))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.4))

    # Left: data + true density.
    xs = np.linspace(X.min() - 5, X.max() + 5, 800)
    true_density = sum(
        (1 / len(centers)) * stats.t.pdf(xs - c, df=3) for c in centers
    )
    ax1.hist(X.ravel(), bins=80, density=True,
             color=ROLE["data"], alpha=0.55, edgecolor="white",
             linewidth=0.4, label="Student-$t$ Mixture Data")
    ax1.plot(xs, true_density, color=ROLE["true"], lw=2.2,
             label="True Density ($K{=}3$, $\\nu{=}3$)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Density")
    ax1.set_title("Heavy-tailed data: $K{=}3$ Student-$t$ mixture")
    ax1.grid(False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=2, frameon=False, fontsize=10)

    # Per-Sample Nats Criterion (PSNC). sklearn's bic() and our
    # _icl_for_gmm() return values on the deviance scale; divide by 2
    # to land in natural-log units before normalising by
    # n * log(K*). See docs/INFORMATION_CRITERIA.md §3.
    K_star = 3            # ground-truth number of components
    n_train = len(X)      # samples the criteria were computed on
    bic_nat = 0.5 * np.asarray(bic)
    icl_nat = 0.5 * np.asarray(icl)
    bic_r = per_sample_nats_criterion(bic_nat, n_samples=n_train, k_star=K_star)
    icl_r = per_sample_nats_criterion(icl_nat, n_samples=n_train, k_star=K_star)
    ax2.plot(Ks, bic_r, color=ROLE["bic"], lw=3.2, label="BIC")
    ax2.plot(Ks, icl_r, color=ROLE["icl"], lw=2.2, label="ICL")
    # Big hollow ring on each minimum, ringed in the SAME colour as
    # the curve it minimises. Drawn but kept out of the legend.
    for vals, color, K_star in [(bic_r, ROLE["bic"], K_bic),
                                (icl_r, ROLE["icl"], K_icl)]:
        ax2.scatter([K_star], [vals[Ks.index(K_star)]],
                    facecolors="none", edgecolors=color,
                    s=420, lw=3.0, zorder=4)
    # PSNC anchors: green at y=0 (perfection), red at y=1 (confusion).
    # Registered before the legend so they appear in it.
    draw_psnc_anchors(ax2)
    ax2.set_xlabel("Number of components $K$")
    ax2.set_ylabel(PSNC_YLABEL)
    ax2.set_title("BIC overestimates $K$, ICL recovers the truth")
    ax2.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    ax2.grid(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=4, frameon=False, fontsize=10)

    fig.suptitle(
        "ICL vs BIC on a Student-$t$ mixture",
        fontsize=15, fontweight="semibold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = os.path.join(HERE, "fig_icl_student.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: held-out log-likelihood vs information criteria
# ---------------------------------------------------------------------------


def make_fig_holdout_vs_ic() -> str:
    use_house_style()
    # 3-component Student mixture; large-enough n that ICL recovers
    # the truth at K=3 (BIC and held-out still overshoot).
    X, _ = _student_mixture_1d(true_K=3, n_per=8000, df=3, sep=18.0, seed=42)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    tr, va = idx[: int(0.7 * len(X))], idx[int(0.7 * len(X)):]
    X_tr, X_va = X[tr], X[va]

    Ks = list(range(2, 16))
    ll_tr, ll_va, bic, icl = [], [], [], []
    for K in Ks:
        gmm = GaussianMixture(
            n_components=K, covariance_type="full",
            random_state=0, n_init=5, max_iter=500, reg_covar=1e-5,
        ).fit(X_tr)
        ll_tr.append(gmm.score(X_tr) * len(X_tr))
        ll_va.append(gmm.score(X_va) * len(X_va))
        bic.append(gmm.bic(X_tr))
        icl.append(_icl_for_gmm(gmm, X_tr))

    # Per-Sample Nats Criterion (PSNC). Each curve is normalised by
    # its own n * log(K*): n_tr for the criteria fitted on the
    # training set, n_va for the held-out log-likelihood. sklearn's
    # bic() and our _icl_for_gmm() are in deviance units, so we
    # divide by 2 first to reach natural-log units. See
    # docs/INFORMATION_CRITERIA.md §3.
    K_star = 3
    n_tr = len(X_tr)
    n_va = len(X_va)
    bic_nat = 0.5 * np.asarray(bic)
    icl_nat = 0.5 * np.asarray(icl)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    # Distinct line styles + thicknesses so curves stay readable when
    # they overlap (which they often do near the basin of K_true).
    # Single big ring on each curve's minimum, no per-point markers.
    series = [
        (per_sample_nats_criterion(ll_tr, n_samples=n_tr, k_star=K_star,
                                   lower_is_better=False),
         ROLE["train"],   dict(lw=1.6, ls=":"),  "Train log-likelihood"),
        (per_sample_nats_criterion(ll_va, n_samples=n_va, k_star=K_star,
                                   lower_is_better=False),
         ROLE["holdout"], dict(lw=2.6, ls="-."), "Held-out log-likelihood"),
        (per_sample_nats_criterion(bic_nat, n_samples=n_tr, k_star=K_star),
         ROLE["bic"],     dict(lw=3.4, ls="--"), "BIC"),
        (per_sample_nats_criterion(icl_nat, n_samples=n_tr, k_star=K_star),
         ROLE["icl"],     dict(lw=2.2, ls="-"),  "ICL"),
    ]
    for arr, color, style, label in series:
        ax.plot(Ks, arr, color=color, label=label, **style)
        k_star = Ks[int(np.argmin(arr))]
        ax.scatter([k_star], [arr[Ks.index(k_star)]],
                   facecolors="none", edgecolors=color,
                   s=420, lw=3.0, zorder=4)

    # "True K" vertical marker. Red is now reserved for the
    # Confusion anchor (horizontal); use the accent (pink) for the
    # ground-truth K marker, and place the label mid-axis so it does
    # not collide visually with the Confusion dashed line at y=1.
    ax.axvline(3, color=ROLE["accent"], lw=1.4, alpha=0.6)
    ax.text(3.05, 0.55, "True $K{=}3$", color=ROLE["accent"],
            fontsize=10, va="center", transform=ax.get_xaxis_transform())
    # PSNC anchors: green at y=0 (perfection), red at y=1 (confusion).
    # Registered before the legend so they appear in it.
    draw_psnc_anchors(ax)

    ax.set_xlabel("Number of components $K$")
    ax.set_ylabel(PSNC_YLABEL)
    ax.set_title("Example of ICL beating held-out likelihood and BIC")
    ax.set_xticks(list(range(2, 16)))
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = os.path.join(HERE, "fig_holdout_vs_ic.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: HDDC parameter counts vs Full-GMM (K=4, p=100, d=10)
# ---------------------------------------------------------------------------


def make_fig_icl_decomposition() -> str:
    use_house_style()
    # Same setup as the hero figure so ICL recovers K=3.
    X, _ = _student_mixture_1d(true_K=3, n_per=4000, df=3, sep=18.0, seed=42)
    n_samples = len(X)
    K_star = 3
    Ks = list(range(2, 13))
    bic, two_H, icl = [], [], []
    for K in Ks:
        gmm = GaussianMixture(n_components=K, covariance_type="full",
                              random_state=0, n_init=5, max_iter=200).fit(X)
        b = gmm.bic(X)
        _, log_resp = gmm._estimate_log_prob_resp(X)
        H = -np.nansum(np.exp(log_resp) * log_resp)
        bic.append(b)
        two_H.append(2 * H)
        icl.append(b + 2 * H)
    bic = np.array(bic); two_H = np.array(two_H)

    # Put both bars on the Per-Sample Nats Criterion axis (units of
    # ``log K*`` nats per sample), like all the other K-sweep figures.
    # First halve the deviance-scale quantities so we're in nats:
    #   BIC_nat = BIC / 2,  H = (2H) / 2.
    # Then anchor the stack at ``min(BIC_nat)`` and divide by
    # ``n · log K*``. Anchoring BOTH bars at the same baseline is what
    # preserves the additive identity ICL_nat = BIC_nat + H visually.
    bic_nat = bic / 2.0
    H = two_H / 2.0
    denom = n_samples * np.log(K_star)
    bic_h = (bic_nat - bic_nat.min()) / denom         # bottom bar
    H_n   = H / denom                                  # top bar
    # Sanity: bic_h + H_n == (icl_nat - min(bic_nat)) / denom.

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.55
    ax.bar(Ks, bic_h, width=width, color=ROLE["bic"], alpha=0.85,
           label="$\\mathrm{BIC}^{\\mathrm{nat}}$ (shifted to min)")
    ax.bar(Ks, H_n, width=width, bottom=bic_h, color=ROLE["icl"],
           alpha=0.85, label="$+\\,H$  (entropy penalty, in nats)")

    # Headroom for the equation band at the top; show ticks up to 1.0
    # only (the unlabelled headroom is intentional). The "Confusion"
    # line at y = 1 is the one full K*-way-uniform entropy unit per
    # sample — kept off-axis here because the figure is about the
    # *decomposition*, not the optimum's distance from chaos.
    y_top = max(1.6, (bic_h + H_n).max() * 1.6)
    ax.set_ylim(0, y_top)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # Mark the chosen K (no text annotation; the legend already names
    # the curves and the axvline is enough).
    K_icl = Ks[int(np.argmin(icl))]
    ax.axvline(K_icl, color=ROLE["accent"], lw=1.2, alpha=0.8)

    ax.set_xlabel("Number of components $K$")
    ax.set_ylabel(PSNC_YLABEL)
    ax.set_title("ICL decomposes as BIC plus an entropy penalty")
    ax.set_xticks(Ks)
    ax.grid(False)
    # Bars with "$+\\,2\\,H$" need a higher-cap legend so the equation box
    # stays readable above; relocate to upper-right with tight padding.
    leg = ax.legend(loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(ROLE["text"])

    # Equation in a dedicated band at the top of the plot. In natural-
    # log units (used by the PSNC axis), ICL = BIC + H — half of the
    # conventional deviance-scale ICL = BIC + 2H.
    ax.text(0.5, 0.92,
            r"$\mathrm{ICL}^{\mathrm{nat}}"
            r" \;=\; \mathrm{BIC}^{\mathrm{nat}} \;+\; H$",
            transform=ax.transAxes, fontsize=14, va="center", ha="center",
            color=ROLE["text"],
            bbox=dict(facecolor="white", edgecolor=ROLE["muted"],
                      boxstyle="round,pad=0.5", lw=0.9, alpha=0.95))
    out = os.path.join(HERE, "fig_icl_decomposition.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 7: HDDC naming map (14 models on the cube + 2 forbidden cells)
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_FIGURES = [
    ("ICL vs BIC (Student mixture)",   make_fig_icl_student),
    ("Held-out vs IC",                 make_fig_holdout_vs_ic),
    ("ICL = BIC + 2H decomposition",   make_fig_icl_decomposition),
    ("Real-world: Roeder galaxies",    demo_icl_galaxies),
]


def main() -> None:
    log.info("ICL-PR figures")
    log.info("=" * 70)
    for label, fn in ALL_FIGURES:
        log.info(f"-- {label}")
        path = fn()
        log.info(f"   -> {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
