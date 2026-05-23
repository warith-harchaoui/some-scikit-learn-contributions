"""Regenerate all figures used in the contribution docs.

Run from this folder:
    python make_figures.py

Each ``make_*`` function below writes a single PNG into ``./``.
Figures are deterministic - same random seed, same matplotlib
configuration - so re-runs produce byte-identical output.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import logging

log = logging.getLogger("make_figures")


# Local style module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (                                            # noqa: E402
    PALETTE, ROLE, CMAP_PURPLE,
    use_house_style,
    per_sample_nats_criterion, PSNC_YLABEL, draw_psnc_anchors,
)


HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIB = os.path.dirname(HERE)

# Load the draft HDDC implementation from pr_hddc/_hddc.py.
_hddc_spec = importlib.util.spec_from_file_location(
    "_draft_hddc", os.path.join(CONTRIB, "pr_hddc", "_hddc.py"),
)
_draft_hddc = importlib.util.module_from_spec(_hddc_spec)
_hddc_spec.loader.exec_module(_draft_hddc)
HighDimensionalGaussianMixture = _draft_hddc.HighDimensionalGaussianMixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _icl_for_gmm(gmm: GaussianMixture, X: np.ndarray) -> float:
    """ICL = BIC + 2 H, lower-is-better."""
    _, log_resp = gmm._estimate_log_prob_resp(X)
    resp = np.exp(log_resp)
    entropy = -np.nansum(resp * log_resp)
    return float(gmm.bic(X) + 2 * entropy)


def _student_mixture_1d(true_K=3, n_per=4000, df=3, seed=42, sep=18.0):
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

def _hddc_param_count_tables():
    """Build the (gmm_rows, hddc_rows) tables shared by the two
    parameter-count figures. Numbers come from Bouveyron, Girard &
    Schmid (2007), Table 1; the helper also asserts the AVE/AEE
    regression-test values cross-checked in ``pr_hddc/test_hddc.py``.
    """
    K, p, d = 4, 100, 10
    rho = K * p + (K - 1)
    tau = d * (p - (d + 1) / 2)
    tau_bar = K * tau

    hddc_rows = [
        ("AVV", rho + tau_bar + 2 * K + d * K),
        ("AEV", rho + tau_bar + K + d * K + 1),
        ("IVV", rho + tau_bar + 3 * K),
        ("IEV", rho + tau_bar + 2 * K + 1),
        ("UVV", rho + tau_bar + 2 * K + 1),
        ("UEV", rho + tau_bar + K + 2),
        ("AVE", rho + K * (tau + d + 1) + 1),
        ("CVE", rho + K * (tau + 1) + d + 1),
        ("AEE", rho + K * (tau + d) + 2),
        ("CEE", rho + K * tau + d + 2),
        ("IVE", rho + K * (tau + 2) + 1),
        ("IEE", rho + K * (tau + 1) + 2),
        ("UVE", rho + K * (tau + 1) + 2),
        ("UEE", rho + K * tau + 3),
    ]
    gmm_rows = [
        ("Spherical GMM",   rho + K),
        ("Diagonal GMM",    rho + K * p),
        ("Same cov GMM",    rho + p * (p + 1) / 2),
        ("Full GMM",        rho + K * p * (p + 1) / 2),
    ]
    hddc_rows.sort(key=lambda r: r[1])
    gmm_rows.sort(key=lambda r: r[1])

    expected = {"AVE": 4228, "AEE": 4225}
    got = {c: v for c, v in hddc_rows}
    for k, want in expected.items():
        assert got[k] == want, f"{k} = {got[k]}, expected {want}"
    return K, p, d, gmm_rows, hddc_rows


def make_fig_hddc_vs_gmm_budget() -> str:
    """HDDC vs the 4 classical GMM variants — log-scale parameter budget.

    Single panel, ranked from most to least parsimonious. The HDDC
    family appears as one bar with an explicit min-max range, so the
    comparison stays clean while showing that even the heaviest HDDC
    sub-model is ~5x cheaper than Full-GMM.
    """
    use_house_style()
    K, p, d, gmm_rows, hddc_rows = _hddc_param_count_tables()

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    hddc_min = min(v for _, v in hddc_rows)
    hddc_max = max(v for _, v in hddc_rows)
    rep = gmm_rows + [(f"HDDC family\n({hddc_min:,}–{hddc_max:,})",
                       (hddc_min + hddc_max) / 2)]
    ys = np.arange(len(rep))
    for y, (lbl, v) in zip(ys, rep):
        is_hddc = lbl.startswith("HDDC")
        c = ROLE["icl"] if is_hddc else ROLE["bic"]
        ax.barh(y, v, color=c, alpha=0.85, height=0.6)
        if is_hddc:
            ax.errorbar(v, y, xerr=[[v - hddc_min], [hddc_max - v]],
                        color=ROLE["text"], lw=1.4, capsize=4)
            ax.text(hddc_max * 1.05, y,
                    f"{hddc_min:,}–{hddc_max:,}",
                    va="center", fontsize=10, color=ROLE["text"])
        else:
            ax.text(v * 1.08, y, f"{int(v):,}",
                    va="center", fontsize=10, color=ROLE["text"])
    ax.set_yticks(ys)
    ax.set_yticklabels([lbl for lbl, _ in rep], fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(100, 100000)
    ax.set_xlabel("Free parameters, log scale (lower is more parsimonious)")
    ax.set_title(
        f"HDDC vs the GMM family at $K{{=}}{K}$, $p{{=}}{p}$, $d{{=}}{d}$: "
        "roughly 5x more parsimonious than Full-GMM"
    )
    ax.grid(False)

    fig.tight_layout()
    out = os.path.join(HERE, "fig_hddc_vs_gmm_budget.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def make_fig_hddc_subfamily() -> str:
    """Zoomed view of the 14 HDDC sub-models alone.

    Same axis units as the family-comparison plot. The 14 sub-models
    cover a narrow band (~45 parameters) because they share the
    dominant orientation cost ``K * tau``; the differences live in
    the eigenvalue parameterisation alone.
    """
    use_house_style()
    K, p, d, _gmm_rows, hddc_rows = _hddc_param_count_tables()

    SIG = {"A": "anisotropic", "I": "isotropic",
           "C": "common_axis", "U": "uniform"}
    NOI = {"V": "free", "E": "tied"}
    DIM = {"V": "free", "E": "tied"}
    _SIG_W = max(len(v) for v in SIG.values())
    _NOI_W = max(len(v) for v in NOI.values())
    _DIM_W = max(len(v) for v in DIM.values())

    def english(code: str) -> str:
        s, n, dm = code[0], code[1], code[2]
        return (
            f"{code} · "
            f"signal={SIG[s]:<{_SIG_W}}, "
            f"noise={NOI[n]:<{_NOI_W}}, "
            f"dim={DIM[dm]:<{_DIM_W}}"
        )

    hddc_min = min(v for _, v in hddc_rows)
    hddc_max = max(v for _, v in hddc_rows)
    pad = max(5, int((hddc_max - hddc_min) * 0.15))

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    ys = np.arange(len(hddc_rows))
    for y, (code, v) in zip(ys, hddc_rows):
        ax.barh(y, v, color=ROLE["icl"], alpha=0.85, height=0.7)
        ax.text(v + pad * 0.15, y, f"{int(v):,}", va="center",
                fontsize=9, color=ROLE["text"])
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [english(c) for c, _ in hddc_rows],
        fontsize=9, family="monospace",
    )
    ax.invert_yaxis()
    ax.set_xlim(hddc_min - pad, hddc_max + pad * 5)
    ax.set_xlabel("Free parameters")
    ax.grid(False)

    # Center the title over the whole figure (not the axes), so the
    # long monospace y-tick labels on the left don't push it visually
    # off-center.
    fig.suptitle(
        f"HDDC sub-models at $K{{=}}{K}$, $p{{=}}{p}$, $d{{=}}{d}$",
        fontsize=13, fontweight="semibold",
    )
    fig.tight_layout()
    out = os.path.join(HERE, "fig_hddc_subfamily.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 6: ICL = BIC + 2H decomposition
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

def make_fig_naming_map() -> str:
    """Three-tier explanation of the 3-letter HDDC code.

    1) Decoder ring: a worked example showing how `AVV` breaks down
       into its three letters and what each axis controls.
    2) Letter dictionary: A/I/C/U, V/E, V/E with one-line meanings.
    3) The 4 × 4 lookup grid: 14 valid cells coloured, 2 invalid
       cells hatched in red. Each valid cell shows the geometric
       code (big) and the Bouveyron-2007 paper bracket (small).

    The reading flow is top-to-bottom: see the example, learn the
    letters, look up your case.
    """
    use_house_style()

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[0.85, 1.0, 3.0],
        hspace=0.55,
        top=0.945, bottom=0.04, left=0.07, right=0.96,
    )

    SIGNAL_COL = ROLE["icl"]   # purple
    NOISE_COL  = ROLE["bic"]   # yellow
    DIM_COL    = ROLE["data"]  # blue

    # ----- TOP: decoder ring with a worked example ------------------------
    ax_ex = fig.add_subplot(gs[0])
    ax_ex.set_xlim(0, 12); ax_ex.set_ylim(0, 4)
    ax_ex.axis("off")

    # Tier-1 label sits on a dedicated line above the example so it
    # never competes with the AVV letters / paper bracket horizontally.
    ax_ex.text(0, 3.85, "Step 1 — Read a code (worked example)",
               ha="left", va="center",
               fontsize=12, fontweight="bold", color=ROLE["text"])

    # Layout: [AVV letters with arrows+labels] ≡ [paper bracket]
    # Centered as a whole around x=6.
    letter_dx = 1.55
    block_w = 2 * letter_dx               # span covered by the 3 letters
    eq_dx   = 1.05                        # space between block and ≡
    bracket_dx = 0.95                     # space between ≡ and bracket
    bracket_text_w = 2.3                  # approx width of "[a_{kj} b_k Q_k d_k]"
    total_w = block_w + eq_dx + bracket_dx + bracket_text_w
    block_x0 = 6 - total_w / 2

    y_letters = 2.55
    for k, (letter, col) in enumerate(
        zip("AVV", [SIGNAL_COL, NOISE_COL, DIM_COL])
    ):
        ax_ex.text(block_x0 + k * letter_dx, y_letters, letter,
                   ha="center", va="center",
                   fontsize=34, fontweight="bold", color=col)

    # Arrows + labels under each letter.
    axis_labels = [
        (block_x0 + 0 * letter_dx, "signal=anisotropic", SIGNAL_COL),
        (block_x0 + 1 * letter_dx, "noise=free",         NOISE_COL),
        (block_x0 + 2 * letter_dx, "dim=free",           DIM_COL),
    ]
    for x, lab, col in axis_labels:
        ax_ex.annotate(
            "", xy=(x, 1.35), xytext=(x, 1.85),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.4),
        )
        ax_ex.text(x, 1.05, lab, ha="center", va="top",
                   fontsize=10, fontweight="semibold", color=col)

    # Paper-bracket alias to the right of the AVV block.
    eq_x = block_x0 + block_w + eq_dx
    # Mathtext for ≡ — Montserrat doesn't ship U+2261, but matplotlib's
    # math font has \equiv natively.
    ax_ex.text(eq_x, y_letters, r"$\equiv$",
               ha="center", va="center",
               fontsize=22, color=ROLE["muted"])
    bracket_x = eq_x + bracket_dx
    ax_ex.text(bracket_x, y_letters,
               "[$a_{kj}\\,b_k\\,Q_k\\,d_k$]",
               ha="left", va="center",
               fontsize=17, color=ROLE["text"])
    ax_ex.text(bracket_x, 1.40,
               "Bouveyron 2007 paper bracket",
               ha="left", va="center",
               fontsize=9, style="italic", color=ROLE["muted"])

    # ----- MIDDLE: letter dictionary --------------------------------------
    ax_top = fig.add_subplot(gs[1])
    ax_top.set_xlim(0, 12); ax_top.set_ylim(0, 4.4)
    ax_top.axis("off")

    ax_top.text(0, 4.25, "Step 2 — Letter dictionary",
                ha="left", va="center",
                fontsize=12, fontweight="bold", color=ROLE["text"])

    sections = [
        (0.0, 4.0, "1st letter — `signal=...`", SIGNAL_COL,
         [("A", "anisotropic    ($a_{kj}$ varies in $k$ AND $j$)"),
          ("I", "isotropic      ($a_k$, per-cluster scalar)"),
          ("C", "common_axis   ($a_j$, per-axis, shared in $k$)"),
          ("U", "uniform        ($a$, one scalar everywhere)")]),
        (4.0, 4.0, "2nd letter — `noise=...`", NOISE_COL,
         [("V", "free   ($b_k$ varies across clusters)"),
          ("E", "tied   ($b$ shared across clusters)")]),
        (8.0, 4.0, "3rd letter — `dim=...`", DIM_COL,
         [("V", "free   ($d_k$ varies across clusters)"),
          ("E", "tied   ($d$ shared across clusters)")]),
    ]
    for x0, w, hdr, axis_col, lines in sections:
        ax_top.add_patch(plt.Rectangle(
            (x0 + 0.1, 0.15), w - 0.2, 3.7,
            facecolor="white", edgecolor=axis_col, lw=1.3,
        ))
        ax_top.text(x0 + w / 2, 3.42, hdr,
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=axis_col)
        for k, (letter, meaning) in enumerate(lines):
            y = 2.7 - 0.7 * k
            ax_top.text(x0 + 0.6, y, letter,
                        ha="center", va="center",
                        fontsize=15, fontweight="bold", color=axis_col)
            ax_top.text(x0 + 1.1, y, meaning,
                        ha="left", va="center",
                        fontsize=9.5, color=ROLE["text"])

    # ----- BOTTOM: the 4×4 validity grid ----------------------------------
    ax = fig.add_subplot(gs[2])
    signals = ["A", "I", "C", "U"]
    nd_pairs = [("V", "V"), ("E", "V"), ("V", "E"), ("E", "E")]
    valid = {
        "AVV", "AEV", "IVV", "IEV", "UVV", "UEV",
        "AVE", "CVE", "AEE", "CEE", "IVE", "IEE", "UVE", "UEE",
    }
    paper = {
        "AVV": "[$a_{kj}\\,b_k\\,Q_k\\,d_k$]",
        "AEV": "[$a_{kj}\\,b\\,Q_k\\,d_k$]",
        "IVV": "[$a_k\\,b_k\\,Q_k\\,d_k$]",
        "IEV": "[$a_k\\,b\\,Q_k\\,d_k$]",
        "UVV": "[$a\\,b_k\\,Q_k\\,d_k$]",
        "UEV": "[$a\\,b\\,Q_k\\,d_k$]",
        "AVE": "[$a_{kj}\\,b_k\\,Q_k\\,d$]",
        "CVE": "[$a_j\\,b_k\\,Q_k\\,d$]",
        "AEE": "[$a_{kj}\\,b\\,Q_k\\,d$]",
        "CEE": "[$a_j\\,b\\,Q_k\\,d$]",
        "IVE": "[$a_k\\,b_k\\,Q_k\\,d$]",
        "IEE": "[$a_k\\,b\\,Q_k\\,d$]",
        "UVE": "[$a\\,b_k\\,Q_k\\,d$]",
        "UEE": "[$a\\,b\\,Q_k\\,d$]",
    }

    nrows, ncols = len(signals), len(nd_pairs)
    cell_w, cell_h = 1.0, 1.0
    for i, s in enumerate(signals):
        for j, (n, d) in enumerate(nd_pairs):
            code = s + n + d
            x0, y0 = j * cell_w, (nrows - 1 - i) * cell_h
            if code in valid:
                ax.add_patch(plt.Rectangle(
                    (x0 + 0.05, y0 + 0.05), cell_w - 0.1, cell_h - 0.1,
                    facecolor=ROLE["icl"], alpha=0.15,
                    edgecolor=ROLE["icl"], lw=1.4,
                ))
                ax.text(x0 + cell_w / 2, y0 + cell_h - 0.30, code,
                        ha="center", va="center",
                        fontsize=22, fontweight="bold", color=ROLE["text"])
                ax.text(x0 + cell_w / 2, y0 + cell_h - 0.65, paper[code],
                        ha="center", va="center",
                        fontsize=10, color=ROLE["muted"])
            else:
                ax.add_patch(plt.Rectangle(
                    (x0 + 0.05, y0 + 0.05), cell_w - 0.1, cell_h - 0.1,
                    facecolor=PALETTE["red_l"], alpha=0.35,
                    edgecolor=ROLE["true"], lw=1.4, hatch="//",
                ))
                # × (U+00D7) is in Montserrat; ✗ (U+2717) is not.
                ax.text(x0 + cell_w / 2, y0 + cell_h - 0.36, "×",
                        ha="center", va="center",
                        fontsize=22, fontweight="bold", color=ROLE["true"])
                ax.text(x0 + cell_w / 2, y0 + cell_h - 0.72,
                        "Not in family\n(signal=C needs dim=E)",
                        ha="center", va="center",
                        fontsize=8, color=ROLE["true"])

    ax.set_xlim(-0.05, ncols * cell_w + 0.05)
    ax.set_ylim(-0.10, nrows * cell_h + 0.70)
    # Column header at the TOP of the grid: noise on top line, dim
    # underneath, both coloured by their axis.
    for j, (n, d) in enumerate(nd_pairs):
        cx = j * cell_w + cell_w / 2
        ax.text(cx, nrows * cell_h + 0.36, f"noise = {n}",
                ha="center", va="bottom",
                fontsize=10, fontweight="semibold", color=NOISE_COL)
        ax.text(cx, nrows * cell_h + 0.16, f"dim = {d}",
                ha="center", va="bottom",
                fontsize=10, fontweight="semibold", color=DIM_COL)
    # The column headers above the grid already encode noise/dim, so
    # repeating them at the bottom is redundant — drop the xticklabels.
    ax.set_xticks([])
    ax.set_yticks([(nrows - 1 - i) * cell_h + cell_h / 2
                   for i in range(nrows)])
    ax.set_yticklabels([f"signal = {s}" for s in signals],
                       fontsize=11, color=SIGNAL_COL,
                       fontweight="semibold")
    ax.tick_params(axis="both", length=0, color="white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)

    # Mini-title above the grid (sits below the column headers).
    ax.text(0, nrows * cell_h + 0.62,
            "Step 3 — Lookup grid (14 valid sub-models, 2 invalid)",
            ha="left", va="bottom", fontsize=12,
            fontweight="bold", color=ROLE["text"])

    fig.suptitle(
        "HDDC naming: a 3-letter code `[signal][noise][dim]`",
        fontsize=14, fontweight="semibold", y=0.985,
    )

    out = os.path.join(HERE, "fig_naming_map.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure: synthetic GMM vs HDDC, rectangular confusion matrices @ K_ICL
# ---------------------------------------------------------------------------

def _signal_subspace_mixture(K=4, p=60, d=2, n_per=120,
                             sigma_signal=4.0, sigma_noise=1.0,
                             center_scale=1.2, seed=0):
    """Synthetic mixture matching HDDC's generative assumption.

    Each cluster has its own random d-dim signal subspace stretched
    with std ``sigma_signal``, plus isotropic noise ``sigma_noise``
    on all p axes. Centers are drawn from N(0, center_scale·I_p), so
    inter-cluster separation is moderate relative to the intra-cluster
    elongation. With ``p`` moderately high, ``n_per`` small, and the
    signal long compared to inter-cluster spacing, diagonal GMM (the
    default sklearn fallback for ill-conditioned data) sees clusters
    as bumpy overlapping clouds along each axis and tends to
    over-split; HDDC's per-cluster low-rank factorisation recovers
    the clusters at the true K.
    """
    rng = np.random.RandomState(seed)
    n = K * n_per
    X = np.zeros((n, p))
    y = np.zeros(n, dtype=int)
    for k in range(K):
        center = rng.randn(p) * center_scale
        Q, _ = np.linalg.qr(rng.randn(p, d))
        scores = rng.randn(n_per, d) * sigma_signal
        noise = rng.randn(n_per, p) * sigma_noise
        X[k * n_per:(k + 1) * n_per] = center + scores @ Q.T + noise
        y[k * n_per:(k + 1) * n_per] = k
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _draw_cm_on_ax(ax, fit, X, y, K_clust, K_classes, title):
    """Draw a rectangular confusion matrix onto ``ax``.

    Rows = clusters (lexsorted by dominant class so the diagonal pops);
    columns = true classes. Cells annotated with raw counts.
    """
    pred = fit.predict(X)
    counts = np.zeros((K_clust, K_classes), dtype=int)
    for c, yc in zip(pred, y):
        counts[int(c), int(yc)] += 1
    rowsum = counts.sum(1, keepdims=True).astype(float)
    rowsum[rowsum == 0] = 1
    M = counts / rowsum
    order = np.lexsort((-M.max(1), M.argmax(1)))
    C = counts[order]
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


def make_fig_hddc_vs_gmm_cm() -> str:
    """Synthetic side-by-side: diag-GMM vs AVV-HDDC, both at K_ICL.

    K_true = 4, p = 60, d = 2 signal axes per cluster, n_per = 120
    (enough for AVV's K·p·d orientation params to stabilise),
    moderate inter-cluster separation. Both estimators sweep K = 2..10
    and pick K* by ICL. The figure shows the two rectangular
    confusion matrices (rows = clusters, columns = true classes) so
    the K_ICL difference between GMM and HDDC is visible in cluster
    space, not only as a number.
    """
    use_house_style()
    K_true, p, d_signal, n_per = 4, 60, 2, 120
    X, y = _signal_subspace_mixture(K=K_true, p=p, d=d_signal, n_per=n_per,
                                    sigma_signal=4.0, sigma_noise=1.0,
                                    center_scale=1.2, seed=0)
    K_grid = list(range(2, 11))

    def _fit_gmm(K):
        return GaussianMixture(
            n_components=K, covariance_type="diag",
            init_params="kmeans", random_state=0,
            n_init=5, max_iter=200, reg_covar=1e-3,
        ).fit(X)

    def _fit_hddc(K):
        return HighDimensionalGaussianMixture(
            n_components=K, model="AVV",
            cattell_threshold=0.5, min_cluster_size=2,
            init_params="kmeans", random_state=0,
            n_init=5, max_iter=200,
        ).fit(X)

    K_gmm, gmm = None, None
    best = np.inf
    for K in K_grid:
        f = _fit_gmm(K)
        v = _icl_for_gmm(f, X)
        if v < best:
            best, K_gmm, gmm = v, K, f

    K_hddc, hddc = None, None
    best = np.inf
    for K in K_grid:
        try:
            f = _fit_hddc(K)
        except Exception:                                       # noqa: BLE001
            continue
        v = f.icl(X)
        if v < best:
            best, K_hddc, hddc = v, K, f

    h = max(4.5, 0.35 * max(K_gmm, K_hddc) + 2.2)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, h))
    _draw_cm_on_ax(ax_l, gmm,  X, y, K_gmm,  K_true,
                   f"Diagonal GMM (K* = {K_gmm})")
    _draw_cm_on_ax(ax_r, hddc, X, y, K_hddc, K_true,
                   f"AVV HDDC (K* = {K_hddc})")
    fig.suptitle(
        f"Synthetic signal-subspace mixture "
        f"(K_true = {K_true}, p = {p}, d = {d_signal}): "
        f"GMM vs HDDC at K_ICL",
        fontsize=13, fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "fig_hddc_vs_gmm_cm.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_FIGURES = [
    ("ICL vs BIC (Student mixture)",  make_fig_icl_student),
    ("Held-out vs IC",                make_fig_holdout_vs_ic),
    ("HDDC vs GMM-family budget",     make_fig_hddc_vs_gmm_budget),
    ("HDDC 14 sub-models",            make_fig_hddc_subfamily),
    ("ICL = BIC + 2H decomposition",  make_fig_icl_decomposition),
    ("HDDC naming map",               make_fig_naming_map),
    ("GMM vs HDDC: synthetic CM @ K_ICL", make_fig_hddc_vs_gmm_cm),
]


def main() -> None:
    for label, fn in ALL_FIGURES:
        log.info(f"-- {label}")
        path = fn()
        log.info(f"   -> {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
