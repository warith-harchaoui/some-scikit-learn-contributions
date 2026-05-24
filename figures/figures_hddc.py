"""HDDC-PR figures: synthetic structural plots + digits + Olivetti demos.

Each ``make_*`` and ``demo_*`` function writes one (or several)
PNGs into this folder. Figures are deterministic.

Run from the repo root or this folder:

    python figures/figures_hddc.py
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
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

log = logging.getLogger("figures_hddc")

# Local style + shared ICL-on-GMM compat helper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (                                            # noqa: E402
    PALETTE, ROLE, CMAP_PURPLE,
    use_house_style,
)
from _icl_compat import icl_gmm as _icl_for_gmm                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRIB = os.path.dirname(HERE)

# Load the draft HDDC implementation from pr_hddc/_hddc.py without
# requiring the PR to have landed in sklearn.
_hddc_spec = importlib.util.spec_from_file_location(
    "_draft_hddc", os.path.join(CONTRIB, "pr_hddc", "_hddc.py"),
)
_draft_hddc = importlib.util.module_from_spec(_hddc_spec)
_hddc_spec.loader.exec_module(_draft_hddc)
HighDimensionalGaussianMixture = _draft_hddc.HighDimensionalGaussianMixture

# Local alias for readability: the demo block uses ``_icl_gmm`` to
# match the BIC / ICL convention spelled out in
# ``docs/INFORMATION_CRITERIA.md``.
_icl_gmm = _icl_for_gmm


# ---------------------------------------------------------------------------
# Helpers
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
            n_init=10, max_iter=300, reg_covar=1e-3,
        ).fit(X)

    def _fit_hddc(K):
        return HighDimensionalGaussianMixture(
            n_components=K, model="AVV",
            cattell_threshold=0.5, min_cluster_size=2,
            init_params="kmeans", random_state=0,
            n_init=10, max_iter=300,
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

# --- helpers used by the digits + Olivetti demos ---
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



def _draw_cm_panel(ax, fit, X, y, K_clust, K_classes, title):
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
        _draw_cm_panel(ax_l, fit_l, X, y, K_l, K_classes, t_l)
        _draw_cm_panel(ax_r, fit_r, X, y, K_r, K_classes, t_r)
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
        threshold=0.5, n_init=10, max_iter=300,
        fig_prefix=os.path.join(HERE, "fig_real_hddc_digits"),
    )



def demo_hddc_olivetti(n_people: int = 10) -> list:
    """GMM vs HDDC on Olivetti faces, K known vs K unknown.

    Limited to ``n_people`` individuals to keep runtime tractable
    while preserving the ``n << p`` regime that motivates HDDC. We
    use the **raw** 4096-dim pixel vectors (no PCA): pre-projecting
    would conflate PCA and HDDC in the comparison. Tractability on
    the HDDC side comes from the SVD-of-data-matrix path in
    ``pr_hddc/_hddc.py`` (cheap when ``p > n``).
    """
    log.info("\n== Demo HDDC-olivetti: GMM vs HDDC, K known vs unknown ==")
    faces = fetch_olivetti_faces()
    X_all, y_all = faces.data, faces.target
    mask = y_all < n_people
    X, y = X_all[mask].astype(float), y_all[mask]
    log.info(f"  n={X.shape[0]}, p={X.shape[1]} (raw pixels), "
             f"K_true={n_people}")
    K_true = n_people
    K_grid = [4, 6, 8, 10, 12, 14]
    return _hddc_vs_gmm_panel(
        name="Olivetti faces", X=X, y=y,
        K_true=K_true, K_grid=K_grid,
        threshold=0.5, n_init=10, max_iter=300,
        fig_prefix=os.path.join(HERE, "fig_real_hddc_olivetti"),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_FIGURES = [
    ("HDDC vs GMM-family budget",          make_fig_hddc_vs_gmm_budget),
    ("HDDC 14 sub-models",                 make_fig_hddc_subfamily),
    ("HDDC naming map",                    make_fig_naming_map),
    ("GMM vs HDDC: synthetic CM @ K_ICL",  make_fig_hddc_vs_gmm_cm),
    ("Real-world: digits",                 demo_hddc_digits),
    ("Real-world: Olivetti faces",         demo_hddc_olivetti),
]


def main() -> None:
    log.info("HDDC-PR figures")
    log.info("=" * 70)
    for label, fn in ALL_FIGURES:
        log.info(f"-- {label}")
        result = fn()
        paths = result if isinstance(result, list) else [result]
        for path in paths:
            if path is None:
                continue
            log.info(f"   -> {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
