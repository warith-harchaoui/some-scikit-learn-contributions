"""Shared style for all contribution figures.

Colour palette from https://harchaoui.org/warith/colors
Typography: matplotlib's default sans-serif.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Palette (harchaoui.org/warith/colors)
# ---------------------------------------------------------------------------

PALETTE = {
    # base
    "red":        "#FF3B30",
    "orange":     "#FF9500",
    "yellow":     "#FFCC00",
    "green":      "#28CD41",
    "blue":       "#007AFF",
    "turquoise":  "#79DBDC",
    "purple":     "#AF52DE",
    "pink":       "#FF2D55",
    # light variants
    "red_l":       "#FFD8D6",
    "orange_l":    "#FFEACC",
    "yellow_l":    "#FFF5CC",
    "green_l":     "#D4F5D9",
    "blue_l":      "#CCE4FF",
    "turquoise_l": "#00FFEF",
    "purple_l":    "#EFDCF8",
    "pink_l":      "#FFD5DD",
    # neutrals
    "silence":   "#000000",
    "neutral":   "#808080",
    "brown":     "#A52A2A",
    "offwhite":  "#F8F8F8",
}

# Conventional role -> colour mapping used across figures.
ROLE = {
    "bic":     PALETTE["yellow"],
    "icl":     PALETTE["purple"],
    "acc":     PALETTE["orange"],
    "true":    PALETTE["red"],
    "data":    PALETTE["blue"],
    "holdout": PALETTE["green"],
    "train":   PALETTE["turquoise"],
    "scree":   PALETTE["purple"],
    "varfrac": PALETTE["blue"],
    "grid":    PALETTE["neutral"],
    "text":    PALETTE["silence"],
    "muted":   PALETTE["neutral"],
    "accent":  PALETTE["pink"],
}


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
# Try to register Montserrat (Google Fonts) the first time the
# style is used. The TTF is downloaded once and cached under
# ``~/.cache/harchaoui-figures-fonts/``; subsequent runs reuse it.
# If the download or registration fails for any reason, we silently
# fall back to matplotlib's default sans-serif so the figures still
# render.

_FONT_FAMILY = "Montserrat"
_FONT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "harchaoui-figures-fonts",
)
# Google Fonts now ships Montserrat as a single variable TTF that
# covers every weight from 100 to 900 (and italics in the matching
# -Italic file). One download → all weights available.
_FONT_FILES = {
    "Montserrat[wght].ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "montserrat/Montserrat%5Bwght%5D.ttf"
    ),
    "Montserrat-Italic[wght].ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "montserrat/Montserrat-Italic%5Bwght%5D.ttf"
    ),
}


def _ensure_montserrat() -> bool:
    """Download (once) and register Montserrat with matplotlib.

    Returns True if Montserrat is available after the call,
    False if download/registration failed (caller should fall back).
    """
    try:
        from matplotlib import font_manager
    except Exception:                                    # noqa: BLE001
        return False
    os.makedirs(_FONT_CACHE_DIR, exist_ok=True)
    any_added = False
    for fname, url in _FONT_FILES.items():
        local = os.path.join(_FONT_CACHE_DIR, fname)
        if not os.path.exists(local):
            try:
                import urllib.request
                urllib.request.urlretrieve(url, local)
            except Exception:                            # noqa: BLE001
                continue
        try:
            font_manager.fontManager.addfont(local)
            any_added = True
        except Exception:                                # noqa: BLE001
            continue
    if any_added:
        # Bust the font_manager's font-name cache so ``Montserrat``
        # resolves on the next ``rcParams`` lookup.
        try:
            font_manager._load_fontmanager(try_read_cache=False)  # type: ignore[attr-defined]
        except Exception:                                # noqa: BLE001
            pass
    # Final check: did matplotlib actually recognise "Montserrat"?
    families = {f.name for f in font_manager.fontManager.ttflist}
    return _FONT_FAMILY in families


def use_house_style() -> None:
    """Apply the global rcParams; call once before plotting."""
    have_montserrat = _ensure_montserrat()
    font_family = _FONT_FAMILY if have_montserrat else "sans-serif"
    mpl.rcParams.update({
        "font.family":        font_family,
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "semibold",
        "axes.labelsize":     11,
        "axes.labelweight":   "regular",
        "axes.labelcolor":    PALETTE["silence"],
        "axes.edgecolor":     PALETTE["neutral"],
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "grid.color":         PALETTE["neutral"],
        "grid.alpha":         0.25,
        "grid.linewidth":     0.6,
        "xtick.color":        PALETTE["silence"],
        "ytick.color":        PALETTE["silence"],
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.frameon":     False,
        "legend.fontsize":    10,
        "figure.facecolor":   "white",
        "figure.dpi":         140,
        "savefig.dpi":        220,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.25,
        "savefig.facecolor":  "white",
        "lines.linewidth":    2.2,
        "lines.solid_capstyle": "round",
    })


def annotate_choice(ax, x: float, y: float, label: str, color: str,
                    dx: float = 0.0, dy: float = 0.0) -> None:
    """Place a small arrow + label at (x, y) pointing to the chosen K."""
    ax.annotate(
        label, xy=(x, y), xytext=(x + dx, y + dy),
        color=color, fontsize=10, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.7),
    )


def style_legend(ax, loc="best") -> None:
    leg = ax.legend(loc=loc)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_color(PALETTE["silence"])


def soften_yticks(ax) -> None:
    ax.tick_params(axis="y", color=PALETTE["neutral"], length=3)
    ax.tick_params(axis="x", color=PALETTE["neutral"], length=3)


# House-brand colormaps: white → palette colour. Useful for heatmaps
# (confusion matrices, scree spectra) that need to read in the same
# visual language as the rest of the figures.
CMAP_PURPLE = LinearSegmentedColormap.from_list(
    "harchaoui_purple", ["#FFFFFF", PALETTE["purple_l"], PALETTE["purple"]],
)
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "harchaoui_blue", ["#FFFFFF", PALETTE["blue_l"], PALETTE["blue"]],
)
CMAP_YELLOW = LinearSegmentedColormap.from_list(
    "harchaoui_yellow", ["#FFFFFF", PALETTE["yellow_l"], PALETTE["yellow"]],
)


# ---------------------------------------------------------------------------
# Rounded-corner helpers for bar charts.
# ---------------------------------------------------------------------------
# Matplotlib's bar / barh produce sharp-rectangle patches. We replace
# them in-place with ``FancyBboxPatch`` instances so the corners get a
# ~10 px rounding while keeping all the other styling (colour, alpha,
# hatch, edge colour) untouched.
#
# Pie chart wedges are *not* rectangles, so a true corner rounding
# requires drawing rounded outer strokes — not portable. The
# pie-chart styling here only adds white inter-slice borders, which
# gives a similar "soft" look without any rendering hacks.

_ROUND_CORNER_PX = 10


def round_bar_corners(ax, radius_px: float = _ROUND_CORNER_PX) -> None:
    """Round the corners of every ``Rectangle`` patch on ``ax``.

    Call this **after** ``ax.bar(...)`` / ``ax.barh(...)`` has drawn
    the bars; works for both horizontal and vertical bar charts.

    Parameters
    ----------
    ax : matplotlib axes
    radius_px : float, default=10
        Corner radius in **points** (so it stays visually constant
        regardless of figure size).
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    # ``boxstyle="round,pad=0,rounding_size=R"`` lets ``FancyBboxPatch``
    # use ``R`` directly in axes-data units. We want a fixed pixel
    # radius, so we translate ``radius_px`` points -> data units via
    # the axis's pixel-per-data ratio, evaluated on the smaller bar
    # dimension so we never round more than the bar's own thickness.
    if not ax.patches:
        return
    fig = ax.figure
    radius_inch = radius_px / fig.dpi
    new_patches = []
    for patch in list(ax.patches):
        if not isinstance(patch, Rectangle) or isinstance(patch, FancyBboxPatch):
            continue
        x, y = patch.get_xy()
        w, h = patch.get_width(), patch.get_height()
        # Convert the requested pixel radius back to data units along
        # each axis (use the smaller side so the rounding never
        # exceeds the bar's own thickness).
        ax_w_pix = abs(ax.transData.transform((1, 0))[0]
                       - ax.transData.transform((0, 0))[0])
        ax_h_pix = abs(ax.transData.transform((0, 1))[1]
                       - ax.transData.transform((0, 0))[1])
        # ``radius_inch * dpi`` is the pixel radius we want.
        radius_data_x = (radius_inch * fig.dpi) / max(ax_w_pix, 1e-9)
        radius_data_y = (radius_inch * fig.dpi) / max(ax_h_pix, 1e-9)
        # Cap at half the smaller bar side to avoid weird overshoot.
        r = min(radius_data_x, radius_data_y, abs(w) / 2, abs(h) / 2)
        rect = FancyBboxPatch(
            (x + r, y + r),
            max(abs(w) - 2 * r, 0),
            max(abs(h) - 2 * r, 0),
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=patch.get_linewidth(),
            edgecolor=patch.get_edgecolor(),
            facecolor=patch.get_facecolor(),
            hatch=patch.get_hatch(),
            alpha=patch.get_alpha(),
            zorder=patch.get_zorder(),
        )
        patch.remove()
        ax.add_patch(rect)
        new_patches.append(rect)
    # autoscale_view keeps any post-bar text annotations aligned.
    ax.autoscale_view()


# ---------------------------------------------------------------------------
# The Per-Sample Nats Criterion (PSNC) axis.
# ---------------------------------------------------------------------------
# Quantities plotted on a y-axis vs. K (NLL, BIC, ICL,
# log-likelihoods, posterior entropy) are in nats. The shared
# y-axis is linear, normalised by ``n * log(R)`` — the entropy of
# ``n`` samples uniformly distributed over ``R`` reference outcomes
# (``R = K*`` for clustering plots in this repo):
#
#     PSNC(K) = (C_K - min_J C_J) / (n * log R)         lower-is-better
#     PSNC(K) = (max_J ll_J - ll_K) / (n * log R)       higher-is-better
#     PSNC_H(K) = H_K / (n * log R)                     entropy (no shift)
#
# Reading the y-axis:
#   y = 0      the curve's own reference (min for lower-is-better,
#              max for higher-is-better), or one-hot assignments
#              for entropy. Drawn as the green "Perfection" line.
#   y = 1      one full R-way uncertainty unit per sample
#              (uniform assignment over the R reference outcomes).
#              Drawn as the red dashed "Confusion" line.
#              NOT a maximum — curves can exceed 1.
#
# IMPORTANT: criteria reported on the deviance scale (sklearn's
# ``gmm.bic(X)``, ICL = BIC + 2H) must be divided by 2 to land in
# natural-log units before plotting.
#
# See ``docs/INFORMATION_CRITERIA.md`` §3 (public, same content as the
# repo-root gist ``PSNC_GIST.md``) for the full derivation,
# motivation, and worked examples (classification + clustering).

PSNC_YLABEL = "Per-Sample Nats Criterion\n(lower is better)"


def draw_psnc_anchors(ax):
    """Draw the PSNC reference lines on ``ax``.

    - Solid green at ``y = 0`` labelled "Perfection" (harchaoui
      palette green: ``PALETTE["green"]``).
    - Dashed red at ``y = 1`` labelled "Confusion" (harchaoui
      palette red: ``PALETTE["red"]``).

    Both lines sit below the curve markers (``zorder=2``) and
    register themselves in the legend; no text is drawn over the
    plot area.
    """
    ax.axhline(0.0, linestyle="-",  linewidth=1.4, alpha=0.7,
               color=PALETTE["green"], zorder=2, label="Perfection")
    ax.axhline(1.0, linestyle="--", linewidth=1.2, alpha=0.8,
               color=PALETTE["red"],   zorder=2, label="Confusion")


def per_sample_nats_criterion(
    values,
    *,
    n_samples: int,
    k_star: int,
    lower_is_better: bool = True,
    already_per_sample: bool = False,
):
    """Convert a curve to the Per-Sample Nats Criterion (PSNC) scale.

    Parameters
    ----------
    values : array-like of shape (n_values,)
        Criterion values across the tested K. **Must already be in
        natural-log units.** For sklearn-style deviance BIC and ICL
        (which return ``-2 log L + ...``), divide by 2 first.
    n_samples : int
        Number of samples used to compute ``values``.
    k_star : int
        Reference number of clusters; sets the entropy unit.
    lower_is_better : bool, default=True
        True for NLL, BIC, ICL. False for log-likelihood.
    already_per_sample : bool, default=False
        If ``values`` are already averaged per sample, the
        denominator is ``log(k_star)``; otherwise it's
        ``n_samples * log(k_star)``.

    Returns
    -------
    y : ndarray
        PSNC values: y = 0 at the curve's reference (min or max),
        y = 1 at one full K*-way uncertainty unit per sample.
    """
    values = np.asarray(values, dtype=float)
    if k_star <= 1:
        raise ValueError("k_star must be greater than 1.")
    denominator = np.log(k_star)
    if not already_per_sample:
        denominator *= n_samples
    if lower_is_better:
        reference = np.nanmin(values)
        y = (values - reference) / denominator
    else:
        reference = np.nanmax(values)
        y = (reference - values) / denominator
    return y


def entropy_psnc(entropy, *, n_samples: int, k_star: int):
    """PSNC of posterior assignment entropy.

    Unlike NLL / BIC / ICL / log-likelihood, entropy has its own
    natural zero (one-hot assignments), so there is no per-curve
    reference subtraction:

        PSNC_H(K) = H_K / (n * log K*).
    """
    if k_star <= 1:
        raise ValueError("k_star must be greater than 1.")
    return np.asarray(entropy, dtype=float) / (n_samples * np.log(k_star))


__all__ = [
    "PALETTE", "ROLE",
    "CMAP_PURPLE", "CMAP_BLUE", "CMAP_YELLOW",
    "use_house_style",
    "annotate_choice", "style_legend", "soften_yticks",
    "round_bar_corners",
    "per_sample_nats_criterion", "entropy_psnc",
    "PSNC_YLABEL", "draw_psnc_anchors",
]
