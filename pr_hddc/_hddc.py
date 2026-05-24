"""High-Dimensional Data Clustering (HDDC).

A family of parsimonious Gaussian mixture models for clustering in
high-dimensional spaces, following Bouveyron, Girard & Schmid (2007)
[Bouveyron2007]_. The full family ``[a_kj b_k Q_k d_k]`` and its 13
documented sub-models are exposed through a single ``model`` string
(plus optional per-axis kwargs).

Naming
------
Three accepted spellings (full discussion in
``docs/HDDC.md`` §1):

1. **Geometric code** (canonical, recommended):
   ``model="AVV"`` etc., where the three letters describe the
   signal-eigenvalue regime (``A``/``I``/``C``/``U``), the noise
   regime (``V``/``E``), and the signal-dimension regime
   (``V``/``E``). The ``V``/``E`` spelling matches mclust's
   convention for "varying / equal across clusters."

2. **Paper bracket** (citation-friendly):
   ``model="akj_bk_Qk_dk"``, ``model="a_b_Qk_d"``, etc. Aliases for
   the geometric code; resolved to it internally.

3. **Per-axis kwargs**: ``signal=``, ``noise=``, ``dim=``. May be
   used either standalone (all three) or together with ``model=``,
   in which case **they must agree on every axis they both specify**
   - silent overrides are not allowed; a mismatch raises
   ``ValueError``.

mclust covariance codes (``"EII"``, ``"VVV"``, ...) are not
recognized as HDDC models; passing one raises with a pointer to the
HDDC table. mclust covariances are full-rank; HDDC covariances are
rank-``d_k``. The model families are different.

For model selection, two lower-is-better criteria are provided:
:meth:`HighDimensionalGaussianMixture.bic`,
:meth:`HighDimensionalGaussianMixture.icl`.

.. [Bouveyron2007] Bouveyron, C., Girard, S., & Schmid, C. (2007).
   "High-dimensional data clustering." Computational Statistics &
   Data Analysis, 52(1), 502-519. https://arxiv.org/abs/math/0604064

Authors
-------
Warith Harchaoui (warith@deraison.ai).
Special thanks to Pierre-Alexandre Mattei (https://pamattei.github.io/).
"""

# License: BSD-3-Clause

import math
import numbers
from typing import Optional, Tuple, Union

import numpy as np
from scipy import linalg
from scipy.special import logsumexp, xlogy

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

_EPS = 1e-10


# --------------------------------------------------------------------------
# Model-name tables
# --------------------------------------------------------------------------

# Paper bracket notation -> geometric V/E code.
# Every row of Table 1 of Bouveyron 2007 appears exactly once.
_PAPER_TO_GEOMETRIC = {
    # free-d sub-models (geometric code ends with V)
    "akj_bk_Qk_dk": "AVV",
    "akj_b_Qk_dk":  "AEV",
    "ak_bk_Qk_dk":  "IVV",
    "ak_b_Qk_dk":   "IEV",
    "a_bk_Qk_dk":   "UVV",
    "a_b_Qk_dk":    "UEV",
    # common-d sub-models (geometric code ends with E)
    "akj_bk_Qk_d":  "AVE",
    "aj_bk_Qk_d":   "CVE",
    "akj_b_Qk_d":   "AEE",
    "aj_b_Qk_d":    "CEE",
    "ak_bk_Qk_d":   "IVE",
    "ak_b_Qk_d":    "IEE",
    "a_bk_Qk_d":    "UVE",
    "a_b_Qk_d":     "UEE",
}

_VALID_GEOMETRIC = frozenset(_PAPER_TO_GEOMETRIC.values())
_VALID_PAPER = frozenset(_PAPER_TO_GEOMETRIC.keys())
_VALID_MODEL_STRINGS = _VALID_GEOMETRIC | _VALID_PAPER

# Per-axis kwarg vocabulary -> geometric letter.
_SIGNAL = {
    "anisotropic": "A",
    "isotropic":   "I",
    "common_axis": "C",
    "uniform":     "U",
}
_NOISE = {"varying": "V", "equal": "E"}
_DIM = {"varying": "V", "equal": "E"}

# All 3-letter mclust covariance codes for full-rank Gaussians.
# Used to give a friendly error when the user accidentally passes one.
_MCLUST_CONFUSABLES = frozenset({
    "EII", "VII", "EEI", "VEI", "EVI", "VVI",
    "EEE", "VEE", "EVE", "VVE", "EEV", "VEV", "EVV", "VVV",
}) - _VALID_GEOMETRIC


# --------------------------------------------------------------------------
# Cattell scree test
# --------------------------------------------------------------------------

def _cattell_scree_test(
    eigvals: np.ndarray,
    threshold: float = 0.5,
    noise_ctrl: float = 1e-8,
) -> int:
    """Estimate intrinsic dimension via the HDclassif Cattell scree rule.

    Direct port of the algorithm shipped in the R package
    ``HDclassif`` (Berge, Bouveyron & Girard, 2012), which is the de
    facto reference HDDC implementation. The rule is:

        d = max{ i : (|λ_i − λ_{i+1}| / max_j |λ_j − λ_{j+1}|) > threshold
                   AND λ_{i+1} > noise_ctrl }

    i.e. the **largest** index where the normalised consecutive drop
    exceeds ``threshold`` *and* the following eigenvalue stays above
    the noise floor. If no index satisfies both conditions, ``d``
    falls back to 1.

    This is intentionally a permissive rule: it can pick d well past
    the first below-threshold dip, as long as a later drop exceeds
    the threshold. That matches HDclassif's behaviour and lets the
    Python implementation reproduce HDclassif fits exactly when the
    Cattell threshold and noise floor are matched.

    Parameters
    ----------
    eigvals : array-like
        Eigenvalues, sorted in non-increasing order.
    threshold : float, default=0.5
        Sensitivity parameter; smaller picks larger ``d``. The local
        default of 0.5 is more conservative than HDclassif's 0.2;
        see ``docs/HDDC.md`` §3 for the rationale.
    noise_ctrl : float, default=1e-8
        Eigenvalues at or below this floor are treated as noise and
        cannot terminate the signal subspace. Mirrors HDclassif's
        ``noise.ctrl`` argument.

    Returns
    -------
    d : int
        Estimated intrinsic dimension, in ``[1, len(eigvals) - 1]``.

    Notes
    -----
    The HDclassif source is essentially::

        dev <- abs(apply(ev, 1, diff))
        max_dev <- apply(dev, 2, max)
        dev <- dev / rep(max_dev, each = p - 1)
        d <- apply((dev > threshold) * (1:(p-1)) * t(ev[, -1] > noise.ctrl),
                   2, which.max)

    Translated: weight position ``i`` by 1..p-1 when both conditions
    hold, then ``which.max`` returns the largest such ``i`` (because
    the weights are monotone in ``i`` and all other entries are 0).
    """
    eigvals = np.asarray(eigvals, dtype=float)
    if not np.all(eigvals[:-1] >= eigvals[1:] - _EPS):
        raise ValueError("eigvals must be sorted in non-increasing order")
    p = eigvals.size
    if p <= 2:
        # HDclassif special-cases this as d = 1.
        return 1

    diffs = np.abs(np.diff(eigvals))  # length p - 1
    max_diff = float(np.max(diffs))
    if max_diff == 0.0:
        return 1

    # Normalised drops in [0, 1] and the "next eigenvalue above noise"
    # mask. Their elementwise product, weighted by position 1..p-1,
    # gives a vector whose argmax (1-indexed) is the desired d.
    norm_diffs = diffs / max_diff
    above_thr = norm_diffs > threshold
    above_noise = eigvals[1:] > noise_ctrl
    eligible = above_thr & above_noise
    if not eligible.any():
        return 1
    weights = np.arange(1, p) * eligible
    # +1 because position i in the diff vector corresponds to keeping
    # the first i eigenvalues as signal (1-indexed dimension).
    return int(np.argmax(weights)) + 1


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------

class HighDimensionalGaussianMixture(ClusterMixin, BaseEstimator):
    """Gaussian Mixture for high-dimensional data (HDDC).

    Parameters
    ----------
    n_components : int, default=1
        Number of mixture components.

    model : str or None, default="AVV"
        HDDC sub-model identifier. Accepts either:

        * a 3-letter geometric code from
          :data:`_VALID_GEOMETRIC` (e.g. ``"AVV"``, ``"UEE"``); or
        * a paper bracket string from
          :data:`_VALID_PAPER` (e.g. ``"akj_bk_Qk_dk"``,
          ``"a_b_Qk_d"``).

        Set to ``None`` to require fully specifying the axes via
        ``signal``, ``noise`` and ``dim``.

    signal : {"anisotropic", "isotropic", "common_axis", "uniform"} or None
        Signal-eigenvalue regime. If set together with ``model``, must
        agree with the model's signal axis; otherwise raises.

    noise : {"varying", "equal"} or None
        Noise-variance regime. Same conflict semantics as ``signal``.

    dim : {"varying", "equal"} or None
        Signal-dimension regime. Same conflict semantics as ``signal``.

    cattell_threshold : float, default=0.5
        Sensitivity for Cattell's scree test, used to select each
        cluster's signal dimensionality ``d_k``. See
        ``docs/HDDC.md`` §3. Ignored when the model forces a common
        ``d`` *and* ``signal_dim`` is also passed.

    signal_dim : int or None, default=None
        If set and the model has ``dim="equal"`` (geometric letter
        ``E``), fixes the shared signal dimensionality to this value.
        Silently ignored for ``dim="varying"`` models.

    tol : float, default=1e-3
        Convergence threshold on the log-likelihood.

    max_iter : int, default=100
        Maximum number of EM iterations.

    n_init : int, default=1
        Number of initializations to explore.

        - With ``init_params="kmeans"`` (recommended), this is the
          KMeans ``n_init``: kmeans++ is run ``n_init`` times and the
          best (lowest inertia) is kept. A **single** HDDC EM
          refinement then runs from that init, since EM is
          deterministic given the kmeans seed.
        - With ``init_params="random"`` (no clustering exploration
          step), ``n_init`` falls back to its classical meaning: run
          ``n_init`` independent EM restarts from random
          responsibilities and keep the one with highest
          log-likelihood.
        - With ``init_params`` an ndarray of labels (deterministic
          initialization), ``n_init`` is ignored.

    init_params : {"kmeans", "random"} or ndarray of shape (n_samples,)
        Responsibility initialization strategy.

    min_cluster_size : int, default=5
        Minimum number of observations assigned to each cluster.

    random_state : int, RandomState instance or None, default=None
        Controls the random initializations.

    verbose : bool, default=False
        Print EM progress.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        The mixing weight of each component.
    means_ : ndarray of shape (n_components, n_features)
        The mean of each mixture component.
    eigenvalues_ : list of ndarray of shape (n_features,)
        Per-cluster eigenvalues of the empirical covariance,
        sorted in non-increasing order. The first ``signal_dims_[k]``
        entries are signal; the rest are absorbed into
        ``noise_variances_[k]`` by the model parameterisation.
    eigenvectors_ : list of ndarray of shape (n_features, n_features)
        Per-cluster orthonormal basis ``Q_k``. Only the first
        ``signal_dims_[k]`` columns are used at predict time
        (the orthogonal complement carries isotropic ``b_k`` noise).
    signal_dims_ : list of int
        Per-cluster intrinsic signal dimension ``d_k`` selected by
        the Cattell scree rule (or forced via ``signal_dim`` on
        ``*E``-dim sub-models).
    noise_variances_ : ndarray of shape (n_components,)
        Per-cluster isotropic noise variance ``b_k``. Tied across
        clusters when the sub-model has ``noise="equal"`` (``*E*``).
    labels_ : ndarray of shape (n_samples,)
        Hard cluster assignment for each training sample,
        ``self.predict(X_train)`` at convergence.
    lower_bound_ : float
        Sample log-likelihood at the EM fixed point of the best
        ``n_init`` initialisation. Matches the role of
        :attr:`sklearn.mixture.GaussianMixture.lower_bound_`.
    n_iter_ : int
        Number of EM iterations used by the best initialisation.
    converged_ : bool
        ``True`` if EM converged (change in log-likelihood < ``tol``)
        before ``max_iter`` for the best initialisation.
    n_features_in_ : int
        Number of features seen at fit time.
    _geometric_model_ : str
        Canonical 3-letter geometric code resolved from ``model`` and
        the per-axis kwargs. Read this attribute (not ``self.model``)
        in user code that depends on the chosen sub-model.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.mixture import HighDimensionalGaussianMixture
    >>> X, _ = make_blobs(n_samples=300, n_features=50, centers=3,
    ...                   random_state=0)
    >>> hgmm = HighDimensionalGaussianMixture(
    ...     n_components=3, model="AVV", random_state=0,
    ... ).fit(X)
    >>> hgmm.bic(X)  # doctest: +SKIP
    """

    _parameter_constraints = {
        "n_components": [Interval(numbers.Integral, 1, None, closed="left")],
        # mclust codes are accepted by the constraint layer so that
        # `_resolve_model` can raise the friendly "mclust != HDDC"
        # error instead of a generic StrOptions message.
        "model": [
            StrOptions(set(_VALID_MODEL_STRINGS) | set(_MCLUST_CONFUSABLES)),
            None,
        ],
        "signal": [StrOptions(set(_SIGNAL)), None],
        "noise": [StrOptions(set(_NOISE)), None],
        "dim": [StrOptions(set(_DIM)), None],
        "cattell_threshold": [Interval(numbers.Real, 0.0, 1.0, closed="neither")],
        "signal_dim": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "tol": [Interval(numbers.Real, 0.0, None, closed="left")],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_init": [Interval(numbers.Integral, 1, None, closed="left")],
        "init_params": [StrOptions({"kmeans", "random"}), np.ndarray],
        "min_cluster_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        model: Optional[str] = "AVV",
        signal: Optional[str] = None,
        noise: Optional[str] = None,
        dim: Optional[str] = None,
        cattell_threshold: float = 0.5,
        signal_dim: Optional[int] = None,
        tol: float = 1e-3,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Union[str, np.ndarray] = "kmeans",
        min_cluster_size: int = 5,
        random_state=None,
        verbose: bool = False,
    ):
        # sklearn convention: __init__ stores only.
        self.n_components = n_components
        self.model = model
        self.signal = signal
        self.noise = noise
        self.dim = dim
        self.cattell_threshold = cattell_threshold
        self.signal_dim = signal_dim
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Model-name resolution (strict no-conflict semantics)
    # ------------------------------------------------------------------ #

    def _resolve_model(self) -> None:
        """Resolve (model, signal, noise, dim) -> single geometric code.

        Strict: when both ``model=`` and per-axis kwargs are set, they
        must agree on every axis they both specify; otherwise raise.
        No silent override. Sets ``self._geometric_model_``.
        """
        has_model = self.model is not None
        has_kwargs = any(v is not None
                         for v in (self.signal, self.noise, self.dim))

        if not has_model and not has_kwargs:
            # Should not happen because `model` has a default of "AVV",
            # but kept for symmetry if a future default changes.
            raise ValueError(
                "Either `model` or all three axis kwargs "
                "(`signal`, `noise`, `dim`) must be set."
            )

        if has_kwargs and not has_model:
            missing = [name for name, v in (
                ("signal", self.signal),
                ("noise", self.noise),
                ("dim", self.dim),
            ) if v is None]
            if missing:
                raise ValueError(
                    f"When `model` is None, all of `signal`, `noise`, "
                    f"`dim` must be set. Missing: {missing}."
                )
            code = (
                _SIGNAL[self.signal]
                + _NOISE[self.noise]
                + _DIM[self.dim]
            )
        else:
            # `model=` is set; decode it.
            m = self.model
            if m in _VALID_GEOMETRIC:
                s_m, n_m, d_m = m[0], m[1], m[2]
            elif m in _PAPER_TO_GEOMETRIC:
                g = _PAPER_TO_GEOMETRIC[m]
                s_m, n_m, d_m = g[0], g[1], g[2]
            elif m in _MCLUST_CONFUSABLES:
                raise ValueError(
                    f"{m!r} is an mclust covariance code, not an HDDC "
                    f"sub-model. mclust uses full-rank Gaussian "
                    f"covariances; HDDC uses rank-d_k structured "
                    f"covariances. Valid HDDC codes: "
                    f"{sorted(_VALID_GEOMETRIC)}. The closest HDDC "
                    f"analogue of mclust's VVV is HDDC's 'AVV'."
                )
            else:
                raise ValueError(
                    f"Unknown model {m!r}. Valid forms: a 3-letter "
                    f"geometric code ({sorted(_VALID_GEOMETRIC)}) or a "
                    f"paper bracket string ({sorted(_VALID_PAPER)})."
                )

            # Strict no-conflict check against axis kwargs.
            for axis_name, kwarg_val, code_letter, table in (
                ("signal", self.signal, s_m, _SIGNAL),
                ("noise",  self.noise,  n_m, _NOISE),
                ("dim",    self.dim,    d_m, _DIM),
            ):
                if kwarg_val is None:
                    continue
                expected = table[kwarg_val]
                if expected != code_letter:
                    inverse = {v: k for k, v in table.items()}
                    model_says = inverse[code_letter]
                    raise ValueError(
                        f"Conflict on `{axis_name}`: model={m!r} "
                        f"implies {axis_name}={model_says!r} "
                        f"({code_letter!r}), but "
                        f"{axis_name}={kwarg_val!r} ({expected!r}) was "
                        f"also passed. Either drop one of them or pass "
                        f"the resulting model code directly."
                    )

            code = s_m + n_m + d_m

        if code not in _VALID_GEOMETRIC:
            raise ValueError(
                f"Resolved model {code!r} is not one of the 14 valid "
                f"HDDC sub-models. In particular, signal='common_axis' "
                f"(C) requires dim='equal' (E)."
            )
        self._geometric_model_ = code

    # ------------------------------------------------------------------ #
    # Parameter count - driven by the geometric model code, per Table 1
    # ------------------------------------------------------------------ #
    def _n_parameters(self) -> int:
        """Number of free parameters for the resolved sub-model.

        Implements the formulas of Table 1 of Bouveyron, Girard &
        Schmid (2007). Conventions:

        - ``rho = K p + (K - 1)`` for means + mixing proportions
        - ``tau_bar = sum_k d_k * (p - (d_k + 1) / 2)`` for free
          orientations (``Q_k``, geometric position 2 onwards is
          always ``Q_k`` in this implementation since common-Q models
          are not in the 14-row table we ship)
        - ``tau = d * (p - (d + 1) / 2)`` for the common-``d``
          sub-models, with ``d`` the shared signal dimension
        - ``D = sum_k d_k``
        """
        K = self.n_components
        p = self.n_features_in_
        d_k = np.asarray(self.signal_dims_, dtype=float)
        D = float(np.sum(d_k))

        rho = K * p + (K - 1)
        tau_bar = float(np.sum(d_k * (p - (d_k + 1.0) / 2.0)))

        m = self._geometric_model_

        # Free-d sub-models (last letter V): use tau_bar directly.
        if m == "AVV":
            return int(rho + tau_bar + 2 * K + D)
        if m == "AEV":
            return int(rho + tau_bar + K + D + 1)
        if m == "IVV":
            return int(rho + tau_bar + 3 * K)
        if m == "IEV":
            return int(rho + tau_bar + 2 * K + 1)
        if m == "UVV":
            return int(rho + tau_bar + 2 * K + 1)
        if m == "UEV":
            return int(rho + tau_bar + K + 2)

        # Common-d sub-models (last letter E): all d_k = d.
        d = float(d_k[0])
        tau = d * (p - (d + 1.0) / 2.0)

        if m == "AVE":
            return int(rho + K * (tau + d + 1) + 1)
        if m == "CVE":
            return int(rho + K * (tau + 1) + d + 1)
        if m == "AEE":
            return int(rho + K * (tau + d) + 2)
        if m == "CEE":
            return int(rho + K * tau + d + 2)
        if m == "IVE":
            return int(rho + K * (tau + 2) + 1)
        if m == "IEE":
            return int(rho + K * (tau + 1) + 2)
        if m == "UVE":
            return int(rho + K * (tau + 1) + 2)
        if m == "UEE":
            return int(rho + K * tau + 3)

        raise ValueError(  # pragma: no cover - guarded by _resolve_model
            f"Unknown geometric model {m!r}"
        )

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    # The expensive exploration of cluster-assignment space lives in
    # KMeans (which we run with kmeans++ and ``n_init = self.n_init``).
    # Once KMeans has returned the best of those runs by inertia, we
    # derive responsibilities, means, and **full** per-cluster
    # covariances; the first M-step then projects those full
    # covariances onto the chosen HDDC configuration (eigendecomp +
    # per-cluster signal-dim selection + constraint projection in
    # ``_apply_model_constraints``). That removes the wasteful pattern
    # of running KMeans inside each of multiple outer HDDC restarts.

    def _initialize_responsibilities(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        rng = self.random_state_
        if isinstance(self.init_params, np.ndarray):
            labels = self.init_params
            if labels.shape[0] != n_samples:
                raise ValueError(
                    "init_params array length does not match n_samples"
                )
            if not np.all(np.isin(labels, np.arange(self.n_components))):
                raise ValueError("init_params contains invalid cluster ids")
            return np.eye(self.n_components)[labels]
        if self.init_params == "kmeans":
            # KMeans absorbs ALL of self.n_init: with kmeans++ seeding
            # it runs ``n_init`` independent starts and keeps the best
            # by inertia. The HDDC EM that follows is then a single
            # deterministic refinement from that good init.
            labels = KMeans(
                n_clusters=self.n_components,
                init="k-means++",
                random_state=rng,
                n_init=max(self.n_init, 1),
            ).fit_predict(X)
            return np.eye(self.n_components)[labels]
        if self.init_params == "random":
            R = rng.rand(n_samples, self.n_components)
            return R / R.sum(axis=1, keepdims=True)
        raise ValueError(f"Unknown init_params {self.init_params!r}")

    # ------------------------------------------------------------------ #
    # Model-constraint projection in the M-step
    # ------------------------------------------------------------------ #

    def _compute_global_signal_dim(self, X: np.ndarray) -> Optional[int]:
        """Common signal dim from the *global* scatter, à la HDclassif.

        Only meaningful when the resolved sub-model has ``d == "E"``
        (tied signal dimension across clusters) and the user did not
        force ``signal_dim``. Returns ``None`` in every other case;
        callers must check.

        The computation mirrors ``HDclassif::hddc_main``: form the
        global covariance from the *single* dataset mean (not
        per-cluster), eigendecompose it, then apply the same Cattell
        rule the M-step uses on per-cluster eigenvalues. The SVD-of-
        data-matrix path is selected when ``p > n`` to avoid an
        expensive ``p x p`` eigendecomposition.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        d : int or None
            Forced common signal dim if computed, else ``None``.
        """
        _, _, d_axis = self._geometric_model_
        # Not a tied-d model, or user forced d: nothing to do.
        if d_axis != "E" or self.signal_dim is not None:
            return None

        n, p = X.shape
        Xc = X - X.mean(axis=0, keepdims=True)
        if p > n:
            # Same SVD trick as the M-step: cheaper than forming the
            # p x p covariance when p > n.
            _, s, _ = np.linalg.svd(Xc, full_matrices=False)
            eigvals = np.zeros(p)
            eigvals[: s.size] = (s * s) / n
            rank_eff = s.size
        else:
            cov = (Xc.T @ Xc) / n
            cov = 0.5 * (cov + cov.T) + _EPS * np.eye(p)
            eigvals_asc, _ = linalg.eigh(cov)
            eigvals = eigvals_asc[::-1]
            rank_eff = p

        d = _cattell_scree_test(eigvals[:rank_eff], self.cattell_threshold)
        return max(1, min(int(d), rank_eff - 1, p - 1))

    def _apply_model_constraints(self) -> None:
        """Project parameters onto the resolved sub-model's constraints.

        Reads ``self._geometric_model_`` (3-letter code). Each letter
        triggers one of the constraint sub-steps.

        Ordering matters: ``d`` is collapsed first so the noise step
        sees the final per-cluster signal dimensions; the noise step
        recomputes ``b_k`` from the per-cluster eigenvalues using
        HDclassif's formula (trace minus signal mass, divided by the
        full noise subspace dimension), then ties it across clusters
        if ``noise == "E"``.
        """
        s, n, d = self._geometric_model_
        p = self.n_features_in_

        # 1) signal dimension regime — must run first so the noise
        #    computation below sees the right ``d_k`` (e.g. when the
        #    user forced ``signal_dim`` on an E-suffix model).
        if d == "E":
            if self.signal_dim is not None:
                shared = int(self.signal_dim)
            elif getattr(self, "_global_signal_dim_", None) is not None:
                # HDclassif-aligned: use the global-covariance Cattell
                # pick computed once in fit(), not the mean of
                # per-cluster picks.
                shared = int(self._global_signal_dim_)
            else:
                shared = int(round(np.mean(self.signal_dims_)))
            shared = max(1, min(shared, p - 1))
            self.signal_dims_ = [shared] * self.n_components

        # 2) noise regime — first recompute per-cluster ``b_k`` from
        #    the *current* eigenvalues and (possibly collapsed) ``d_k``;
        #    then, if tied, apply HDclassif's mixing-proportion-weighted
        #    formula:
        #        b_tied = Σ_k π_k (trace_k − sum signal_k)
        #                 ─────────────────────────────────
        #                          p − Σ_k π_k d_k
        weighted_noise_mass = 0.0
        weighted_d = 0.0
        for k in range(self.n_components):
            dk = self.signal_dims_[k]
            trace_k = float(np.sum(self.eigenvalues_[k]))
            signal_sum = float(np.sum(self.eigenvalues_[k][:dk]))
            noise_mass = max(trace_k - signal_sum, 0.0)
            if dk < p:
                self.noise_variances_[k] = max(noise_mass / (p - dk), _EPS)
            else:  # pragma: no cover
                self.noise_variances_[k] = _EPS
            weighted_noise_mass += self.weights_[k] * noise_mass
            weighted_d += self.weights_[k] * dk

        if n == "E":
            denom = max(p - weighted_d, _EPS)
            b_tied = max(weighted_noise_mass / denom, _EPS)
            self.noise_variances_ = np.full(self.n_components, b_tied)

        # 3) signal-eigenvalue regime
        if s == "A":
            pass  # free per cluster, per axis
        elif s == "I":
            # Isotropic within each cluster.
            for k in range(self.n_components):
                dk = self.signal_dims_[k]
                v = float(np.mean(self.eigenvalues_[k][:dk]))
                self.eigenvalues_[k][:dk] = v
        elif s == "C":
            # Common per-axis across clusters (requires d == "E", which
            # _resolve_model already enforces).
            dk = self.signal_dims_[0]
            shared = np.mean(
                np.stack([self.eigenvalues_[k][:dk]
                          for k in range(self.n_components)]),
                axis=0,
            )
            for k in range(self.n_components):
                self.eigenvalues_[k][:dk] = shared
        elif s == "U":
            # Uniform scalar shared across clusters and axes.
            vals = []
            for k in range(self.n_components):
                dk = self.signal_dims_[k]
                vals.append(np.mean(self.eigenvalues_[k][:dk]))
            v = float(np.mean(vals))
            for k in range(self.n_components):
                dk = self.signal_dims_[k]
                self.eigenvalues_[k][:dk] = v
        else:
            raise ValueError(  # pragma: no cover
                f"unknown signal regime {s!r}"
            )

    # ------------------------------------------------------------------ #
    # E and M steps
    # ------------------------------------------------------------------ #

    def _compute_log_density(self, X: np.ndarray, k: int) -> np.ndarray:
        """Log-density of component ``k`` evaluated at each row of ``X``.

        Uses the HDDC factorisation of the per-component covariance:
        signal eigenvalues on the first ``d_k`` axes of ``Q_k``, and
        an isotropic ``b_k`` on the orthogonal complement.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        k : int
            Component index in ``[0, n_components)``.

        Returns
        -------
        log_density : ndarray of shape (n_samples,)
            ``log N(x_i ; mu_k, Sigma_k)`` under the HDDC
            parameterisation.
        """
        diff = X - self.means_[k]
        eigvals = np.maximum(self.eigenvalues_[k], _EPS)
        dk = self.signal_dims_[k]
        b_k = max(self.noise_variances_[k], _EPS)
        p = self.n_features_in_

        log_det = (
            float(np.sum(np.log(eigvals[:dk])))
            + (p - dk) * math.log(b_k)
        )

        Qk = self.eigenvectors_[k][:, :dk]
        proj = diff @ Qk
        signal = np.sum(proj ** 2 / eigvals[:dk], axis=1)
        residual = diff - proj @ Qk.T
        noise = np.sum(residual ** 2, axis=1) / b_k

        return -0.5 * (log_det + signal + noise + p * math.log(2 * math.pi))

    def _e_step(self, X: np.ndarray) -> float:
        """E-step: refresh ``self._responsibilities`` and return log L(X).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        log_likelihood : float
            Sample log-likelihood ``sum_i log p(x_i)`` under the
            current parameters. Used by the outer EM loop both to
            check convergence (change since previous iteration < tol)
            and to compare final fits across ``n_init`` restarts.
        """
        n = X.shape[0]
        log_resp = np.empty((n, self.n_components))
        for k in range(self.n_components):
            log_resp[:, k] = self._compute_log_density(X, k)
        log_resp = log_resp + np.log(self.weights_)
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        self._responsibilities = np.exp(log_resp - log_norm)
        return float(np.sum(log_norm))

    def _repair_cluster_sizes(self, X: np.ndarray, Nk: np.ndarray) -> np.ndarray:
        """Reassign points until every cluster has >= ``min_cluster_size`` mass.

        Repairs under-sized clusters by hard-reassigning points with
        the highest residual responsibility budget. Mutates
        ``self._responsibilities`` in place and returns the updated
        ``Nk`` (column-sum vector).

        Invariants:

        1. ``n_missing`` rounds *up* so a single iteration always
           lifts ``Nk[k_min]`` strictly past ``min_cluster_size``
           (otherwise sub-half-unit deficits make the loop spin).
        2. Stolen rows are zeroed across all clusters before being
           set to 1 in ``k_min``, so row sums stay at 1 and other
           clusters' ``Nk`` are accounted for correctly.

        The feasibility check in ``fit`` (``K * min_cluster_size <=
        n``) guarantees this loop terminates.
        """
        n = X.shape[0]
        steal_guard = 0
        while np.min(Nk) < self.min_cluster_size:
            steal_guard += 1
            if steal_guard > 4 * self.n_components:
                # Safety net — should never trigger given the feasibility
                # check, but it keeps a bad initialisation from spinning.
                break
            k_min = int(np.argmin(Nk))
            p_steal = 1.0 - self._responsibilities[:, k_min]
            p_steal = np.clip(p_steal, _EPS, None)
            p_steal /= p_steal.sum()
            n_missing = int(math.ceil(self.min_cluster_size - Nk[k_min]))
            n_missing = max(1, min(n_missing, n))
            idx = self.random_state_.choice(
                n, size=n_missing, replace=False, p=p_steal,
            )
            self._responsibilities[idx, :] = 0.0
            self._responsibilities[idx, k_min] = 1.0
            Nk = self._responsibilities.sum(axis=0)
        return Nk

    def _eigendecompose_cluster(
        self, Xc: np.ndarray, Nk_k: float, p: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return ``(eigvals, eigvecs, rank_eff)`` for one cluster's scatter.

        Two paths give the same eigendecomposition of
        ``(1/Nk)·X_c^T X_c`` with very different cost:

        * ``eigh(cov)``: ``O(n p^2 + p^3)`` — cheap when ``n >= p``.
        * ``svd(Xc)``  : ``O(min(n,p)^2 · max(n,p))`` — cheap when ``p > n``.

        For HDDC's signature regime (``n << p``) the SVD path is
        orders of magnitude faster (~4800× at ``n=100, p=4096``) and
        is what HDclassif uses internally. We pick whichever is
        cheaper for the given cluster.

        Parameters
        ----------
        Xc : ndarray of shape (n_samples, p)
            Centred, responsibility-weighted data for one cluster.
        Nk_k : float
            Effective cluster count ``sum_i tau_{i,k}`` (>= 1).
        p : int
            Ambient feature dimension.

        Returns
        -------
        eigvals : ndarray of shape (p,)
            Eigenvalues sorted non-increasing, zero-padded beyond
            ``rank_eff``.
        eigvecs : ndarray of shape (p, p)
            Corresponding eigenvectors; only the first ``rank_eff``
            columns carry signal in the SVD branch.
        rank_eff : int
            Number of well-defined eigenvalues (``min(n, p)``).
        """
        n_samples = Xc.shape[0]
        if p > n_samples:
            # Economy SVD of the centred+weighted data matrix.
            # ``s`` has length ``min(n, p) = n``; the rows of ``Vt``
            # (shape (n, p)) give the orthonormal signal directions,
            # ``s²/Nk`` the corresponding eigenvalues.
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            eigvals = np.zeros(p)
            eigvals[:n_samples] = (s * s) / Nk_k
            # ``eigenvectors_[k]`` is kept as a (p, p) array for
            # downstream uniformity, but only the first ``n_samples``
            # columns carry real information; the rest sit in the
            # null space of X_c^T X_c and are never read (log-density
            # only slices ``[:, :dk]`` with ``dk < n_samples`` by
            # construction).
            eigvecs = np.zeros((p, p))
            eigvecs[:, :n_samples] = Vt.T
            rank_eff = n_samples
        else:
            cov = (Xc.T @ Xc) / Nk_k
            cov = 0.5 * (cov + cov.T) + _EPS * np.eye(p)
            eigvals_asc, eigvecs_asc = linalg.eigh(cov)
            eigvals = eigvals_asc[::-1]
            eigvecs = eigvecs_asc[:, ::-1]
            rank_eff = p
        return eigvals, eigvecs, rank_eff

    def _m_step(self, X: np.ndarray) -> None:
        """One EM M-step: refresh ``weights_``, ``means_`` and
        per-cluster ``(eigenvalues_, eigenvectors_, signal_dims_,
        noise_variances_)``, then collapse to the resolved sub-model.

        Splits naturally into three sub-steps:

        1. ``_repair_cluster_sizes`` — fix under-populated clusters.
        2. Per-cluster ``_eigendecompose_cluster`` and Cattell pick.
        3. ``_apply_model_constraints`` — collapse the free fit to
           the geometry of the resolved sub-model.
        """
        n, p = X.shape
        Nk = self._responsibilities.sum(axis=0)
        Nk = self._repair_cluster_sizes(X, Nk)

        self.weights_ = Nk / n
        self.means_ = (self._responsibilities.T @ X) / Nk[:, None]

        for k in range(self.n_components):
            w = np.sqrt(self._responsibilities[:, k])[:, None]
            Xc = (X - self.means_[k]) * w  # shape (n, p)

            eigvals, eigvecs, rank_eff = self._eigendecompose_cluster(
                Xc, Nk[k], p
            )
            self.eigenvalues_[k] = eigvals
            self.eigenvectors_[k] = eigvecs

            # Per-cluster signal dim via scree test; the constraint
            # projection in _apply_model_constraints will collapse
            # these if the resolved model has dim=="E". Capped at
            # ``rank_eff - 1`` so the noise subspace is non-empty.
            d_full = _cattell_scree_test(
                self.eigenvalues_[k][:rank_eff], self.cattell_threshold,
            )
            self.signal_dims_[k] = max(1, min(d_full, rank_eff - 1, p - 1))
            # Per-cluster noise variance follows HDclassif's formula:
            #     b_k = (sum(all_eigvals) - sum(signal_eigvals)) / (p - dk)
            # i.e. average over the *full* noise subspace dimension
            # (p - dk), not just the empirically realised rank. When
            # n > p the two are identical; when n < p the null-space
            # directions count as zero-variance contributors to the
            # average, which is the convention HDclassif documents.
            dk = self.signal_dims_[k]
            if dk < p:
                trace_total = float(np.sum(self.eigenvalues_[k]))
                signal_sum = float(np.sum(self.eigenvalues_[k][:dk]))
                self.noise_variances_[k] = max(
                    (trace_total - signal_sum) / (p - dk),
                    _EPS,
                )
            else:  # pragma: no cover - guarded by min above
                self.noise_variances_[k] = _EPS

        self._apply_model_constraints()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X, y=None):
        """Fit the HDDC model on ``X``.

        Resolves the requested sub-model (geometric code / paper
        bracket / per-axis kwargs), validates input, runs ``n_init``
        EM trials from the configured initialiser, and keeps the
        fit with the highest log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Will be coerced to ``float64``.
        y : Ignored
            Present for sklearn API compatibility.

        Returns
        -------
        self : object
            The fitted estimator. See the class-level
            ``Attributes`` section for the list of fitted attributes.

        Raises
        ------
        ValueError
            If the sub-model spec resolves with a conflict, an
            unknown code (including any mclust covariance code), or
            if ``n_components * min_cluster_size > n_samples``.
        """
        self._validate_params()
        # Resolves model + per-axis kwargs to self._geometric_model_,
        # raising on conflict, unknown spelling, or mclust code.
        self._resolve_model()
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        n, p = X.shape

        # Compute the global common signal dimension once, up-front,
        # when the resolved sub-model uses a tied d ("E" suffix) and
        # the user did not force ``signal_dim``. This mirrors
        # HDclassif's behaviour: on the *D family it eigendecomposes
        # the **global** scatter matrix (one mean across all data,
        # not per-cluster), applies the Cattell scree rule once, and
        # locks the resulting common dimension for every EM
        # iteration. Averaging per-cluster Cattell picks (the
        # previous approach) gave systematically different d's on
        # high-K datasets like digits.
        self._global_signal_dim_ = self._compute_global_signal_dim(X)

        # Feasibility check: the steal mechanism in _m_step can only
        # satisfy ``K * min_cluster_size <= n``. If the requested
        # configuration is infeasible, raise early with a clear message
        # rather than spin in an unsatisfiable steal loop.
        if self.n_components * self.min_cluster_size > n:
            raise ValueError(
                "HighDimensionalGaussianMixture: n_components"
                f" * min_cluster_size = {self.n_components}"
                f" * {self.min_cluster_size}"
                f" = {self.n_components * self.min_cluster_size}"
                f" exceeds n_samples = {n}. Reduce either n_components"
                " or min_cluster_size."
            )

        # Determine how many EM restarts to do AFTER the kmeans init:
        # - "kmeans" / ndarray: KMeans already absorbed self.n_init by
        #   keeping the best of n_init kmeans++ runs. A single EM
        #   refinement from that init is then deterministic-given-rng,
        #   so only one outer pass is needed.
        # - "random": there is no clustering exploration step to
        #   absorb the budget; keep self.n_init random restarts.
        if isinstance(self.init_params, np.ndarray) \
           or self.init_params == "kmeans":
            n_outer = 1
        else:
            n_outer = self.n_init

        # Seed all inits from a single master RNG so the loop works
        # uniformly whether ``self.random_state`` is None, an int, or
        # a RandomState/Generator instance. Each init draws a fresh
        # integer seed; this keeps inits deterministic given the
        # master and avoids ``int + RandomState`` TypeErrors.
        master_rng = check_random_state(self.random_state)
        init_seeds = master_rng.randint(0, np.iinfo(np.int32).max, size=n_outer)

        results = [
            self._run_em(X, seed=int(init_seeds[i]), label=i)
            for i in range(n_outer)
        ]
        self._select_best_init(results)
        return self

    def _run_em(self, X: np.ndarray, seed: int, label: int) -> dict:
        """Run one EM trial from a fresh init and return the resulting fit.

        Returns a dict snapshotting the trial's fitted attributes so
        ``_select_best_init`` can compare across trials without
        relying on ``self`` state (which the next trial would
        overwrite).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        seed : int
            Per-trial RNG seed drawn from the master RNG.
        label : int
            Trial index, used only for ``verbose`` logging.
        """
        n, p = X.shape
        self.random_state_ = check_random_state(seed)
        self._responsibilities = self._initialize_responsibilities(X)
        self.weights_ = self._responsibilities.sum(axis=0) / n
        self.means_ = (
            (self._responsibilities.T @ X) / self.weights_[:, None] / n
        )
        # Placeholders; the first M-step replaces them with the
        # eigendecomposition of each cluster's full empirical
        # covariance and then projects onto the resolved HDDC
        # configuration via ``_apply_model_constraints``.
        self.eigenvalues_ = [np.ones(p) for _ in range(self.n_components)]
        self.eigenvectors_ = [np.eye(p) for _ in range(self.n_components)]
        self.signal_dims_ = [1] * self.n_components
        self.noise_variances_ = np.ones(self.n_components)

        prev = -np.inf
        ll = -np.inf
        converged = False
        for it in range(self.max_iter):
            self._m_step(X)
            ll = self._e_step(X)
            if self.verbose:
                print(f"[init {label}] iter {it:4d}  ll={ll:.6f}")
            if abs(ll - prev) < self.tol:
                converged = True
                break
            prev = ll

        return dict(
            weights_=self.weights_.copy(),
            means_=self.means_.copy(),
            eigenvalues_=[e.copy() for e in self.eigenvalues_],
            eigenvectors_=[v.copy() for v in self.eigenvectors_],
            signal_dims_=list(self.signal_dims_),
            noise_variances_=self.noise_variances_.copy(),
            _responsibilities=self._responsibilities.copy(),
            n_iter_=it + 1,
            converged_=converged,
            lower_bound_=ll,
        )

    def _select_best_init(self, results: list) -> None:
        """Pick the highest-log-likelihood trial and set fitted attrs.

        Mirrors ``GaussianMixture``'s "keep the best of ``n_init``
        random restarts" pattern. Sets every fitted attribute on
        ``self`` from the winning trial's snapshot.
        """
        best = max(results, key=lambda r: r["lower_bound_"])
        for k, v in best.items():
            setattr(self, k, v)
        # ``lower_bound_`` mirrors ``GaussianMixture.lower_bound_`` —
        # log-likelihood at the EM fixed point of the winning init.
        # (For vanilla EM the ELBO equals the log-likelihood; the name
        # ``lower_bound_`` is kept for sklearn API parity.)
        self.labels_ = self._responsibilities.argmax(axis=1)

    def _check_fitted(self) -> None:
        check_is_fitted(self, "weights_")

    def _validate_X_predict(self, X) -> np.ndarray:
        """Validate ``X`` against the shape and dtype seen at fit time.

        Thin wrapper around :func:`sklearn.utils.validation.validate_data`
        with ``reset=False`` so the feature count and dtype must match
        the training data. Used by every predict-time method
        (``predict``, ``predict_proba``, ``score_samples``).
        """
        return validate_data(self, X, dtype=np.float64, reset=False)

    def predict(self, X):
        """Predict the hard cluster assignment for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to assign. Must have the same number of features
            as the data passed to :meth:`fit`.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the maximum-posterior component per sample, in
            ``[0, n_components)``. Equivalent to
            ``predict_proba(X).argmax(axis=1)``.
        """
        self._check_fitted()
        X = self._validate_X_predict(X)
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """Posterior cluster responsibilities for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        resp : ndarray of shape (n_samples, n_components)
            Soft cluster assignment ``tau_{ik} = P(z_i = k | x_i)``.
            Each row sums to 1.
        """
        self._check_fitted()
        X = self._validate_X_predict(X)
        n = X.shape[0]
        log_resp = np.empty((n, self.n_components))
        for k in range(self.n_components):
            log_resp[:, k] = self._compute_log_density(X, k)
        log_resp = log_resp + np.log(self.weights_)
        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
        return np.exp(log_resp)

    def score_samples(self, X):
        """Log-density of the fitted mixture, per sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        log_prob : ndarray of shape (n_samples,)
            ``log p(x_i)`` under the fitted mixture, with the HDDC
            per-component covariance parameterisation.
        """
        self._check_fitted()
        X = self._validate_X_predict(X)
        n = X.shape[0]
        log_resp = np.empty((n, self.n_components))
        for k in range(self.n_components):
            log_resp[:, k] = self._compute_log_density(X, k)
        log_resp = log_resp + np.log(self.weights_)
        return logsumexp(log_resp, axis=1)

    def score(self, X, y=None):
        """Mean log-density of ``X`` under the fitted mixture.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.
        y : Ignored
            Present for sklearn API compatibility.

        Returns
        -------
        avg_log_prob : float
            ``mean(score_samples(X))``. Higher is better.
        """
        return float(np.mean(self.score_samples(X)))

    def fit_predict(self, X, y=None):
        """Fit the model and return hard cluster labels for ``X``.

        Convenience method equivalent to ``self.fit(X).labels_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Present for sklearn API compatibility.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Hard cluster assignment per training sample.
        """
        return self.fit(X, y).labels_

    # ------------------------------------------------------------------ #
    # Information criteria  (lower-is-better, matching GaussianMixture)
    # ------------------------------------------------------------------ #

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Same sign convention as :meth:`sklearn.mixture.GaussianMixture.bic`
        (lower is better).

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        self._check_fitted()
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * math.log(
            X.shape[0]
        )

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Same sign convention as :meth:`sklearn.mixture.GaussianMixture.aic`
        (lower is better).

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        aic : float
            The lower the better.
        """
        self._check_fitted()
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

    def icl(self, X):
        """Integrated Completed Likelihood criterion for the current model on the input X.

        Same sign convention as :meth:`bic` (lower is better). See
        ``docs/INFORMATION_CRITERIA.md`` §2 for the derivation.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        icl : float
            The lower the better.

        Notes
        -----
        ``ICL = BIC + 2 H`` with
        ``H = - sum_i sum_k tau_{ik} log tau_{ik}`` the entropy of the
        posterior responsibilities at the fitted parameters. Uses
        :func:`scipy.special.xlogy` so zero responsibilities contribute
        exactly zero (no ``eps`` clip, no ``RuntimeWarning``), and
        ``ICL`` reduces to ``BIC`` on a hard partition.

        References
        ----------
        .. [1] :doi:`Biernacki, C., Celeux, G., & Govaert, G. (2000).
           "Assessing a mixture model for clustering with the integrated
           completed likelihood." IEEE TPAMI, 22(7), 719-725.
           <10.1109/34.865189>`
        """
        self._check_fitted()
        resp = self.predict_proba(X)
        entropy = -float(xlogy(resp, resp).sum())
        return self.bic(X) + 2.0 * entropy

    # ------------------------------------------------------------------ #
    # sklearn tags
    # ------------------------------------------------------------------ #
    def __sklearn_tags__(self):
        """Mark HDDC as non-deterministic w.r.t. initialization.

        EM with kmeans / random init does not converge to the same
        local optimum for arbitrary input shapes; the upstream
        ``check_estimators_dtypes`` and similar checks that compare
        fits across calls would otherwise fail.
        """
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
