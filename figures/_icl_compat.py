"""Shared ICL-for-``GaussianMixture`` helper.

Tiny compat shim: until the ``pr_gmm_icl`` PR lands and
:meth:`sklearn.mixture.GaussianMixture.icl` becomes available
upstream, every script in this repo that wants to compute ICL on a
plain ``GaussianMixture`` does:

    ICL = BIC + 2 H,
    H   = - sum_i sum_k tau_{ik} log tau_{ik}

i.e. the lower-is-better convention this contribution argues for.
Three call sites used to inline this six-line snippet; this module
is the single source of truth they now share. **Delete this file
once the ICL PR is merged** — every caller will switch to
``gmm.icl(X)``.

Examples
--------
>>> from sklearn.mixture import GaussianMixture
>>> from _icl_compat import icl_gmm
>>> gmm = GaussianMixture(n_components=3).fit(X)
>>> icl_gmm(gmm, X)  # lower is better
12345.6
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def icl_gmm(gmm: GaussianMixture, X: np.ndarray) -> float:
    """Integrated Completed Likelihood (lower-is-better) of a fitted GMM.

    Equivalent to ``gmm.icl(X)`` once the ICL PR lands.

    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture
        A fitted estimator (``check_is_fitted`` is *not* enforced
        here — the caller is the test harness, not user input).
    X : ndarray of shape (n_samples, n_features)
        The samples to evaluate ICL on. Typically the training data.

    Returns
    -------
    icl : float
        ``BIC(X) + 2 H`` with H the entropy of the soft
        responsibilities at the EM fixed point.
    """
    # ``_estimate_log_prob_resp`` already lives on GaussianMixture
    # and returns (log p(X), log responsibilities) — exactly what we
    # need to plug into ``2H``.
    _, log_resp = gmm._estimate_log_prob_resp(X)
    resp = np.exp(log_resp)
    # ``nansum`` because ``log_resp`` may underflow to ``-inf`` when
    # a responsibility is exactly 0 (rare but possible on perfectly
    # separated data); the matching ``resp`` factor is then exactly
    # 0 too, so the term is 0 — `nansum` enforces that.
    entropy = -np.nansum(resp * log_resp)
    return float(gmm.bic(X) + 2.0 * entropy)
