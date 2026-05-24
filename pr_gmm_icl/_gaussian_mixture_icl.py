"""
Patch snippet for ``sklearn/mixture/_gaussian_mixture.py``.

This file is **not** meant to be imported. It is the exact text of the new
``icl`` method to be inserted in the ``GaussianMixture`` class, right after
the existing ``bic`` method.

Sign convention
---------------
``ICL = BIC + 2 * H``, where ``H = - sum_i sum_k tau_ik log tau_ik`` is the
entropy of the (soft) responsibilities at the EM fixed point.

This is the "lower is better" convention obtained by multiplying the
higher-is-better Biernacki-Celeux-Govaert (2000) ICL by ``-2``, and is
consistent with ``GaussianMixture.bic``.

For a hard (MAP) partition, ``tau_ik in {0, 1}`` gives ``H = 0`` and
``ICL = BIC``. Soft / overlapping partitions are penalized.

References
----------
- Biernacki, C., Celeux, G., & Govaert, G. (2000). Assessing a mixture model
  for clustering with the integrated completed likelihood. IEEE TPAMI, 22(7),
  719-725.
"""

# --- INSERT AFTER ``def bic(self, X)`` IN GaussianMixture ------------------
#
# ``numpy`` is already imported as ``np`` at the top of
# ``sklearn/mixture/_gaussian_mixture.py`` upstream, so copy only the
# ``def icl(self, X)`` block below. The import line that follows is
# kept solely so this file is lint-clean when scanned in isolation.

import numpy as np  # noqa: F401 — present upstream; here for standalone lint.
from scipy.special import xlogy  # noqa: F401 — same rationale as numpy above.
from sklearn.utils.validation import check_is_fitted  # noqa: F401 — same.


def icl(self, X):
    """Integrated Completed Likelihood criterion for the current model on the input X.

    You can refer to this :ref:`mathematical section <bic_icl>` for more
    details regarding the formulation of the ICL used.

    Parameters
    ----------
    X : array of shape (n_samples, n_dimensions)
        The input samples.

    Returns
    -------
    icl : float
        The lower the better.

    Notes
    -----
    The ICL value is

    .. math::

        \\mathrm{ICL}(K)
        = \\mathrm{BIC}(K) + 2 \\, H,
        \\qquad
        H = -\\sum_{i=1}^{n} \\sum_{k=1}^{K}
            \\tau_{ik} \\log \\tau_{ik},

    where :math:`\\tau_{ik}` are the soft responsibilities at the EM
    fixed point. The convention matches :meth:`bic` (lower is better).
    On a hard partition (:math:`\\tau \\in \\{0, 1\\}`),
    :math:`H = 0` and ICL reduces to BIC.

    On data with heavy tails (e.g. Student-:math:`t` mixtures), BIC tends
    to overestimate the number of Gaussian components needed to "cover"
    the tails because adding a redundant component always improves the
    fit by more than the BIC parameter penalty. ICL penalizes the
    additional component because the resulting cluster assignment is
    ambiguous, and therefore typically recovers the true number of
    components.

    References
    ----------
    .. [1] :doi:`Biernacki, C., Celeux, G., & Govaert, G. (2000).
       "Assessing a mixture model for clustering with the integrated
       completed likelihood." IEEE TPAMI, 22(7), 719-725.
       <10.1109/34.865189>`
    """
    # Explicit fitted check so users get NotFittedError before
    # ``_estimate_log_prob_resp`` would otherwise raise AttributeError
    # on the unset ``weights_`` / ``means_`` / covariance attributes.
    check_is_fitted(self)
    # ``_estimate_log_prob_resp`` returns (log p(X), log responsibilities);
    # recomputed cheaply on demand here, mirroring how ``bic`` recomputes
    # ``score`` on demand (no caching in the upstream class).
    _, log_resp = self._estimate_log_prob_resp(X)
    # xlogy(0, 0) = 0 by definition, so zero responsibilities contribute
    # exactly zero with no eps clip and no RuntimeWarning.
    resp = np.exp(log_resp)
    entropy = -xlogy(resp, resp).sum()
    return self.bic(X) + 2.0 * entropy

# --- END SNIPPET -----------------------------------------------------------
