# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
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
# Upstream ``sklearn/mixture/_gaussian_mixture.py`` already imports
# ``numpy as np``, ``math``, ``check_is_fitted``, and ``validate_data``;
# the imports below are kept only so this file is lint-clean in
# isolation. They are duplicates upstream — drop when copying.

import math  # noqa: F401 — present upstream; here for standalone lint.
import numpy as np  # noqa: F401
from scipy.special import xlogy  # noqa: F401
from sklearn.utils.validation import check_is_fitted, validate_data  # noqa: F401


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
    # Single validated pass through the data, mirroring the
    # ``score_samples`` / ``predict_proba`` idiom in this module:
    # ``check_is_fitted`` + ``validate_data(..., reset=False)`` then
    # ``_estimate_log_prob_resp`` once. We do **not** call
    # ``self.bic(X)`` here — that would re-validate ``X`` and re-call
    # ``_estimate_log_prob_resp``; the inline form computes BIC
    # from ``log_prob_norm.sum()`` directly.
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
    n_samples = X.shape[0]
    bic_value = (
        -2.0 * log_prob_norm.sum()
        + self._n_parameters() * math.log(n_samples)
    )
    # ``xlogy(0, 0) = 0`` by definition, so zero responsibilities
    # contribute exactly zero with no ``eps`` clip and no
    # ``RuntimeWarning`` — ICL reduces to BIC on a hard partition.
    resp = np.exp(log_resp)
    entropy = -xlogy(resp, resp).sum()
    return bic_value + 2.0 * entropy

# --- END SNIPPET -----------------------------------------------------------
