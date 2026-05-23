.. Snippet to splice into ``doc/modules/mixture.rst`` as a new
.. subsection just after the existing GaussianMixture section.
.. Add a ``.. _hddc:`` label so cross-references from the class
.. docstring resolve.

.. _hddc:

HighDimensionalGaussianMixture (HDDC)
-------------------------------------

:class:`HighDimensionalGaussianMixture` is a parsimonious Gaussian
mixture for the high-dimensional, small-sample regime
(:math:`n \lesssim p`) where :class:`GaussianMixture` with
``covariance_type="full"`` over-parameterises and ``"diag"`` /
``"spherical"`` discard too much covariance structure.

It implements the HDDC family of Bouveyron, Girard & Schmid (2007):
each cluster's covariance is factorised as a low-rank signal
subspace plus an isotropic noise variance,

.. math::

   \Sigma_k = Q_k\, \mathrm{diag}(a_{k1}, \ldots, a_{kd_k})\, Q_k^\top
              \;+\; b_k \, (I_p - Q_k Q_k^\top),

where :math:`Q_k` is the matrix of the :math:`d_k` leading
eigenvectors of cluster :math:`k`'s covariance and :math:`b_k` is
the common variance on the orthogonal noise subspace. Constraints
on the eigenvalues :math:`a_{kj}`, on :math:`b_k`, and on
:math:`d_k` (free vs. equal across clusters) define the 14
sub-models of Bouveyron 2007 Table 1, selected via the ``model``
string parameter.

**Three accepted spellings for** ``model``\ **.** The same 14
sub-models can be requested as:

- a 3-letter **geometric code** (recommended; e.g. ``"AVV"``,
  ``"IEE"``),
- a **paper bracket alias** matching Bouveyron 2007 (e.g.
  ``"akj_bk_Qk_dk"``),
- per-axis kwargs (``signal=``, ``noise=``, ``dim=``).

All three are validated to resolve to the same internal
representation; conflicts raise.

**Parameter count.** ``_n_parameters`` is computed exactly from
Table 1 of the paper. As a regression sanity check, the count
matches the paper's published numbers row-by-row (e.g.
``[a_kj * b * Q_k * d] = 4225`` for :math:`K=4, p=100, d=10`).

**Model selection.** :class:`HighDimensionalGaussianMixture`
exposes :meth:`~mixture.HighDimensionalGaussianMixture.bic` and
:meth:`~mixture.HighDimensionalGaussianMixture.icl` in the
"lower is better" convention used elsewhere in
:mod:`sklearn.mixture`. ICL is particularly useful for HDDC
because the high-dimensional regime makes BIC's over-counting more
severe — see :ref:`bic_icl` for the ICL story.

**Initialisation.** With ``init_params="kmeans"`` (default), the
``n_init`` parameter is consumed entirely by a single
``KMeans(n_init=n_init)`` call: kmeans++ explores cluster-assignment
space exhaustively, and the best of those runs (by inertia) seeds a
**single** HDDC EM refinement. This avoids the wasteful pattern of
nested KMeans-inside-each-HDDC-restart that would happen if ``n_init``
were interpreted as "EM restarts" the way it is for
:class:`GaussianMixture`. See the ``n_init`` docstring for the
exact contract under ``"random"`` and ndarray ``init_params``.

**Strict :math:`d_k` cap.** Each cluster's signal dimension is
selected per cluster via the relative-drop Cattell scree test on
its eigenvalues, capped at :math:`p - 1` so the orthogonal noise
subspace is non-empty.

.. rubric:: References

* Bouveyron, C., Girard, S., & Schmid, C. (2007). High-Dimensional
  Data Clustering. *Computational Statistics & Data Analysis*,
  52(1), 502-519.
* Berge, L., Bouveyron, C., & Girard, S. (2012). HDclassif: an R
  package for model-based clustering and discriminant analysis of
  high-dimensional data. *Journal of Statistical Software*, 46(6).
