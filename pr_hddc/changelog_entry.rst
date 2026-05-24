.. Add to ``doc/whats_new/upcoming_changes/sklearn.mixture/`` as a new
.. ``<issue_number>.feature.rst`` fragment.

A new estimator :class:`mixture.HighDimensionalGaussianMixture`
implements the High-Dimensional Data Clustering family of Bouveyron,
Girard & Schmid (2007). All 14 sub-models of the paper's Table 1 are
exposed via a single ``model`` string (geometric 3-letter codes,
paper-bracket aliases, and per-axis kwargs are all accepted). The
estimator exposes :meth:`~mixture.HighDimensionalGaussianMixture.bic`,
:meth:`~mixture.HighDimensionalGaussianMixture.aic`, and
:meth:`~mixture.HighDimensionalGaussianMixture.icl` selection
criteria, all in the lower-is-better convention used by
:class:`~mixture.GaussianMixture`. By :user:`Warith Harchaoui
<warith-harchaoui>`.
