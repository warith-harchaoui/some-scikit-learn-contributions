.. Snippet to append inside ``doc/modules/mixture.rst``, inside the dropdown
.. titled "Selecting the number of components in a classical Gaussian
.. Mixture model".  Add a new ``.. _bic_icl:`` label just before, so the
.. cross-reference from ``GaussianMixture.icl`` resolves.

.. _bic_icl:

.. dropdown:: Selecting the number of components with BIC and ICL

  In addition to BIC (see above), :class:`GaussianMixture` provides ICL
  via :meth:`GaussianMixture.icl`. Both criteria follow the
  "lower is better" convention.

  - BIC penalizes by ``nu log n``.
  - ICL adds to BIC twice the entropy of the posterior responsibilities,
    ``ICL = BIC + 2 H`` with ``H = - sum_i sum_k tau_ik log tau_ik``.

  Because :math:`H \ge 0`, we always have :math:`\mathrm{ICL} \ge
  \mathrm{BIC}`. The two coincide on a hard partition and diverge in
  proportion to how overlapping the components are. ICL was introduced by
  Biernacki, Celeux and Govaert (2000) precisely to discourage spurious
  components that exist only to refine an already well-modeled region of
  feature space - a regime in which BIC tends to overestimate the number
  of components.

  **When ICL helps in practice.** Whenever the true data-generating
  process has heavier tails than a Gaussian, fitting a Gaussian mixture
  forces extra Gaussian components to "cover" the tails. BIC rewards the
  added likelihood and the parameter penalty is too mild to compensate;
  ICL rewards classification certainty and rejects those extra
  components. A canonical illustration is fitting a Gaussian mixture to
  a heavy-tailed Student-:math:`t` mixture: BIC drifts to higher
  :math:`K` while ICL recovers the true :math:`K` (see
  ``test_gaussian_mixture_icl_student_mixture``).

  .. rubric:: References

  * Biernacki, C., Celeux, G., & Govaert, G. (2000). Assessing a mixture
    model for clustering with the integrated completed likelihood.
    *IEEE TPAMI*, 22(7), 719-725.
