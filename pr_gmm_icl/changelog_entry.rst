.. Add this entry to the appropriate ``doc/whats_new/upcoming_changes/``
.. file. Follow the towncrier-style convention used by sklearn on main
.. (one ``.rst`` fragment per change, named ``<issue_number>.<type>.rst``).
.. Adjust the issue number once the PR is opened.

:class:`mixture.GaussianMixture` now provides the
:meth:`~mixture.GaussianMixture.icl` method, which computes the
Integrated Completed Likelihood criterion of Biernacki, Celeux and
Govaert (2000) alongside the existing
:meth:`~mixture.GaussianMixture.bic`. ICL adds twice the entropy of the
posterior responsibilities to BIC and is robust to heavy-tailed
mixtures, where BIC tends to overestimate the number of components.
By :user:`Warith Harchaoui <warith-harchaoui>`.
