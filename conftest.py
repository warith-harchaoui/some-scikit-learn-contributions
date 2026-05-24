"""Root conftest that wires the draft PR code into ``sklearn.mixture``.

The two ``pr_*/test_*.py`` test files are written as if both PRs have
already landed in scikit-learn (they ``from sklearn.mixture import
HighDimensionalGaussianMixture`` etc.). This conftest performs the
monkey-patching once per pytest session so the tests can run against
the local PR drafts without an editable sklearn install.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import sklearn.mixture as _skm
from sklearn.mixture import GaussianMixture

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, path: str):
    """Load a file as a module and register it in sys.modules.

    Registering in sys.modules under the target name is what makes
    pickle.dumps work — pickle stores classes by their __module__
    attribute and imports that module by name at unpickle time.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# pr_gmm_icl: attach the `icl` method to GaussianMixture.
if not hasattr(GaussianMixture, "icl"):
    _icl_mod = _load(
        "_pr_gmm_icl_patch",
        os.path.join(_HERE, "pr_gmm_icl", "_gaussian_mixture_icl.py"),
    )
    GaussianMixture.icl = _icl_mod.icl

# pr_hddc: inject HighDimensionalGaussianMixture into sklearn.mixture.
# Register the loaded module as ``sklearn.mixture._hddc`` (the path it
# will live at once the PR lands) so pickle can round-trip estimator
# instances via sklearn's check_estimator pickle test.
if not hasattr(_skm, "HighDimensionalGaussianMixture"):
    _hddc_mod = _load(
        "sklearn.mixture._hddc",
        os.path.join(_HERE, "pr_hddc", "_hddc.py"),
    )
    _skm.HighDimensionalGaussianMixture = _hddc_mod.HighDimensionalGaussianMixture
    _hddc_mod.HighDimensionalGaussianMixture.__module__ = "sklearn.mixture._hddc"
