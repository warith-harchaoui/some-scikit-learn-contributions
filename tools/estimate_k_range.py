"""Estimate a sensible ``K_grid = [K_min, K_max]`` for a clustering
problem, **without fitting any mixture model**.

The idea (provided by the user):

1. Sequentially grow ``k-means++`` centres on the input data, one
   at a time, up to a hard ceiling. Record the seeding inertia
   :math:`\\Phi_K = \\sum_i \\min_{k \\le K} \\| x_i - c_k \\|^2`
   at each step. No Lloyd's iterations — the cost is
   :math:`O(K \\cdot n \\cdot p)` per run.

2. Repeat ``R = 20`` times with different seeds to wash out the
   randomness in the kmeans++ first-pick + sampling.

3. For each ``K``, compute the relative gain
   :math:`g_K = (\\Phi_{K-1} - \\Phi_K) / \\Phi_{K-1}` per run, then
   aggregate over runs with the **75th-percentile**
   (``gain_summary_K = q75(g_K over runs)``). The q75 is robust to
   lucky / unlucky seedings.

4. **Stop** at the first ``K`` where ``gain_summary_K < epsilon``
   for ``patience`` consecutive values of ``K``. That's the
   elbow.

5. Return ``K_min = 2``, ``K_max = K_stop``.

Why this is useful
------------------
:func:`auto_mixture.auto_select_mixture` searches over a user-
supplied ``K_grid``. Picking that grid by hand is annoying for
unknown data; using ``estimate_k_range`` produces a grid that
adapts to the data's intrinsic complexity in well under a second
on typical inputs.

Default constants are the values the user spelled out:

    min_cluster_size = 10
    K_hard_max       = min(100, n // min_cluster_size)
    epsilon          = 0.02
    patience         = 3
    R                = 20
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np


log = logging.getLogger("estimate_k_range")


@dataclass
class KRangeEstimate:
    """Result of :func:`estimate_k_range`."""
    K_min: int
    K_max: int
    K_hard_max: int
    inertia_per_run: np.ndarray   # shape (R, K_hard_max), Phi_K per run
    gain_per_run: np.ndarray      # shape (R, K_hard_max-1), per-step rel. gain
    gain_summary: np.ndarray      # shape (K_hard_max-1,), q75 over runs
    epsilon: float
    patience: int
    notes: dict = field(default_factory=dict)


def _grow_kmeans_pp_inertia(
    X: np.ndarray, k_max: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sequentially place ``k_max`` kmeans++ centres on ``X``.

    Returns
    -------
    inertias : ndarray of shape (k_max,)
        ``inertias[k]`` is the seeding inertia after placing
        ``k + 1`` centres (i.e. ``Phi_{k+1}``).
    """
    n = X.shape[0]
    # First centre uniformly at random.
    first_idx = int(rng.integers(n))
    d2 = np.sum((X - X[first_idx]) ** 2, axis=1)
    inertias = np.empty(k_max, dtype=float)
    inertias[0] = float(d2.sum())

    for k in range(1, k_max):
        total = float(d2.sum())
        if total <= 0.0:
            # All points coincident with a centre; no further gain.
            inertias[k:] = 0.0
            return inertias
        probs = d2 / total
        # ``rng.choice`` with explicit probabilities is the kmeans++
        # sampling rule.
        idx = int(rng.choice(n, p=probs))
        new_d2 = np.sum((X - X[idx]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)
        inertias[k] = float(d2.sum())
    return inertias


def estimate_k_range(
    X: np.ndarray,
    *,
    min_cluster_size: int = 10,
    k_hard_max: int = 100,
    epsilon: float = 0.02,
    patience: int = 3,
    n_runs: int = 20,
    random_state: int = 0,
) -> KRangeEstimate:
    """Adaptive ``(K_min, K_max)`` for a clustering problem.

    Implements the q75-of-relative-gains elbow rule on k-means++
    seeding inertia. See the module docstring for the algorithm.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    min_cluster_size : int, default=10
        Lower bound on samples-per-cluster; caps ``K_hard_max`` at
        ``n // min_cluster_size``.
    k_hard_max : int, default=100
        Absolute cap on the largest K considered.
    epsilon : float, default=0.02
        Below this relative gain, the curve is "flat".
    patience : int, default=3
        Number of consecutive flat steps that triggers the stop.
    n_runs : int, default=20
        Number of independent kmeans++ seedings (R).
    random_state : int, default=0
        Seeds ``rng = np.random.default_rng(random_state + r)``
        for each run ``r``.

    Returns
    -------
    KRangeEstimate
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2 * min_cluster_size:
        raise ValueError(
            f"Need at least 2*min_cluster_size={2 * min_cluster_size} "
            f"samples, got n={n}."
        )

    K_hard_max = min(int(k_hard_max), n // int(min_cluster_size))
    if K_hard_max < 2:
        raise ValueError(
            f"K_hard_max={K_hard_max} < 2 — relax min_cluster_size "
            f"or supply more data."
        )

    # Inertia table: (n_runs, K_hard_max). Column k holds Phi_{k+1}.
    inertia = np.empty((n_runs, K_hard_max), dtype=float)
    for r in range(n_runs):
        rng = np.random.default_rng(random_state + r)
        inertia[r] = _grow_kmeans_pp_inertia(X, K_hard_max, rng)

    # gain[r, k-1] = (Phi_{k} - Phi_{k+1}) / Phi_{k}
    # i.e. the relative drop when going from K=k to K=k+1.
    prev = inertia[:, :-1]
    nxt = inertia[:, 1:]
    # Guard against Phi_k == 0 (data exactly representable by k centres).
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(prev > 0, (prev - nxt) / prev, 0.0)
    # q75 across runs, robust to a few lucky / unlucky seedings.
    gain_summary = np.quantile(gain, 0.75, axis=0)

    # Walk K=2, 3, ... and look for ``patience`` consecutive flat
    # gains. ``gain_summary[i]`` corresponds to the step K=(i+1)->K=(i+2),
    # so we report K_stop = i+2 once the rule fires.
    K_stop = None
    consecutive = 0
    for i, g in enumerate(gain_summary):
        K = i + 2                       # K we are *about* to stop at
        if g < epsilon:
            consecutive += 1
            if consecutive >= patience:
                K_stop = K
                break
        else:
            consecutive = 0

    notes: dict = {}
    if K_stop is None:
        K_stop = K_hard_max
        notes["hit_K_hard_max"] = True

    log.info(
        "estimate_k_range: n=%d  K_hard_max=%d  R=%d  -> "
        "K_min=2  K_max=%d%s",
        n, K_hard_max, n_runs, K_stop,
        "  (HIT CEILING — consider increasing k_hard_max)"
        if notes else "",
    )

    return KRangeEstimate(
        K_min=2,
        K_max=int(K_stop),
        K_hard_max=int(K_hard_max),
        inertia_per_run=inertia,
        gain_per_run=gain,
        gain_summary=gain_summary,
        epsilon=float(epsilon),
        patience=int(patience),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Standalone CLI for sanity-checking on built-in datasets.
# ---------------------------------------------------------------------------
def _cli(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="estimate_k_range",
        description="Pick K_grid = [2, K_max] by k-means++ elbow detection.",
    )
    parser.add_argument(
        "dataset",
        choices=("digits", "iris", "wine", "olivetti"),
        nargs="?", default="digits",
    )
    parser.add_argument("--input", type=str, default=None,
                        help=".npz containing an 'X' key.")
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--k-hard-max", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=20)
    args = parser.parse_args(argv)

    if args.input:
        npz = np.load(args.input, allow_pickle=True)
        X = np.asarray(npz["X"], dtype=float) if "X" in npz.files \
            else np.asarray(npz[npz.files[0]], dtype=float)
        label = f"npz:{args.input}"
    else:
        from sklearn import datasets as _ds
        loaders = {
            "digits":   lambda: _ds.load_digits(return_X_y=True)[0],
            "iris":     lambda: _ds.load_iris(return_X_y=True)[0],
            "wine":     lambda: _ds.load_wine(return_X_y=True)[0],
            "olivetti": lambda: _ds.fetch_olivetti_faces().data,
        }
        X = loaders[args.dataset]()
        label = args.dataset

    log.info("dataset=%s  n=%d  p=%d", label, X.shape[0], X.shape[1])
    result = estimate_k_range(
        X,
        min_cluster_size=args.min_cluster_size,
        k_hard_max=args.k_hard_max,
        epsilon=args.epsilon,
        patience=args.patience,
        n_runs=args.n_runs,
    )
    log.info("K_grid = [%d, %d]   (K_hard_max=%d, epsilon=%.3f, patience=%d)",
             result.K_min, result.K_max, result.K_hard_max,
             result.epsilon, result.patience)
    # Tiny preview of the gain curve so the user can sanity-check.
    log.info("first 10 q75-gains: %s",
             ", ".join(f"{g:.3f}" for g in result.gain_summary[:10]))
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(_cli())
