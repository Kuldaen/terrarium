"""Probabilistic calibration metrics for ensemble forecasts (phenomenon-agnostic).

Conventions: ``samples`` has the ensemble on the last axis, ``[..., n]``;
``y`` (the realised outcome) has shape ``[...]``.
"""

from __future__ import annotations

import numpy as np


def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score, ensemble estimator (lower is better).

    CRPS = E|X - y| - 0.5 E|X - X'|. Uses the "fair" spread term
    (n / (n - 1) correction) so small ensembles are not rewarded for being
    under-dispersed.
    """
    x = np.sort(np.asarray(samples, np.float64), axis=-1)
    y = np.asarray(y, np.float64)[..., None]
    n = x.shape[-1]
    skill = np.abs(x - y).mean(-1)
    if n == 1:
        return skill
    i = np.arange(1, n + 1)
    # sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - n - 1) x_(i)
    spread = 2.0 * ((2 * i - n - 1) * x).sum(-1) / (n * (n - 1))
    return skill - 0.5 * spread


def pit(samples: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomised probability integral transform in [0, 1].

    Uniform when the forecast is calibrated. Ties (point forecasts, discrete
    outcomes) are broken uniformly so they are not mistaken for miscalibration.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(samples)
    y = np.asarray(y)[..., None]
    n = x.shape[-1]
    less = (x < y).sum(-1)
    eq = (x == y).sum(-1)
    u = rng.random(less.shape)
    return (less + u * (eq + 1)) / (n + 1)


def pit_ks(p: np.ndarray) -> float:
    """Kolmogorov–Smirnov distance of PIT values from Uniform(0, 1)."""
    p = np.sort(np.ravel(p))
    n = len(p)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max((i / n - p).max(), (p - (i - 1) / n).max()))


def coverage(samples: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    """Whether y falls in the central ``level`` interval of the ensemble (bool)."""
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=-1)
    y = np.asarray(y)
    return (y >= lo) & (y <= hi)


def coverage_from_pit(p: np.ndarray, level: float) -> np.ndarray:
    """Tie-aware central-interval coverage from randomised PIT values (bool).

    Equivalent to ``coverage`` for continuous forecasts, but not inflated when
    outcomes are discrete (e.g. retreat quantised by joint spacing).
    """
    return (p >= (1 - level) / 2) & (p <= (1 + level) / 2)


def wasserstein1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1-D Wasserstein distance between two sample sets along the last axis."""
    a = np.sort(a, -1)
    b = np.sort(b, -1)
    q = np.linspace(0, 1, max(a.shape[-1], b.shape[-1]) * 4 + 1)[1:-1]
    return np.abs(np.quantile(a, q, axis=-1) - np.quantile(b, q, axis=-1)).mean(0)


def brier(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Brier score of probabilities p for binary outcomes o."""
    return (np.asarray(p, np.float64) - np.asarray(o, np.float64)) ** 2
