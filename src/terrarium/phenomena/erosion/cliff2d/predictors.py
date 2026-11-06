"""Predictors scored by the calibration report, plus reference baselines.

A predictor answers: given a start state, an action, an interval Δt and that
interval's forcing summary, what is the distribution of outcomes? It returns
``n_samples`` draws per query. A learned surrogate implements the same
interface (see ``Predictor``).

Baselines
---------
oracle         the simulator itself, re-run with fresh failure seeds under the
               realised daily forcing. Calibrated by construction for the
               intrinsic (collapse) randomness; validates the metric code and
               gives the cost a surrogate must beat. Uses privileged input
               (full daily forcing), so it is a reference, not a competitor.
climatology    ignores state and forcing; samples outcomes seen in training at
               the same Δt. Calibrated on average, not sharp. CRPS skill is
               reported relative to this.
linear         deterministic per-Δt least squares on forcing summary + state
               features. Sharp, typically badly calibrated: a point forecast
               of a lumpy, multi-modal outcome.
linear_gauss   the same mean plus Gaussian residual noise. Better calibrated
               marginally, but cannot represent discrete, joint-quantised
               collapse outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .dataset import DT_DAYS, EvalSet, PairDataset, branch_rollout
from .summary import cliff_outcomes, state_features


@dataclass
class Query:
    """Batch of evaluation queries (N cases; arrays over Δt are [N, nDT, ...])."""

    k: np.ndarray  # [N] start-state index
    m: np.ndarray  # [N] future index
    dts: tuple[int, ...]
    rock0: np.ndarray  # [N, NZ, NX]
    talus0: np.ndarray  # [N]
    strength: np.ndarray  # [N, NZ, NX]
    top_row: np.ndarray  # [N]
    rev_crest: np.ndarray  # [N] (NaN = none)
    summary: np.ndarray  # [N, nDT, F]
    ev: EvalSet  # for privileged access (oracle only)

    @classmethod
    def from_eval(cls, ev: EvalSet, k: np.ndarray, m: np.ndarray) -> "Query":
        return cls(
            k=k, m=m, dts=ev.dts,
            rock0=ev.start_rock[k].astype(np.float32),
            talus0=ev.start_talus[k],
            strength=ev.strength[k].astype(np.float32),
            top_row=ev.top_row[k],
            rev_crest=ev.rev_crest[k],
            summary=ev.summary[k, m],
            ev=ev,
        )


class Predictor(Protocol):
    name: str

    def predict(self, q: Query, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return outcome samples [N, nDT, n_samples, n_outcomes]."""
        ...


def _design(summary: np.ndarray, sfeat: np.ndarray, rev_crest: np.ndarray) -> np.ndarray:
    has_rev = np.isfinite(rev_crest).astype(np.float32)
    crest = np.nan_to_num(rev_crest, nan=0.0)
    cols = [summary[:, 1:], sfeat, has_rev[:, None], crest[:, None], np.ones((len(summary), 1), np.float32)]
    return np.concatenate(cols, 1).astype(np.float64)


class OraclePredictor:
    name = "oracle"

    def __init__(self, device: str | None = None):
        self.device = device

    def predict(self, q: Query, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        ev, N, n = q.ev, len(q.k), n_samples
        rep = lambda a: np.repeat(a, n, axis=0)
        init = {k: rep(v) for k, v in ev.init_for(q.k).items()}
        forcing = {k: rep(v) for k, v in ev.forcing_for(q.k, q.m).items()}
        seeds = rng.integers(0, 2**31 - 1, size=N * n)
        rocks, _ = branch_rollout(init, rep(q.rock0), rep(q.talus0), rep(q.rev_crest), forcing, seeds, q.dts, self.device)
        out = cliff_outcomes(rep(q.rock0)[:, None], rocks, rep(q.top_row)[:, None])  # [N*n, nDT, 3]
        return out.reshape(N, n, len(q.dts), -1).transpose(0, 2, 1, 3)


class _Fitted:
    """Shared fitting on training pairs, one model per Δt."""

    def __init__(self, train: PairDataset, n_fit: int = 4000, seed: int = 0, dts: tuple[int, ...] = DT_DAYS):
        rng = np.random.default_rng(seed)
        b = train.sample(n_fit, rng)
        self.dts = dts
        self.pool, self.coef, self.resid_std = {}, {}, {}
        sf = state_features(b["rock0"], b["top_row"], b["talus0"])
        X = _design(b["summary"], sf, b["rev_crest"])
        for d in dts:
            sel = b["dt_days"] == d
            y = b["outcomes"][sel].astype(np.float64)
            self.pool[d] = y
            Xd = X[sel]
            coef, *_ = np.linalg.lstsq(Xd, y, rcond=1e-6)
            self.coef[d] = coef
            self.resid_std[d] = (y - Xd @ coef).std(0)

    def mean(self, q: Query, di: int) -> np.ndarray:
        sf = state_features(q.rock0, q.top_row, q.talus0)
        X = _design(q.summary[:, di], sf, q.rev_crest)
        return X @ self.coef[q.dts[di]]


class ClimatologyPredictor:
    name = "climatology"

    def __init__(self, fitted: _Fitted):
        self.f = fitted

    def predict(self, q: Query, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        out = []
        for d in q.dts:
            pool = self.f.pool[d]
            out.append(pool[rng.integers(0, len(pool), size=(len(q.k), n_samples))])
        return np.stack(out, 1).astype(np.float32)


class LinearPredictor:
    name = "linear"

    def __init__(self, fitted: _Fitted):
        self.f = fitted

    def predict(self, q: Query, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        mu = np.stack([self.f.mean(q, i) for i in range(len(q.dts))], 1)  # [N, nDT, 3]
        return np.repeat(mu[:, :, None], n_samples, 2).astype(np.float32)


class LinearGaussPredictor:
    name = "linear_gauss"

    def __init__(self, fitted: _Fitted):
        self.f = fitted

    def predict(self, q: Query, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        out = []
        for i, d in enumerate(q.dts):
            mu = self.f.mean(q, i)[:, None]
            eps = rng.normal(size=(len(q.k), n_samples, mu.shape[-1]))
            out.append(mu + eps * self.f.resid_std[d])
        return np.stack(out, 1).astype(np.float32)


def baselines(train: PairDataset, device: str | None = None, n_fit: int = 4000) -> list[Predictor]:
    f = _Fitted(train, n_fit=n_fit)
    return [OraclePredictor(device), ClimatologyPredictor(f), LinearPredictor(f), LinearGaussPredictor(f)]
