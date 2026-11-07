"""Interval forcing summaries and scalar outcomes derived from rock fields.

Forcing summaries are what a large-Δt surrogate is conditioned on, so they
keep threshold/extreme information (max storm, exceedance fractions) rather
than only means. Outcomes are computed from rock fields with the same function
for simulator truth and surrogate samples, so they are scored identically.
"""

from __future__ import annotations

import numpy as np

from .params import Grid

FORCING_CHANNELS = ("hs", "surge", "rain", "wind", "slr")

SUMMARY_FEATURES = (
    "dt_years",
    "hs_max",
    "hs_p95",
    "frac_hs_gt2",
    "frac_hs_gt3",
    "mean_h2",
    "surge_max",
    "rain_mean",
    "rain_max",
    "wind_on2_mean",
    "slr_start",
    "slr_delta",
    "tidal_amp",
)

OUTCOMES = ("top_retreat", "toe_retreat", "volume_lost")

ROCK_THRESHOLD = 0.5


def stack_forcing(forcing: dict[str, np.ndarray]) -> np.ndarray:
    """dict of [B, T] arrays -> [B, T, C] in FORCING_CHANNELS order."""
    f = dict(forcing)
    f.setdefault("wind", np.zeros_like(f["hs"]))
    return np.stack([f[k] for k in FORCING_CHANNELS], axis=-1).astype(np.float32)


def unstack_forcing(arr: np.ndarray, tidal_amp: np.ndarray) -> dict[str, np.ndarray]:
    out = {k: np.ascontiguousarray(arr[..., i]) for i, k in enumerate(FORCING_CHANNELS)}
    out["tidal_amp"] = np.asarray(tidal_amp, np.float32)
    return out


def summarize_forcing(daily: np.ndarray, tidal_amp: np.ndarray) -> np.ndarray:
    """daily: [..., T, C] forcing over the interval; tidal_amp: [...].

    Returns [..., len(SUMMARY_FEATURES)] float32.
    """
    hs, surge, rain, wind, slr = (daily[..., i] for i in range(len(FORCING_CHANNELS)))
    T = daily.shape[-2]
    won = np.maximum(wind, 0.0)
    feats = [
        np.full(hs.shape[:-1], T / 365.25),
        hs.max(-1),
        np.percentile(hs, 95, axis=-1),
        (hs > 2.0).mean(-1),
        (hs > 3.0).mean(-1),
        (hs**2).mean(-1),
        surge.max(-1),
        rain.mean(-1),
        rain.max(-1),
        (won**2).mean(-1) / 100.0,
        slr[..., 0],
        slr[..., -1] - slr[..., 0],
        np.broadcast_to(tidal_amp, hs.shape[:-1]),
    ]
    return np.stack(feats, -1).astype(np.float32)


def step_summaries(daily: np.ndarray, tidal_amp: np.ndarray, step_days) -> np.ndarray:
    """Forcing summaries for consecutive intervals.

    daily: [N, T, C] with T >= sum(step_days); returns [N, S, F].
    """
    out, t = [], 0
    for d in step_days:
        out.append(summarize_forcing(daily[:, t : t + d], tidal_amp))
        t += d
    return np.stack(out, 1)


def _first_rock_col(rows: np.ndarray, nx: int) -> np.ndarray:
    """rows: [..., nx] bool -> index of first True, nx if none."""
    any_ = rows.any(-1)
    return np.where(any_, rows.argmax(-1), nx)


def cliff_outcomes(
    rock0: np.ndarray,
    rock1: np.ndarray,
    top_row: np.ndarray,
    grid: Grid = Grid(),
    toe_band: tuple[float, float] = (0.0, 4.0),
) -> np.ndarray:
    """Scalar outcomes between two rock fields.

    rock0, rock1: [..., NZ, NX]; top_row: broadcastable to [...].
    Returns [..., 3] in OUTCOMES order:
      top_retreat  landward movement of the cliff-top edge (m)
      toe_retreat  mean landward face movement over z in toe_band (m)
      volume_lost  rock area removed per metre of coast (m^3/m)
    """
    r0 = np.asarray(rock0, np.float32)
    r1 = np.asarray(rock1, np.float32)
    nz, nx = grid.nz, grid.nx
    lead = r0.shape[:-2]
    tr = np.broadcast_to(np.asarray(top_row), lead).astype(np.int64)
    idx = tr[..., None, None]
    top0 = np.take_along_axis(r0, np.broadcast_to(idx, lead + (1, nx)), axis=-2)[..., 0, :]
    top1 = np.take_along_axis(r1, np.broadcast_to(idx, lead + (1, nx)), axis=-2)[..., 0, :]
    top_ret = (_first_rock_col(top1 > ROCK_THRESHOLD, nx) - _first_rock_col(top0 > ROCK_THRESHOLD, nx)) * grid.dx

    z = grid.z0 + (np.arange(nz) + 0.5) * grid.dz
    band = (z >= toe_band[0]) & (z < toe_band[1])
    f0 = _first_rock_col(r0[..., band, :] > ROCK_THRESHOLD, nx)
    f1 = _first_rock_col(r1[..., band, :] > ROCK_THRESHOLD, nx)
    toe_ret = (f1 - f0).mean(-1) * grid.dx

    vol = np.clip(r0 - r1, 0.0, None).sum((-1, -2)) * grid.dx * grid.dz
    return np.stack([top_ret, toe_ret, vol], -1).astype(np.float32)


STATE_FEATURES = ("top_x", "toe_x", "overhang", "talus")


def state_features(rock: np.ndarray, top_row: np.ndarray, talus: np.ndarray, grid: Grid = Grid(), toe_band: tuple[float, float] = (0.0, 4.0)) -> np.ndarray:
    """Coarse geometry descriptors of a start state: [..., len(STATE_FEATURES)].

    top_x     cliff-top edge position (m)
    toe_x     mean face position over the toe band (m)
    overhang  toe_x - top_x: positive when the toe is cut back beneath the top
    talus     talus volume at the toe (m^3/m)
    """
    r = np.asarray(rock, np.float32) > ROCK_THRESHOLD
    nz, nx = grid.nz, grid.nx
    lead = r.shape[:-2]
    tr = np.broadcast_to(np.asarray(top_row), lead).astype(np.int64)[..., None, None]
    top = np.take_along_axis(r, np.broadcast_to(tr, lead + (1, nx)), axis=-2)[..., 0, :]
    top_x = _first_rock_col(top, nx) * grid.dx
    z = grid.z0 + (np.arange(nz) + 0.5) * grid.dz
    band = (z >= toe_band[0]) & (z < toe_band[1])
    toe_x = _first_rock_col(r[..., band, :], nx).mean(-1) * grid.dx
    return np.stack([top_x, toe_x, toe_x - top_x, np.broadcast_to(talus, lead)], -1).astype(np.float32)
