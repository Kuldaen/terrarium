"""Stochastic forcing generator: daily wave height, surge, rain, sea level.

Forcing is generated up front as explicit arrays so that the same storm
history can be replayed under different actions (common random numbers).
"""

from __future__ import annotations

import numpy as np

from .params import Climate


def generate_forcing(
    batch: int,
    days: int,
    rng: np.random.Generator,
    climate: Climate = Climate(),
    start_day: int = 0,
) -> dict[str, np.ndarray]:
    """Return dict of float32 arrays, each [B, T]:

    hs     significant wave height offshore (m)
    surge  storm surge (m)
    rain   normalised rain index (≈0..3)
    wind   onshore wind component (m/s, negative = offshore)
    slr    sea-level offset relative to present MSL (m)
    and per-cliff [B] array ``tidal_amp``.
    """
    t = np.arange(start_day, start_day + days)
    season = np.cos(2 * np.pi * (t - 15) / 365.25)  # +1 mid-January

    # background log-Hs with AR(1) noise
    eps = rng.normal(0.0, climate.hs_noise, size=(batch, days))
    noise = np.empty_like(eps)
    noise[:, 0] = eps[:, 0] / np.sqrt(1 - climate.hs_ar**2)
    for k in range(1, days):
        noise[:, k] = climate.hs_ar * noise[:, k - 1] + eps[:, k]
    hs = climate.hs_mean * (1 + climate.hs_seasonal * season)[None, :] * np.exp(noise)

    surge = np.zeros((batch, days))
    storm = np.zeros((batch, days))
    # storms: seasonally modulated Poisson arrivals
    rate = climate.storms_per_year / 365.25 * (1 + 0.8 * season)
    arrivals = rng.random((batch, days)) < rate[None, :]
    for b, k in zip(*np.nonzero(arrivals)):
        peak = climate.storm_peak_min + rng.exponential(climate.storm_peak_scale)
        dur = rng.integers(climate.storm_duration_range[0], climate.storm_duration_range[1] + 1)
        for d in range(dur):
            if k + d >= days:
                break
            shape = 1.0 - 0.5 * abs(d - (dur - 1) / 2) / max(dur / 2, 1)
            h = peak * shape
            hs[b, k + d] = max(hs[b, k + d], h)
            surge[b, k + d] = max(surge[b, k + d], climate.surge_per_hs * h)
            storm[b, k + d] = 1.0

    rain = rng.gamma(0.6, climate.rain_mean / 0.6, size=(batch, days))
    rain *= 1 + 0.5 * season[None, :]
    rain += storm * rng.gamma(2.0, climate.rain_storm_boost / 2.0, size=(batch, days))
    rain = np.clip(rain, 0.0, 3.0)

    # wind speed correlated with waves; daily direction gives the onshore
    # component (signed: negative = offshore). Storm days skew onshore.
    speed = climate.wind_mean * (hs / climate.hs_mean) ** climate.wind_hs_exponent
    speed *= np.exp(rng.normal(0.0, climate.wind_noise, size=(batch, days)))
    theta = rng.uniform(-np.pi, np.pi, size=(batch, days))
    onshore = np.cos(theta)
    onshore = np.where(storm > 0, climate.storm_onshore_bias + (1 - climate.storm_onshore_bias) * onshore, onshore)
    wind = np.clip(speed, 0.0, 35.0) * onshore

    slr = (climate.slr_mm_per_year / 1000.0 / 365.25) * t[None, :].repeat(batch, 0)
    tidal_amp = rng.uniform(*climate.tidal_amp_range, size=batch)

    return {
        "hs": hs.astype(np.float32),
        "surge": surge.astype(np.float32),
        "rain": rain.astype(np.float32),
        "wind": wind.astype(np.float32),
        "slr": slr.astype(np.float32),
        "tidal_amp": tidal_amp.astype(np.float32),
    }
