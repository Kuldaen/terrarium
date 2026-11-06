"""Sanity checks for the cliff2d simulator.

Runs a batch of random cliffs for N years and writes:
  profiles.png     cross-sections at yearly snapshots for a few cliffs
  retreat.png      cliff-top / notch position over time + retreat-rate histogram
  collapses.png    collapse volume size–frequency (log–log CCDF)
  ensemble.png     one cliff, many storm futures: spread of 5-year outcomes

Usage: python experiments/erosion/cliff2d_sanity.py --batch 64 --years 5 --out outputs/sanity
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terrarium.phenomena.erosion.cliff2d import Cliff2D, Grid, generate_cliffs, generate_forcing


def plot_profiles(rollout, grid: Grid, idx, path):
    snaps = rollout.snapshots["rock"].astype(np.float32)
    ext = [0, grid.nx * grid.dx, grid.z0, grid.z0 + grid.nz * grid.dz]
    fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4), sharey=True)
    for ax, b in zip(np.atleast_1d(axes), idx):
        S = snaps.shape[1]
        cols = plt.cm.viridis(np.linspace(0, 1, S))
        ax.imshow(snaps[b, -1] > 0.5, origin="lower", extent=ext, cmap="Greys", alpha=0.35, aspect="auto")
        for s in range(S):
            ax.contour(snaps[b, s], levels=[0.5], extent=ext, colors=[cols[s]], linewidths=1.2)
        ax.axhline(0, color="tab:blue", lw=0.8, ls="--")
        ax.set_xlim(8, 30)
        ax.set_title(f"cliff {b}")
        ax.set_xlabel("x (m, landward)")
    np.atleast_1d(axes)[0].set_ylabel("z (m rel. MSL)")
    fig.suptitle("Profiles at yearly snapshots (dark→light = time)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_retreat(rollout, years, path):
    S = rollout.series
    t = np.arange(S["top_x"].shape[1]) / 365.25
    top = S["top_x"] - S["top_x"][:, :1]
    notch = S["notch_x"] - S["notch_x"][:, :1]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for b in range(min(12, top.shape[0])):
        ax[0].plot(t, top[b], lw=1)
        ax[1].plot(t, notch[b], lw=1)
    ax[0].set_title("Cliff-top retreat (m)")
    ax[1].set_title("Notch-back retreat (m)")
    for a in ax[:2]:
        a.set_xlabel("years")
    ax[2].hist(top[:, -1] / years, bins=20, color="tab:orange")
    ax[2].set_title("Mean cliff-top retreat rate (m/yr)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_collapses(rollout, path):
    v = rollout.events["fail_vol"]
    v = np.sort(v[v > 0])[::-1]
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(v):
        ax.loglog(v, np.arange(1, len(v) + 1) / len(v), ".", ms=4)
    ax.set_xlabel("collapse volume (m³ per m of coast)")
    ax.set_ylabel("P(V ≥ v)")
    ax.set_title(f"Collapse size–frequency (n={len(v)})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return v


def ensemble(grid, years, n, seed, path):
    """Same cliff, n different storm futures."""
    rng = np.random.default_rng(seed)
    one = generate_cliffs(1, rng, grid)
    init = {k: np.repeat(v, n, axis=0) for k, v in one.items()}
    forcing = generate_forcing(n, int(years * 365.25), rng)
    forcing["tidal_amp"][:] = forcing["tidal_amp"][0]
    sim = Cliff2D(init, grid)
    r = sim.run(forcing, seeds=np.arange(n) + 1000, snapshot_every=int(years * 365.25))
    final = r.snapshots["rock"][:, -1].astype(np.float32)
    ext = [0, grid.nx * grid.dx, grid.z0, grid.z0 + grid.nz * grid.dz]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(one["rock"][0] > 0.5, origin="lower", extent=ext, cmap="Greys", alpha=0.3, aspect="auto")
    pr = np.ma.masked_where(final.mean(0) <= 0.0, final.mean(0))
    ax[0].imshow(pr, origin="lower", extent=ext, cmap="magma_r", alpha=0.7, aspect="auto", vmin=0, vmax=1)
    for b in range(min(n, 12)):
        ax[0].contour(final[b], levels=[0.5], extent=ext, colors="k", linewidths=0.5, alpha=0.5)
    ax[0].set_xlim(8, 30)
    ax[0].set_title(f"{n} futures of one cliff after {years} yr\n(shading = P(rock remains))")
    top = r.series["top_x"][:, -1] - r.series["top_x"][:, 0]
    ax[1].hist(top, bins=20, color="tab:red")
    ax[1].set_title(f"Cliff-top retreat after {years} yr across futures")
    ax[1].set_xlabel("m")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--ensemble", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/sanity")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid()
    rng = np.random.default_rng(a.seed)
    days = int(a.years * 365.25)
    init = generate_cliffs(a.batch, rng, grid)
    forcing = generate_forcing(a.batch, days, rng)
    sim = Cliff2D(init, grid, device=a.device)
    t = time.time()
    r = sim.run(forcing, seeds=np.arange(a.batch), snapshot_every=365)
    dt = time.time() - t
    print(f"simulated {a.batch} cliffs x {days} days in {dt:.1f}s on {sim.device} "
          f"({a.batch * days / dt:,.0f} cliff-days/s)")
    plot_profiles(r, grid, [0, 1, 2, 3], out / "profiles.png")
    plot_retreat(r, a.years, out / "retreat.png")
    v = plot_collapses(r, out / "collapses.png")
    top = r.series["top_x"][:, -1] - r.series["top_x"][:, 0]
    print(f"cliff-top retreat: mean {top.mean() / a.years:.2f} m/yr, "
          f"p10 {np.percentile(top, 10) / a.years:.2f}, p90 {np.percentile(top, 90) / a.years:.2f}")
    print(f"collapses: {len(v)} total, median {np.median(v) if len(v) else 0:.2f} m³/m, max {v.max() if len(v) else 0:.2f}")
    et = ensemble(grid, a.years, a.ensemble, a.seed + 1, out / "ensemble.png")
    print(f"ensemble top retreat: mean {et.mean():.2f} m, std {et.std():.2f} m, range [{et.min():.2f}, {et.max():.2f}]")


if __name__ == "__main__":
    main()
