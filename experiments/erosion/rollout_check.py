"""Trajectory check for the Seaglass surrogate, plus a "what if" plan demo.

1. Chained vs direct vs simulator. For every eval case (start state x storm
   future), compare yearly cumulative outcomes from
     - the simulator (truth, re-run with the stored seeds),
     - the surrogate chained in 1-year steps (``rollout``),
     - the surrogate jumping directly to 1, 2 and 5 years.
   Chained and direct 5-year outcomes should agree in distribution
   (self-consistency); both should match the simulator.

2. Asset-loss timing. An asset sits ``--setback`` metres behind the starting
   cliff-top edge. Compare P(lost by year y) from the surrogate with the
   simulator's frequency — the quantity a planner actually acts on.

3. What-if plans. For a few start states, compare plans under common random
   numbers (same storm futures, same surrogate noise):
     none            no intervention
     armour_now      revetment (crest 3 m) from year 0
     armour_year2    revetment from the start of year 3
   Simulator (solid) vs surrogate (dashed) P(asset lost by year).

Usage:
  python experiments/erosion/rollout_check.py --eval data/eval --surrogate runs/mac/last.pt --out outputs/rollout
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terrarium.core import metrics as M
from terrarium.core.torch_utils import pick_device
from terrarium.phenomena.erosion.cliff2d.dataset import EvalSet, simulate_plan
from terrarium.phenomena.erosion.cliff2d.summary import OUTCOMES, cliff_outcomes, stack_forcing, step_summaries
from terrarium.phenomena.erosion.cliff2d.surrogate import FlowSurrogate, rollout

YEAR = 360
DIRECT = (1, 2, 5)  # years with a trained direct-jump Δt (360, 720, 1800 days)


def cases(ev: EvalSet, max_starts: int):
    K = min(ev.K, max_starts) if max_starts else ev.K
    kk, mm = np.meshgrid(np.arange(K), np.arange(ev.M), indexing="ij")
    return kk.ravel(), mm.ravel()


def truth_trajectories(ev, k, m, years, plan=None, device=None):
    init = ev.init_for(k)
    forcing = ev.forcing_for(k, m)
    plan = np.repeat(ev.rev_crest[k][:, None], years, 1) if plan is None else plan
    rock, tal = simulate_plan(init, init["rock"], ev.start_talus[k], forcing, ev.seeds[k, m], [YEAR] * years, plan, device)
    out = cliff_outcomes(rock[:, :1], rock[:, 1:], ev.top_row[k][:, None])  # [N, years, 3]
    return out, rock, tal


def score_block(samples, truth, rng):
    """samples [N, n], truth [N] -> dict of scalar metrics."""
    p = M.pit(samples, truth, rng)
    return {
        "crps": float(M.crps_ensemble(samples, truth).mean()),
        "pit_ks": M.pit_ks(p),
        "cover90": float(M.coverage_from_pit(p, 0.9).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--out", default="outputs/rollout")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-starts", type=int, default=0)
    ap.add_argument("--setback", type=float, default=2.0, help="asset distance behind the starting cliff-top edge (m)")
    ap.add_argument("--demo-starts", type=int, default=4)
    ap.add_argument("--crest", type=float, default=3.0)
    ap.add_argument("--flow-steps", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sim-device", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Y = a.years
    ev = EvalSet(a.eval)
    model = FlowSurrogate.load(a.surrogate, pick_device(a.device))
    k, m = cases(ev, a.max_starts)
    N = len(k)
    daily = ev.forcing[k, m]
    amp = ev.tidal_amp[k]
    q = dict(rock0=ev.start_rock[k].astype(np.float32), strength=ev.strength[k].astype(np.float32), top_row=ev.top_row[k], talus0=ev.start_talus[k])
    lines = ["# Seaglass rollout check", "", f"{N} cases ({len(np.unique(k))} starts × {ev.M} futures), "
             f"{a.n_samples} surrogate samples per case, {Y} yearly steps, asset setback {a.setback} m.", ""]

    # ---- 1. truth, chained, direct ----
    t = time.time()
    truth, _, _ = truth_trajectories(ev, k, m, Y, device=a.sim_device)
    t_sim = time.time() - t
    # the re-run must reproduce the stored eval outcomes exactly where they overlap
    for y in DIRECT:
        if y <= Y and y * YEAR in ev.dts:
            err = np.abs(truth[:, y - 1] - ev.outcomes[k, m, ev.dts.index(y * YEAR)]).max()
            assert err < 1e-3, f"simulator re-run does not reproduce eval truth at year {y} (max err {err})"
    plan = np.repeat(ev.rev_crest[k][:, None], Y, 1)
    t = time.time()
    ch = rollout(model, **q, step_days=[YEAR] * Y, summaries=step_summaries(daily, amp, [YEAR] * Y), rev_plan=plan,
                 n_samples=a.n_samples, seed=1, flow_steps=a.flow_steps, keep_fields=False)["outcomes"]  # [N, n, Y, 3]
    t_ch = time.time() - t
    direct = {}
    for y in DIRECT:
        if y <= Y:
            d = y * YEAR
            direct[y] = rollout(model, **q, step_days=[d], summaries=step_summaries(daily, amp, [d]), rev_plan=plan[:, :1],
                                n_samples=a.n_samples, seed=2, flow_steps=a.flow_steps, keep_fields=False)["outcomes"][:, :, 0]

    lines += ["## Yearly cumulative outcomes vs simulator", "",
              f"Cost for all cases: simulator {t_sim:.1f}s (1 future each), chained surrogate {t_ch:.1f}s ({a.n_samples} futures each).", ""]
    for oi, o in enumerate(OUTCOMES):
        if o == "toe_retreat":
            continue
        lines += [f"### {o}", "", "| year | mode | CRPS | PIT KS | cover90 |", "|---|---|---|---|---|"]
        for y in range(1, Y + 1):
            r = score_block(ch[:, :, y - 1, oi], truth[:, y - 1, oi], rng)
            lines.append(f"| {y} | chained | {r['crps']:.3f} | {r['pit_ks']:.2f} | {r['cover90']:.2f} |")
            if y in direct:
                r = score_block(direct[y][:, :, oi], truth[:, y - 1, oi], rng)
                lines.append(f"| {y} | direct | {r['crps']:.3f} | {r['pit_ks']:.2f} | {r['cover90']:.2f} |")
        lines.append("")
    if Y in direct:
        oi = OUTCOMES.index("volume_lost")
        w = np.mean([M.wasserstein1(ch[k == kk, :, Y - 1, oi].ravel(), direct[Y][k == kk, :, oi].ravel()) for kk in np.unique(k)])
        lines += [f"**Self-consistency:** W1 between chained and direct {Y}-year volume lost, per start: {w:.3f} m³/m.", ""]

    # ---- 2. asset-loss timing ----
    ti = OUTCOMES.index("top_retreat")
    lost_true = truth[:, :, ti] >= a.setback  # [N, Y]
    p_model = (ch[:, :, :, ti] >= a.setback).mean(1)  # [N, Y]
    lines += ["## Asset-loss timing", "", "| year | simulator P(lost) | surrogate P(lost) | Brier |", "|---|---|---|---|"]
    for y in range(Y):
        lines.append(f"| {y + 1} | {lost_true[:, y].mean():.2f} | {p_model[:, y].mean():.2f} | {M.brier(p_model[:, y], lost_true[:, y]).mean():.3f} |")
    lines.append("")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    yrs = np.arange(1, Y + 1)
    for oi, o in enumerate(("top_retreat", "volume_lost")):
        j = OUTCOMES.index(o)
        for name, arr, style, col in (("simulator", truth[:, :, j], "-", "k"), ("chained", ch[:, :, :, j].reshape(-1, Y), "--", "C0")):
            ax[oi].plot(yrs, np.median(arr, 0), style, color=col, label=f"{name} median")
            ax[oi].fill_between(yrs, *np.percentile(arr, [5, 95], 0), color=col, alpha=0.12)
        for y, dsm in direct.items():
            ax[oi].errorbar(y, np.median(dsm[:, :, j]), yerr=[[np.median(dsm[:, :, j]) - np.percentile(dsm[:, :, j], 5)], [np.percentile(dsm[:, :, j], 95) - np.median(dsm[:, :, j])]], fmt="s", color="C3", capsize=3, label="direct" if y == min(direct) else None)
        ax[oi].set_title(f"{o}: median and 5–95% band")
        ax[oi].set_xlabel("year")
        ax[oi].legend(fontsize=8)
    ax[2].plot(yrs, lost_true.mean(0), "k-", label="simulator")
    ax[2].plot(yrs, p_model.mean(0), "C0--", label="surrogate (chained)")
    ax[2].set_title(f"P(asset {a.setback} m back is lost by year)")
    ax[2].set_ylim(0, 1)
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=120)
    plt.close(fig)

    # ---- 3. what-if plans (common random numbers) ----
    starts = np.unique(k)[: a.demo_starts]
    sel = np.isin(k, starts)
    kd, md = k[sel], m[sel]
    nan = np.full((len(kd), Y), np.nan, np.float32)
    plans = {
        "none": nan,
        "armour_now": np.full((len(kd), Y), a.crest, np.float32),
        "armour_year2": np.broadcast_to(np.where(np.arange(Y) >= 2, a.crest, np.nan), (len(kd), Y)).astype(np.float32),
    }
    qd = {kk: v[sel] for kk, v in q.items()}
    sums = step_summaries(daily[sel], amp[sel], [YEAR] * Y)
    fig, ax = plt.subplots(1, len(starts), figsize=(4 * len(starts), 3.6), sharey=True)
    lines += ["## What-if plans (common random numbers)", "", "P(asset lost by final year)", "",
              "| start | plan | simulator | surrogate |", "|---|---|---|---|"]
    for name, pl in plans.items():
        tr, _, _ = truth_trajectories(ev, kd, md, Y, plan=pl, device=a.sim_device)
        sm = rollout(model, **qd, step_days=[YEAR] * Y, summaries=sums, rev_plan=pl, n_samples=a.n_samples, seed=3,
                     flow_steps=a.flow_steps, keep_fields=False)["outcomes"]
        for i, s0 in enumerate(starts):
            c = kd == s0
            pt = (tr[c, :, ti] >= a.setback).mean(0)
            ps = (sm[c][:, :, :, ti] >= a.setback).mean((0, 1))
            line, = np.atleast_1d(ax)[i].plot(yrs, pt, "-", label=f"{name}")
            np.atleast_1d(ax)[i].plot(yrs, ps, "--", color=line.get_color())
            lines.append(f"| {s0} | {name} | {pt[-1]:.2f} | {ps[-1]:.2f} |")
    for i, s0 in enumerate(starts):
        a_ = np.atleast_1d(ax)[i]
        a_.set_title(f"start {s0}: P(asset lost)")
        a_.set_xlabel("year")
        a_.set_ylim(-0.02, 1.02)
    np.atleast_1d(ax)[0].legend(fontsize=8, title="solid = sim, dashed = surrogate")
    fig.tight_layout()
    fig.savefig(out / "what_if.png", dpi=120)
    plt.close(fig)

    (out / "rollout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
