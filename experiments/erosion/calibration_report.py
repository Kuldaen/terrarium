"""Calibration report: how well does each predictor's *distribution* of outcomes
match the simulator's, as a function of Δt?

For every evaluation case (start state k, storm future m, interval Δt) the
predictor is given the start state, action and that future's forcing summary,
and draws n samples. Scored against the realised simulator outcome:

  CRPS / CRPSS     accuracy of the whole predictive distribution (skill vs climatology)
  PIT KS           distance of PIT histogram from uniform (0 = calibrated)
  cover50/cover90  hit rate of central intervals (should be 0.50 / 0.90)
  W1 (marginal)    per start state: predicted outcomes pooled over futures vs
                   the simulator's spread over futures (Wasserstein-1)
  s/sample         wall time per predictive sample

Usage:
  python experiments/erosion/calibration_report.py --train data/train --eval data/eval --out outputs/calib
  (generate data first with experiments/erosion/make_dataset.py)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terrarium.core import metrics as M
from terrarium.phenomena.erosion.cliff2d.dataset import EvalSet, PairDataset
from terrarium.phenomena.erosion.cliff2d.predictors import Query, baselines
from terrarium.phenomena.erosion.cliff2d.summary import OUTCOMES

UNITS = {"top_retreat": "m", "toe_retreat": "m", "volume_lost": "m³/m"}


def score(samples: np.ndarray, truth: np.ndarray, rng) -> dict:
    """samples [N, nDT, n, O], truth [N, nDT, O] -> metrics [nDT, O] per key."""
    s = np.moveaxis(samples, 2, -1)  # [N, nDT, O, n]
    crps = M.crps_ensemble(s, truth).mean(0)
    p = M.pit(s, truth, rng)  # [N, nDT, O]
    ks = np.array([[M.pit_ks(p[:, d, o]) for o in range(p.shape[2])] for d in range(p.shape[1])])
    c50 = M.coverage_from_pit(p, 0.5).mean(0)
    c90 = M.coverage_from_pit(p, 0.9).mean(0)
    return {"crps": crps, "pit_ks": ks, "cover50": c50, "cover90": c90, "pit": p}


def marginal_w1(samples: np.ndarray, truth: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Per start state, pooled predicted outcomes vs simulator spread over futures."""
    out = []
    for kk in np.unique(k):
        sel = k == kk
        pred = np.moveaxis(samples[sel], (0, 2), (-2, -1))  # [nDT, O, Nk, n]
        pred = pred.reshape(pred.shape[:2] + (-1,))
        tru = np.moveaxis(truth[sel], 0, -1)  # [nDT, O, Nk]
        out.append(M.wasserstein1(pred, tru))
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--out", default="outputs/calib")
    ap.add_argument("--n-samples", type=int, default=32, help=">= 19 so a 90%% interval is resolvable")
    ap.add_argument("--oracle-samples", type=int, default=8, help="the oracle is expensive; it is exchangeable with truth, so small n stays unbiased")
    ap.add_argument("--max-cases", type=int, default=0, help="limit number of (start, future) cases; 0 = all")
    ap.add_argument("--skip", nargs="*", default=[], help="predictor names to skip (e.g. oracle)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--surrogate", nargs="*", default=[], help="surrogate checkpoint(s) to score")
    ap.add_argument("--surrogate-steps", type=int, default=20, help="flow-matching sampling steps")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    ev = EvalSet(a.eval)
    train = PairDataset(a.train, dts=ev.dts)
    kk, mm = np.meshgrid(np.arange(ev.K), np.arange(ev.M), indexing="ij")
    k, m = kk.ravel(), mm.ravel()
    if a.max_cases:
        k, m = k[: a.max_cases], m[: a.max_cases]
    q = Query.from_eval(ev, k, m)
    truth = ev.outcomes[k, m]  # [N, nDT, O]
    print(f"{len(k)} cases ({ev.K} starts x {ev.M} futures), Δt = {ev.dts} days")

    preds = [p for p in baselines(train, a.device) if p.name not in a.skip]
    if a.surrogate:
        from terrarium.phenomena.erosion.cliff2d.surrogate import SurrogatePredictor

        for path in a.surrogate:
            sp = SurrogatePredictor(path, steps=a.surrogate_steps)
            if any(p.name == sp.name for p in preds):
                sp.name = f"{sp.name}:{Path(path).parent.name}"
            preds.append(sp)
    results, samples_by = {}, {}
    for p in preds:
        n = a.oracle_samples if p.name == "oracle" else a.n_samples
        t = time.time()
        s = p.predict(q, n, rng)
        dt = time.time() - t
        r = score(s, truth, rng)
        r["w1"] = marginal_w1(s, truth, k)
        r["sec_per_sample"] = dt / (len(k) * n)
        r["n_samples"] = n
        results[p.name] = r
        samples_by[p.name] = s
        print(f"  {p.name:<13} {dt:7.1f}s")

    clim = results["climatology"]["crps"]
    for r in results.values():
        r["crpss"] = 1.0 - r["crps"] / np.maximum(clim, 1e-9)

    # ---- markdown report ----
    lines = ["# Seaglass calibration report", "",
             f"{len(k)} cases = {ev.K} start states × {ev.M} storm futures; "
             f"{a.n_samples} samples per case ({a.oracle_samples} for the oracle); Δt ∈ {list(ev.dts)} days.", "",
             "Calibrated: PIT KS → 0, cover50 ≈ 0.50, cover90 ≈ 0.90. CRPSS > 0 beats climatology.", ""]
    for oi, o in enumerate(OUTCOMES):
        lines += [f"## {o} ({UNITS[o]})", "",
                  "| predictor | Δt (d) | CRPS | CRPSS | PIT KS | cover50 | cover90 | W1 marginal |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            for di, d in enumerate(ev.dts):
                lines.append(f"| {name} | {d} | {r['crps'][di, oi]:.3f} | {r['crpss'][di, oi]:+.2f} | "
                             f"{r['pit_ks'][di, oi]:.2f} | {r['cover50'][di, oi]:.2f} | {r['cover90'][di, oi]:.2f} | {r['w1'][di, oi]:.3f} |")
        lines.append("")
    lines += ["## Cost", "", "| predictor | seconds per sample (all Δt) |", "|---|---|"]
    lines += [f"| {n} | {r['sec_per_sample']:.2e} |" for n, r in results.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    json.dump({n: {kk_: (v.tolist() if isinstance(v, np.ndarray) and kk_ != "pit" else v)
                   for kk_, v in r.items() if kk_ != "pit"} for n, r in results.items()},
              open(out / "metrics.json", "w"), indent=1)

    # ---- plots ----
    names = list(results)
    fig, ax = plt.subplots(2, len(OUTCOMES), figsize=(5 * len(OUTCOMES), 7), sharex=True)
    yrs = np.array(ev.dts) / 365.25
    for oi, o in enumerate(OUTCOMES):
        for n in names:
            ax[0, oi].plot(yrs, results[n]["crps"][:, oi], "o-", label=n)
            ax[1, oi].plot(yrs, results[n]["cover90"][:, oi], "o-", label=n)
        ax[0, oi].set_title(f"{o}: CRPS ({UNITS[o]}) ↓")
        ax[1, oi].axhline(0.9, color="k", ls="--", lw=0.8)
        ax[1, oi].set_title(f"{o}: 90% interval coverage (target 0.9)")
        ax[1, oi].set_xlabel("Δt (years)")
        ax[1, oi].set_ylim(0, 1.02)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "skill_vs_dt.png", dpi=120)
    plt.close(fig)

    oi = OUTCOMES.index("volume_lost")
    fig, ax = plt.subplots(len(names), len(ev.dts), figsize=(2.6 * len(ev.dts), 2.0 * len(names)), sharex=True)
    for i, n in enumerate(names):
        for di, d in enumerate(ev.dts):
            ax[i, di].hist(results[n]["pit"][:, di, oi], bins=10, range=(0, 1), color="tab:blue")
            ax[i, di].axhline(len(k) / 10, color="k", ls="--", lw=0.8)
            ax[i, di].set_yticks([])
            if i == 0:
                ax[i, di].set_title(f"Δt = {d} d")
            if di == 0:
                ax[i, di].set_ylabel(n)
    fig.suptitle("PIT histograms, volume lost (flat = calibrated; U = overconfident; ∩ = underconfident)")
    fig.tight_layout()
    fig.savefig(out / "pit_volume.png", dpi=120)
    plt.close(fig)

    # one start state at the longest Δt: simulator spread vs predictors
    k0 = int(k[0])
    sel = k == k0
    di = len(ev.dts) - 1
    fig, ax = plt.subplots(1, len(OUTCOMES), figsize=(5 * len(OUTCOMES), 3.6))
    for oi, o in enumerate(OUTCOMES):
        tru = truth[sel, di, oi]
        bins = np.linspace(min(tru.min(), min(samples_by[n][sel, di, :, oi].min() for n in names)),
                           max(tru.max(), max(samples_by[n][sel, di, :, oi].max() for n in names)) + 1e-6, 25)
        ax[oi].hist(tru, bins=bins, density=True, color="k", alpha=0.35, label="simulator futures")
        for n in names:
            ax[oi].hist(samples_by[n][sel, di, :, oi].ravel(), bins=bins, density=True, histtype="step", lw=1.5, label=n)
        ax[oi].set_title(f"start {k0}: {o} after {ev.dts[di]} d")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "example_start.png", dpi=120)
    plt.close(fig)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
