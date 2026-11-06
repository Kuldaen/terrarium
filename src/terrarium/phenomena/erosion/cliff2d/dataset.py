"""Dataset generation for Δt-conditioned surrogates.

Two products:

Training chains (``generate_training``)
    Each chain is one cliff run under one storm history with monthly
    snapshots. Any pair of snapshots (s, s + Δt) is a training example, so one
    run yields many (state, forcing-summary, action, Δt) -> next-state pairs.
    Stored as shard directories of .npy files (memory-mappable).

Evaluation ensembles (``generate_eval``)
    A set of start states, each continued under many independent storm
    futures. This gives a real distribution of outcomes per start state and
    Δt to score a surrogate's calibration against.

Δt values are multiples of the 30-day snapshot spacing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .forcing import generate_forcing
from .geology import generate_cliffs
from .params import Climate, Grid, Physics
from .sim import Cliff2D
from .summary import FORCING_CHANNELS, OUTCOMES, SUMMARY_FEATURES, cliff_outcomes, stack_forcing, summarize_forcing, unstack_forcing

SNAP_DAYS = 30
DT_DAYS = (30, 90, 360, 720, 1800)
STATIC_KEYS = ("strength", "joints", "top_row")


def sample_revetment(n: int, rng: np.random.Generator, prob: float) -> np.ndarray:
    """Revetment crest per cliff: NaN (none) with prob 1 - prob, else U(1, 4) m."""
    crest = rng.uniform(1.0, 4.0, size=n).astype(np.float32)
    crest[rng.random(n) >= prob] = np.nan
    return crest


def _save(dir_: Path, arrays: dict[str, np.ndarray]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(dir_ / f"{k}.npy", v)


def _load(dir_: Path, mmap: bool = True) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p, mmap_mode="r" if mmap else None) for p in sorted(dir_.glob("*.npy"))}


def _manifest(kind: str, **extra) -> dict:
    return {
        "kind": kind,
        "grid": asdict(Grid()),
        "physics": asdict(Physics()),
        "climate": asdict(Climate()),
        "forcing_channels": list(FORCING_CHANNELS),
        "summary_features": list(SUMMARY_FEATURES),
        "outcomes": list(OUTCOMES),
        "snap_days": SNAP_DAYS,
        **extra,
    }


# --------------------------------------------------------------------------
# Training chains
# --------------------------------------------------------------------------


def generate_training(
    out_dir: str | Path,
    n_chains: int,
    years: float = 5.0,
    shard_size: int = 64,
    max_burn_in_days: int = 720,
    revetment_prob: float = 0.4,
    seed: int = 0,
    device: str | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec_days = int(round(years * 365.25 / SNAP_DAYS)) * SNAP_DAYS
    n_shards = (n_chains + shard_size - 1) // shard_size
    for si in range(n_shards):
        rng = np.random.default_rng([seed, si])
        n = min(shard_size, n_chains - si * shard_size)
        burn = int(rng.integers(0, max_burn_in_days // SNAP_DAYS + 1)) * SNAP_DAYS
        total = burn + rec_days
        init = generate_cliffs(n, rng)
        forcing = generate_forcing(n, total, rng)
        crest = sample_revetment(n, rng, revetment_prob)
        sim = Cliff2D(init, device=device)
        sim.set_actions(revetment_crest=crest)
        seeds = rng.integers(0, 2**31 - 1, size=n)
        r = sim.run(forcing, seeds=seeds, snapshot_every=SNAP_DAYS)
        first = burn // SNAP_DAYS
        rock = r.snapshots["rock"][:, first:]
        steps = r.snapshot_steps[first:]
        talus = np.where(steps[None, :] > 0, r.series["talus"][:, np.maximum(steps - 1, 0)], 0.0)
        _save(
            out / f"shard_{si:05d}",
            {
                "rock": rock,
                "talus": talus.astype(np.float32),
                "strength": init["strength"].astype(np.float32),
                "joints": init["joints"],
                "top_row": init["top_row"],
                "rev_crest": crest,
                "forcing": stack_forcing(forcing)[:, burn:],
                "tidal_amp": forcing["tidal_amp"],
            },
        )
    (out / "manifest.json").write_text(json.dumps(_manifest("train", n_chains=n_chains, rec_days=rec_days, seed=seed), indent=2))
    return out


class PairDataset:
    """All valid (chain, start snapshot, Δt) pairs across training shards.

    ``__getitem__`` returns a dict of numpy arrays for one pair, which a
    PyTorch ``Dataset`` can wrap directly; ``sample`` draws a random batch.
    """

    def __init__(self, root: str | Path, dts: tuple[int, ...] = DT_DAYS):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.shards = [_load(p) for p in sorted(self.root.glob("shard_*"))]
        self.dts = tuple(dts)
        index = []
        for si, sh in enumerate(self.shards):
            n, S = sh["rock"].shape[:2]
            for d in self.dts:
                k = d // SNAP_DAYS
                for c in range(n):
                    for s in range(S - k):
                        index.append((si, c, s, d))
        self.index = np.array(index, np.int64).reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        si, c, s, d = (int(x) for x in self.index[i])
        sh = self.shards[si]
        k = d // SNAP_DAYS
        r0 = np.asarray(sh["rock"][c, s], np.float32)
        r1 = np.asarray(sh["rock"][c, s + k], np.float32)
        daily = np.asarray(sh["forcing"][c, s * SNAP_DAYS : s * SNAP_DAYS + d])
        return {
            "rock0": r0,
            "rock1": r1,
            "talus0": np.float32(sh["talus"][c, s]),
            "talus1": np.float32(sh["talus"][c, s + k]),
            "strength": np.asarray(sh["strength"][c], np.float32),
            "top_row": np.int64(sh["top_row"][c]),
            "rev_crest": np.float32(sh["rev_crest"][c]),
            "dt_days": np.int64(d),
            "summary": summarize_forcing(daily, sh["tidal_amp"][c]),
            "outcomes": cliff_outcomes(r0, r1, sh["top_row"][c]),
        }

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.choice(len(self), size=min(n, len(self)), replace=False)
        items = [self[int(i)] for i in idx]
        return {k: np.stack([it[k] for it in items]) for k in items[0]}


# --------------------------------------------------------------------------
# Evaluation ensembles
# --------------------------------------------------------------------------


def branch_rollout(
    init: dict[str, np.ndarray],
    rock: np.ndarray,
    talus: np.ndarray,
    rev_crest: np.ndarray,
    forcing: dict[str, np.ndarray],
    seeds: np.ndarray,
    dts: tuple[int, ...] = DT_DAYS,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Continue N cliffs from given states; return rock [N, nDT, NZ, NX] (f16) and talus [N, nDT]."""
    sim = Cliff2D(init, device=device)
    sim.set_state({"rock": rock.astype(np.float32), "talus": talus.astype(np.float32), "t": np.array(0)})
    sim.set_actions(revetment_crest=rev_crest)
    rocks, tal = [], []
    prev = 0
    for d in dts:
        r = sim.run(forcing, seeds=seeds, snapshot_every=d - prev, days=d - prev)
        rocks.append(r.snapshots["rock"][:, -1])
        tal.append(sim.talus.numpy().copy())
        prev = d
    return np.stack(rocks, 1), np.stack(tal, 1)


def generate_eval(
    out_dir: str | Path,
    n_starts: int,
    n_futures: int,
    burn_in_days: int = 360,
    revetment_prob: float = 0.4,
    dts: tuple[int, ...] = DT_DAYS,
    seed: int = 12345,
    device: str | None = None,
) -> Path:
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    K, M, H = n_starts, n_futures, max(dts)

    # start states: fresh cliffs after a burn-in under their own history
    init = generate_cliffs(K, rng)
    crest = sample_revetment(K, rng, revetment_prob)
    burn_forcing = generate_forcing(K, burn_in_days, rng)
    sim = Cliff2D(init, device=device)
    sim.set_actions(revetment_crest=crest)
    sim.run(burn_forcing, seeds=rng.integers(0, 2**31 - 1, size=K), snapshot_every=burn_in_days)
    st = sim.get_state()
    amp = burn_forcing["tidal_amp"]  # tidal range is a site property, shared by its futures

    # M futures per start
    rep = lambda a: np.repeat(a, M, axis=0)
    init_rep = {k: rep(v) for k, v in init.items()}
    fut = generate_forcing(K * M, H, rng, start_day=burn_in_days)
    fut["tidal_amp"] = rep(amp)
    seeds = rng.integers(0, 2**31 - 1, size=K * M)
    rocks, tal = branch_rollout(init_rep, rep(st["rock"]), rep(st["talus"]), rep(crest), fut, seeds, dts, device)

    daily = stack_forcing(fut)  # [K*M, H, C]
    summaries = np.stack([summarize_forcing(daily[:, :d], fut["tidal_amp"]) for d in dts], 1)
    outcomes = cliff_outcomes(rep(st["rock"])[:, None], rocks, rep(init["top_row"])[:, None])

    shape = lambda a: a.reshape((K, M) + a.shape[1:])
    _save(
        out,
        {
            "start_rock": st["rock"].astype(np.float32),  # exact: fractional cells matter for short Δt
            "start_talus": st["talus"],
            "strength": init["strength"].astype(np.float32),
            "joints": init["joints"],
            "top_row": init["top_row"],
            "rev_crest": crest,
            "tidal_amp": amp,
            "forcing": shape(daily),
            "seeds": shape(seeds),
            "rock": shape(rocks),
            "talus": shape(tal),
            "summary": shape(summaries),
            "outcomes": shape(outcomes),
            "dt_days": np.array(dts),
        },
    )
    (out / "manifest.json").write_text(json.dumps(_manifest("eval", n_starts=K, n_futures=M, burn_in_days=burn_in_days, dts=list(dts), seed=seed), indent=2))
    return out


class EvalSet:
    """Loaded evaluation ensembles. Arrays are [K, M, nDT, ...] where applicable."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        a = _load(self.root, mmap=False)
        self.__dict__.update(a)
        self.dts = tuple(int(d) for d in a["dt_days"])
        self.K, self.M = a["outcomes"].shape[:2]

    def init_for(self, k: np.ndarray) -> dict[str, np.ndarray]:
        out = {"rock": self.start_rock[k].astype(np.float32), **{s: np.asarray(getattr(self, s)[k]) for s in STATIC_KEYS}}
        out["strength"] = out["strength"].astype(np.float32)
        return out

    def forcing_for(self, k: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
        return unstack_forcing(self.forcing[k, m], self.tidal_amp[k])
