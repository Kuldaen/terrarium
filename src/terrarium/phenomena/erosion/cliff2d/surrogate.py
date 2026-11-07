"""Seaglass surrogate: a Δt-conditioned generative model of cliff evolution.

Given the current rock field (and optionally the strength field), an action,
an interval Δt and that interval's forcing summary, it *samples* the rock
removed over the interval and the talus volume at its end. Many samples =
many possible futures.

Model
-----
Conditional flow matching (rectified flow). A U-Net maps a noisy target,
the conditioning images and an embedding of (flow time τ, conditioning
vector) to a velocity; sampling integrates noise → target with Euler steps.

Target channels (in [-1, 1]):
  0  rock removed over the interval, 2·(rock0 − rock1) − 1  (mostly −1)
  1  talus at the end, broadcast as a constant map           (read back as mean)

Predicting *removed* rock (clipped to [0, rock0]) makes rock monotonically
non-increasing by construction — a physical prior that costs nothing.

Conditioning images: rock0, and log-strength if ``use_strength`` (the
"known geology" setting; off = geology must be inferred from the visible
profile, the realistic "survey only" setting).
Conditioning vector: forcing summary (incl. Δt in years), log Δt,
revetment flag + crest, talus at start.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictors import Query
from .summary import SUMMARY_FEATURES, cliff_outcomes

TALUS_SCALE = 20.0  # m^3/m; talus/TALUS_SCALE*2-1 maps typical range into [-1, 1]
N_COND = len(SUMMARY_FEATURES) + 4


# --------------------------------------------------------------------------
# conditioning
# --------------------------------------------------------------------------


def cond_vector(summary: np.ndarray, dt_days: np.ndarray, rev_crest: np.ndarray, talus0: np.ndarray) -> np.ndarray:
    """[N, N_COND] raw (unnormalised) conditioning vector."""
    has = np.isfinite(rev_crest).astype(np.float32)
    crest = np.nan_to_num(rev_crest, nan=0.0).astype(np.float32)
    return np.concatenate(
        [summary, np.log(np.asarray(dt_days, np.float32))[:, None], has[:, None], crest[:, None], (np.asarray(talus0) / TALUS_SCALE)[:, None]],
        1,
    ).astype(np.float32)


def cond_images(rock0: np.ndarray, strength: np.ndarray | None) -> np.ndarray:
    """[N, C, NZ, NX] conditioning images in roughly [-1, 1]."""
    chans = [2.0 * rock0 - 1.0]
    if strength is not None:
        chans.append(np.log(np.clip(strength, 1e-3, None)) / 0.7)
    return np.stack(chans, 1).astype(np.float32)


def encode_target(rock0, rock1, talus1) -> np.ndarray:
    removed = np.clip(rock0 - rock1, 0.0, 1.0)
    tal = np.clip(np.asarray(talus1) / TALUS_SCALE, 0.0, 1.0)[:, None, None] * np.ones_like(rock0)
    return np.stack([2.0 * removed - 1.0, 2.0 * tal - 1.0], 1).astype(np.float32)


def decode_target(x: np.ndarray, rock0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    removed = np.clip((x[:, 0] + 1.0) / 2.0, 0.0, None)
    removed[removed < 0.05] = 0.0  # sampling noise is not erosion
    removed = np.minimum(removed, rock0)
    rock1 = rock0 - removed
    rock1[rock1 < 0.02] = 0.0  # snap dust to air
    talus1 = np.clip((x[:, 1].mean((-1, -2)) + 1.0) / 2.0, 0.0, None) * TALUS_SCALE
    return rock1, talus1


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------


@dataclass
class SurrogateConfig:
    base_ch: int = 32
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    emb_dim: int = 128
    use_strength: bool = True
    attn_heads: int = 4

    @property
    def in_ch(self) -> int:
        return 2 + 1 + (1 if self.use_strength else 0)


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = _gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(emb, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(e)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + scale) + shift
        return self.c2(F.silu(h)) + self.skip(x)


class Attn(nn.Module):
    """Global self-attention at the coarsest level: lets the toe "see" the top."""

    def __init__(self, c: int, heads: int):
        super().__init__()
        self.n = _gn(c)
        self.a = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.n(x).flatten(2).transpose(1, 2)
        y, _ = self.a(t, t, t, need_weights=False)
        return x + y.transpose(1, 2).reshape(B, C, H, W)


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t[:, None].float() * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class UNet(nn.Module):
    def __init__(self, cfg: SurrogateConfig):
        super().__init__()
        self.cfg = cfg
        E, C = cfg.emb_dim, [cfg.base_ch * m for m in cfg.ch_mult]
        self.t_mlp = nn.Sequential(nn.Linear(E, E), nn.SiLU(), nn.Linear(E, E))
        self.c_mlp = nn.Sequential(nn.Linear(N_COND, E), nn.SiLU(), nn.Linear(E, E))
        self.inp = nn.Conv2d(cfg.in_ch, C[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        c = C[0]
        for i, co in enumerate(C):
            self.down.append(nn.ModuleList([ResBlock(c, co, E), ResBlock(co, co, E)]))
            c = co
            self.pool.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(C) - 1 else nn.Identity())
        self.mid1, self.attn, self.mid2 = ResBlock(c, c, E), Attn(c, cfg.attn_heads), ResBlock(c, c, E)
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(C))):
            self.up.append(nn.ModuleList([ResBlock(c + co, co, E), ResBlock(co, co, E)]))
            c = co
        self.out = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv2d(c, 2, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, xt, cimg, tau, cvec):
        e = self.t_mlp(_timestep_embedding(tau, self.cfg.emb_dim)) + self.c_mlp(cvec)
        h = self.inp(torch.cat([xt, cimg], 1))
        skips = []
        for (r1, r2), pool in zip(self.down, self.pool):
            h = r2(r1(h, e), e)
            skips.append(h)
            h = pool(h)
        h = self.mid2(self.attn(self.mid1(h, e)), e)
        for (r1, r2), s in zip(self.up, reversed(skips)):
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = r2(r1(torch.cat([h, s], 1), e), e)
        return self.out(h)


# --------------------------------------------------------------------------
# surrogate wrapper
# --------------------------------------------------------------------------


class FlowSurrogate:
    def __init__(self, cfg: SurrogateConfig, cond_mean: np.ndarray, cond_std: np.ndarray, device: torch.device):
        self.cfg, self.device = cfg, device
        self.net = UNet(cfg).to(device)
        self.cond_mean = torch.tensor(np.asarray(cond_mean, np.float32), device=device)
        self.cond_std = torch.tensor(np.asarray(cond_std, np.float32), device=device)

    # ---- batch preparation ----
    def tensors(self, rock0, strength, cvec_raw):
        cimg = cond_images(rock0, strength if self.cfg.use_strength else None)
        c = (torch.tensor(cvec_raw, device=self.device) - self.cond_mean) / self.cond_std
        return torch.tensor(cimg, device=self.device), c

    # ---- training ----
    def loss(self, x1, cimg, cvec):
        x0 = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - tau)[:, None, None, None] * x0 + tau[:, None, None, None] * x1
        v = self.net(xt, cimg, tau, cvec)
        return F.mse_loss(v.float(), (x1 - x0).float())

    # ---- sampling ----
    @torch.no_grad()
    def sample(self, cimg, cvec, steps: int = 20, generator: torch.Generator | None = None, net: nn.Module | None = None, x0: torch.Tensor | None = None):
        """Integrate noise -> target. Pass ``x0`` to fix the noise (common random numbers)."""
        net = net or self.net
        x = x0.clone() if x0 is not None else torch.randn((cimg.shape[0], 2) + tuple(cimg.shape[-2:]), device=cimg.device, generator=generator)
        for i in range(steps):
            tau = torch.full((x.shape[0],), i / steps, device=x.device)
            x = x + net(x, cimg, tau, cvec) / steps
        return x

    def state_dict(self) -> dict:
        return {"cfg": asdict(self.cfg), "cond_mean": self.cond_mean.cpu().numpy(), "cond_std": self.cond_std.cpu().numpy(), "net": self.net.state_dict()}

    @classmethod
    def load(cls, path: str, device: torch.device, ema: bool = True) -> "FlowSurrogate":
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        cfg["ch_mult"] = tuple(cfg["ch_mult"])
        s = cls(SurrogateConfig(**cfg), ck["cond_mean"], ck["cond_std"], device)
        s.net.load_state_dict(ck["ema"] if ema and "ema" in ck else ck["net"])
        s.net.eval()
        return s


class SurrogatePredictor:
    """Calibration-report adapter (implements ``predictors.Predictor``)."""

    def __init__(self, path: str, device: str | None = None, steps: int = 20, batch: int = 128, name: str | None = None):
        from terrarium.core.torch_utils import pick_device

        self.dev = pick_device(device)
        self.s = FlowSurrogate.load(path, self.dev)
        self.steps, self.batch = steps, batch
        self.name = name or ("surrogate" + ("" if self.s.cfg.use_strength else "_nostrength"))

    def predict(self, q: Query, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        N, D = len(q.k), len(q.dts)
        out = np.zeros((N, D, n_samples, 3), np.float32)
        g = torch.Generator(device=self.dev).manual_seed(int(rng.integers(2**31)))
        # flatten (case, sample) and chunk
        rep = np.repeat(np.arange(N), n_samples)
        for di, d in enumerate(q.dts):
            cv = cond_vector(q.summary[:, di], np.full(N, d), q.rev_crest, q.talus0)
            for s0 in range(0, len(rep), self.batch):
                idx = rep[s0 : s0 + self.batch]
                cimg, c = self.s.tensors(q.rock0[idx], q.strength[idx], cv[idx])
                x = self.s.sample(cimg, c, self.steps, g).float().cpu().numpy()
                rock1, _ = decode_target(x, q.rock0[idx])
                o = cliff_outcomes(q.rock0[idx], rock1, q.top_row[idx])
                flat = np.arange(s0, s0 + len(idx))
                out[flat // n_samples, di, flat % n_samples] = o
        return out


# --------------------------------------------------------------------------
# rollouts: chained steps under an action plan
# --------------------------------------------------------------------------


def _noise(shape, seed: int, step: int, chunk: int, device) -> torch.Tensor:
    """Deterministic noise per (seed, step, chunk): the same seed gives the same
    model randomness across plans, so plan comparisons use common random numbers."""
    g = torch.Generator(device="cpu").manual_seed((seed * 1_000_003 + step * 10_007 + chunk) % (2**63 - 1))
    return torch.randn(shape, generator=g).to(device)


def rollout(
    model: FlowSurrogate,
    rock0: np.ndarray,
    strength: np.ndarray,
    top_row: np.ndarray,
    talus0: np.ndarray,
    step_days,
    summaries: np.ndarray,
    rev_plan: np.ndarray,
    n_samples: int = 1,
    seed: int = 0,
    flow_steps: int = 20,
    batch: int = 128,
    keep_fields: bool = True,
) -> dict[str, np.ndarray]:
    """Sample trajectories by chaining Δt jumps from sampled states.

    rock0, strength  [N, NZ, NX]   start states (strength ignored if the model doesn't use it)
    top_row, talus0  [N]
    step_days        S step lengths in days, e.g. [360] * 5; any Δt the model was trained on
    summaries        [N, S, F]     forcing summary of each step's interval (see ``step_summaries``)
    rev_plan         [N, S]        revetment crest in force during each step (NaN = none)
    n_samples        futures per start; sample j of start i uses the same noise for any plan

    Returns
      rock      [N, n, S+1, NZ, NX] float16 (if keep_fields)
      talus     [N, n, S+1]
      outcomes  [N, n, S, 3] cumulative from the start (top retreat, toe retreat, volume lost)
    """
    N, S = len(rock0), len(step_days)
    rep = lambda a: np.repeat(np.asarray(a), n_samples, axis=0)
    cur, tal = rep(rock0).astype(np.float32), rep(talus0).astype(np.float32)
    st, tr = rep(strength).astype(np.float32), rep(top_row)
    start = cur.copy()
    M = len(cur)
    rocks = [cur.astype(np.float16)] if keep_fields else []
    tals, outs = [tal.copy()], []
    for s, d in enumerate(step_days):
        cv = cond_vector(rep(summaries[:, s]), np.full(M, d), rep(rev_plan[:, s]), tal)
        nxt = np.empty_like(cur)
        ntal = np.empty_like(tal)
        for ci, s0 in enumerate(range(0, M, batch)):
            sl = slice(s0, min(s0 + batch, M))
            cimg, c = model.tensors(cur[sl], st[sl], cv[sl])
            x0 = _noise((sl.stop - sl.start, 2) + cur.shape[1:], seed, s, ci, model.device)
            x = model.sample(cimg, c, flow_steps, x0=x0).float().cpu().numpy()
            nxt[sl], ntal[sl] = decode_target(x, cur[sl])
        cur, tal = nxt, ntal
        if keep_fields:
            rocks.append(cur.astype(np.float16))
        tals.append(tal.copy())
        outs.append(cliff_outcomes(start, cur, tr))
    res = {
        "talus": np.stack(tals, 1).reshape(N, n_samples, S + 1),
        "outcomes": np.stack(outs, 1).reshape(N, n_samples, S, 3),
    }
    if keep_fields:
        res["rock"] = np.stack(rocks, 1).reshape((N, n_samples, S + 1) + rock0.shape[1:])
    return res
