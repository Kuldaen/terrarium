"""Batched 2D cross-shore cliff erosion simulator (NVIDIA Warp).

One daily step = two kernel launches:

1. ``_cliff_step`` (one thread per cliff): bed/face extraction, talus removal,
   stochastic cantilever failure + block removal, series recording.
2. ``_face_erosion`` (one thread per cliff row): wave attack and weathering of
   the seaward-most rock cell in each row.

Runs on CPU or CUDA (``device="cuda"``). All randomness comes from the
``forcing`` arrays and per-cliff ``seeds``.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from terrarium.core.interfaces import Rollout

from .params import Grid, Physics

wp.config.quiet = True


@wp.struct
class _P:
    nx: int
    nz: int
    dx: float
    dz: float
    z0: float
    gamma: float
    alpha: float
    n_phases: int
    k_wave: float
    impact_offset: float
    impact_width_h: float
    impact_width_min: float
    low_tide_margin: float
    k_weather: float
    weather_base: float
    bulking: float
    tan_phi: float
    k_talus: float
    talus_dissip: float
    rho_g: float
    tensile: float
    rain_weak: float
    wave_weak: float
    hs_ref: float
    hazard_base: float
    hazard_kappa: float
    min_overhang: int
    overtop: float
    max_joint_reach: int


@wp.func
def _bed_z(p: _P, bed: wp.array2d(dtype=wp.int32), b: int, i: int) -> float:
    return p.z0 + float(bed[b, i]) * p.dz


@wp.func
def _eta(p: _P, eta0: float, amp: float, k: int) -> float:
    return eta0 + amp * wp.cos(2.0 * wp.pi * float(k) / float(p.n_phases))


@wp.func
def _wave_at(
    p: _P,
    bed: wp.array2d(dtype=wp.int32),
    shallow: wp.array3d(dtype=wp.int32),
    b: int,
    k: int,
    fc: int,
    eta: float,
    hs: float,
    floor_z: float,
) -> float:
    """Wave height reaching a face whose seaward-most rock cell is column fc.

    Depth-limited by the platform floor in front of the cliff (small ledges on
    the face do not break waves) and decayed across the shallow (surf-zone)
    width seaward of it; ``shallow[b, k, fc]`` counts shallow columns < fc.
    """
    if fc <= 0:
        return 0.0
    d = eta - wp.min(_bed_z(p, bed, b, fc - 1), floor_z)
    if d <= 0.0:
        return 0.0
    h = wp.min(hs, p.gamma * d)
    return h * wp.exp(-p.alpha * float(shallow[b, k, fc]) * p.dx)


@wp.kernel
def _cliff_step(
    p: _P,
    t: int,
    t_abs: int,
    rock: wp.array3d(dtype=float),
    strength: wp.array3d(dtype=float),
    joints: wp.array2d(dtype=wp.int32),
    top_row: wp.array(dtype=wp.int32),
    talus: wp.array(dtype=float),
    hs: wp.array2d(dtype=float),
    surge: wp.array2d(dtype=float),
    rain: wp.array2d(dtype=float),
    slr: wp.array2d(dtype=float),
    tidal_amp: wp.array(dtype=float),
    seeds: wp.array(dtype=wp.int32),
    # scratch / outputs
    bed: wp.array2d(dtype=wp.int32),
    coltop: wp.array2d(dtype=wp.int32),
    face: wp.array2d(dtype=wp.int32),
    shallow: wp.array3d(dtype=wp.int32),
    toe_z: wp.array(dtype=float),
    talus_h: wp.array(dtype=float),
    s_toe_x: wp.array2d(dtype=float),
    s_notch_x: wp.array2d(dtype=float),
    s_top_x: wp.array2d(dtype=float),
    s_talus: wp.array2d(dtype=float),
    e_fail_vol: wp.array2d(dtype=float),
    e_fail_z: wp.array2d(dtype=float),
):
    b = wp.tid()
    nx = p.nx
    nz = p.nz
    tr = top_row[b]

    # --- bed heights (contiguous rock from the bottom) ---
    for i in range(nx):
        k = int(0)
        while k < nz and rock[b, k, i] > 0.0:
            k += 1
        bed[b, i] = k
        top = int(-1)
        for jj in range(nz):
            if rock[b, jj, i] > 0.0:
                top = jj
        coltop[b, i] = top

    # --- face column per row ---
    for j in range(nz):
        f = int(nx)
        for i in range(nx):
            if rock[b, j, i] > 0.0:
                f = i
                break
        face[b, j] = f

    # --- cantilever failure: find most critical notch ---
    h_now = hs[b, t]
    r_now = rain[b, t]
    weak = (1.0 - p.rain_weak * wp.min(r_now, 1.0)) * (1.0 - p.wave_weak * wp.min(h_now / p.hs_ref, 1.0))
    best_r = float(0.0)
    best_n = int(-1)
    min_above = int(nx)
    if tr < nz:
        min_above = face[b, tr]
    n = tr - 1
    while n >= 1:
        xc = face[b, n]
        if xc > 0 and xc < nx:
            L = xc - min_above
            if L >= p.min_overhang:
                hb = float(tr - n) * p.dz
                Lm = float(L) * p.dx
                sigma = 3.0 * p.rho_g * Lm * Lm / hb
                T = p.tensile * strength[b, n, xc] * weak
                r = sigma / T
                if r > best_r:
                    best_r = r
                    best_n = n
        if face[b, n] < min_above:
            min_above = face[b, n]
        n -= 1

    fail_vol = float(0.0)
    fail_z = float(-999.0)
    if best_n >= 0:
        lam = wp.min(p.hazard_base * wp.exp(p.hazard_kappa * (best_r - 1.0)), 5.0)
        prob = 1.0 - wp.exp(-lam)
        state = wp.rand_init(seeds[b], t_abs)
        u = wp.randf(state)
        if u < prob:
            xc = face[b, best_n]
            # crack snaps to the nearest joint landward of the notch back
            xcrack = xc
            for k in range(joints.shape[1]):
                jx = joints[b, k]
                if jx >= xc:
                    if jx - xc <= p.max_joint_reach:
                        xcrack = jx
                    break
            vol = float(0.0)
            for j in range(best_n + 1, nz):
                for i in range(xcrack):
                    vol += rock[b, j, i]
                    rock[b, j, i] = 0.0
            fail_vol = vol * p.dx * p.dz
            fail_z = p.z0 + (float(best_n) + 1.0) * p.dz
            talus[b] = talus[b] + fail_vol * p.bulking
            # refresh faces / beds above the notch
            for j in range(best_n + 1, nz):
                f = int(nx)
                for i in range(nx):
                    if rock[b, j, i] > 0.0:
                        f = i
                        break
                face[b, j] = f
            for i in range(xcrack):
                k = int(0)
                while k < nz and rock[b, k, i] > 0.0:
                    k += 1
                bed[b, i] = k
                top = int(-1)
                for jj in range(nz):
                    if rock[b, jj, i] > 0.0:
                        top = jj
                coltop[b, i] = top
    e_fail_vol[b, t] = fail_vol
    e_fail_z[b, t] = fail_z

    amp = tidal_amp[b]
    msl = slr[b, t]
    eta0 = msl + surge[b, t]

    # --- prefix counts of shallow (surf-zone) columns per tidal phase ---
    thresh = h_now / p.gamma
    for k in range(p.n_phases):
        eta = _eta(p, eta0, amp, k)
        c = int(0)
        shallow[b, k, 0] = 0
        for i in range(nx):
            if eta - _bed_z(p, bed, b, i) < thresh:
                c += 1
            shallow[b, k, i + 1] = c

    # --- toe: lowest floor in front of the face within the intertidal band ---
    j_lo = int(wp.max(0.0, (msl - amp - 1.0 - p.z0) / p.dz))
    j_hi = int(wp.min(float(nz - 1), (msl + amp + 3.0 - p.z0) / p.dz))
    # notch back = most landward face among "cliff rows" (face column rises
    # >= 2 m above the row, which excludes platform steps). The toe is the
    # lowest face within 4 m seaward of the notch back; the talus rests on
    # the floor in front of it.
    wall = int(2.0 / p.dz)
    reach = int(4.0 / p.dx)
    f_notch = int(0)
    for j in range(j_lo, j_hi + 1):
        fj = face[b, j]
        if fj > 0 and fj < nx:
            if coltop[b, fj] - j >= wall and fj > f_notch:
                f_notch = fj
    f_toe = f_notch
    for j in range(j_lo, j_hi + 1):
        fj = face[b, j]
        if fj > 0 and fj < nx and fj >= f_notch - reach:
            f_toe = fj
            break
    tz = float(p.z0)
    if f_toe > 0:
        tz = _bed_z(p, bed, b, f_toe - 1)
    toe_z[b] = tz

    # --- talus removal by waves (tide-averaged H^2 at the toe) ---
    h2 = float(0.0)
    for k in range(p.n_phases):
        h = _wave_at(p, bed, shallow, b, k, f_toe, _eta(p, eta0, amp, k), h_now, tz)
        h2 += h * h
    h2 = h2 / float(p.n_phases)
    v = wp.max(talus[b] - p.k_talus * h2, 0.0)
    talus[b] = v
    talus_h[b] = wp.sqrt(2.0 * v * p.tan_phi)

    s_toe_x[b, t] = float(f_toe) * p.dx
    s_notch_x[b, t] = float(f_notch) * p.dx
    top_f = nx
    if tr < nz:
        top_f = face[b, tr]
    s_top_x[b, t] = float(top_f) * p.dx
    s_talus[b, t] = v


@wp.kernel
def _face_erosion(
    p: _P,
    t: int,
    rock: wp.array3d(dtype=float),
    strength: wp.array3d(dtype=float),
    bed: wp.array2d(dtype=wp.int32),
    face: wp.array2d(dtype=wp.int32),
    shallow: wp.array3d(dtype=wp.int32),
    toe_z: wp.array(dtype=float),
    talus_h: wp.array(dtype=float),
    rev_crest: wp.array(dtype=float),
    hs: wp.array2d(dtype=float),
    surge: wp.array2d(dtype=float),
    rain: wp.array2d(dtype=float),
    slr: wp.array2d(dtype=float),
    tidal_amp: wp.array(dtype=float),
):
    b, j = wp.tid()
    fc = face[b, j]
    if fc <= 0 or fc >= p.nx:
        return
    z = p.z0 + (float(j) + 0.5) * p.dz
    amp = tidal_amp[b]
    eta0 = slr[b, t] + surge[b, t]
    s = strength[b, j, fc]
    rate = float(0.0)

    # protection by talus wedge or revetment
    ht = talus_h[b]
    z_talus_top = toe_z[b] + ht
    crest = rev_crest[b]
    protected = z < z_talus_top or z < crest

    if not protected and z > eta0 - amp - p.low_tide_margin:
        h_off = hs[b, t]
        talus_w = ht / p.tan_phi
        fsum = float(0.0)
        for k in range(p.n_phases):
            eta = _eta(p, eta0, amp, k)
            h = _wave_at(p, bed, shallow, b, k, fc, eta, h_off, toe_z[b])
            h = h * wp.exp(-p.talus_dissip * talus_w)
            if crest > -100.0:
                h = h * p.overtop
            if h > 0.0:
                w = wp.max(p.impact_width_h * h, p.impact_width_min)
                dzi = (z - (eta + p.impact_offset * h)) / w
                fsum += h * h * wp.exp(-dzi * dzi)
        rate += p.k_wave * fsum / float(p.n_phases) / s

    # subaerial weathering above the highest water
    if z > eta0 + amp + 0.5:
        rate += p.k_weather * (p.weather_base + rain[b, t]) / s

    if rate > 0.0:
        rock[b, j, fc] = wp.max(rock[b, j, fc] - rate, 0.0)


@wp.kernel
def _snapshot(
    rock: wp.array3d(dtype=float),
    snaps: wp.array4d(dtype=wp.float16),
    s: int,
):
    b, j, i = wp.tid()
    snaps[b, s, j, i] = wp.float16(rock[b, j, i])


class Cliff2D:
    """Batched ground-truth simulator implementing ``core.Simulator``."""

    def __init__(
        self,
        init: dict[str, np.ndarray],
        grid: Grid = Grid(),
        physics: Physics = Physics(),
        device: str | None = None,
    ):
        wp.init()
        self.device = device or ("cuda" if wp.is_cuda_available() else "cpu")
        self.grid = grid
        self.physics = physics
        self.batch_size = init["rock"].shape[0]
        B, nz, nx = self.batch_size, grid.nz, grid.nx
        assert init["rock"].shape[1:] == (nz, nx)

        p = _P()
        p.nx, p.nz, p.dx, p.dz, p.z0 = nx, nz, grid.dx, grid.dz, grid.z0
        ph = physics
        p.gamma, p.alpha, p.n_phases = ph.gamma, ph.alpha_dissip, ph.n_tide_phases
        p.k_wave, p.impact_offset = ph.k_wave, ph.impact_offset
        p.impact_width_h, p.impact_width_min = ph.impact_width_h, ph.impact_width_min
        p.low_tide_margin = ph.low_tide_margin
        p.k_weather, p.weather_base = ph.k_weather, ph.weather_base
        p.bulking, p.tan_phi = ph.talus_bulking, math.tan(math.radians(ph.talus_angle_deg))
        p.k_talus, p.talus_dissip = ph.k_talus, ph.talus_dissip
        p.rho_g, p.tensile = ph.rho_g, ph.tensile_strength
        p.rain_weak, p.wave_weak, p.hs_ref = ph.rain_weakening, ph.wave_weakening, ph.hs_ref
        p.hazard_base, p.hazard_kappa = ph.hazard_base, ph.hazard_kappa
        p.min_overhang = ph.min_overhang_cells
        p.overtop = ph.revetment_overtop_factor
        p.max_joint_reach = int(round(2.0 / grid.dx))
        self._p = p

        d = self.device
        self.rock = wp.array(init["rock"], dtype=float, device=d)
        self.strength = wp.array(init["strength"], dtype=float, device=d)
        self.joints = wp.array(init["joints"], dtype=wp.int32, device=d)
        self.top_row = wp.array(init["top_row"], dtype=wp.int32, device=d)
        self.talus = wp.array(init.get("talus", np.zeros(B, np.float32)), dtype=float, device=d)
        self.rev_crest = wp.full(B, -1e3, dtype=float, device=d)
        self.bed = wp.zeros((B, nx), dtype=wp.int32, device=d)
        self.coltop = wp.zeros((B, nx), dtype=wp.int32, device=d)
        self.face = wp.zeros((B, nz), dtype=wp.int32, device=d)
        self.shallow = wp.zeros((B, physics.n_tide_phases, nx + 1), dtype=wp.int32, device=d)
        self.toe_z = wp.zeros(B, dtype=float, device=d)
        self.talus_h = wp.zeros(B, dtype=float, device=d)
        self.t = 0  # absolute day index into forcing

    # ---- state management (for branching futures) ----
    def get_state(self) -> dict[str, np.ndarray]:
        return {
            "rock": self.rock.numpy().copy(),
            "talus": self.talus.numpy().copy(),
            "rev_crest": self.rev_crest.numpy().copy(),
            "t": np.array(self.t),
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.rock.assign(state["rock"])
        self.talus.assign(state["talus"])
        if "rev_crest" in state:
            self.rev_crest.assign(state["rev_crest"])
        self.t = int(state.get("t", 0))

    def set_actions(self, revetment_crest: np.ndarray | None = None) -> None:
        """revetment_crest: [B] crest elevation in m rel. MSL (nan / <= -100 = none)."""
        if revetment_crest is not None:
            rc = np.nan_to_num(np.asarray(revetment_crest, np.float32), nan=-1e3)
            self.rev_crest.assign(rc)

    # ---- rollout ----
    def run(
        self,
        forcing: dict[str, np.ndarray],
        seeds: np.ndarray,
        snapshot_every: int = 30,
        days: int | None = None,
    ) -> Rollout:
        """Advance ``days`` steps, reading forcing[:, self.t : self.t + days]."""
        B, nz, nx = self.batch_size, self.grid.nz, self.grid.nx
        d = self.device
        t0 = self.t
        T = days if days is not None else forcing["hs"].shape[1] - t0
        sl = slice(t0, t0 + T)
        fz = {k: wp.array(np.ascontiguousarray(forcing[k][:, sl]), dtype=float, device=d) for k in ("hs", "surge", "rain", "slr")}
        amp = wp.array(forcing["tidal_amp"], dtype=float, device=d)
        sd = wp.array(np.asarray(seeds, np.int32), dtype=wp.int32, device=d)

        series = {k: wp.zeros((B, T), dtype=float, device=d) for k in ("toe_x", "notch_x", "top_x", "talus")}
        events = {k: wp.zeros((B, T), dtype=float, device=d) for k in ("fail_vol", "fail_z")}
        snap_steps = list(range(0, T + 1, snapshot_every))
        snaps = wp.zeros((B, len(snap_steps), nz, nx), dtype=wp.float16, device=d)

        si = 0
        for k in range(T + 1):
            if si < len(snap_steps) and snap_steps[si] == k:
                wp.launch(_snapshot, dim=(B, nz, nx), inputs=[self.rock, snaps, si], device=d)
                si += 1
            if k == T:
                break
            # seeds offset by absolute time so branching from a state stays reproducible
            wp.launch(
                _cliff_step,
                dim=B,
                inputs=[
                    self._p, k, t0 + k, self.rock, self.strength, self.joints, self.top_row, self.talus,
                    fz["hs"], fz["surge"], fz["rain"], fz["slr"], amp, sd,
                    self.bed, self.coltop, self.face, self.shallow, self.toe_z, self.talus_h,
                    series["toe_x"], series["notch_x"], series["top_x"], series["talus"],
                    events["fail_vol"], events["fail_z"],
                ],
                device=d,
            )
            wp.launch(
                _face_erosion,
                dim=(B, nz),
                inputs=[
                    self._p, k, self.rock, self.strength, self.bed, self.face, self.shallow,
                    self.toe_z, self.talus_h, self.rev_crest,
                    fz["hs"], fz["surge"], fz["rain"], fz["slr"], amp,
                ],
                device=d,
            )
        self.t = t0 + T
        return Rollout(
            snapshots={"rock": snaps.numpy()},
            snapshot_steps=np.array(snap_steps) + t0,
            series={k: v.numpy() for k, v in series.items()},
            events={k: v.numpy() for k, v in events.items()},
            meta={"grid": self.grid, "t0": t0},
        )
