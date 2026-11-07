"""Initial cliff geometry and a heterogeneous rock-strength field.

Strength is the only place rock structure lives. Fractures are not drawn as
geometry; they are weak cells inside the strength field, and collapses find
their own crack path through it (see ``sim._cliff_step``). The field combines:

* horizontal **layers** with lognormal strengths,
* **multi-scale correlated heterogeneity** (smoothed noise at several
  wavelengths), so rock has hard and soft patches at many scales,
* **joint traces**: dipping (either way), wavy, non-persistent fractures
  with intact rock bridges, each with its own weakness,
* **bedding-plane partings** at layer boundaries, also with gaps.
"""

from __future__ import annotations

import numpy as np

from .params import Geology, Grid


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma_cells: float) -> np.ndarray:
    """Unit-variance Gaussian random field with Gaussian correlation length sigma (cells)."""
    nz, nx = shape
    w = rng.normal(size=shape)
    kz = np.fft.fftfreq(nz)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    filt = np.exp(-2 * (np.pi * sigma_cells) ** 2 * (kz**2 + kx**2))
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * filt))
    return f / (f.std() + 1e-9)


def _runs(rng: np.random.Generator, n: int, mean_on: float, mean_off: float) -> np.ndarray:
    """Boolean mask of length n alternating on/off runs with exponential lengths (cells)."""
    out = np.zeros(n, bool)
    i, on = 0, rng.random() < mean_on / (mean_on + mean_off)
    while i < n:
        L = max(1, int(round(rng.exponential(mean_on if on else mean_off))))
        out[i : i + L] = on
        i += L
        on = not on
    return out


def generate_cliffs(
    batch: int,
    rng: np.random.Generator,
    grid: Grid = Grid(),
    geo: Geology = Geology(),
) -> dict[str, np.ndarray]:
    """Return initial state arrays.

    rock      [B, NZ, NX] float32 rock fraction (1 = intact)
    strength  [B, NZ, NX] float32 relative strength (1 = reference soft rock),
              with joints/bedding written in as weak cells
    joints    [B, NJ] int32 column where each joint trace meets the cliff top
              (informational; sorted, padded with nx)
    top_row   [B] int32 index of the top rock row of the cliff
    """
    nz, nx, dx, dz = grid.nz, grid.nx, grid.dx, grid.dz
    x = (np.arange(nx) + 0.5) * dx
    z = grid.z0 + (np.arange(nz) + 0.5) * dz

    rock = np.zeros((batch, nz, nx), np.float32)
    strength = np.zeros((batch, nz, nx), np.float32)
    max_joints = int(nx * dx / 0.3) + 1
    joints = np.full((batch, max_joints), nx, np.int32)
    top_row = np.zeros(batch, np.int32)

    for b in range(batch):
        toe_x = geo.cliff_toe_x + rng.uniform(-1.0, 1.0)
        toe_z = geo.toe_z_mean + rng.uniform(-geo.toe_z_jitter, geo.toe_z_jitter)
        top_z = geo.cliff_top_z + rng.uniform(-1.5, 1.0)
        top_row[b] = int(np.floor((top_z - grid.z0) / dz)) - 1

        # ---- geometry: platform + cliff face ----
        z_plat = toe_z - (toe_x - x) * geo.platform_slope
        plat = z[:, None] < z_plat[None, :]
        rough = rng.integers(-geo.face_roughness_cells, geo.face_roughness_cells + 1, size=nz) * dx
        x_face = toe_x + geo.face_slope * np.maximum(z - toe_z, 0.0) + rough
        cliff = (x[None, :] >= x_face[:, None]) & (z[:, None] < top_z)
        rock[b] = (plat | cliff).astype(np.float32)

        # ---- layers ----
        layer_s = np.empty(nz, np.float32)
        boundaries = []
        zc = grid.z0
        while zc < grid.z0 + nz * dz:
            th = rng.uniform(*geo.layer_thickness_range)
            sel = (z >= zc) & (z < zc + th)
            layer_s[sel] = np.exp(rng.normal(0.0, geo.layer_strength_sigma))
            zc += th
            boundaries.append(zc)
        log_s = np.log(layer_s)[:, None] * np.ones((1, nx))

        # ---- multi-scale correlated heterogeneity ----
        for wl_m, amp in geo.heterogeneity_octaves:
            log_s += amp * _smooth_noise(rng, (nz, nx), wl_m / dx)
        log_s += rng.normal(0.0, geo.cell_strength_sigma, size=(nz, nx))
        s = np.exp(log_s)

        # ---- bedding-plane partings at layer boundaries ----
        for zb in boundaries:
            j = int(round((zb - grid.z0) / dz))
            if 0 <= j < nz:
                gaps = _runs(rng, nx, geo.bedding_on_m / dx, geo.bedding_off_m / dx)
                s[j, gaps] *= geo.bedding_weakness

        # ---- joint traces: dipping, wavy, with intact bridges ----
        tops = []
        xj = rng.uniform(0, geo.joint_spacing_mean)
        while xj < nx * dx:
            dip = np.radians(rng.normal(geo.joint_dip_mean_deg, geo.joint_dip_sigma_deg))
            lean = rng.choice([-1.0, 1.0])
            weak = rng.uniform(*geo.joint_weakness_range)
            present = _runs(rng, nz, geo.joint_on_m / dz, geo.joint_off_m / dz)
            xc = xj / dx  # in cells, at the cliff-top row
            wobble = 0.0
            for j in range(nz - 1, -1, -1):
                # cot(dip) cells sideways per cell down, plus a random walk
                wobble += rng.normal(0.0, geo.joint_wobble_cells)
                xi = int(round(xc + lean * (nz - 1 - j) * dz / np.tan(dip) / dx + wobble))
                if present[j] and 0 <= xi < nx:
                    s[j, xi] *= weak
            tops.append(int(xj / dx))
            xj += geo.joint_spacing_mean * np.exp(rng.normal(-0.5 * geo.joint_spacing_sigma**2, geo.joint_spacing_sigma))
        tops = sorted(set(t for t in tops if t < nx))[:max_joints]
        joints[b, : len(tops)] = tops

        strength[b] = np.clip(s, 0.05, None).astype(np.float32)

    return {"rock": rock, "strength": strength, "joints": joints, "top_row": top_row}
