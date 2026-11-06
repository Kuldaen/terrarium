"""Initial cliff geometry, layered strength and joint sets."""

from __future__ import annotations

import numpy as np

from .params import Geology, Grid


def generate_cliffs(
    batch: int,
    rng: np.random.Generator,
    grid: Grid = Grid(),
    geo: Geology = Geology(),
) -> dict[str, np.ndarray]:
    """Return initial state arrays.

    rock      [B, NZ, NX] float32 rock fraction (1 = intact)
    strength  [B, NZ, NX] float32 relative strength (1 = reference soft rock)
    joints    [B, NJ] int32 column indices of vertical joints (sorted, padded with nx)
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

        # platform: below toe_z at the toe, sloping down seaward
        z_plat = toe_z - (toe_x - x) * geo.platform_slope  # [NX]
        plat = z[:, None] < z_plat[None, :]
        # cliff face: x >= toe_x + slope*(z - toe_z) + roughness
        rough = rng.integers(-geo.face_roughness_cells, geo.face_roughness_cells + 1, size=nz) * dx
        x_face = toe_x + geo.face_slope * np.maximum(z - toe_z, 0.0) + rough
        cliff = (x[None, :] >= x_face[:, None]) & (z[:, None] < top_z)
        rock[b] = (plat | cliff).astype(np.float32)

        # horizontal layers with lognormal strengths
        layer_s = np.empty(nz, np.float32)
        zc = grid.z0
        while zc < grid.z0 + nz * dz:
            th = rng.uniform(*geo.layer_thickness_range)
            s = np.exp(rng.normal(0.0, geo.layer_strength_sigma))
            sel = (z >= zc) & (z < zc + th)
            layer_s[sel] = s
            zc += th
        cell = np.exp(rng.normal(0.0, geo.cell_strength_sigma, size=(nz, nx)))
        strength[b] = (layer_s[:, None] * cell).astype(np.float32)

        # vertical joints
        xs = []
        xj = rng.uniform(0, geo.joint_spacing_mean)
        while xj < nx * dx and len(xs) < max_joints:
            xs.append(int(xj / dx))
            xj += geo.joint_spacing_mean * np.exp(rng.normal(-0.5 * geo.joint_spacing_sigma**2, geo.joint_spacing_sigma))
        uniq = sorted(set(xs))
        joints[b, : len(uniq)] = uniq

    return {"rock": rock, "strength": strength, "joints": joints, "top_row": top_row}
