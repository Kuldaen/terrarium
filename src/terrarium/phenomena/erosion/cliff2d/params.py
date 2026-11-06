"""Parameters for the 2D cross-shore cliff simulator.

Units: metres, days. Elevations are relative to present mean sea level (MSL).
x increases landward (x = 0 is the seaward edge of the domain).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    nx: int = 160  # cross-shore cells
    nz: int = 80  # vertical cells
    dx: float = 0.25  # m
    dz: float = 0.25  # m
    z0: float = -4.0  # elevation of grid bottom (m, rel. MSL)

    def z_of_row(self, j: int) -> float:
        return self.z0 + (j + 0.5) * self.dz


@dataclass(frozen=True)
class Physics:
    # --- hydrodynamics ---
    gamma: float = 0.78  # depth-limited breaking ratio H <= gamma * d
    alpha_dissip: float = 0.03  # 1/m, wave decay across shallow platform
    n_tide_phases: int = 12  # discrete tidal phases integrated per day
    # --- wave erosion ---
    k_wave: float = 0.03  # cells/day per (m^2 of H^2) per unit strength
    impact_offset: float = 0.25  # impact peak at eta + offset * H
    impact_width_h: float = 0.6  # vertical spread of impact ∝ H
    impact_width_min: float = 0.3  # m
    low_tide_margin: float = 0.3  # no wave erosion below (lowest water - margin)
    # --- subaerial weathering ---
    k_weather: float = 0.0006  # cells/day per unit rain at unit strength
    weather_base: float = 0.2  # rain-independent weathering factor
    # --- talus ---
    talus_bulking: float = 1.3
    talus_angle_deg: float = 35.0
    k_talus: float = 0.1  # m^2/day removed per m^2 of H_toe^2
    talus_dissip: float = 0.15  # 1/m of talus width, wave height reduction
    # --- cantilever failure ---
    rho_g: float = 22000.0  # N/m^3 unit weight of rock
    tensile_strength: float = 15000.0  # Pa, effective jointed rock-mass tension at strength 1
    rain_weakening: float = 0.3  # fractional strength loss at rain = 1
    wave_weakening: float = 0.2  # fractional strength loss at Hs >= hs_ref
    hs_ref: float = 4.0
    hazard_base: float = 0.02  # 1/day at sigma/T = 1
    hazard_kappa: float = 4.0
    min_overhang_cells: int = 1
    # --- revetment ---
    revetment_overtop_factor: float = 0.5  # H multiplier for face above crest


@dataclass(frozen=True)
class Geology:
    cliff_toe_x: float = 16.0  # m, initial toe position
    cliff_top_z: float = 14.0  # m
    toe_z_mean: float = -0.3  # platform elevation at the toe
    toe_z_jitter: float = 0.4
    platform_slope: float = 1.0 / 40.0
    face_slope: float = 0.08  # horizontal m per vertical m (near vertical, landward-leaning)
    face_roughness_cells: int = 1
    layer_thickness_range: tuple[float, float] = (0.8, 3.5)
    layer_strength_sigma: float = 0.4  # lognormal sigma of per-layer strength
    cell_strength_sigma: float = 0.1
    joint_spacing_mean: float = 1.5  # m
    joint_spacing_sigma: float = 0.5  # lognormal sigma


@dataclass(frozen=True)
class Climate:
    tidal_amp_range: tuple[float, float] = (1.0, 2.5)  # m, semi-range
    hs_mean: float = 1.0
    hs_seasonal: float = 0.4  # amplitude of seasonal cycle (winter peak)
    hs_ar: float = 0.8  # AR(1) coefficient of daily log-Hs noise
    hs_noise: float = 0.25
    storms_per_year: float = 8.0
    storm_peak_min: float = 2.5
    storm_peak_scale: float = 1.0  # exponential excess over min
    storm_duration_range: tuple[int, int] = (1, 4)
    surge_per_hs: float = 0.12  # m surge per m of storm Hs
    rain_mean: float = 0.25  # normalised daily rain index
    rain_storm_boost: float = 0.8
    slr_mm_per_year: float = 4.0
