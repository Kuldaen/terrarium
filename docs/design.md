# Terrarium — design notes

Terrarium builds fast, calibrated, stochastic world models of slow physical
phenomena (coastal erosion first; wildfire, hydrology and animal migration
later) so that agents can plan interventions by sampling many possible futures.

## Thesis

1. **Large-timestep prediction.** Classical simulators are bound to the
   fastest process (individual waves, fire-front spread). The quantity
   decision-makers care about (cliff-top position in 5 years) evolves on a
   slow manifold. A learned model conditioned on Δt can jump months–years per
   step.
2. **Many futures, not one.** Outcomes are path-dependent and stochastic
   (storm sequence, collapse timing). A deterministic regressor averages
   futures into an impossible mean (smooth notch, no blocks). The surrogate
   must be generative; cheap sampling turns one forecast into a distribution.
3. **Futures guide actions.** Agents choose interventions by evaluating the
   distribution of outcomes (expected cost + tail risk, e.g. CVaR), then
   replan as observations arrive.

## Architecture

```
core/                  shared across phenomena
  interfaces           Simulator / Forcing / Action protocols
  surrogate            Δt-conditioned generative model (later)
  gate                 stochastic fidelity gate (later)
  metrics              calibration: coverage, CRPS, event-size distributions
  planner              sample-based planning with common random numbers
phenomena/
  erosion/cliff2d      2D cross-section ground-truth simulator (Warp)
  erosion/cliff3d      voxel version (later)
  wildfire/ hydrology/ migration/
experiments/
```

### Design rules

- **Randomness is an explicit input.** Forcing (wave height, surge, rain) and
  failure noise seeds are arrays passed into the simulator. This gives common
  random numbers: two actions compared under identical storm histories.
- **Aleatoric vs epistemic split.** Storms are aleatoric (sampled forcing).
  Rock strength, joint spacing, initial notch depth are hidden parameters
  (epistemic) that surveys can reduce — information gathering becomes an action.
- **Actions are first-class inputs** (revetment crest, asset setback, …), so
  training data includes interventions.
- **Statistical evaluation.** Score retreat-distance distributions, volume
  loss, collapse size–frequency and coverage — not per-voxel MSE.

## Relation to PROWL-2

PROWL-2 (Odyssey) co-trains agents and a world model with a fidelity gate that
withholds start states the model mispredicts and a developer agent that seeks
mispredicted trajectories. Terrarium maps onto it directly:

| PROWL-2               | Terrarium                                             |
|-----------------------|-------------------------------------------------------|
| real environment      | ground-truth simulator (+ lidar surveys)              |
| developer agent       | spends simulator compute where the surrogate is wrong |
| fidelity gate         | **stochastic** gate (see below)                       |
| task curriculum       | start states ranked by learning potential (near-failure cliffs) |

Extensions Terrarium needs:

- **Stochastic gate.** Single-trajectory deviation is ill-posed for a
  stochastic model. Either replay with matched noise (same storm sequence), or
  test calibration: does the true trajectory fall inside the predicted
  ensemble at the right rate (coverage / CRPS across states)?
- **Δt-aware gate.** Trust is per (state, Δt): a model may be reliable for
  1-year jumps but not 5-year jumps from a given state.
- **Decision-relevant error.** Prioritise simulator calls where model error
  would change the agent's decision.

## Large-Δt training

- Condition the surrogate on Δt (flow-map / shortcut / consistency style).
- Self-consistency: two Δt/2 steps ≈ one Δt step in distribution; lets long
  jumps be trained mostly from short-step data.
- Forcing summaries per interval must preserve threshold information
  (exceedance hours, max storm), not just means.

## Simulation stack

NVIDIA Warp (Python kernels → CPU/CUDA), batched over independent cliffs,
zero-copy into PyTorch for online data generation during training. If
irregular collapse logic becomes the bottleneck in 3D, move that part to Rust
(PyO3) and keep dense updates in Warp.

## cliff2d — the first prototype

A batched 2D cross-shore cliff simulator. Cell state: rock fraction in
[0, 1] on an (NZ × NX) grid; per-cliff talus volume.

Processes per daily step:

1. **Tide + surge + SLR** give water levels (discrete tidal phases).
2. **Wave attack** on each row's seaward face cell. Incident Hs is
   depth-limited at the face (H ≤ γ·d) and dissipated across the shallow
   platform (exp(−α·W_shallow)) — wider platforms protect the cliff. Impact
   pressure peaks slightly above still water level. Erosion ∝ force / strength.
3. **Talus protection.** Fallen debris forms a wedge at the toe that shields
   the lower face and is removed by waves (∝ H²).
4. **Subaerial weathering** of the face above water: rain (boosted by the
   onshore wind component — wind-driven rain) plus **salt spray** ∝ (onshore
   wind)² · (0.5 + 0.5·Hs), decaying exponentially with height above the
   highest water. Wind-driven rain also lowers failure strength. Wind is a
   forcing series correlated with storms (speed ∝ Hs^0.6, storm days skew
   onshore). Not modelled: aeolian sand abrasion/talus deflation (relevant
   for sandy-beach and dune-backed coasts — a separate phenomenon).
5. **Cantilever failure.** For each notch, bending stress
   σ = 3ρgL²/h of the overhanging slab vs. effective tensile strength
   (reduced by rain and wave loading). Failure is a stochastic hazard
   λ = λ₀·exp(κ(σ/T − 1)) drawn with per-cliff seeds; the crack snaps to the
   nearest joint landward of the notch, giving variable block sizes.
6. **Actions.** Revetment crest elevation (protects the face below it,
   reduces overtopping wave height). Asset setback is evaluated from the
   recorded cliff-top edge.

Outputs: rock-fraction snapshots, talus, cliff-top and toe position time
series, failure events (step, volume, notch row).
