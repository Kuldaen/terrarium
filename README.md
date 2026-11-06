# Terrarium

Calibrated, stochastic world models of slow physical phenomena — coastal
erosion first, then wildfire, hydrology and animal migration — so agents can
plan interventions by sampling many possible futures.

See [docs/design.md](docs/design.md) for the thesis and architecture.

| Codename | Phenomenon | Package |
|----------|------------|---------|
| **Seaglass** | coastal & cliff erosion | `terrarium.phenomena.erosion` |
| **Ignis** | wildfire | `terrarium.phenomena.wildfire` (planned) |

## Layout

```
src/terrarium/
  core/                     phenomenon-agnostic interfaces (surrogate, gate, planner to come)
  phenomena/erosion/cliff2d 2D cross-shore cliff simulator (NVIDIA Warp, CPU/CUDA)
experiments/erosion/        sanity checks and studies
tests/
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest
python experiments/erosion/cliff2d_sanity.py --batch 64 --years 5 --out outputs/sanity
```

```python
import numpy as np
from terrarium.phenomena.erosion.cliff2d import Cliff2D, generate_cliffs, generate_forcing

rng = np.random.default_rng(0)
init = generate_cliffs(256, rng)                 # geometry, layered strength, joints
forcing = generate_forcing(256, 5 * 365, rng)    # explicit storm/surge/rain/SLR arrays
sim = Cliff2D(init)                              # device="cuda" when available
sim.set_actions(revetment_crest=np.full(256, 3.0))
rollout = sim.run(forcing, seeds=np.arange(256), snapshot_every=30)
rollout.snapshots["rock"]   # [B, S, NZ, NX] float16
rollout.series["top_x"]     # [B, T] cliff-top edge position
rollout.events["fail_vol"]  # [B, T] collapse volumes
```

### Datasets and calibration

```bash
# training chains (monthly snapshots; any snapshot pair is a Δt example) + eval ensembles
python experiments/erosion/make_dataset.py --out data --chains 128 --starts 16 --futures 16
# score predictors' outcome distributions against the simulator per Δt
python experiments/erosion/calibration_report.py --train data/train --eval data/eval --out outputs/calib
```

Baseline results: [docs/results/calibration-baselines.md](docs/results/calibration-baselines.md).
A learned surrogate plugs in by implementing `predictors.Predictor`.

Randomness enters only through `forcing` and `seeds`, so the same storm
history can be replayed under different actions (common random numbers), and
`get_state` / `set_state` branch futures from any point.
