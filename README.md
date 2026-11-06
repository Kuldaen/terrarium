# Terrarium

Calibrated, stochastic world models of slow physical phenomena — coastal
erosion first, then wildfire, hydrology and animal migration — so agents can
plan interventions by sampling many possible futures.

See [docs/design.md](docs/design.md) for the thesis and architecture.

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

Randomness enters only through `forcing` and `seeds`, so the same storm
history can be replayed under different actions (common random numbers), and
`get_state` / `set_state` branch futures from any point.
