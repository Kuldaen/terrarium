"""Phenomenon-agnostic interfaces.

Every phenomenon exposes a batched ground-truth simulator with the same shape
of contract, so the surrogate, fidelity gate and planner in ``core`` can be
reused across erosion, wildfire, hydrology, ...

Conventions
-----------
* Batch-first arrays: leading dim ``B`` indexes independent worlds.
* Randomness is an explicit input. ``forcing`` (exogenous drivers, e.g. storm
  sequences) and ``seeds`` (intrinsic noise, e.g. failure draws) fully
  determine a rollout given state and actions. Re-using them gives common
  random numbers across counterfactual actions.
* ``state`` is a dict of numpy arrays that can be saved and restored, so a
  planner can branch many futures from one start state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Rollout:
    """Result of running a simulator for some steps."""

    snapshots: dict[str, np.ndarray]  # [B, S, ...] state fields at snapshot times
    snapshot_steps: np.ndarray  # [S] step index of each snapshot
    series: dict[str, np.ndarray]  # [B, T] scalar time series
    events: dict[str, np.ndarray] = field(default_factory=dict)  # [B, T] sparse event data
    meta: dict[str, Any] = field(default_factory=dict)


class Simulator(Protocol):
    batch_size: int

    def get_state(self) -> dict[str, np.ndarray]: ...

    def set_state(self, state: dict[str, np.ndarray]) -> None: ...

    def set_actions(self, **actions: np.ndarray) -> None: ...

    def run(
        self,
        forcing: dict[str, np.ndarray],
        seeds: np.ndarray,
        snapshot_every: int,
    ) -> Rollout: ...
