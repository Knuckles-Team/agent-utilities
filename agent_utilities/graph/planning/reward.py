"""Reward decomposition — unified planning surface (Plan 03 Step 4).

Re-exports the reward decomposition logic from
:mod:`agent_utilities.graph.reward_decomposition` so the unified
``graph/planning/`` package is the single planning entrypoint.

STRANGLER NOTE: ``reward_decomposition`` remains the live implementation; these
re-exports are the *same* objects (identity preserved).
"""

from __future__ import annotations

from ..reward_decomposition import (
    DecomposedRewardRecord,
    RewardDecomposer,
    StepOutcome,
    StepReward,
    TrajectoryOutcome,
    TrajectoryReward,
)

__all__ = [
    "DecomposedRewardRecord",
    "RewardDecomposer",
    "StepOutcome",
    "StepReward",
    "TrajectoryOutcome",
    "TrajectoryReward",
]
