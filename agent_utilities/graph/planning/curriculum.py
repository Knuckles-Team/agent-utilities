"""Horizon curriculum — unified planning surface (Plan 03 Step 4).

Re-exports the horizon curriculum scheduling logic from
:mod:`agent_utilities.graph.horizon_curriculum` so the unified
``graph/planning/`` package is the single planning entrypoint.

STRANGLER NOTE: ``horizon_curriculum`` remains the live implementation; these
re-exports are the *same* objects (identity preserved).
"""

from __future__ import annotations

from ..horizon_curriculum import (
    CurriculumStage,
    HorizonCurriculum,
    HorizonStageConfig,
    MacroAction,
    PromotionPolicy,
    SubgoalCheckpoint,
)

__all__ = [
    "CurriculumStage",
    "HorizonCurriculum",
    "HorizonStageConfig",
    "MacroAction",
    "PromotionPolicy",
    "SubgoalCheckpoint",
]
