"""AU-SEMANTIC-R019.6.1: the reactive placement-mining tick is gone."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.core.schedule_engine import _MAINTENANCE_REF_ALLOWLIST

_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = (
    "agent_utilities/knowledge_graph/core/engine_tasks.py",
    "agent_utilities/core/schedule_engine.py",
)


@pytest.mark.spec("AU-SEMANTIC-R019.6.1")
@pytest.mark.parametrize("rel", _SOURCES)
def test_scheduler_sources_have_no_placement_mining_reference(rel: str) -> None:
    assert "placement_mining" not in (_ROOT / rel).read_text(encoding="utf-8")


@pytest.mark.spec("AU-SEMANTIC-R019.6.1")
def test_scheduled_job_set_omits_placement_mining_reactive() -> None:
    assert "placement_mining_reactive" not in _MAINTENANCE_REF_ALLOWLIST
