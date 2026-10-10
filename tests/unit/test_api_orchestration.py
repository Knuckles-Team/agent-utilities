"""Public API re-exports the orchestrator."""

from __future__ import annotations

import pytest

from agent_utilities.api import orchestration
from agent_utilities.orchestration import manager


@pytest.mark.spec("AU-BOUNDARY-R013.2")
def test_public_orchestrator_is_the_internal_definition() -> None:
    assert orchestration.Orchestrator is manager.Orchestrator
    assert orchestration.__all__ == ["Orchestrator"]
