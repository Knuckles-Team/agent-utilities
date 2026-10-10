"""Public API exports resolve to the canonical agent dispatch definitions."""

from __future__ import annotations

import pytest

from agent_utilities.api import agent_dispatch as public
from agent_utilities.orchestration import agent_dispatch as core


@pytest.mark.spec("AU-BOUNDARY-R013.4")
def test_public_agent_dispatch_exports_are_the_canonical_objects() -> None:
    assert public.__all__ == [
        "KIND_ORCHESTRATOR_TASK",
        "AgentTurnEnvelope",
        "enqueue_agent_turn",
    ]
    for name in public.__all__:
        assert getattr(public, name) is getattr(core, name)
