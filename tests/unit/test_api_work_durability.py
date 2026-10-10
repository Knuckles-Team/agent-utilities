"""Public API exports resolve to the canonical work durability definitions."""

from __future__ import annotations

import pytest

from agent_utilities.api import work_durability as public
from agent_utilities.knowledge_graph.core import work_durability as core


@pytest.mark.spec("AU-BOUNDARY-R013.3")
def test_public_work_durability_exports_are_the_canonical_objects() -> None:
    assert public.__all__
    for name in public.__all__:
        assert getattr(public, name) is getattr(core, name)
