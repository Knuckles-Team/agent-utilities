"""Shared autouse fixture for the mandatory-marking seam.

``ContextCompiler.compile`` runs every candidate through the policy
``enforce`` gate, which resolves the mandatory-marking store on every call
regardless of whether a marking was ever applied -- so every test touching
it needs one installed. Centralized here since the identical fixture body
was independently defined in most of this directory's test modules.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.ontology.permissioning import (
    clear_markings,
    use_marking_authority,
)
from tests.retrieval.fakes import _FakeMarkingStore


@pytest.fixture(autouse=True)
def _clean_state():
    reset_company_brain()
    clear_markings()
    with use_marking_authority(_FakeMarkingStore()):
        yield
    reset_company_brain()
    clear_markings()
