#!/usr/bin/python
"""Regression: MemoryEngine.consolidate()/compact_traces() must not ImportError.

Both methods previously imported non-existent modules (`.consolidation`,
`.memory_compaction`); they are now wired to the real SynthesisEngine (KG-2.1)
and MemoryHygiene (KG-2.17).

CONCEPT:KG-2.1
"""

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.memory.memory_engine import MemoryEngine
from agent_utilities.knowledge_graph.memory.optimization_engine import SynthesisProposal

pytestmark = pytest.mark.concept("KG-2.1")


def _engine():
    return IntelligenceGraphEngine(db_path=":memory:")


def test_consolidate_runs_and_returns_proposals_list():
    me = MemoryEngine(engine=_engine())
    proposals = me.consolidate(dry_run=True)
    # No ImportError; returns a list of SynthesisProposal (empty on a fresh graph).
    assert isinstance(proposals, list)
    assert all(isinstance(p, SynthesisProposal) for p in proposals)


def test_consolidate_handles_no_engine():
    assert MemoryEngine(engine=None).consolidate(dry_run=True) == []


def test_compact_traces_runs_and_returns_int():
    me = MemoryEngine(engine=_engine())
    n = me.compact_traces(agent_id="agent-1", threshold=10)
    # No ImportError; returns a non-negative count.
    assert isinstance(n, int)
    assert n >= 0


def test_compact_traces_handles_no_engine():
    assert MemoryEngine(engine=None).compact_traces() == 0
