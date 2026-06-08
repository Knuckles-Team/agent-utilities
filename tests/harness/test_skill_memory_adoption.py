#!/usr/bin/python
"""Skill evolution routes through the unified EvolvingMemoryStore (adoption).

CONCEPT:KG-2.1
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_utilities.harness.agentic_evolution_engine import AgenticEvolutionEngine
from agent_utilities.harness.evolving_memory import MemoryBank

pytestmark = pytest.mark.concept("KG-2.1")


def _engine():
    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()  # builds the EvolvingMemoryStore
    return eng


def test_create_skill_from_execution_records_to_store():
    eng = _engine()
    eng._skill_factory = MagicMock()
    eng._skill_factory.create_from_execution.return_value = SimpleNamespace(
        name="parse_json", description="safely parse JSON", id="sk1"
    )

    skill = eng.create_skill_from_execution("task", "result", success=True)
    assert skill.id == "sk1"  # original return preserved

    records = eng._memory_store.query(MemoryBank.SKILL)
    assert len(records) == 1
    assert "parse_json" in records[0].content
    assert records[0].metadata["source"] == "execution"


def test_merge_skills_records_survivor():
    eng = _engine()
    eng._skill_merger = MagicMock()
    eng._skill_merger.merge.return_value = SimpleNamespace(
        name="merged_skill", description="unified", id="m1"
    )

    merged = eng.merge_skills(object(), object())
    assert merged.id == "m1"
    recs = eng._memory_store.query(MemoryBank.SKILL)
    assert any(r.metadata["source"] == "merge" for r in recs)


def test_record_skill_is_noop_without_content():
    eng = _engine()
    eng._record_skill(SimpleNamespace(), "execution")  # no name/desc → skipped
    assert eng._memory_store.query(MemoryBank.SKILL) == []
