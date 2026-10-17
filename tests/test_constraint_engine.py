"""Hierarchical constraints: block, warn, escalate."""

from __future__ import annotations

import pytest

from agent_utilities.harness.constraint_engine import (
    ConstraintEngine,
    ConstraintLevel,
    HierarchicalConstraint,
)


def _engine(**kw) -> ConstraintEngine:
    engine = ConstraintEngine()
    engine.add_constraint(
        HierarchicalConstraint(id="c", description="Always validate inputs", **kw)
    )
    return engine


@pytest.mark.parametrize(
    ("level", "allowed", "violations"),
    [(ConstraintLevel.MIDDLEWARE, False, 1), (ConstraintLevel.PROMPT, True, 0)],
)
def test_tool_call_enforcement_by_level(level, allowed, violations):
    engine = _engine(
        current_level=level,
        applies_to=["delete_file"],
        condition="not_allowed",
        action="block",
    )
    ok, found = engine.check_tool_call("delete_file")
    assert ok is allowed and len(found) == violations
    assert all(v.auto_blocked for v in found)
    assert len(engine.get_all_constraints()) == 1


def test_escalation_manual_and_automatic():
    engine = _engine(current_level=ConstraintLevel.PROMPT, escalation_threshold=3)
    assert engine.escalate_constraint("c", "r") == ConstraintLevel.TOOL_DESCRIPTION
    assert engine.escalate_constraint("c", "r") == ConstraintLevel.MIDDLEWARE

    auto = _engine(current_level=ConstraintLevel.PROMPT, escalation_threshold=3)
    constraint = auto.get_constraint("c")
    assert constraint is not None
    constraint.violation_count = 5
    assert "c" in auto.auto_escalate_all()
    assert constraint.current_level == ConstraintLevel.TOOL_DESCRIPTION


def test_prompt_constraints_text_and_ordering():
    text = _engine(
        current_level=ConstraintLevel.PROMPT, applies_to=["process"]
    ).get_prompt_constraints()
    assert "c" in text and "validate inputs" in text
    assert (
        ConstraintLevel.PROMPT
        < ConstraintLevel.TOOL_DESCRIPTION
        < ConstraintLevel.MIDDLEWARE
        < ConstraintLevel.TOOL_IMPLEMENTATION
    )
