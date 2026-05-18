"""CONCEPT:KG-2.4"""

import pytest

from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    FormalStateMachine,
)


def test_state_machine_valid_transition():
    fsm = FormalStateMachine("START")
    fsm.add_state("END")
    fsm.add_transition("START", "END", "finish")

    actions = fsm.get_available_actions()
    assert actions == ["finish"]

    new_state = fsm.transition("finish")
    assert new_state == "END"
    assert fsm.current_state == "END"


def test_state_machine_invalid_transition():
    fsm = FormalStateMachine("START")
    fsm.add_state("END")
    fsm.add_transition("START", "END", "finish")

    with pytest.raises(ValueError, match="No valid transition found"):
        fsm.transition("invalid_action")


def test_state_machine_conditions():
    fsm = FormalStateMachine("START")
    fsm.add_state("END")
    fsm.add_transition(
        "START", "END", "finish", condition=lambda ctx: ctx.get("is_done", False)
    )

    actions = fsm.get_available_actions({"is_done": False})
    assert not actions

    actions = fsm.get_available_actions({"is_done": True})
    assert actions == ["finish"]

    with pytest.raises(ValueError, match="No valid transition found"):
        fsm.transition("finish", {"is_done": False})

    fsm.transition("finish", {"is_done": True})
    assert fsm.current_state == "END"


def test_state_machine_invariants():
    fsm = FormalStateMachine("START")
    fsm.add_state("END")
    fsm.add_transition("START", "END", "finish")

    # Invariant: context value must be positive
    fsm.add_invariant(lambda state, ctx: ctx.get("value", 0) > 0)

    # Fails invariant
    with pytest.raises(ValueError, match="Invariant violation"):
        fsm.transition("finish", {"value": -1})

    assert fsm.current_state == "START"

    # Passes invariant
    fsm.transition("finish", {"value": 5})
    assert fsm.current_state == "END"
