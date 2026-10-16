"""Harness-evolution SHACL gate (CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard) — the surpass over HarnessX.

These tests reproduce HarnessX's τ³-Bench Telecom failure (5 same-dimension edits
accumulating sub-threshold coupling) and show our SHACL gate **blocks** it — a
formal guarantee the paper's per-edit pass@2 gate explicitly lacks.

EH-431: ``HarnessGate`` validates through the engine's real
``shacl_validate_ad_hoc`` surface (see ``harness_gate.py``), never local
``pyshacl`` — these tests exercise the default (ambient-engine) construction
and require a real engine; they skip cleanly (via the standard AU
engine-availability check) when none is available in this environment.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.harness_gate import HarnessGate

pytestmark = pytest.mark.engine


@pytest.fixture()
def gate(engine_graph):
    """A gate bound to the real ephemeral engine (the shapes run in EG)."""
    return HarnessGate(engine=engine_graph)


def test_diversified_edits_pass(gate):
    # The same five edits spread across five dimensions: no concentration → ships.
    dims = ["D2_context", "D4_tools", "D7_control", "D3_memory", "D1_model"]
    edits = [
        {"id": f"edit:r{r}", "dimension": d, "round": r, "status": "shipped"}
        for r, d in zip(range(2, 7), dims, strict=True)
    ]
    assert gate.check_facts(edits).passed


def test_no_regression_seesaw_blocks_accepted_variant(gate):
    edits = [
        {
            "id": "edit:x",
            "dimension": "D4_tools",
            "round": 4,
            "status": "shipped",
            "regresses": ["task:solved_earlier"],
        }
    ]
    variants = [{"id": "variant:1", "status": "accepted", "applies": ["edit:x"]}]
    verdict = gate.check_facts(edits, variants=variants)
    assert not verdict.passed
    assert any("regression" in r.lower() for r in verdict.reasons), verdict.reasons


def test_reward_hacking_pathology_blocks(gate):
    edits = [{"id": "edit:rh", "dimension": "D6_eval", "round": 5, "status": "shipped"}]
    pathologies = [
        {"id": "path:rh", "kind": "reward_hacking", "exhibited_by": "edit:rh"}
    ]
    verdict = gate.check_facts(edits, pathologies=pathologies)
    assert not verdict.passed
    assert any("reward-hacking" in r.lower() for r in verdict.reasons), verdict.reasons


# ── Substitution-algebra type safety (CONCEPT:AU-KG.ontology.harness-gate) ──────────────────────
def test_hook_contract_blocks_modifying_read_only_hook(gate):
    # An edit that modifies a field at a read-only hook (task_end) violates the
    # hook contract the paper enforces at runtime — here a reasoned SHACL check.
    edits = [
        {
            "id": "edit:bad",
            "dimension": "D8_obs",
            "round": 2,
            "status": "shipped",
            "at_hook": "task_end",
            "modifies_field": "final_answer",
            "operation": "replace",
        }
    ]
    verdict = gate.check_facts(edits)
    assert not verdict.passed
    assert any("hook contract" in r.lower() for r in verdict.reasons), verdict.reasons


def test_edit_at_writable_hook_passes(gate):
    edits = [
        {
            "id": "edit:ok",
            "dimension": "D2_context",
            "round": 2,
            "status": "shipped",
            "at_hook": "task_start",
            "modifies_field": "system_prompt",
            "operation": "replace",
        }
    ]
    assert gate.check_facts(edits).passed


def test_singleton_exclusion_blocks_two_processors_same_group_and_hook(gate):
    edits = [{"id": "edit:p", "dimension": "D4_tools", "round": 2, "status": "shipped"}]
    processors = [
        {
            "id": "proc:a",
            "hook": "before_tool",
            "singleton_group": "approval",
            "status": "accepted",
        },
        {
            "id": "proc:b",
            "hook": "before_tool",
            "singleton_group": "approval",
            "status": "accepted",
        },
    ]
    verdict = gate.check_facts(edits, processors=processors)
    assert not verdict.passed
    assert any("singleton" in r.lower() for r in verdict.reasons), verdict.reasons


def test_singleton_distinct_groups_pass(gate):
    edits = [{"id": "edit:p", "dimension": "D4_tools", "round": 2, "status": "shipped"}]
    processors = [
        {
            "id": "proc:a",
            "hook": "before_tool",
            "singleton_group": "approval",
            "status": "accepted",
        },
        {
            "id": "proc:b",
            "hook": "before_tool",
            "singleton_group": "rate_limit",
            "status": "accepted",
        },
    ]
    assert gate.check_facts(edits, processors=processors).passed
