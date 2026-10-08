"""Deletion proof: the topology-authority gate is clean on the tree and
catches every planted second authority."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_topology_authority", ROOT / "scripts" / "check_topology_authority.py"
)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gate
_SPEC.loader.exec_module(gate)

SELECTOR = "agent_utilities/graph/team_composer.py"
ELSEWHERE = "agent_utilities/graph/other.py"


def test_the_tree_holds_one_topology_authority() -> None:
    assert gate.check_tree(ROOT) == []


@pytest.mark.parametrize(
    ("source", "path", "rule"),
    [
        (
            'q = "MATCH (t:TopologyTemplate) SET t.success_rate = 1"',
            ELSEWHERE,
            "cypher-topology-write",
        ),
        (
            "def f(best):\n    return best.success_rate > 0.7",
            SELECTOR,
            "success-rate-selection",
        ),
        (
            'def f(best):\n    return best.get("success_rate")',
            SELECTOR,
            "success-rate-selection",
        ),
        (
            'def f(e, n):\n    e._upsert_node("SubagentPatternDecision", n, {})',
            ELSEWHERE,
            "pattern-outcome-store",
        ),
        (
            'def f(d, o):\n    return d.choose("au.swarm.topology", o, None)',
            ELSEWHERE,
            "second-topology-selector",
        ),
    ],
)
def test_each_planted_second_authority_is_named(
    source: str, path: str, rule: str
) -> None:
    assert [v.rule for v in gate.check_source(source, path)] == [rule]


def test_success_rate_outside_selection_modules_is_not_this_gate_s_business() -> None:
    assert gate.check_source("def f(t):\n    return t.success_rate", ELSEWHERE) == []
