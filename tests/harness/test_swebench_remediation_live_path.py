"""CONCEPT:AHE-3.23 — SWE-failure-driven remediation: unresolved -> gap Concept -> regression gate."""

from __future__ import annotations

from agent_utilities.harness.swebench_corpus import SweBenchInstance
from agent_utilities.harness.swebench_harness import InstanceResult
from agent_utilities.harness.swebench_remediation import (
    build_failure_records,
    make_swebench_regression_gate,
    remediate,
)
from agent_utilities.runtime import DevWorkspace, LocalWorkspace
from agent_utilities.runtime.events import FileEditAction


class _RecordingEngine:
    """Honours the file_gap_topic contract: add_node + link_nodes."""

    def __init__(self):
        self.nodes: list[tuple[str, str, dict]] = []
        self.links: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes.append((node_id, node_type, properties or {}))

    def link_nodes(self, source_id, target_id, rel_type="", properties=None):
        self.links.append((source_id, target_id, rel_type))


def _results():
    return [
        InstanceResult("ok-1", "r/a", resolved=True),
        InstanceResult(
            "bad-1", "r/b", resolved=False, error="patch failed", trace_run_id="t1"
        ),
        InstanceResult("bad-2", "r/b", resolved=False, fail_to_pass_total=2),
    ]


def test_only_unresolved_become_failure_records():
    records = build_failure_records(_results())
    assert {r.name for r in records} == {"bad-1", "bad-2"}
    assert all(r.anomaly_type == "swebench_unresolved" for r in records)


def test_remediate_files_gap_concepts_for_unresolved():
    engine = _RecordingEngine()
    summary = remediate(_results(), engine, run_cycle=False)
    assert summary["unresolved"] == 2
    # one failure_gap Concept per distinct failure pattern
    gap_nodes = [n for n in engine.nodes if n[1] == "Concept"]
    assert len(gap_nodes) == summary["patterns"] >= 1
    assert all(props.get("kind") == "failure_gap" for _, _, props in gap_nodes)
    assert all(props.get("source") == "swebench" for _, _, props in gap_nodes)


def test_remediate_runs_golden_cycle_with_gap_topics():
    engine = _RecordingEngine()
    captured = {}

    class _Loop:
        def run_one_cycle(self, *, topics=None, max_topics=5):
            captured["topics"] = topics
            return {"ok": True, "topics_seen": len(topics or [])}

    summary = remediate(_results(), engine, golden_loop=_Loop(), run_cycle=True)
    assert summary["cycle"]["ok"] is True
    assert captured["topics"] == summary["gaps"]  # the filed gaps drive the cycle


async def test_regression_gate_reruns_exact_instance():
    inst = SweBenchInstance(
        instance_id="calc-1",
        repo="fixtures/calc",
        base_commit="HEAD",
        problem_statement="add returns a-b",
        fail_to_pass=["test_calc.py::test_add"],
        setup_commands=[
            "git init -q .",
            "printf 'def add(a, b):\\n    return a - b\\n' > calc.py",
            "printf 'from calc import add\\n\\ndef test_add():\\n    assert add(2,3)==5\\n' > test_calc.py",
        ],
    )

    async def fixing_solver(instance, workspace):
        await workspace.act(
            FileEditAction(path="calc.py", old="return a - b", new="return a + b")
        )
        return "patch"

    gate = make_swebench_regression_gate(
        inst,
        workspace_factory=lambda i: DevWorkspace(LocalWorkspace(), run_id="rg"),
        solve=fixing_solver,
    )
    assert await gate() is True  # the remediation makes the exact instance re-resolve
