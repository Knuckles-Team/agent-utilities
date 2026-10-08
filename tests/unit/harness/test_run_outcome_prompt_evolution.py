"""Run outcomes feed propose-only prompt evolution (AU-HARNESS-R007).

CONCEPT:AU-AHE.optimization.run-outcome-prompt-evolution. An in-memory graph
double holds attributed ``RunTrace`` rows. A fake optimizer stands in for the
program optimizer, and one test drives the default native ``eg-program`` path
through a deterministic engine fake. No live graph, model or DSPy install.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent_utilities.harness import program_optimization as po
from agent_utilities.harness import run_outcome_prompt_evolution as roe
from agent_utilities.observability.trace_ontology import TRACE_CURSOR_NODE_LABEL
from agent_utilities.prompting.structured import StructuredPrompt
from agent_utilities.security.persistence_privacy import persistence_reference

AGENT = "deploy_agent"
_REF = "eg:{ns}:" + "a" * 64


def _compiled_state(optimizer: str = "bootstrap_few_shot") -> dict[str, Any]:
    return {
        "id": _REF.format(ns="candidate"),
        "program_ref": _REF.format(ns="program"),
        "optimizer": optimizer,
        "execution": "native_kernel",
        "candidate_role": "proposal",
        "demonstration_refs": [_REF.format(ns="example")],
        "artifact_refs": [],
        "composition_refs": [],
        "instruction_ref": None,
        "tool_policy_ref": None,
        "model_profile_ref": None,
        "evidence_refs": [_REF.format(ns="evaluation")],
        "source_refs": [_REF.format(ns="corpus")],
        "proof_ids": [],
        "contradiction_ids": [],
        "modalities": ["text"],
    }


@dataclass
class _Result:
    compiled_state: dict[str, Any]
    confidence: float = 0.75
    optimizer: str = "eg-program"


@dataclass
class _FakeOptimizer:
    result: _Result | None = field(default_factory=lambda: _Result(_compiled_state()))
    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def __call__(self, artifact, trainset, *, engine):
        assert artifact["task"] == AGENT
        self.calls.append(list(trainset))
        return self.result


class _Graph:
    """Minimal graph double: RunTrace rows, cursor checkpoints, version nodes."""

    def __init__(self, traces: list[dict[str, Any]]) -> None:
        self.traces = traces
        self.nodes: dict[str, tuple[str, dict[str, Any]]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.fail_add_node = False

    def query_cypher(self, cypher: str, params: dict[str, Any] | None = None):
        params = params or {}
        if TRACE_CURSOR_NODE_LABEL in cypher:
            return self._label_rows(
                TRACE_CURSOR_NODE_LABEL, consumer_ref=params["consumer_ref"]
            )
        if "prompt_version" in cypher:
            return self._label_rows(
                "prompt_version",
                prompt_id=params["prompt_id"],
                status="proposal",
                origin=params["origin"],
            )
        if "RunTrace" in cypher:
            return [
                row
                for row in self.traces
                if row["attribution_ref"] == params.get("agent_ref")
                and row["event_sequence"] > params["after_sequence"]
            ]
        return []

    def _label_rows(self, label: str, **match: Any) -> list[dict[str, Any]]:
        return [
            {"id": node_id, **props}
            for node_id, (node_label, props) in self.nodes.items()
            if node_label == label and all(props.get(k) == v for k, v in match.items())
        ]

    def add_node(self, node_id: str, label: str, properties: dict[str, Any]) -> None:
        if self.fail_add_node and label == "prompt_version":
            raise RuntimeError("graph write refused")
        self.nodes[node_id] = (label, dict(properties))

    def add_edge(self, source: str, target: str, rel_type: str) -> None:
        self.edges.append((source, target, rel_type))


def _trace(index: int, *, success: bool, agent: str = AGENT) -> dict[str, Any]:
    return {
        "id": f"trace:{index:04d}",
        "attribution_ref": persistence_reference(
            "agent", agent, namespace="execution-trace"
        ),
        "status": "completed" if success else "failed",
        "task_digest": f"digest-task-{index}",
        "result_digest": f"digest-result-{index}",
        "reward": None,
        "event_sequence": 1000 + index,
    }


def _traces(total: int, failures: int) -> list[dict[str, Any]]:
    return [_trace(i, success=i >= failures) for i in range(total)]


@pytest.fixture
def prompt_path(tmp_path: Path) -> Path:
    path = tmp_path / f"{AGENT}.json"
    path.write_text(
        json.dumps(
            {
                "task": AGENT,
                "type": "prompt",
                "prompt_version": "0.1.0",
                "instructions": {"core_directive": "Deploy services safely."},
            }
        ),
        encoding="utf-8",
    )
    return path


def _versions(graph: _Graph) -> list[tuple[str, dict[str, Any]]]:
    return [
        (node_id, props)
        for node_id, (label, props) in graph.nodes.items()
        if label == "prompt_version"
    ]


def test_proposes_candidate_with_trace_provenance(prompt_path: Path) -> None:
    graph = _Graph(_traces(10, failures=3))
    optimizer = _FakeOptimizer()
    before = prompt_path.read_bytes()

    report = roe.propose_prompt_from_run_outcomes(
        graph, AGENT, prompt_path, optimizer=optimizer
    )

    assert report["status"] == "proposed"
    assert report["trace_count"] == 10
    [(version_id, props)] = _versions(graph)
    baseline = StructuredPrompt.load(prompt_path)
    assert props["status"] == "proposal"
    assert props["origin"] == roe.PROPOSAL_ORIGIN
    assert props["prompt_id"] == AGENT
    assert props["parent_hash"] == baseline.version_hash()
    assert props["version_hash"] != baseline.version_hash()
    assert props["task_count"] == 10
    assert json.loads(props["program_compiled_state_json"]) == _compiled_state()
    derived = {(s, t) for s, t, rel in graph.edges if rel == "was_derived_from"}
    assert derived == {(version_id, f"trace:{i:04d}") for i in range(10)}
    # Propose-only: the live prompt file is untouched.
    assert prompt_path.read_bytes() == before
    # The optimizer sees reference-only rows; failures carry no response.
    [rows] = optimizer.calls
    failed = [row for row in rows if not row["success"]]
    assert len(failed) == 3 and all(row["response"] == "" for row in failed)
    assert all(row["trace_ref"].startswith("trace:") for row in rows)


def test_pending_candidate_blocks_a_second_proposal(prompt_path: Path) -> None:
    graph = _Graph(_traces(10, failures=3))
    roe.propose_prompt_from_run_outcomes(
        graph, AGENT, prompt_path, optimizer=_FakeOptimizer()
    )
    graph.traces.extend(_trace(100 + i, success=False) for i in range(10))
    second = _FakeOptimizer()

    report = roe.propose_prompt_from_run_outcomes(
        graph, AGENT, prompt_path, optimizer=second
    )

    assert report["status"] == "pending_review"
    assert second.calls == []
    assert len(_versions(graph)) == 1


def test_cursor_limits_the_next_pass_to_new_outcomes(prompt_path: Path) -> None:
    graph = _Graph(_traces(10, failures=3))
    roe.propose_prompt_from_run_outcomes(
        graph, AGENT, prompt_path, optimizer=_FakeOptimizer()
    )
    # A reviewer rejects the candidate; old traces stay below the cursor.
    for _label, props in graph.nodes.values():
        if props.get("origin") == roe.PROPOSAL_ORIGIN:
            props["status"] = "rejected"
    optimizer = _FakeOptimizer()

    report = roe.propose_prompt_from_run_outcomes(
        graph, AGENT, prompt_path, optimizer=optimizer
    )

    assert report["status"] == "no_data"
    assert optimizer.calls == []


@pytest.mark.parametrize(
    ("traces", "expected"),
    [
        (_traces(4, failures=2), "no_data"),
        (_traces(10, failures=0), "no_failures"),
        (
            [_trace(i, success=False, agent="other_agent") for i in range(10)],
            "no_data",
        ),
    ],
)
def test_idle_passes_do_not_call_the_optimizer(
    prompt_path: Path, traces: list[dict[str, Any]], expected: str
) -> None:
    graph = _Graph(traces)
    optimizer = _FakeOptimizer()

    report = roe.propose_prompt_from_run_outcomes(
        graph, AGENT, prompt_path, optimizer=optimizer
    )

    assert report["status"] == expected
    assert optimizer.calls == []
    assert _versions(graph) == []


def test_outcome_reward_overrides_trace_status(prompt_path: Path) -> None:
    traces = _traces(10, failures=0)
    traces[0]["reward"] = 0.1  # completed run, but the evaluator scored it low
    graph = _Graph(traces)
    optimizer = _FakeOptimizer()

    report = roe.propose_prompt_from_run_outcomes(
        graph, AGENT, prompt_path, optimizer=optimizer
    )

    assert report["status"] == "proposed"
    assert sum(1 for row in optimizer.calls[0] if not row["success"]) == 1


@pytest.mark.parametrize("persist_fails", [False, True])
def test_failed_pass_keeps_the_cursor(prompt_path: Path, persist_fails: bool) -> None:
    graph = _Graph(_traces(10, failures=3))
    graph.fail_add_node = persist_fails
    optimizer = _FakeOptimizer(result=_Result(_compiled_state()))
    if not persist_fails:
        optimizer.result = None

    report = roe.propose_prompt_from_run_outcomes(
        graph, AGENT, prompt_path, optimizer=optimizer
    )

    assert report["status"] == "error"
    cursor_nodes = [
        n for n, (label, _p) in graph.nodes.items() if label == TRACE_CURSOR_NODE_LABEL
    ]
    assert cursor_nodes == []
    assert _versions(graph) == []


def test_sweep_without_graph_authority_is_idle(prompt_path: Path) -> None:
    assert roe.run_prompt_evolution_sweep(None)["status"] == "no_data"
    assert roe.run_prompt_evolution_sweep(object())["status"] == "no_data"


def test_sweep_reports_per_agent_and_isolates_failures(
    prompt_path: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    graph = _Graph(
        _traces(10, failures=3)
        + [_trace(200 + i, success=False, agent="broken") for i in range(10)]
    )

    report = roe.run_prompt_evolution_sweep(
        graph,
        targets={AGENT: prompt_path, "broken": broken},
        optimizer=_FakeOptimizer(),
    )

    assert report["status"] == "proposed"
    assert report["agents"][AGENT]["status"] == "proposed"
    assert report["agents"]["broken"]["status"] == "error"


def test_packaged_prompt_targets_lists_base_prompts() -> None:
    targets = roe.packaged_prompt_targets()
    assert "architect" in targets
    assert all(path.suffix == ".json" for path in targets.values())


class _NativeProgramGraph(_Graph):
    """Graph double that also serves the native ``eg-program`` job."""

    def optimize_program(self, request: dict[str, Any]) -> dict[str, Any]:
        examples = request["corpus"]["examples"]
        row = {
            **_compiled_state(request["optimizer"]),
            "id": _REF.format(ns="candidate"),
            "program_ref": request["program"]["program_ref"],
            "demonstration_refs": [
                e["example_ref"] for e in examples if e["outcome"] == "success"
            ],
            "instruction_ref": request["program"]["signature"]["instruction_ref"],
            "evidence_refs": request["baseline"]["evidence_refs"],
            "source_refs": [request["corpus"]["corpus_ref"]],
            "kind": "program_candidate",
            "confidence": 0.9,
            "selected": True,
        }
        return {"status": "proposed", "result": {"rows": [row]}}


def test_default_optimizer_runs_the_native_program_job(prompt_path: Path) -> None:
    graph = _NativeProgramGraph(_traces(10, failures=3))

    report = roe.propose_prompt_from_run_outcomes(graph, AGENT, prompt_path)

    assert report["status"] == "proposed"
    [(_version_id, props)] = _versions(graph)
    state = json.loads(props["program_compiled_state_json"])
    assert state["optimizer"] == "bootstrap_few_shot"
    assert len(state["demonstration_refs"]) == 7
    assert props["reward"] == pytest.approx(0.9)


def test_optimization_sweep_schedules_prompt_evolution(monkeypatch) -> None:
    calls: list[Any] = []

    def fake_sweep(engine):
        calls.append(engine)
        return {"status": "proposed", "agents": {}}

    monkeypatch.setattr(roe, "run_prompt_evolution_sweep", fake_sweep)
    po.reset_target_backoff()
    sentinel = object()

    report = po.run_optimization_sweep(sentinel, targets=["prompt_evolution"])

    assert "prompt_evolution" in po.SCHEDULABLE_TARGETS
    assert calls == [sentinel]
    assert report["optimized"] == ["prompt_evolution"]
    assert report["propose_only"] is True
