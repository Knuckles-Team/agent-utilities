"""CI-failure -> LLM re-cycling loop (G15,
``agent_utilities.observability.ci_recycle``) — the self-healing-CI piece of
the SDLC loop (``reports/autonomous-sdlc-loop-design.md`` §5.2), incorporating
the three Atomic Task Graph paper ideas
(``reports/paper-analysis-2607.01942.md``). Mirrors the fake-KG style of
``test_observability_lifecycle_orchestrator.py`` / ``test_observability_escalation_policy.py``.

@pytest.mark.concept("AU-OS.host.report-only-remediation-proposal")
"""

from __future__ import annotations

from typing import Any

import pytest

import agent_utilities.knowledge_graph.memory.native_ingest as native_ingest
import agent_utilities.observability.ci_recycle as cr
import agent_utilities.observability.health_ingest as hi

pytestmark = pytest.mark.concept("AU-OS.host.report-only-remediation-proposal")


class _Capture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, entities, relationships=None, *, source, domain, **kw):
        self.calls.append(
            {
                "entities": entities,
                "relationships": relationships or [],
                "source": source,
            }
        )
        return {"nodes": len(entities), "edges": len(relationships or [])}


class _FakeEngine:
    """Serves out_edges + get_nodes_by_label from tables (mirrors
    localized_repair's/lifecycle_orchestrator's fake-engine conventions)."""

    def __init__(
        self,
        edges: list[tuple[str, str, str]] | None = None,
        by_label: dict[str, list[tuple[str, dict]]] | None = None,
    ) -> None:
        self._edges = edges or []
        self._by_label = by_label or {}

    def out_edges(self, node_id: str, data: bool = False):
        return [(s, t, {"rel_type": r}) for (s, r, t) in self._edges if s == node_id]

    def get_nodes_by_label(self, label: str, limit: int = 0):
        return self._by_label.get(label, [])


# --- diagnosis --------------------------------------------------------------#
@pytest.mark.parametrize(
    "log,expected",
    [
        ("ruff check failed: 3 errors", "lint"),
        ("mypy: error: incompatible types", "typecheck"),
        ("FAILED tests/test_x.py::test_y - AssertionError", "test"),
        ("failed to build: Dockerfile:12", "build"),
        ("Error: The operation was canceled. timed out after 6000s", "timeout"),
        ("pre-commit hook id: ruff failed", "pre_commit"),
        ("some completely unrelated gibberish", "unknown"),
    ],
)
def test_diagnose_classifies_common_failure_logs(log, expected):
    assert cr._diagnose(log) == expected


# --- report-only propose_ci_repair ------------------------------------------#
def test_propose_ci_repair_is_report_only_and_diagnoses(monkeypatch):
    """A failed :PipelineRun -> a diagnosis + repair proposal, nothing dispatched
    (no CI re-trigger, no MR/PR opened — only a :CIRepairProposal is written)."""
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)
    engine = _FakeEngine()

    out = cr.propose_ci_repair(
        {"id": "pr1", "repo": "org/repo"},
        engine=engine,
        context={"log_excerpt": "ruff check failed", "repo": "org/repo"},
    )

    assert out["capped"] is False
    assert out["escalation"] is None
    proposal = out["proposal"]
    assert proposal["status"] == "proposed"
    assert proposal["failureClass"] == "lint"
    assert proposal["attempt"] == 1
    assert proposal["pipelineRun"] == "pr1"
    assert "develop_spec" in proposal["boundCapability"]
    assert proposal["modelTier"] == "small"  # well-scoped lint fix -> cheap tier
    # written as a :CIRepairProposal + hasRepairProposal edge, nothing else
    assert len(cap.calls) == 1
    assert {e["type"] for e in cap.calls[0]["entities"]} == {"CIRepairProposal"}
    assert {r["type"] for r in cap.calls[0]["relationships"]} == {"hasRepairProposal"}


def test_unknown_failure_class_routes_to_standard_tier(monkeypatch):
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    out = cr.propose_ci_repair(
        {"id": "pr2"}, engine=_FakeEngine(), context={"log_excerpt": "???"}
    )
    assert out["proposal"]["failureClass"] == "unknown"
    assert out["proposal"]["modelTier"] == "standard"


# --- paper idea #1: localized repair region ---------------------------------#
def test_repair_region_is_localized_from_the_failed_step(monkeypatch):
    """A CI job graph where the failing job (build) has downstream jobs (deploy)
    but an unrelated sibling job (docs) — only the failed job's descendants are
    invalidated."""
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    engine = _FakeEngine(
        edges=[
            ("build", "TRANSITION_TO", "deploy"),
            ("docs", "TRANSITION_TO", "publish_docs"),
        ]
    )
    out = cr.propose_ci_repair(
        {"id": "pr3"},
        engine=engine,
        context={
            "log_excerpt": "docker build failed",
            "failed_step": "build",
            "all_steps": ["build", "deploy", "docs", "publish_docs"],
        },
    )
    proposal = out["proposal"]
    assert proposal["invalidatedRegion"] == ["build", "deploy"]
    assert set(proposal["preservedRegion"]) == {"docs", "publish_docs"}


# --- paper idea #2: reuse-lookup ---------------------------------------------#
def test_a_second_equivalent_failure_reuses_the_first_repair(monkeypatch):
    """Two DIFFERENT pipeline runs on the same repo with the same failure class
    -> the second reuses the first's diagnosis instead of re-deriving it."""
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    first_proposal = {
        "id": "cirepair:pr1:aaa",
        "type": "CIRepairProposal",
        "pipelineRun": "pr1",
        "repo": "org/repo",
        "kind": "ci_repair",
        "fromStage": "pipeline_run",
        "toStage": "code_change",
        "transition": "lint",
        "targetType": "org/repo",
        "failureClass": "lint",
        "signature": "aaa",
        "status": "proposed",
    }
    engine = _FakeEngine(
        by_label={"CIRepairProposal": [(first_proposal["id"], first_proposal)]}
    )

    out = cr.propose_ci_repair(
        {"id": "pr2", "repo": "org/repo"},
        engine=engine,
        context={"log_excerpt": "ruff check failed", "repo": "org/repo"},
    )
    proposal = out["proposal"]
    assert proposal["status"] == "reused"
    assert proposal["reusedFrom"] == first_proposal["id"]
    assert proposal["reuseScore"] == 1.0


def test_different_repo_same_class_does_not_reuse(monkeypatch):
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    first_proposal = {
        "id": "cirepair:pr1:aaa",
        "kind": "ci_repair",
        "fromStage": "pipeline_run",
        "toStage": "code_change",
        "transition": "lint",
        "targetType": "org/other-repo",
        "failureClass": "lint",
    }
    engine = _FakeEngine(
        by_label={"CIRepairProposal": [(first_proposal["id"], first_proposal)]}
    )
    out = cr.propose_ci_repair(
        {"id": "pr2", "repo": "org/repo"},
        engine=engine,
        context={"log_excerpt": "ruff check failed", "repo": "org/repo"},
    )
    assert out["proposal"]["status"] == "proposed"
    assert "reusedFrom" not in out["proposal"]


# --- retry cap -> escalation -------------------------------------------------#
def test_retry_cap_exceeded_escalates_instead_of_proposing(monkeypatch):
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    prior = [
        (
            f"cirepair:pr1:{i}",
            {"pipelineRun": "pr1", "attempt": i, "failureClass": "test"},
        )
        for i in range(1, 4)  # 3 prior attempts already made (== default cap)
    ]
    engine = _FakeEngine(by_label={"CIRepairProposal": prior})

    out = cr.propose_ci_repair(
        {"id": "pr1", "repo": "org/repo"},
        engine=engine,
        context={"log_excerpt": "pytest failed"},
        retry_cap=3,
    )
    assert out["capped"] is True
    assert out["proposal"] is None
    assert out["escalation"] is not None
    assert "red_ci_past_cap" in out["escalation"]["signals"]


def test_below_retry_cap_keeps_proposing(monkeypatch):
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    prior = [("cirepair:pr1:1", {"pipelineRun": "pr1", "attempt": 1})]
    engine = _FakeEngine(by_label={"CIRepairProposal": prior})
    out = cr.propose_ci_repair(
        {"id": "pr1"}, engine=engine, context={"log_excerpt": "x"}, retry_cap=3
    )
    assert out["capped"] is False
    assert out["proposal"]["attempt"] == 2


# --- sweep -------------------------------------------------------------------#
def test_sweep_failed_pipelines_scans_and_proposes(monkeypatch):
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    engine = _FakeEngine(
        by_label={
            "PipelineRun": [
                ("pr_ok", {"status": "success"}),
                ("pr_bad", {"status": "failed", "repo": "org/repo"}),
            ],
            "CheckRun": [],
        }
    )
    out = cr.sweep_failed_pipelines(engine=engine)
    assert out["scanned"] == 1
    assert out["proposed"] == 1
    assert out["escalated"] == 0


# --- guards --------------------------------------------------------------- #
def test_no_engine_is_a_noop_for_sweep(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    out = cr.sweep_failed_pipelines()
    assert out == {"scanned": 0, "proposed": 0, "reused": 0, "escalated": 0}


def test_propose_ci_repair_never_writes_without_an_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(hi, "_engine", lambda: None)
    monkeypatch.setattr(
        native_ingest, "ingest_entities", lambda *a, **k: calls.append(1)
    )
    out = cr.propose_ci_repair({"id": "pr1"}, engine=None, context={"log_excerpt": "x"})
    assert out["proposal"] is not None  # still computed/returned (report-only)
    assert calls == []  # but never persisted without a reachable engine
