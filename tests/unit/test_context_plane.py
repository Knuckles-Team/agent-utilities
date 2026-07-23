"""Universal context plane + ops diagnosis + shared source paths.

CONCEPT:AU-KG.retrieval.route-question-its-domain (context plane), KG-2.137 (ops diagnosis), KG-2.135 (path norm).
"""

from __future__ import annotations

import pytest

from agent_utilities.core.source_paths import normalize_path, repo_of
from agent_utilities.knowledge_graph.retrieval import context_plane
from agent_utilities.knowledge_graph.retrieval.ops_context import diagnose_ops

_CANON = "/home/agent-user/workspace/agent-packages/agent-utilities/x.py"


# ── E1: shared source-path util ───────────────────────────────────────────────
@pytest.mark.concept("AU-KG.retrieval.every-usage-published-symbol")
def test_source_paths_normalize_and_repo_of():
    # CONCEPT:AU-KG.retrieval.every-usage-published-symbol — the ``/au/`` mount alias
    # folds to a portable ``repo://`` URI (host-independent), not a homelab-specific
    # absolute path.
    assert (
        normalize_path("/au/agent_utilities/x.py")
        == "repo://agent-utilities/agent_utilities/x.py"
    )
    assert normalize_path(_CANON) == _CANON
    assert repo_of(_CANON) == "agent-utilities"
    assert (
        repo_of("/home/agent-user/workspace/open-source-libraries/aider/m.py")
        == "oss/aider"
    )
    assert repo_of("") == "unknown"


# ── A3: ops diagnosis over live-shaped task data ──────────────────────────────
class FakeOpsEngine:
    """Reproduces ``engine_tasks.py``'s ``list_tasks``/``lane_metrics`` shapes
    (CONCEPT:AU-KG.retrieval.ops-context migrated off raw Cypher onto these two typed
    engine primitives) with the same live-shaped numbers the old Cypher-query fake
    encoded: whole-queue status totals + a per-lane pending/running snapshot."""

    def list_tasks(self):
        def _jobs(n, status):
            return [{"job_id": f"{status}-{i}"} for i in range(n)]

        return {
            "pending": _jobs(336, "pending"),
            "running": _jobs(6, "running"),
            "dead_letter": _jobs(250, "dead_letter"),
            "failed": _jobs(283, "failed"),
            "completed": _jobs(1340, "completed"),
        }

    def lane_metrics(self):
        return {
            "ingestion": {"pending": 113, "running": 3},
            "maint": {"pending": 175, "running": 2},
        }


@pytest.mark.concept("AU-KG.retrieval.ops-context")
def test_diagnose_ops_health_flags_backing_up_lane():
    res = diagnose_ops(FakeOpsEngine(), query="", intent="health")
    assert res["status"] == "ok" and res["domain"] == "ops"
    assert "336 pending" in res["answer"] and "250 dead-lettered" in res["answer"]
    # maint: 175 pending vs 2 running -> backing-up signal
    sigs = res["sections"]["signals"]
    assert any(s.get("lane") == "maint" and s["kind"] == "backing_up" for s in sigs)
    assert any(s["kind"] == "dead_letter" for s in sigs)
    assert res["capability_id"] == "ops:health:queue"


@pytest.mark.concept("AU-KG.retrieval.ops-context")
def test_diagnose_ops_why_focuses_named_lane():
    res = diagnose_ops(
        FakeOpsEngine(), query="why is the maint lane backing up", intent="why"
    )
    assert "Lane 'maint'" in res["answer"]
    assert "graph-os-host restart" in res["answer"]  # remediation surfaced
    assert res["capability_id"] == "ops:why:maint"


@pytest.mark.concept("AU-KG.retrieval.ops-context")
def test_diagnose_ops_degrades_on_empty_engine():
    class Empty:
        def list_tasks(self):
            return {}

        def lane_metrics(self):
            return {}

    res = diagnose_ops(Empty(), query="health")
    assert res["status"] == "ok"
    assert "healthy" in res["answer"] or "0 pending" in res["answer"]


# ── A1: the context plane registry + routing ──────────────────────────────────
@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_infer_domain():
    assert context_plane.infer_domain("why is the maint lane backing up") == "ops"
    assert context_plane.infer_domain("how does run_agent work") == "code"
    assert context_plane.infer_domain("the task queue dead_letter backlog") == "ops"


# ── BUG-3 (kg-exhaustive-smoke.md): default is `entity`, NOT a blind `code` ───
@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_infer_domain_plain_concept_question_does_not_default_to_code():
    # The exact repro: a plain-English KG/concept question that hits none of the
    # ops/troubleshoot/code hints used to fall through to the hardcoded "code"
    # default and get a false "no code symbol matched" answer.
    domain = context_plane.infer_domain("What is the 1:1:1 traceability rule?")
    assert domain != "code"
    assert domain == "entity"


@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_infer_domain_still_routes_real_code_questions_to_code():
    # A code keyword still routes to `code` even with no snake_case identifier.
    assert context_plane.infer_domain("what does this function do") == "code"
    assert context_plane.infer_domain("show me the class definition") == "code"
    # A bare snake_case identifier is its own signal.
    assert context_plane.infer_domain("what is build_code_context for") == "code"


@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_synthesize_context_default_domain_routes_to_entity_not_code():
    class _EmptyEngine:
        def query_cypher(self, cypher, params):
            return []

    res = context_plane.synthesize_context(
        _EmptyEngine(), query="What is the 1:1:1 traceability rule?"
    )
    assert res["domain"] == "entity"
    assert res["status"] == "ok"


@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_list_context_domains_has_builtins():
    domains = {d["domain"] for d in context_plane.list_context_domains()}
    assert {"code", "ops"} <= domains


@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_synthesize_context_routes_to_ops_builtin():
    res = context_plane.synthesize_context(
        FakeOpsEngine(), domain="ops", query="health", intent="health"
    )
    assert res["domain"] == "ops" and res["status"] == "ok"
    assert "pending" in res["answer"]


@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_synthesize_context_infers_domain_from_query():
    res = context_plane.synthesize_context(
        FakeOpsEngine(), query="why is the maint lane backing up", intent="why"
    )
    assert res["domain"] == "ops"
    assert "maint" in res["answer"]


@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_synthesize_context_unknown_domain_errors():
    res = context_plane.synthesize_context(object(), domain="nonsense", query="x")
    assert res["status"] == "error"
    assert "code" in res["available_domains"] and "ops" in res["available_domains"]


@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_register_custom_provider_overrides(monkeypatch):
    def provider(engine, *, query, intent, **opts):
        return {"status": "ok", "answer": f"custom:{query}:{intent}"}

    monkeypatch.setitem(context_plane._PROVIDERS, "tickets", provider)
    res = context_plane.synthesize_context(
        None, domain="tickets", query="P1", intent="triage"
    )
    assert res["answer"] == "custom:P1:triage"
    assert res["domain"] == "tickets"  # plane fills domain
