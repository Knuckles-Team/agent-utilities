"""Universal context plane + ops diagnosis + shared source paths.

CONCEPT:AU-KG.retrieval.route-question-its-domain (context plane), KG-2.137 (ops diagnosis), KG-2.135 (path norm).
"""

from __future__ import annotations

import pytest

from agent_utilities.core.source_paths import normalize_path, repo_of
from agent_utilities.knowledge_graph.retrieval import context_plane
from agent_utilities.knowledge_graph.retrieval.ops_context import diagnose_ops

_CANON = "/home/apps/workspace/agent-packages/agent-utilities/x.py"


# ── E1: shared source-path util ───────────────────────────────────────────────
@pytest.mark.concept("AU-KG.retrieval.every-usage-published-symbol")
def test_source_paths_normalize_and_repo_of():
    assert normalize_path("/au/agent_utilities/x.py") == _CANON.replace(
        "x.py", "agent_utilities/x.py"
    )
    assert normalize_path(_CANON) == _CANON
    assert repo_of(_CANON) == "agent-utilities"
    assert (
        repo_of("/home/apps/workspace/open-source-libraries/aider/m.py") == "oss/aider"
    )
    assert repo_of("") == "unknown"


# ── A3: ops diagnosis over live-shaped task data ──────────────────────────────
class FakeOpsEngine:
    def query_cypher(self, cypher, params):
        if "WHERE t.status IN ['pending','running']" in cypher:
            return [
                {"lane": "ingestion", "status": "pending", "n": 113},
                {"lane": "ingestion", "status": "running", "n": 3},
                {"lane": "maint", "status": "pending", "n": 175},
                {"lane": "maint", "status": "running", "n": 2},
            ]
        if "WHERE t.status IN ['failed','dead_letter']" in cypher:
            return [{"lane": "maint", "status": "dead_letter", "n": 250}]
        if "t.status = 'dead_letter'" in cypher:
            return [{"id": "task:abc", "lane": "maint", "tkind": "scheduled_job"}]
        if "count(t) AS n" in cypher:  # whole-queue status counts
            return [
                {"status": "pending", "n": 336},
                {"status": "running", "n": 6},
                {"status": "dead_letter", "n": 250},
                {"status": "failed", "n": 283},
                {"status": "completed", "n": 1340},
            ]
        return []


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
        def query_cypher(self, c, p):
            return []

    res = diagnose_ops(Empty(), query="health")
    assert res["status"] == "ok"
    assert "healthy" in res["answer"] or "0 pending" in res["answer"]


# ── A1: the context plane registry + routing ──────────────────────────────────
@pytest.mark.concept("AU-KG.retrieval.route-question-its-domain")
def test_infer_domain():
    assert context_plane.infer_domain("why is the maint lane backing up") == "ops"
    assert context_plane.infer_domain("how does run_agent work") == "code"
    assert context_plane.infer_domain("the task queue dead_letter backlog") == "ops"


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
