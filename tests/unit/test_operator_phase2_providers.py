"""Phase-2 context providers + connector coverage.

CONCEPT:KG-2.138 (deploy provider), KG-2.139 (entity provider), OS-5.48 (connectors).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_utilities.knowledge_graph.ingestion.connector_coverage import (
    assess_connector_coverage,
    enumerate_expected_connectors,
)
from agent_utilities.knowledge_graph.retrieval import context_plane
from agent_utilities.knowledge_graph.retrieval.deploy_context import deploy_status
from agent_utilities.knowledge_graph.retrieval.entity_context import entity_context

_NOW = datetime(2026, 6, 21, tzinfo=UTC)


# ── KG-2.138 deploy provider ──────────────────────────────────────────────────
class _NoRoutes:
    def query_cypher(self, c, p):
        return []


@pytest.mark.concept("KG-2.138")
def test_deploy_status_reports_canonical_and_restart_honesty():
    res = deploy_status(_NoRoutes(), query="run_agent", intent="status")
    assert res["status"] == "ok" and res["domain"] == "deploy"
    assert "Canonical checkout" in res["answer"]
    # the honest "loaded rev unknown → restart to guarantee live" line
    assert "restart" in res["answer"].lower()
    assert res["sections"]["canonical"][0]["root"]
    assert res["capability_id"].startswith("deploy:status:")


# ── KG-2.139 entity provider ──────────────────────────────────────────────────
class _CensusEngine:
    def query_cypher(self, cypher, params):
        if "labels(n)[0] AS label" in cypher:
            return [
                {"label": "Code", "n": 15535},
                {"label": "Document", "n": 40},
                {"label": "Task", "n": 2200},
            ]
        if "MATCH (n:Document)" in cypher:
            return [{"id": "doc:1", "name": "intro"}]
        return []


@pytest.mark.concept("KG-2.139")
def test_entity_health_census():
    res = entity_context(_CensusEngine(), query="", intent="health", domain="entity")
    assert "15535" in res["answer"] and "Code=15535" in res["answer"]
    assert res["capability_id"] == "entity:entity:health"


@pytest.mark.concept("KG-2.139")
def test_entity_focus_named_type():
    res = entity_context(
        _CensusEngine(), query="show Document nodes", intent="list", domain="entity"
    )
    assert "Document: 40" in res["answer"]
    assert any(c.get("name") == "intro" for c in res["citations"])


@pytest.mark.concept("KG-2.139")
def test_entity_unknown_domain_degrades():
    class Empty:
        def query_cypher(self, c, p):
            return []

    res = entity_context(
        Empty(), query="open tickets", intent="health", domain="tickets"
    )
    assert "no 'tickets' entities" in res["answer"]


@pytest.mark.concept("KG-2.139")
def test_plane_routes_enterprise_domains_to_entity():
    domains = {d["domain"] for d in context_plane.list_context_domains()}
    assert {"deploy", "entity", "tickets", "deploys", "process"} <= domains
    res = context_plane.synthesize_context(
        _CensusEngine(), domain="deploys", query="", intent="health"
    )
    # 'deploys' routed to the entity provider, which was told its domain
    assert res["domain"] == "deploys"


# ── OS-5.48 connector coverage ────────────────────────────────────────────────
@pytest.mark.concept("OS-5.48")
def test_enumerate_expected_connectors_includes_delta_handlers():
    expected = enumerate_expected_connectors()
    # the delta-handler set is always present (gitlab/jira/confluence/…)
    assert "gitlab" in expected and "jira" in expected
    assert "fleet" not in expected  # excluded


@pytest.mark.concept("OS-5.48")
def test_assess_connector_coverage_flags_dark_and_stale():
    expected = ["gitlab", "jira", "confluence"]
    fresh = {
        "gitlab": (_NOW - timedelta(days=1)).isoformat(),  # fresh
        "jira:host": (_NOW - timedelta(days=30)).isoformat(),  # stale
        # confluence absent -> dark
    }
    rep = assess_connector_coverage(expected, fresh, sla_days=7, now=_NOW)
    assert rep["covered"] == 2
    assert rep["missing"] == ["confluence"]
    assert [s["connector"] for s in rep["stale"]] == ["jira"]
    assert rep["coverage_pct"] == pytest.approx(66.7, abs=0.1)
