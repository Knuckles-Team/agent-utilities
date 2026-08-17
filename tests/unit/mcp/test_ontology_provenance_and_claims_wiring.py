"""BUG-266: wire ``classification_claims`` and ``repository_provenance`` to
both the gateway (REST) and MCP surfaces (CONCEPT:AU-KG.ontology.classification
-claim-multi-category / repository-provenance-snapshot).

``guardrail-surface-parity`` reported both ontology capability modules as
reachable from NEITHER surface — nothing imported them. Fixed by adding two
new focused tools, ``ontology_classification_claims`` and
``ontology_repository_provenance``, registered in
``register_ontology_tools`` (so they are import-reachable from the ``kg_server``
surface root) and given a REST twin via ``ACTION_TOOL_ROUTES`` (served by the
existing generic ``_make_tool_endpoint`` factory — no bespoke handler, so
there is exactly one action core per capability, never parallel handlers that
can drift, matching AGENTS.md "Two surfaces by default").

These tests exercise the REAL tools, registered by ``register_ontology_tools``
and dispatched through the REAL ``_execute_tool`` (the same dispatcher the
generic REST twin and every MCP client use) — not a mock of either — following
the same convention as ``test_ontology_catalogue_live_path.py``. The
engine-dependent write paths (``record_claim``/``ClassificationPromotionLedger``/
``ingest_graph_slice``) are already covered in isolation by
``test_classification_claims.py`` and ``test_repository_provenance.py``; what
these tests prove is the NEW seam — that the tool body reaches those real
functions with the right arguments, not a parallel reimplementation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import ontology_tools


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures every ``@mcp.tool``-registered function.

    ``register_ontology_tools`` ALSO assigns directly into the module-level
    ``kg_server.REGISTERED_TOOLS`` (the real dispatch table), so registering
    against this fake is enough to make ``_execute_tool`` reach the real tool.
    """

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        def _deco(fn):
            self.tools[kwargs.get("name", fn.__name__)] = fn
            return fn

        return _deco


@pytest.fixture
def registered() -> dict[str, object]:
    """Register the real ontology tools (populates kg_server.REGISTERED_TOOLS)."""
    mcp = _CollectingMCP()
    ontology_tools.register_ontology_tools(mcp)
    return mcp.tools


def test_both_new_tools_are_registered_on_the_real_dispatch_table(registered):
    assert "ontology_classification_claims" in registered
    assert "ontology_repository_provenance" in registered
    # Reachable from the SAME table _execute_tool (and the generic REST twin,
    # _make_tool_endpoint) both dispatch through — one action core, no drift.
    assert "ontology_classification_claims" in kg_server.REGISTERED_TOOLS
    assert "ontology_repository_provenance" in kg_server.REGISTERED_TOOLS


def test_both_tools_have_a_rest_twin_in_action_tool_routes():
    assert kg_server.ACTION_TOOL_ROUTES["ontology_classification_claims"] == (
        "/ontology/classification-claims"
    )
    assert kg_server.ACTION_TOOL_ROUTES["ontology_repository_provenance"] == (
        "/ontology/repository-provenance"
    )


# ── ontology_classification_claims: real record -> query -> promote chain ──


class _StubClaimsEngine:
    """Round-trips claim nodes through ``query_cypher`` — same convention as
    ``test_classification_claims.py``'s own ``_StubEngine``."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def link_nodes(self, source, target, rel_type, properties=None) -> None:
        pass

    def query_cypher(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        if "ClassificationClaimLifecycleEvent" in query:
            return []
        if "ClassificationClaim" in query and "Fragment" not in query:
            rows = [
                n for n in self.nodes.values() if n.get("type") == "ClassificationClaim"
            ]
            sid = params.get("sid")
            if sid is not None:
                rows = [r for r in rows if r.get("subject_id") == sid]
            status = params.get("status")
            if status is not None:
                rows = [r for r in rows if r.get("status") == status]
            return rows
        return []


async def _claims(**kwargs) -> dict:
    raw = await kg_server._execute_tool("ontology_classification_claims", **kwargs)
    return json.loads(raw)


async def test_record_then_query_reaches_the_real_record_and_query_claims(
    registered, monkeypatch
):
    """The tool must reach ``claim_from_raw`` + ``record_claim`` on 'record'
    and ``query_claims`` on 'query' — proven by a real claim surviving the
    round trip, not by asserting a mock was called."""
    engine = _StubClaimsEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    raw_claim = {
        "subject_id": "artifact:demo",
        "category": "security-critical",
        "value": "true",
        "evidence_refs": ["fragment:demo#1"],
        "method": "observed",
    }
    recorded = await _claims(
        action="record",
        claim_json=json.dumps(raw_claim),
        source_snapshot="commit:abc123",
    )
    assert recorded["status"] == "success"
    assert recorded["claim"]["subject_id"] == "artifact:demo"
    assert recorded["claim"]["status"] == "promoted"  # observed -> always promoted

    queried = await _claims(action="query", subject_id="artifact:demo")
    assert [c["subject_id"] for c in queried["claims"]] == ["artifact:demo"]

    categories = await _claims(action="categories", subject_id="artifact:demo")
    assert categories["categories"] == ["security-critical"]


async def test_promote_reaches_the_real_classification_promotion_ledger(
    registered, monkeypatch
):
    engine = _StubClaimsEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    raw_claim = {
        "subject_id": "artifact:candidate",
        "category": "ownership",
        "value": "team-kg",
        "evidence_refs": ["fragment:demo#2"],
        "method": "generated",
    }
    recorded = await _claims(
        action="record",
        claim_json=json.dumps(raw_claim),
        source_snapshot="commit:def456",
        policy_approved=True,
    )
    assert recorded["claim"]["status"] == "candidate"

    reviewed = await _claims(action="review", claim_json=json.dumps(recorded["claim"]))
    assert reviewed["claim"]["status"] == "reviewed"

    promoted = await _claims(
        action="promote",
        claim_json=json.dumps(reviewed["claim"]),
        reviewer="unit-test",
    )
    assert promoted["claim"]["status"] == "promoted"
    assert promoted["claim"]["reviewer"] == "unit-test"


async def test_claims_tool_fails_closed_with_no_active_engine(registered, monkeypatch):
    monkeypatch.setattr(
        kg_server, "_get_engine", lambda: (_ for _ in ()).throw(RuntimeError)
    )
    result = await _claims(action="query", subject_id="artifact:demo")
    assert result == {"status": "error", "error": "active engine required"}


# ── ontology_repository_provenance: real snapshot/branch/tag/change_event ──


async def _provenance(**kwargs) -> dict:
    raw = await kg_server._execute_tool("ontology_repository_provenance", **kwargs)
    return json.loads(raw)


async def test_snapshot_action_reaches_the_real_ingest_graph_slice(
    registered, monkeypatch
):
    """Proves the tool constructs a REAL ``RepositorySnapshot`` and hands its
    ``to_graph_slice()`` output to the ONE canonical envelope-ingest write
    path — never a second, parallel write route."""
    from agent_utilities.knowledge_graph.ingestion import envelope_ingest

    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    calls: list[dict[str, Any]] = []

    def _fake_ingest(engine, connector, entities, relationships=None, **kwargs):
        calls.append(
            {
                "connector": connector,
                "entities": entities,
                "relationships": relationships,
                **kwargs,
            }
        )
        return {"status": "success"}

    monkeypatch.setattr(envelope_ingest, "ingest_graph_slice", _fake_ingest)

    result = await _provenance(
        action="snapshot", repo_id="agent-utilities", commit_sha="deadbeef", ref="main"
    )
    assert result["status"] == "success"
    assert result["id"] == "repo_snapshot:agent-utilities:deadbeef"
    assert len(calls) == 1
    entities = calls[0]["entities"]
    assert entities[0]["node_type"] == "RepositorySnapshot"
    assert entities[0]["repo_id"] == "agent-utilities"
    assert entities[0]["commit_sha"] == "deadbeef"
    relationships = calls[0]["relationships"]
    assert relationships[0]["relationship"] == "SNAPSHOT_OF"
    assert relationships[0]["target"] == "commit:deadbeef"
    assert calls[0]["source_instance"] == "agent-utilities"


async def test_branch_action_reaches_the_real_ingest_graph_slice(
    registered, monkeypatch
):
    from agent_utilities.knowledge_graph.ingestion import envelope_ingest

    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        envelope_ingest,
        "ingest_graph_slice",
        lambda engine, connector, entities, relationships=None, **kw: (
            calls.append((connector, entities, relationships)) or {"status": "success"}
        ),
    )

    result = await _provenance(
        action="branch", repo_id="agent-utilities", name="main", commit_sha="cafefeed"
    )
    assert result["status"] == "success"
    connector, entities, relationships = calls[0]
    assert connector == "repository_provenance"
    assert entities[0]["node_type"] == "Branch"
    assert relationships[0]["relationship"] == "POINTS_AT"


async def test_provenance_tool_validates_required_fields_before_writing(
    registered, monkeypatch
):
    """A missing required field must fail closed with a clear error, never
    reach the write path with a half-built node."""
    from agent_utilities.knowledge_graph.ingestion import envelope_ingest

    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    called = False

    def _boom(*a, **k):
        nonlocal called
        called = True
        return {"status": "success"}

    monkeypatch.setattr(envelope_ingest, "ingest_graph_slice", _boom)

    result = await _provenance(action="snapshot", repo_id="agent-utilities")
    assert result["status"] == "error"
    assert not called


async def test_provenance_tool_fails_closed_with_no_active_engine(
    registered, monkeypatch
):
    monkeypatch.setattr(
        kg_server, "_get_engine", lambda: (_ for _ in ()).throw(RuntimeError)
    )
    result = await _provenance(action="snapshot", repo_id="x", commit_sha="y")
    assert result == {"status": "error", "error": "active engine required"}
