"""Tests for the graph_ops_causal MCP tool (Codex X-2).

CONCEPT:AU-KG.enrichment.ops-causal-graph

Mirrors the ``_CollectingMCP`` + ``kg_server._get_engine`` monkeypatch pattern
used across the other MCP tool-surface tests (e.g.
``tests/unit/test_engine_surface_tools.py``) — no live engine required.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.ops_causal_tools import register_ops_causal_tools
from tests.kg_recording_backend import RecordingGraphBackend


class _CollectingMCP:
    """Minimal FastMCP stand-in that captures ``@mcp.tool``-registered functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


@pytest.fixture
def tool(monkeypatch):
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    return mcp.tools["graph_ops_causal"]


_LINKS = json.dumps(
    [
        {"source": "commit:bad123", "target": "svc:checkout", "rel_type": "affects"},
        {
            "source": "commit:bad123",
            "target": "incident:INC001",
            "rel_type": "caused_incident",
        },
        {"source": "svc:checkout", "target": "agent:checkout", "rel_type": "part_of"},
        {"source": "agent:checkout", "target": "trace:1", "rel_type": "executed_by"},
        {"source": "policy:pci", "target": "svc:checkout", "rel_type": "governs"},
        {"source": "policy:pci", "target": "evidence:ev1", "rel_type": "has_evidence"},
    ]
)


def _call(tool_fn, **overrides):
    """Invoke the tool function with EVERY parameter explicit.

    Calling a ``@mcp.tool``-decorated function directly (bypassing the real
    FastMCP/pydantic validation layer) leaves any omitted parameter as its raw
    ``pydantic.Field(...)`` sentinel rather than the resolved default — the
    same caveat ``test_engine_surface_tools.py`` documents, so every call here
    supplies the full parameter set explicitly.
    """
    defaults = dict(
        action="root_cause",
        node_id="",
        links_json="[]",
        depth=6,
        max_results=10,
        incident_history_json="[]",
        now=0.0,
        materialize_claims=True,
        as_claim=False,
    )
    defaults.update(overrides)
    return tool_fn(**defaults)


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    assert "graph_ops_causal" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_ops_causal") is not None


def test_root_cause_action(tool):
    out = json.loads(_call(tool, action="root_cause", node_id="trace:1", links_json=_LINKS))
    assert out["surface"] == "ops_causal"
    assert out["action"] == "root_cause"
    result = out["result"]
    assert result[0]["node_id"] == "commit:bad123"
    assert result[0]["is_root"] is True


def test_blast_radius_action(tool):
    out = json.loads(
        _call(tool, action="blast_radius", node_id="commit:bad123", links_json=_LINKS)
    )
    ids = {r["node_id"] for r in out["result"]}
    assert "svc:checkout" in ids
    assert "trace:1" in ids
    assert "incident:INC001" in ids


def test_change_risk_action(tool):
    history = json.dumps([{"node_id": "incident:INC001", "severity": 0.9}])
    out = json.loads(
        _call(
            tool,
            action="change_risk",
            node_id="commit:bad123",
            links_json=_LINKS,
            incident_history_json=history,
        )
    )
    result = out["result"]
    assert result["node_id"] == "commit:bad123"
    assert result["historical_severity"] == pytest.approx(0.9)
    assert len(result["contributing_incidents"]) == 1


def test_control_evidence_action(tool):
    out = json.loads(
        _call(tool, action="control_evidence", node_id="policy:pci", links_json=_LINKS)
    )
    result = out["result"]
    assert "svc:checkout" in result["governs"]
    assert result["is_consistent"] is True


def test_missing_node_id_returns_error(tool):
    out = json.loads(_call(tool, action="root_cause", node_id="", links_json=_LINKS))
    assert "error" in out


def test_invalid_links_json_returns_error(tool):
    out = json.loads(
        _call(tool, action="root_cause", node_id="trace:1", links_json="not json")
    )
    assert "error" in out


def test_unknown_action_returns_error(tool):
    out = json.loads(
        _call(tool, action="not_a_real_action", node_id="trace:1", links_json=_LINKS)
    )
    assert "error" in out


def test_join_action_materializes_edges_via_engine_backend(monkeypatch):
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    backend = RecordingGraphBackend()

    class _FakeEngine:
        pass

    engine = _FakeEngine()
    engine.backend = backend
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool_fn = mcp.tools["graph_ops_causal"]
    out = json.loads(_call(tool_fn, action="join", links_json=_LINKS))
    assert out["result"]["nodes_written"] == 0
    assert out["result"]["edges_written"] == 6
    # NOTE: ``RecordingGraphBackend.edges`` tuples always carry ``None`` for the
    # relationship type here — the shared fake's edge-batch handler still reads
    # the retired ``"type"`` key, while ``write_entities`` (core/materialization.py)
    # now requires the canonical ``"relationship"`` key ("type"/"rel_type"/
    # "relationship_type"/"relation" are explicitly retired aliases that raise).
    # ``edge_props`` carries the full per-row dict (including "relationship"),
    # so assert on that instead of touching the shared test double.
    matches = [
        props
        for props in backend.edge_props
        if props.get("source") == "commit:bad123"
        and props.get("target") == "svc:checkout"
    ]
    assert matches and matches[0]["relationship"] == "affects"


def test_join_action_without_engine_backend_errors(tool):
    out = json.loads(_call(tool, action="join", links_json=_LINKS))
    assert "error" in out


# --------------------------------------------------------------------------- #
# W2 wire-up #3 — ops-causal -> claim loop (CONCEPT:AU-KG.enrichment.ops-causal-graph).
# Live-path proof: the root_cause ACTION itself (the EXISTING analysis entry
# point, not a new opt-in write call) persists the finding as a real,
# queryable :Claim through the SAME CandidateInsight -> register_claim_
# materialization pipeline the mining flywheel uses, as a side effect of the
# default materialize_claims=True.
# --------------------------------------------------------------------------- #


class _ClaimRecordingEngine:
    """Fakes the ``IntelligenceGraphEngine`` surface ``_materialize_root_cause_claims``
    calls: ``add_node`` (claim persist), ``add_edge`` (DERIVED_FROM provenance),
    ``register_materialization`` (TMS registration)."""

    def __init__(self) -> None:
        self.nodes: dict[str, tuple[str, dict]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.registered: list[str] = []
        self.backend = None

    def add_node(self, node_id, label, properties=None):  # noqa: ANN001
        self.nodes[node_id] = (label, dict(properties or {}))

    def add_edge(self, source, target, relationship_type=""):  # noqa: ANN001
        self.edges.append((source, target, relationship_type))

    def register_materialization(self, derived_id):  # noqa: ANN001
        self.registered.append(derived_id)
        return {"registered": True}


def test_root_cause_action_materializes_claims_by_default(monkeypatch):
    """The EXISTING root_cause call, with a live engine, writes real :Claim nodes."""
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    engine = _ClaimRecordingEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    tool_fn = mcp.tools["graph_ops_causal"]

    out = json.loads(
        _call(tool_fn, action="root_cause", node_id="trace:1", links_json=_LINKS)
    )
    assert out["action"] == "root_cause"
    # The root cause commit:bad123 clears the confidence floor (unweighted
    # score=1.0 for a direct causal edge) and is materialized.
    assert out["claims_materialized"], "expected at least one materialized claim"
    claim_id = out["claims_materialized"][0]

    # The Claim node actually landed on the engine, status=proposal, never
    # self-verified — the SAME propose-only floor the mining flywheel uses.
    label, props = engine.nodes[claim_id]
    assert label == "Claim"
    assert props["status"] == "proposal"
    assert props["is_verified"] is False
    assert "commit:bad123" in props["claim_text"]

    # DERIVED_FROM provenance to the real causal path + TMS registration ran.
    assert any(
        target == "commit:bad123" and rel == "DERIVED_FROM"
        for _src, target, rel in engine.edges
    )
    assert claim_id in engine.registered


def test_root_cause_action_materialize_claims_false_skips_write(monkeypatch):
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    engine = _ClaimRecordingEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    tool_fn = mcp.tools["graph_ops_causal"]

    out = json.loads(
        _call(
            tool_fn,
            action="root_cause",
            node_id="trace:1",
            links_json=_LINKS,
            materialize_claims=False,
        )
    )
    assert out["claims_materialized"] == []
    assert engine.nodes == {}


def test_blast_radius_action_never_materializes_claims(monkeypatch):
    """Only root_cause writes; the other read-only analyses stay pure reads."""
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    engine = _ClaimRecordingEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    tool_fn = mcp.tools["graph_ops_causal"]

    out = json.loads(
        _call(tool_fn, action="blast_radius", node_id="commit:bad123", links_json=_LINKS)
    )
    assert "claims_materialized" not in out
    assert engine.nodes == {}


# --------------------------------------------------------------------------- #
# W3.5 — ``as_claim`` opt-in: propose ONE finding through the SAME governed
# ClaimFlywheel lifecycle ``graph_claims propose`` uses, ActionPolicy-gated
# exactly like ``claim_tools._gate`` (CONCEPT:AU-KG.enrichment.ops-causal-graph,
# CONCEPT:AU-KG.evolution.mining-flywheel).
# --------------------------------------------------------------------------- #


def test_as_claim_false_leaves_response_byte_identical(monkeypatch):
    """Default (``as_claim=False``) behavior is unchanged: no claim_id/
    claim_transition/claim_denied/claim_error key appears for either action,
    even with a live engine present (i.e. NOT merely because as_claim's own
    ``engine is not None`` guard never fires)."""
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    engine = _ClaimRecordingEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    tool_fn = mcp.tools["graph_ops_causal"]

    root_cause_out = json.loads(
        _call(
            tool_fn,
            action="root_cause",
            node_id="trace:1",
            links_json=_LINKS,
            materialize_claims=False,
            as_claim=False,
        )
    )
    blast_radius_out = json.loads(
        _call(
            tool_fn,
            action="blast_radius",
            node_id="commit:bad123",
            links_json=_LINKS,
            as_claim=False,
        )
    )
    for out in (root_cause_out, blast_radius_out):
        for key in (
            "claim_id",
            "claim_transition",
            "claim_denied",
            "claim_error",
            "claim_write_errors",
        ):
            assert key not in out, f"{key} leaked into as_claim=False response"
    assert root_cause_out["result"][0]["node_id"] == "commit:bad123"
    assert engine.nodes == {}  # no write happened at all


def test_root_cause_as_claim_proposes_exactly_one_claim_with_evidence_ids(
    monkeypatch,
):
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    engine = _ClaimRecordingEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "agent_utilities.mcp.tools.ops_causal_tools._gate",
        lambda kind, target, reason: (
            True,
            {"decision": "allow", "tier": "auto", "reason": "test"},
        ),
    )
    tool_fn = mcp.tools["graph_ops_causal"]

    out = json.loads(
        _call(
            tool_fn,
            action="root_cause",
            node_id="trace:1",
            links_json=_LINKS,
            materialize_claims=False,  # isolate as_claim from the OLDER W2 path
            as_claim=True,
        )
    )
    assert out["claim_id"] == "claim:ops_causal:root_cause:trace:1"
    assert out["claim_transition"]["to_state"] == "proposed"
    assert out["claim_transition"]["from_state"] == ""

    # Exactly one Claim was written — never one-per-ranked-candidate like the
    # older materialize_claims path. (A second node, the flywheel's own
    # ClaimLifecycleEvent audit record, is expected — that's the lifecycle
    # trail itself, not a second claim.)
    claim_nodes = [nid for nid, (label, _p) in engine.nodes.items() if label == "Claim"]
    assert claim_nodes == ["claim:ops_causal:root_cause:trace:1"]
    label, props = engine.nodes["claim:ops_causal:root_cause:trace:1"]
    assert label == "Claim"
    assert props["status"] == "proposal"
    assert props["is_verified"] is False
    assert props["claim_type"] == "finding"
    assert "commit:bad123" in props["claim_text"]  # the structured one-liner

    # Evidence refs: every node id the root-cause causal walk actually
    # touched (the seed + every ranked candidate's own path — commit:bad123
    # AND policy:pci are both topological-source roots of trace:1 in
    # _LINKS), not just the top candidate's two endpoints.
    assert set(props["source_ids"]) == {
        "trace:1",
        "commit:bad123",
        "svc:checkout",
        "agent:checkout",
        "policy:pci",
    }
    # DERIVED_FROM provenance edges to each evidence id (register_claim_
    # materialization — the SAME shared seam every other claim producer uses).
    derived_targets = {t for _s, t, rel in engine.edges if rel == "DERIVED_FROM"}
    assert derived_targets == set(props["source_ids"])

    # Confidence = the top candidate's own path_strength (never the
    # tie-breaker `score`) — an unbroken, unweighted chain is 1.0.
    assert props["confidence"] == pytest.approx(1.0)

    # PROV-O generator tagging.
    assert props["metadata"]["finding_type"] == "OpsCausalFinding"
    assert props["metadata"]["ops_causal_action"] == "root_cause"
    assert props["metadata"]["was_generated_by"] == "mcp:graph_ops_causal"
    assert props["metadata"]["generated_at_time"]  # non-empty ISO timestamp


def test_blast_radius_as_claim_proposes_exactly_one_claim_with_evidence_ids(
    monkeypatch,
):
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    engine = _ClaimRecordingEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "agent_utilities.mcp.tools.ops_causal_tools._gate",
        lambda kind, target, reason: (
            True,
            {"decision": "allow", "tier": "auto", "reason": "test"},
        ),
    )
    tool_fn = mcp.tools["graph_ops_causal"]

    out = json.loads(
        _call(
            tool_fn,
            action="blast_radius",
            node_id="commit:bad123",
            links_json=_LINKS,
            as_claim=True,
        )
    )
    assert out["claim_id"] == "claim:ops_causal:blast_radius:commit:bad123"
    assert out["claim_transition"]["to_state"] == "proposed"

    label, props = engine.nodes["claim:ops_causal:blast_radius:commit:bad123"]
    assert label == "Claim"
    assert props["status"] == "proposal"
    assert set(props["source_ids"]) == {
        "commit:bad123",
        "svc:checkout",
        "incident:INC001",
        "agent:checkout",
        "trace:1",
    }
    # blast_radius has no per-node score — documented conservative default.
    assert props["confidence"] == pytest.approx(0.4)
    assert props["metadata"]["ops_causal_action"] == "blast_radius"
    assert props["metadata"]["was_generated_by"] == "mcp:graph_ops_causal"


def test_as_claim_denied_policy_returns_analysis_with_claim_denied_note(
    monkeypatch,
):
    """A denied ActionPolicy verdict must never block the read-only answer —
    it only adds a ``claim_denied`` note, and nothing is ever written."""
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    engine = _ClaimRecordingEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "agent_utilities.mcp.tools.ops_causal_tools._gate",
        lambda kind, target, reason: (
            False,
            {"decision": "queue_approval", "tier": "approval_required"},
        ),
    )
    tool_fn = mcp.tools["graph_ops_causal"]

    out = json.loads(
        _call(
            tool_fn,
            action="root_cause",
            node_id="trace:1",
            links_json=_LINKS,
            materialize_claims=False,
            as_claim=True,
        )
    )
    # The read-only analysis is fully intact.
    assert out["result"][0]["node_id"] == "commit:bad123"
    assert out["result"][0]["is_root"] is True
    # The denial is surfaced, never silently swallowed.
    assert out["claim_denied"]["decision"] == "queue_approval"
    # Nothing was proposed and nothing was written.
    assert "claim_id" not in out
    assert engine.nodes == {}


def test_as_claim_no_findings_proposes_nothing(monkeypatch):
    """as_claim=true on an analysis with no findings (no causal ancestors for
    this seed) never fabricates a claim."""
    mcp = _CollectingMCP()
    register_ops_causal_tools(mcp)
    engine = _ClaimRecordingEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    tool_fn = mcp.tools["graph_ops_causal"]

    # commit:bad123 is a topological source in _LINKS — it has no ancestors.
    out = json.loads(
        _call(
            tool_fn,
            action="root_cause",
            node_id="commit:bad123",
            links_json=_LINKS,
            materialize_claims=False,
            as_claim=True,
        )
    )
    assert out["result"] == []
    assert "claim_id" not in out
    assert "claim_denied" not in out
    assert engine.nodes == {}
