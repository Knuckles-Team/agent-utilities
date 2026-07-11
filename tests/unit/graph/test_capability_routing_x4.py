"""X-4 — ontology-driven tool/agent routing: end-to-end tests.

Exercises the single top-level entry point,
:func:`~agent_utilities.graph.routing.enrichers.capability_routing.
route_capability_request`, against a fake engine (no live engine ANN — these
tests deliberately exercise the bounded in-process fallback path so the
ontology-subsumption-aware ``CapabilityIndex``/hierarchy plumbing is proven end
to end, not just mocked at the engine boundary). All four X-4 acceptance
scenarios from the task live here:

1. a request routes to a tool whose capability SUBSUMES the need, not just the
   ANN-nearest tool;
2. a policy/scope filter excludes an ineligible tool;
3. the durable bandit prefers a historically-better tool;
4. explainability returns the full eligibility feature set (subsumption path +
   policy + tenant + calibrated reward).
"""

from __future__ import annotations

import re
import types
from typing import Any

from agent_utilities.graph.routing.enrichers.capability_designation import (
    record_capability_outcome,
)
from agent_utilities.graph.routing.enrichers.capability_routing import (
    RoutingCandidate,
    explain_routing_eligibility,
    route_capability_request,
)
from agent_utilities.knowledge_graph.ontology.capability_hierarchy import (
    load_capability_hierarchy,
)

_HIERARCHY = load_capability_hierarchy()

_RETURN_ALIAS_RE = re.compile(r"n\.(\w+)\s+AS\s+(\w+)")
_SET_ASSIGN_RE = re.compile(r"n\.(\w+)\s*=\s*\$(\w+)")


class _SharedCypherBackend:
    """Executes SET/RETURN against the SAME dict the fake graph exposes as node
    properties — a durable write is immediately visible on the next property
    read, exactly like a real engine (see ``test_capability_descriptor.py``'s
    identically-named fixture rationale)."""

    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.nodes = nodes

    def execute(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        nid = str(params.get("id"))
        if "SET" in query:
            node = self.nodes.setdefault(nid, {})
            for prop, param in _SET_ASSIGN_RE.findall(query):
                node[prop] = params.get(param)
            return []
        node = self.nodes.get(nid, {})
        return [{alias: node.get(prop) for prop, alias in _RETURN_ALIAS_RE.findall(query)}]


def _make_engine(nodes: dict[str, dict[str, Any]]) -> Any:
    """A fake engine with node_ids/_get_node_properties (no query_unified/semantic_search
    — forces the in-process CapabilityIndex fallback path) and a durable backend
    sharing the SAME node store."""
    graph = types.SimpleNamespace(
        node_ids=lambda: list(nodes.keys()),
        _get_node_properties=lambda nid: nodes.get(nid, {}),
    )
    return types.SimpleNamespace(graph=graph, backend=_SharedCypherBackend(nodes))


# ---------------------------------------------------------------------------
# 1. Subsumption drives selection beyond raw ANN nearest-neighbour
# ---------------------------------------------------------------------------
def test_routes_to_the_subsuming_tool_not_the_ann_nearest_wrong_branch_tool():
    nodes = {
        # High cosine similarity to the query direction, but DNSCapability does
        # NOT subsume TransportCapability — must be excluded.
        "dns_tool": {
            "type": "tool",
            "embedding": [0.99, 0.02],
            "capabilities": ["DNSCapability"],
        },
        # Low cosine similarity, but EncryptedTransport ⊑ TransportCapability ⊑
        # ServiceCapability — the ontology-correct match for the request.
        "mtls_tool": {
            "type": "tool",
            "embedding": [0.1, 0.05],
            "capabilities": ["EncryptedTransport"],
        },
    }
    engine = _make_engine(nodes)

    out = route_capability_request(
        engine,
        "need a secure channel",
        required_capability_type="TransportCapability",
        k=5,
        embed_fn=lambda q: [1.0, 0.0],
        capability_hierarchy=_HIERARCHY,
    )

    assert [c.id for c in out] == ["mtls_tool"]
    assert out[0].eligibility["subsumption_paths"]["TransportCapability"] == [
        "EncryptedTransport",
        "TransportCapability",
    ]


def test_subsumption_is_on_by_default_without_an_explicit_hierarchy_argument():
    """``route_capability_request`` is the top-level X-4 entry point: subsumption
    is ON by default (the bundled ontology's singleton), unlike the lower-level
    primitives it composes (``CapabilityIndex``, ``engine_filtered_search``),
    which stay opt-in. Omitting ``capability_hierarchy`` entirely must still
    surface the subsuming tool — not silently degrade to flat exact match."""
    nodes = {
        "mtls_tool": {
            "type": "tool",
            "embedding": [1.0, 0.0],
            "capabilities": ["EncryptedTransport"],
        },
    }
    engine = _make_engine(nodes)
    out = route_capability_request(
        engine,
        "need a secure channel",
        required_capability_type="TransportCapability",
        k=5,
        embed_fn=lambda q: [1.0, 0.0],
    )
    assert [c.id for c in out] == ["mtls_tool"]


# ---------------------------------------------------------------------------
# 2. Policy/scope filter excludes an ineligible tool
# ---------------------------------------------------------------------------
def test_policy_scope_filter_excludes_an_ineligible_tool():
    nodes = {
        "cleared_tool": {
            "type": "tool",
            "embedding": [1.0, 0.0],
            "capabilities": ["DNSCapability"],
            "policy_tags": ["gpu_allowed"],
        },
        "uncleared_tool": {
            "type": "tool",
            "embedding": [0.99, 0.05],  # marginally closer isn't enough to qualify
            "capabilities": ["DNSCapability"],
        },
    }
    engine = _make_engine(nodes)

    out = route_capability_request(
        engine,
        "resolve a hostname",
        required_capability_type="DNSCapability",
        k=5,
        policy_tags=["gpu_allowed"],
        embed_fn=lambda q: [1.0, 0.0],
        capability_hierarchy=_HIERARCHY,
    )

    assert [c.id for c in out] == ["cleared_tool"]
    assert out[0].eligibility["policy_matched"] is True


def test_tenant_scope_filter_excludes_a_wrong_tenant_tool():
    nodes = {
        "tenant_a_tool": {
            "type": "tool",
            "embedding": [1.0, 0.0],
            "capabilities": ["DNSCapability"],
            "tenant": "tenant-a",
        },
        "tenant_b_tool": {
            "type": "tool",
            "embedding": [0.99, 0.02],
            "capabilities": ["DNSCapability"],
            "tenant": "tenant-b",
        },
    }
    engine = _make_engine(nodes)

    out = route_capability_request(
        engine,
        "resolve a hostname",
        required_capability_type="DNSCapability",
        k=5,
        tenant="tenant-a",
        embed_fn=lambda q: [1.0, 0.0],
        capability_hierarchy=_HIERARCHY,
    )

    assert [c.id for c in out] == ["tenant_a_tool"]


# ---------------------------------------------------------------------------
# 3. The durable bandit prefers a historically-better tool
# ---------------------------------------------------------------------------
def test_bandit_prefers_the_historically_better_tool_over_a_marginally_closer_one():
    nodes = {
        "tool_a": {
            "type": "tool",
            "embedding": [1.0, 0.0],  # slightly higher raw cosine to the query
            "capabilities": ["DNSCapability"],
        },
        "tool_b": {
            "type": "tool",
            "embedding": [0.995, 0.03],  # marginally lower cosine
            "capabilities": ["DNSCapability"],
        },
    }
    engine = _make_engine(nodes)
    query_embed = lambda q: [1.0, 0.0]  # noqa: E731

    # Before any outcomes: tool_a (higher raw cosine) ranks first.
    before = route_capability_request(
        engine,
        "resolve",
        required_capability_type="DNSCapability",
        k=5,
        embed_fn=query_embed,
        capability_hierarchy=_HIERARCHY,
    )
    assert [c.id for c in before] == ["tool_a", "tool_b"]

    # tool_b earns a strong, durable success history; tool_a fails repeatedly.
    for _ in range(6):
        record_capability_outcome(engine, "tool_b", success=True)
    for _ in range(6):
        record_capability_outcome(engine, "tool_a", success=False)

    after = route_capability_request(
        engine,
        "resolve",
        required_capability_type="DNSCapability",
        k=5,
        embed_fn=query_embed,
        capability_hierarchy=_HIERARCHY,
    )
    assert [c.id for c in after] == ["tool_b", "tool_a"]
    assert after[0].eligibility["reward"] > 0.5
    assert after[1].eligibility["reward"] < 0.5


# ---------------------------------------------------------------------------
# 4. Explainability returns the full eligibility feature set
# ---------------------------------------------------------------------------
def test_explain_routing_eligibility_reports_subsumption_policy_and_reward():
    nodes = {
        "mtls_tool": {
            "type": "tool",
            "embedding": [1.0, 0.0],
            "capabilities": ["EncryptedTransport"],
            "tenant": "tenant-a",
            "policy_tags": ["cleared"],
        },
    }
    engine = _make_engine(nodes)

    report = explain_routing_eligibility(
        engine,
        "mtls_tool",
        required_capability_type="TransportCapability",
        tenant="tenant-a",
        policy_tags=["cleared"],
        capability_hierarchy=_HIERARCHY,
    )

    assert report["eligible"] is True
    assert report["capabilities_matched"] is True
    assert report["missing_caps"] == []
    assert report["subsumption_paths"] == {
        "TransportCapability": ["EncryptedTransport", "TransportCapability"]
    }
    assert report["tenant_match"] is True
    assert report["policy_matched"] is True
    assert report["reward"] == 0.5
    assert "eligible" in report and isinstance(report["eligible"], bool)


def test_explain_routing_eligibility_reports_why_an_ineligible_candidate_fails():
    nodes = {
        "dns_tool": {
            "type": "tool",
            "embedding": [1.0, 0.0],
            "capabilities": ["DNSCapability"],
        },
    }
    engine = _make_engine(nodes)

    report = explain_routing_eligibility(
        engine,
        "dns_tool",
        required_capability_type="TransportCapability",
        capability_hierarchy=_HIERARCHY,
    )
    assert report["eligible"] is False
    assert report["missing_caps"] == ["TransportCapability"]
    assert report["subsumption_paths"] == {}


def test_explain_routing_eligibility_never_raises_for_an_unknown_entity():
    engine = _make_engine({})
    report = explain_routing_eligibility(
        engine,
        "no_such_tool",
        required_capability_type="DNSCapability",
        capability_hierarchy=_HIERARCHY,
    )
    assert report["eligible"] is False
    assert report["missing_caps"] == ["DNSCapability"]


# ---------------------------------------------------------------------------
# Misc — no matching embedding / no query embedding available
# ---------------------------------------------------------------------------
def test_route_capability_request_returns_empty_list_when_embedding_unavailable():
    engine = _make_engine({"a": {"type": "tool", "embedding": [1.0], "capabilities": ["x"]}})
    out = route_capability_request(
        engine, "q", required_capability_type="x", embed_fn=lambda q: None
    )
    assert out == []


def test_routing_candidate_is_a_plain_dataclass_with_id_score_eligibility():
    c = RoutingCandidate(id="tool:a", score=0.9, eligibility={"eligible": True})
    assert c.id == "tool:a"
    assert c.score == 0.9
    assert c.eligibility["eligible"] is True
