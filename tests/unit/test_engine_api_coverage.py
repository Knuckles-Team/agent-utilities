"""Coverage contract: every epistemic-graph engine domain is reachable as BOTH an
MCP tool and a REST route, and dispatch reaches the engine client (CONCEPT:AU-ECO.mcp.full-api-mcp-surface).

agent-utilities is the native API/MCP layer for the engine. These tests assert the
full low-level surface (the 19 client sub-clients) is exposed in lockstep on both
surfaces and that a representative method in each domain actually dispatches into
the ``epistemic_graph`` client — without needing a live engine (the client is
monkeypatched).
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools

# A representative, JSON-friendly method per engine domain (method name + sample
# kwargs). Picks one method that exercises the dispatch for every sub-client.
_REPRESENTATIVE: dict[str, tuple[str, dict]] = {
    "nodes": ("has", {"node_id": "n1"}),
    "edges": ("has", {"source_id": "a", "target_id": "b"}),
    "graph": ("topological_sort", {}),
    # Bounded call — the unbounded-analytics OOM guard in engine_tools._dispatch
    # rejects the whole-graph 'pagerank'/'betweenness_centrality'/
    # 'degree_centrality_all' (and an unseeded 'personalized_pagerank') fail-loud
    # before dispatch, so the representative smoke test for this domain must use
    # a bounded call.
    "analytics": (
        "personalized_pagerank",
        {"seed_nodes": [["n1", 1.0]], "damping": 0.85, "iterations": 10},
    ),
    "lifecycle": ("metrics", {}),
    "reasoning": ("reason", {}),
    "ledger": ("get", {}),
    "channels": ("list", {}),
    "tenants": ("list", {}),
    "resharding": ("catalog_list", {}),
    "consensus": (
        "register_identity",
        {"agent_id": "a", "role": "Worker", "teams": [], "signature": "s"},
    ),
    "finance": ("var", {"returns": [0.1, -0.2, 0.05], "confidence": 0.95}),
    "datascience": ("compute_stats", {"data": [[1.0, 2.0], [3.0, 4.0]]}),
    "query": ("sql", {"query": "SELECT 1"}),
    "txn": ("begin", {}),
    "timeseries": ("range", {"series_id": "s", "from_": 0, "to": 10}),
    "rdf": ("sparql", {"query": "SELECT * WHERE {?s ?p ?o}"}),
    "streaming": ("list_triggers", {"graph": "g"}),
    "blob": ("gc", {}),
}


def test_every_engine_domain_has_mcp_tool_and_rest_route():
    kg_server.ensure_tools_registered()
    assert engine_tools.ENGINE_DOMAINS, "engine surface should be discovered"
    for domain in engine_tools.ENGINE_DOMAINS:
        tool = f"engine_{domain}"
        assert tool in kg_server.REGISTERED_TOOLS, f"{tool} missing MCP tool"
        assert tool in kg_server.ACTION_TOOL_ROUTES, f"{tool} missing REST route"
        assert kg_server.ACTION_TOOL_ROUTES[tool] == f"/engine/{domain}"


def test_engine_routes_are_mounted():
    kg_server.ensure_tools_registered()

    class _App:
        def __init__(self) -> None:
            self.paths: set[str] = set()

        def add_route(self, path, handler, methods=None):  # noqa: ANN001
            self.paths.add(path)

    app = _App()
    kg_server._mount_rest_routes(app)
    for domain in engine_tools.ENGINE_DOMAINS:
        assert f"/engine/{domain}" in app.paths


def test_every_verbose_engine_op_exists_in_manifest():
    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS

    by_tool: dict[str, set[str]] = {}
    for op in GRAPHOS_ACTIONS:
        if op["tool"].startswith("engine_"):
            by_tool.setdefault(op["tool"], set()).add(op["action"])
    for domain, methods in engine_tools.ENGINE_DOMAINS.items():
        assert by_tool.get(f"engine_{domain}", set()) == set(methods), (
            f"verbose manifest drift for engine_{domain}"
        )


@pytest.mark.parametrize("domain", list(_REPRESENTATIVE))
def test_representative_method_dispatches_to_client(monkeypatch, domain):
    """The tool dispatches the action+params into the matching client sub-client."""
    kg_server.ensure_tools_registered()
    method, params = _REPRESENTATIVE[domain]
    assert method in engine_tools.ENGINE_DOMAINS[domain], (
        f"representative {domain}.{method} not in discovered surface"
    )

    calls: list[tuple[str, str, dict]] = []

    class _FakeSub:
        def __getattr__(self, name):
            def _call(**kwargs):
                calls.append((domain, name, kwargs))
                return {"ok": True, "domain": domain, "method": name}

            return _call

    class _FakeClient:
        def __getattr__(self, name):
            return _FakeSub()

    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: _FakeClient())

    tool = kg_server.REGISTERED_TOOLS[f"engine_{domain}"]
    import asyncio

    out = asyncio.run(tool(action=method, params_json=json.dumps(params), graph=""))
    result = json.loads(out)
    assert result.get("ok") is True, f"dispatch failed: {result}"
    assert calls == [(domain, method, params)], f"dispatch mismatch: {calls}"


def test_unknown_action_is_rejected_with_action_list():
    kg_server.ensure_tools_registered()
    tool = kg_server.REGISTERED_TOOLS["engine_finance"]
    import asyncio

    out = asyncio.run(tool(action="not_a_real_method", params_json="{}", graph=""))
    result = json.loads(out)
    assert "error" in result
    assert "var" in result["actions"]
