"""NE-041 acceptance (AU-ADOPT-B): the ``engine_<domain>`` wrapper's top-level
``graph`` field is a WRAPPER-ONLY selector (routes which connection/session a
call uses) -- it must be forwarded into the underlying engine method call
only when that method's own signature declares a ``graph`` parameter, and
stripped (never forwarded) otherwise.

``753db329`` (U-96/U-98/U-101, U-74, U-18, U-38) and the pre-existing
``tests/unit/test_engine_api_coverage.py`` / ``tests/unit/
test_engine_tenants_rest_unsupported_field.py`` cover: full domain/method
schema discovery and MCP-tool/REST-route parity
(``test_every_engine_domain_has_mcp_tool_and_rest_route``), an unknown field
producing a structured 400 instead of an opaque 500
(``test_unknown_field_is_rejected_before_any_dispatch``), and a duplicated
``graph`` selector (top-level + inside ``params_json``) being rejected as a
caller error. None of those exercises the injection/stripping decision
itself with a REAL, introspectable target-method signature -- every fake
sub-client in that suite is either a generic ``__getattr__`` catch-all
closure (whose ``**kwargs`` signature can never satisfy ``"graph" in
sig.parameters``, so the injection branch in
``engine_tools._dispatch`` is never actually exercised either way) or already
supplies ``graph`` explicitly in ``params_json`` (bypassing the injection
branch entirely). This file proves both halves of that branch directly:
``tenants.list()`` (no ``graph`` parameter) must never receive one, and
``streaming.list_triggers(graph)`` (a real declared ``graph`` parameter) must
receive the session's resolved graph authority when the caller left the
wrapper's own ``graph`` field empty.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("epistemic_graph.client")

from agent_utilities.knowledge_graph.core.session import current_session
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools


class _FakeTenantsSub:
    """A REAL method with a REAL, introspectable signature -- no ``graph``
    parameter, exactly like the actual ``TenantsClient.list``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def list(self) -> dict:
        self.calls.append({})
        return {"ok": True, "domain": "tenants"}


class _FakeStreamingSub:
    """A REAL method with a REAL, introspectable signature that DOES declare
    ``graph``, exactly like the actual ``StreamingClient.list_triggers``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def list_triggers(self, graph: str) -> dict:
        self.calls.append({"graph": graph})
        return {"ok": True, "domain": "streaming", "graph": graph}


class _FakeClient:
    def __init__(self, tenants: _FakeTenantsSub, streaming: _FakeStreamingSub) -> None:
        self.tenants = tenants
        self.streaming = streaming


@pytest.fixture(autouse=True)
def _fresh_client_pool(monkeypatch):
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)
    yield
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)


def test_wrapper_only_graph_selector_is_stripped_when_the_method_does_not_declare_it(
    monkeypatch,
) -> None:
    """``tenants.list()`` takes no ``graph`` parameter -- the wrapper's own
    routing selector must never be forwarded into the call. Forwarding it
    would raise ``TypeError: list() got an unexpected keyword argument
    'graph'`` against a real engine client."""
    kg_server.ensure_tools_registered()
    tenants = _FakeTenantsSub()
    streaming = _FakeStreamingSub()
    monkeypatch.setattr(
        engine_tools, "_client_for", lambda graph: _FakeClient(tenants, streaming)
    )

    tool = kg_server.REGISTERED_TOOLS["engine_tenants"]
    out = asyncio.run(tool(action="list", params_json="{}", graph=""))
    result = json.loads(out)

    assert result == {"ok": True, "domain": "tenants"}
    assert tenants.calls == [{}], (
        "wrapper-only graph selector leaked into a call the target method "
        f"does not declare: {tenants.calls!r}"
    )


def test_wrapper_only_graph_selector_is_injected_when_the_method_declares_it(
    monkeypatch,
) -> None:
    """``streaming.list_triggers(graph)`` DOES declare ``graph`` (BUG-4) --
    the resolved selector must be threaded into the call using the session's
    verified graph authority when the caller left the wrapper's own field
    empty."""
    kg_server.ensure_tools_registered()
    tenants = _FakeTenantsSub()
    streaming = _FakeStreamingSub()
    monkeypatch.setattr(
        engine_tools, "_client_for", lambda graph: _FakeClient(tenants, streaming)
    )

    session = current_session()
    assert session is not None and session.graph, "test fixture must supply a session"

    tool = kg_server.REGISTERED_TOOLS["engine_streaming"]
    out = asyncio.run(tool(action="list_triggers", params_json="{}", graph=""))
    result = json.loads(out)

    assert result == {"ok": True, "domain": "streaming", "graph": session.graph}
    assert streaming.calls == [{"graph": session.graph}]


def test_caller_supplied_graph_in_params_json_is_never_overwritten(
    monkeypatch,
) -> None:
    """If the caller already put ``graph`` inside ``params_json`` (the
    channel the duplicate-selector-conflict test in
    ``test_engine_tenants_rest_unsupported_field.py`` exercises for a method
    that does NOT accept it), the injection branch must not touch it for a
    method that DOES accept it -- the caller's explicit value wins, the
    wrapper never silently substitutes its own resolution."""
    kg_server.ensure_tools_registered()
    tenants = _FakeTenantsSub()
    streaming = _FakeStreamingSub()
    monkeypatch.setattr(
        engine_tools, "_client_for", lambda graph: _FakeClient(tenants, streaming)
    )

    tool = kg_server.REGISTERED_TOOLS["engine_streaming"]
    out = asyncio.run(
        tool(
            action="list_triggers",
            params_json=json.dumps({"graph": "caller-chosen-graph"}),
            graph="",
        )
    )
    result = json.loads(out)

    assert result == {
        "ok": True,
        "domain": "streaming",
        "graph": "caller-chosen-graph",
    }
    assert streaming.calls == [{"graph": "caller-chosen-graph"}]
