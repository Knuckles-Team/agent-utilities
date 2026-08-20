"""NE-044 sub-gate 2 acceptance: does ``cc9063c8``'s "reachability + honesty
fixes for the graph-os intent surface" actually hold, end to end, through the
REAL registered MCP tool coroutines (not just the internal functions cc9063c8
touched)?

Three things are asserted, all against the real ``graph_code`` tool
(``agent_utilities/mcp/tools/analyze_suite.py``) registered exactly as
``register_analysis_tools`` + ``register_analyze_suite_tools`` wire it in
production, dispatched through the real ``kg_server._execute_tool`` core:

1. **Verbose/granular reachability resolves** — calling ``graph_code`` (the
   always-registered granular tool; ``MCP_TOOL_MODE=intent`` only hides it
   from a session's default *list*, never from ``REGISTERED_TOOLS``/
   ``_execute_tool`` — see ``verbose_tools.tool_mode``'s docstring) with a
   real (non-broken) fake engine returns a real, grounded answer.

2. **Intent-level reachability resolves the SAME capability** — the ``ask``
   intent verb (``intent_tools.dispatch_intent``), with ``graph_code`` left
   as the REAL registered tool (not re-faked), routes unpinned natural
   language to it and gets the identical grounded answer — proving the
   intent surface is not a dead end onto a capability the verbose surface
   can reach but the intent surface cannot (the BUG-040 defect class).

3. **An unavailable capability is reported honestly, not as a fake success
   or a silent empty result** — the exact same ``graph_code
   action=code_context`` call, with the engine's Cypher reads raising
   ``EngineCircuitOpenError`` (a real breaker-open engine outage, not a
   genuinely-empty result), must come back as ``error.code ==
   "engine_degraded"`` -- never a confident "no such symbol"/empty-citations
   answer that looks identical to a real miss (BUG-004, reached here through
   the full MCP dispatch path, not just ``build_code_context`` directly).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.engine_breaker import (
    EngineCircuitOpenError,
)
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import intent_tools

_CANON = "/home/agent-user/workspace/agent-packages/agent-utilities/agent_utilities/orchestration/intent_dispatch.py"


class _RealAnchorEngine:
    """A genuinely working (non-broken) engine: resolves the well-known
    symbol ``dispatch_intent`` to a real anchor, exactly like
    ``tests/unit/test_code_context.py``'s ``FakeEngine`` does for its own
    positive-path tests -- reused here in miniature so this file does not
    depend on cross-file test imports."""

    def query_cypher(self, cypher: str, params: dict[str, Any]) -> list[dict]:
        if "c.name = $tok" in cypher or "CONTAINS $tok" in cypher:
            return [
                {
                    "id": f"code:{_CANON}::dispatch_intent",
                    "name": "dispatch_intent",
                    "file_path": _CANON,
                    "line": 42,
                    "language": "python",
                    "kind": "function",
                    "instance": None,
                    "source_system": None,
                }
            ]
        return []


class _BreakerOpenEngine:
    """A real engine OUTAGE (breaker open), not a genuinely-empty result."""

    def query_cypher(self, cypher: str, params: dict[str, Any]) -> list[dict]:
        raise EngineCircuitOpenError("engine")


class _FakeMCP:
    """Captures the tool coroutines the two registration functions define,
    identically to ``tests/unit/mcp/test_evidence_bundle_envelope.py``'s own
    harness -- the established, already-proven pattern for driving the REAL
    registered tool through ``kg_server._execute_tool`` with no live engine
    process."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *, name: str, description: str = "", tags: Any = None):
        def _decorator(fn):
            self.tools[name] = fn
            return fn

        return _decorator


def _register_real_graph_code_tool() -> _FakeMCP:
    from agent_utilities.mcp.tools import analysis_tools, analyze_suite

    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)
    analyze_suite.register_analyze_suite_tools(fake)
    assert kg_server.REGISTERED_TOOLS.get("graph_code") is fake.tools["graph_code"]
    return fake


def test_verbose_surface_graph_code_reachability_resolves(monkeypatch) -> None:
    """The always-registered granular tool, called directly (the "verbose"
    path), reaches the real code_context capability and gets a grounded
    answer -- not an error, not an empty result."""
    _register_real_graph_code_tool()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: _RealAnchorEngine())

    bundle = asyncio.run(
        kg_server._execute_tool(
            "graph_code",
            action="code_context",
            query="dispatch_intent",
            target="how",
        )
    ).model_dump()

    assert bundle["error"] is None
    assert bundle["evidence_spans"], "a real anchor must produce a real citation"
    assert "dispatch_intent" in bundle["answer_candidate"]


@pytest.mark.asyncio
async def test_intent_level_reachability_resolves_the_same_capability(
    monkeypatch,
) -> None:
    """The ``ask`` intent verb reaches the SAME real ``graph_code`` tool
    (left genuinely registered, not re-faked) for unpinned natural-language
    code-context wording, and gets the same real, grounded answer -- proving
    the intent surface is not a reachability dead end onto a capability the
    verbose surface can already reach (the BUG-040 defect class, exercised
    here for the read/ask surface rather than the destructive-write one
    ``tests/unit/mcp/test_intent_surface_gating.py`` already covers)."""
    _register_real_graph_code_tool()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: _RealAnchorEngine())

    intent = (
        "How is dispatch_intent implemented in agent-utilities? "
        "Return code context with cited files."
    )
    result = await intent_tools.dispatch_intent("ask", intent)

    assert result["executed"] is True
    assert result["routing"]["chosen_tool"] == "graph_code"
    assert result["routing"]["action"] == "code_context"
    bundle = result["result"]
    if isinstance(bundle, str):
        import json

        bundle = json.loads(bundle)
    elif hasattr(bundle, "model_dump"):
        bundle = bundle.model_dump()
    assert bundle["error"] is None
    assert "dispatch_intent" in bundle["answer_candidate"]


def test_unavailable_capability_reports_honest_degraded_state_not_fake_success(
    monkeypatch,
) -> None:
    """BUG-004, exercised through the full MCP dispatch path: a real engine
    outage (breaker open) during ``graph_code action=code_context`` must
    come back as an honest ``error.code == "engine_degraded"``, never a
    confident-looking empty/miss answer indistinguishable from a real
    "no such symbol" result."""
    _register_real_graph_code_tool()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: _BreakerOpenEngine())

    bundle = asyncio.run(
        kg_server._execute_tool(
            "graph_code",
            action="code_context",
            query="dispatch_intent",
            target="how",
        )
    ).model_dump()

    assert bundle["error"] is not None
    assert bundle["error"]["code"] == "engine_degraded"
    # The old false-confident phrasing must not silently survive the full
    # dispatch path either.
    assert "may not be ingested" not in bundle["answer_candidate"]
