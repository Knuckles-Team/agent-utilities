"""CX-AU-03 characterization for ``register_analysis_tools._run_analysis_action``
(CCN 289 pre-refactor) in ``agent_utilities/mcp/tools/analysis_tools.py``.

This pins the OBSERVED action-dispatch behaviour of the giant if/elif chain
BEFORE it is decomposed into a dict-dispatch of module-level handlers. It is
written and made green against the UNMODIFIED function, in a commit that
lands before the refactor commit -- see AGENTS.md's two-commit
characterize-then-refactor discipline.

Scope, deliberately narrow: every case here reaches its assertion WITHOUT
calling any method on the (fake) engine, so a bare ``object()`` stands in for
``IntelligenceGraphEngine`` everywhere except the "no engine" case. This is
not full coverage of all ~60 actions -- it is enough to pin (a) the
dispatch-on-`action` behaviour itself, (b) the "no engine" guard, and (c) the
unknown-action fallback, which is exactly what a dict-dispatch refactor of
the *chain itself* can get wrong (an unreachable branch, a dropped action, a
changed fallback message) even when no individual handler body is touched.
"""

from __future__ import annotations

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import analysis_tools


def _get_handler():
    """Register the tools group (idempotent-ish: harmless to call twice in
    one process) and return the raw ``_run_analysis_action`` coroutine
    function via the module global it sets itself."""
    from fastmcp import FastMCP

    mcp = FastMCP("test-cx-au-03")
    analysis_tools.register_analysis_tools(mcp)
    handler = analysis_tools._analysis_core_handler
    assert handler is not None, (
        "register_analysis_tools did not set _analysis_core_handler"
    )
    return handler


async def _call(monkeypatch, action, **kwargs):
    # Calling the raw handler directly (rather than through FastMCP or
    # ``kg_server._execute_tool``) bypasses pydantic's Field-default
    # resolution, so every omitted param must be passed explicitly as its
    # real resolved default -- otherwise it binds to the raw ``FieldInfo``
    # sentinel, which is truthy, silently defeating every `if not <param>:`
    # guard in the function. This mirrors what ``_execute_tool`` does for
    # every other caller.
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    handler = _get_handler()
    call_kwargs = {
        "query": "",
        "top_k": 10,
        "node_id": "",
        "depth": 2,
        "target": "",
    }
    call_kwargs.update(kwargs)
    return await handler(action=action, **call_kwargs)


async def test_no_engine_returns_inactive_error(monkeypatch):
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    handler = _get_handler()
    out = await handler(
        action="inspect", query="", top_k=10, node_id="", depth=2, target=""
    )
    assert out == "Error: IntelligenceGraphEngine not active."


async def test_blast_radius_requires_node_id(monkeypatch):
    out = await _call(monkeypatch, "blast_radius")
    assert out == "Error: node_id required for blast_radius"


async def test_inspect_requires_target_query_or_node_id(monkeypatch):
    out = await _call(monkeypatch, "inspect")
    assert out == "Error: target (or query/node_id) required for inspect"


async def test_contradictions_requires_query(monkeypatch):
    out = await _call(monkeypatch, "contradictions")
    assert out == "Error: contradictions needs the new claim text in `query`."


async def test_evolve_code_requires_query(monkeypatch):
    out = await _call(monkeypatch, "evolve_code")
    assert out == "Error: evolve_code needs a task description in `query`."


async def test_night_shift_requires_target(monkeypatch):
    out = await _call(monkeypatch, "night_shift")
    assert out == "Error: night_shift needs the vault root path in `target`."


async def test_recommend_requires_query(monkeypatch):
    out = await _call(monkeypatch, "recommend")
    assert out == "Error: recommend needs a query/intent in `query`."


async def test_unknown_action_falls_back_with_the_action_name(monkeypatch):
    out = await _call(monkeypatch, "definitely-not-a-real-action")
    assert out == "Error: Unknown analyze action 'definitely-not-a-real-action'"


async def test_each_validation_action_is_independently_routed(monkeypatch):
    # A dict-dispatch bug class this specifically guards against: two action
    # names accidentally mapped to the SAME handler (a copy/paste of the
    # dispatch-table entry), which would make one of these two return the
    # OTHER action's message instead of its own.
    blast = await _call(monkeypatch, "blast_radius")
    inspect_ = await _call(monkeypatch, "inspect")
    assert blast != inspect_
    assert "blast_radius" in blast
    assert "inspect" in inspect_
