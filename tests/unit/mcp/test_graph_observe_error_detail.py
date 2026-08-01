"""MCP-surface tests for `graph_observe action=error_detail` (D-23/D-24).

Exercises the REAL registered tool coroutine directly (mirroring the live-path
pattern used by ``test_evidence_bundle_envelope.py``), proving that a caller
holding an opaque ``error.detail_ref`` from a failed operation can actually
resolve it through graph-os:

* a resolvable ref round-trips through the MCP surface;
* an unknown ref returns a clean "not resolved" response, not a 500;
* a caller from a different tenant authority scope is refused, with the same
  clean shape (never confirming the ref exists in a foreign tenant);
* an unauthenticated caller is refused with a clean ``permission_denied``
  envelope, not an unhandled exception;
* the SAME sanitization contract applies on this resolution path.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.security.error_surface import public_error_payload


class _FakeMCP:
    """Captures the tool coroutines ``register_analyze_suite_tools`` registers."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *, name: str, description: str = "", tags: Any = None):
        def _decorator(fn):
            self.tools[name] = fn
            return fn

        return _decorator


def _register_graph_observe():
    from agent_utilities.mcp.tools import analyze_suite

    fake = _FakeMCP()
    analyze_suite.register_analyze_suite_tools(fake)
    return fake.tools["graph_observe"]


def _tenant_actor(tenant_id: str) -> ActorContext:
    return ActorContext(
        actor_id=f"agent:{tenant_id}",
        actor_type=ActorType.AI_AGENT,
        roles=("member",),
        tenant_id=tenant_id,
        authenticated=True,
    )


def test_graph_observe_error_detail_round_trips_a_resolvable_ref() -> None:
    graph_observe = _register_graph_observe()
    with use_actor(_tenant_actor("acme")):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            payload = public_error_payload(exc)
        detail_ref = payload["error"]["detail_ref"]

        out = asyncio.run(
            graph_observe(action="error_detail", query=detail_ref, top_k=20)
        )

    decoded = json.loads(out)
    assert decoded["resolved"] is True
    assert decoded["detail_ref"] == detail_ref
    assert decoded["error_class"] == "RuntimeError"
    assert decoded["failing_layer"]
    # tenant_id is internal bookkeeping for the authority check, not echoed.
    assert "tenant_id" not in decoded


def test_graph_observe_error_detail_unknown_ref_fails_cleanly_not_a_500() -> None:
    graph_observe = _register_graph_observe()
    with use_actor(_tenant_actor("acme")):
        out = asyncio.run(
            graph_observe(
                action="error_detail",
                query="correlation:does-not-exist",
                top_k=20,
            )
        )

    decoded = json.loads(out)
    assert decoded == {"resolved": False, "detail_ref": "correlation:does-not-exist"}


def test_graph_observe_error_detail_refuses_a_different_tenant_cleanly() -> None:
    graph_observe = _register_graph_observe()
    with use_actor(_tenant_actor("acme")):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            payload = public_error_payload(exc)
    detail_ref = payload["error"]["detail_ref"]

    with use_actor(_tenant_actor("globex")):
        out = asyncio.run(
            graph_observe(action="error_detail", query=detail_ref, top_k=20)
        )

    decoded = json.loads(out)
    # Refused, with the SAME shape as an unknown ref — a foreign-tenant caller
    # cannot even confirm the ref exists.
    assert decoded == {"resolved": False, "detail_ref": detail_ref}


def test_graph_observe_error_detail_fails_closed_without_verified_authority() -> None:
    graph_observe = _register_graph_observe()
    # An actor bound but lacking verified tenant authority (unauthenticated)
    # -> PermissionError -> a clean permission_denied envelope, never an
    # unhandled exception/500. (Every served MCP call path always binds SOME
    # actor via ActorContextMiddleware / verified_tool_session_scope, so this
    # is the realistic "denied" shape, not a wholly unbound context.)
    unauthenticated = ActorContext(
        actor_id="anon", tenant_id="acme", authenticated=False
    )
    with use_actor(unauthenticated):
        out = asyncio.run(
            graph_observe(
                action="error_detail",
                query="correlation:does-not-exist",
                top_k=20,
            )
        )

    decoded = json.loads(out)
    assert decoded["status"] == "failed"
    assert decoded["error"]["code"] == "permission_denied"


def test_graph_observe_error_detail_sanitizes_secret_material() -> None:
    graph_observe = _register_graph_observe()
    sensitive = "Bearer sk-proj-example-abcdefghijklmnopqrstuvwxyz012345"
    with use_actor(_tenant_actor("acme")):
        try:
            raise RuntimeError(sensitive)
        except RuntimeError as exc:
            payload = public_error_payload(exc)
        detail_ref = payload["error"]["detail_ref"]

        out = asyncio.run(
            graph_observe(action="error_detail", query=detail_ref, top_k=20)
        )

    assert "sk-proj-example-abcdefghijklmnopqrstuvwxyz012345" not in out
