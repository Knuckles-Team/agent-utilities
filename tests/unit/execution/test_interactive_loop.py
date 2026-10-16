"""CONCEPT:AU-ORCH.execution.held-turn-registry-mid / OS-5.11 — Interactive loop: held-turn resume, sidecar isolation, run tokens.

Covers the mid-turn pause/resume registry, the ``/api/runs/{id}/tool-result`` live route, per-run
UDS socket isolation via typed process stamps, and run-scoped token mint/validate (expiry + scope).
"""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities.core.execution.held_turns import HeldTurnRegistry
from agent_utilities.security.run_token import (
    TokenError,
    decode_token,
    mint_token,
    validate_token,
)
from agent_utilities.security.sidecar_runtime import (
    ProcessStamp,
    SidecarRuntime,
    socket_path,
)

pytestmark = pytest.mark.concept(id="AU-ORCH.execution.held-turn-registry-mid")


# ── held-turn registry ──────────────────────────────────────────────


async def test_held_turn_resume():
    reg = HeldTurnRegistry()

    async def waiter():
        return await reg.wait_for_result("run1", "tu1", timeout=5)

    task = asyncio.ensure_future(waiter())
    await asyncio.sleep(0.05)
    assert reg.is_waiting("run1", "tu1")
    assert reg.resolve("run1", "tu1", {"answer": "42"}) is True
    out = await task
    assert out == {"answer": "42"}
    assert reg.is_waiting("run1") is False  # cleaned up


async def test_held_turn_timeout_raises():
    reg = HeldTurnRegistry()
    with pytest.raises(asyncio.TimeoutError):
        await reg.wait_for_result("r", "t", timeout=0.05)


def test_resolve_unknown_returns_false():
    reg = HeldTurnRegistry()
    assert reg.resolve("nope", "x", {}) is False


async def test_resolve_any():
    reg = HeldTurnRegistry()
    task = asyncio.ensure_future(reg.wait_for_result("r", "tu", timeout=5))
    await asyncio.sleep(0.05)
    assert reg.resolve_any("r", {"k": 1}) is True
    assert await task == {"k": 1}


# ── sidecar isolation ───────────────────────────────────────────────


def test_socket_paths_isolated_by_namespace():
    a = socket_path(ProcessStamp(namespace="alpha"), "run1")
    b = socket_path(ProcessStamp(namespace="beta"), "run1")
    assert a != b
    assert "alpha" in str(a) and "beta" in str(b)


def test_socket_paths_isolated_by_run():
    s = ProcessStamp(namespace="ns")
    assert socket_path(s, "runA") != socket_path(s, "runB")


def test_runtime_isolation_and_allocate_idempotent():
    rt1 = SidecarRuntime(ProcessStamp(namespace="one"))
    rt2 = SidecarRuntime(ProcessStamp(namespace="two"))
    assert rt1.isolated_from(rt2, "shared-run") is True
    assert rt1.allocate("x") == rt1.allocate("x")  # idempotent
    rt1.release("x")


def test_stamp_key_is_filesystem_safe():
    key = ProcessStamp().key()
    assert "/" not in key and " " not in key


# ── run-scoped token (OS-5.11) ──────────────────────────────────────


def test_token_roundtrip():
    tok = mint_token(
        "run1", project="p", endpoints=("/api/x",), operations=("read", "write")
    )
    decoded = decode_token(tok)
    assert decoded.run_id == "run1"
    assert decoded.project == "p"
    assert "/api/x" in decoded.endpoints


def test_token_tamper_detected():
    tok = mint_token("run1")
    body, _sig = tok.split(".", 1)
    forged = body + ".AAAA"
    with pytest.raises(TokenError):
        decode_token(forged)


def test_token_expiry_enforced():
    tok = mint_token("run1", ttl_seconds=10, now=0.0)
    with pytest.raises(TokenError):
        validate_token(tok, now=100.0)  # past expiry
    assert validate_token(tok, now=1.0).run_id == "run1"


def test_token_endpoint_scope_enforced():
    tok = mint_token("run1", endpoints=("/api/allowed",))
    with pytest.raises(TokenError):
        validate_token(tok, endpoint="/api/forbidden")
    assert validate_token(tok, endpoint="/api/allowed").run_id == "run1"


def test_token_wildcard_endpoint():
    tok = mint_token("run1", endpoints=("*",))
    assert validate_token(tok, endpoint="/anything").run_id == "run1"


# ── live route ──────────────────────────────────────────────────────
