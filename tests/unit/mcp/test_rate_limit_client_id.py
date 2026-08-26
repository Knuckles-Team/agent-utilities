"""Tests for the per-caller rate-limit bucket key (server_factory.py).

CONCEPT:AU-OS.observability.no-op-without-metrics adjacent — this closes a
real "empty tools/list" failure shape: ``RateLimitingMiddleware`` defaults
every caller onto one shared literal ``"global"`` bucket
(``fastmcp.server.middleware.rate_limiting.RateLimitingMiddleware
._get_client_identifier``), so a handful of concurrent legitimate MCP
clients (this fleet routinely runs several against graph-os at once)
exhausts the shared 20-token/10rps budget and gets a JSON-RPC ``-32000
"Rate limit exceeded for client: global"`` error on ANY request type,
including ``tools/list`` itself -- indistinguishable, to a caller that does
not check the ``error`` field first, from a genuinely empty tool list.
``_rate_limit_client_id`` keys the bucket per authenticated caller instead,
so unrelated legitimate callers no longer starve each other.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.mcp import server_factory


def _token(client_id=None, sub=None, tenant_id=None):
    tok = MagicMock()
    tok.client_id = client_id
    claims = {}
    if sub is not None:
        claims["sub"] = sub
    if tenant_id is not None:
        claims["tenant_id"] = tenant_id
    tok.claims = claims
    return tok


def test_rate_limit_client_id_anonymous_without_token():
    with patch(
        "fastmcp.server.dependencies.get_access_token", side_effect=RuntimeError
    ):
        assert server_factory._rate_limit_client_id(MagicMock()) == "anonymous"


def test_rate_limit_client_id_anonymous_when_token_has_no_identity_fields():
    with patch(
        "fastmcp.server.dependencies.get_access_token",
        return_value=_token(),
    ):
        assert server_factory._rate_limit_client_id(MagicMock()) == "anonymous"


def test_rate_limit_client_id_stable_for_the_same_caller():
    tok = _token(client_id="lane-a", sub="user-1", tenant_id="tenant-x")
    with patch("fastmcp.server.dependencies.get_access_token", return_value=tok):
        first = server_factory._rate_limit_client_id(MagicMock())
        second = server_factory._rate_limit_client_id(MagicMock())
    assert first == second
    assert first != "anonymous"
    assert first.startswith("caller_")


def test_rate_limit_client_id_distinct_for_different_callers():
    tok_a = _token(client_id="lane-a", sub="user-1", tenant_id="tenant-x")
    tok_b = _token(client_id="lane-b", sub="user-2", tenant_id="tenant-x")
    with patch("fastmcp.server.dependencies.get_access_token", return_value=tok_a):
        key_a = server_factory._rate_limit_client_id(MagicMock())
    with patch("fastmcp.server.dependencies.get_access_token", return_value=tok_b):
        key_b = server_factory._rate_limit_client_id(MagicMock())
    assert key_a != key_b


def test_rate_limit_client_id_never_leaks_raw_claims_into_the_key():
    tok = _token(client_id="lane-a", sub="super-secret-subject", tenant_id="tenant-x")
    with patch("fastmcp.server.dependencies.get_access_token", return_value=tok):
        key = server_factory._rate_limit_client_id(MagicMock())
    assert "super-secret-subject" not in key


async def test_configure_middleware_keys_rate_limiter_per_caller_not_one_shared_global_bucket():
    """CONCEPT:AU-ECO.mcp — the regression this guards: an un-keyed
    ``RateLimitingMiddleware`` buckets every caller under the single
    literal string "global", so a burst of legitimate concurrent MCP
    clients (multiple agent lanes, the harness, service-account bridges)
    can exhaust the WHOLE server's shared budget and see requests --
    including ``tools/list`` -- rejected with a rate-limit error easily
    mistaken for an empty result.
    """
    from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

    mock_args = MagicMock()
    mock_args.auth_type = "none"
    mock_args.eunomia_type = "none"

    middlewares = server_factory._configure_middleware(mock_args, server_name="graph-os")

    rate_limiters = [m for m in middlewares if isinstance(m, RateLimitingMiddleware)]
    assert len(rate_limiters) == 1
    rate_limiter = rate_limiters[0]
    assert rate_limiter.get_client_id is server_factory._rate_limit_client_id
    assert rate_limiter.global_limit is False
    # The default (broken) shape this replaces: get_client_id is None, so
    # _get_client_identifier() resolves to the literal "global" for every
    # caller. With the fix wired in, two DIFFERENT callers resolve to two
    # DIFFERENT bucket keys instead of colliding on one shared budget.
    with patch(
        "fastmcp.server.dependencies.get_access_token",
        return_value=_token(client_id="lane-a", sub="user-1"),
    ):
        identifier_a = await rate_limiter._get_client_identifier(MagicMock())
    with patch(
        "fastmcp.server.dependencies.get_access_token",
        return_value=_token(client_id="lane-b", sub="user-2"),
    ):
        identifier_b = await rate_limiter._get_client_identifier(MagicMock())
    assert identifier_a != "global"
    assert identifier_b != "global"
    assert identifier_a != identifier_b
