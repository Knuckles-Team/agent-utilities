"""TLS-contract tests for the one remote MCP toolset factory."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agent_utilities.mcp.toolset_factory import (
    _coerce_httpx_timeout,
    _httpx_client_factory,
    build_http_toolset,
)


def test_remote_mcp_factory_uses_resolved_tls_context(monkeypatch) -> None:
    context = object()
    trust = SimpleNamespace(
        httpx_kwargs=lambda: {"verify": context, "trust_env": False}
    )
    create = MagicMock(return_value=object())
    monkeypatch.setattr(
        "agent_utilities.core.http_client.create_async_http_client", create
    )

    _httpx_client_factory(trust, 12.0)(headers={"X-Test": "1"})

    kwargs = create.call_args.kwargs
    assert kwargs["verify"] is context
    assert kwargs["trust_env"] is False
    assert kwargs["headers"] == {"X-Test": "1"}


def test_remote_mcp_api_has_no_boolean_verify_control() -> None:
    parameters = inspect.signature(build_http_toolset).parameters
    assert "verify" not in parameters
    assert {"tls_service", "tls_profile", "tls_profile_ref"} <= set(parameters)


def test_remote_mcp_factory_accepts_refreshing_auth() -> None:
    parameters = inspect.signature(build_http_toolset).parameters
    assert "auth" in parameters


class _ForeignTimeout:
    """Stand-in for ``httpx2.Timeout`` — structurally identical to
    :class:`httpx.Timeout` (same ``connect``/``read``/``write``/``pool``
    attributes) but deliberately NOT a subclass of it, exactly like fastmcp's
    vendored ``httpx2`` package's ``Timeout`` class. Regression pin for
    D-SNI-3: ``httpx.Timeout(foreign_instance)`` silently assigns the whole
    foreign object (not a float) to each attribute, and code that later does
    ``time.monotonic() + timeout.connect`` raises ``TypeError: unsupported
    operand type(s) for +: 'float' and 'Timeout'``.
    """

    def __init__(self, connect: float, read: float, write: float, pool: float) -> None:
        self.connect = connect
        self.read = read
        self.write = write
        self.pool = pool


def test_coerce_httpx_timeout_none_uses_default() -> None:
    result = _coerce_httpx_timeout(None, 42.0)
    assert isinstance(result, httpx.Timeout)
    assert result.connect == result.read == result.write == result.pool == 42.0


def test_coerce_httpx_timeout_passes_through_native_httpx_timeout() -> None:
    native = httpx.Timeout(5.0, read=9.0)
    result = _coerce_httpx_timeout(native, 42.0)
    assert result is native


def test_coerce_httpx_timeout_accepts_numeric() -> None:
    result = _coerce_httpx_timeout(7, 42.0)
    assert isinstance(result, httpx.Timeout)
    assert result.connect == 7.0


def test_coerce_httpx_timeout_duck_types_foreign_timeout_object() -> None:
    """D-SNI-3 regression: a same-shaped Timeout from a DIFFERENT package
    (fastmcp's ``httpx2``) must be rebuilt as a genuine local ``httpx.Timeout``
    with float fields, not passed through as an opaque foreign object."""
    foreign = _ForeignTimeout(connect=30.0, read=15.0, write=30.0, pool=30.0)

    result = _coerce_httpx_timeout(foreign, 42.0)

    assert isinstance(result, httpx.Timeout)
    assert result is not foreign
    assert result.connect == 30.0
    assert result.read == 15.0
    assert result.write == 30.0
    assert result.pool == 30.0
    # Pin the exact failure mode this guards against: every field must be a
    # real float, never the foreign Timeout instance itself.
    import time

    assert isinstance(time.monotonic() + result.connect, float)


def test_coerce_httpx_timeout_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="unsupported MCP client timeout type"):
        _coerce_httpx_timeout(object(), 42.0)


def test_remote_mcp_factory_normalizes_foreign_timeout_object(monkeypatch) -> None:
    """End-to-end through the factory callback fastmcp actually invokes:
    handing it a foreign (non-``httpx``) Timeout-shaped object must not reach
    ``create_async_http_client`` un-normalized."""
    trust = SimpleNamespace(httpx_kwargs=lambda: {})
    create = MagicMock(return_value=object())
    monkeypatch.setattr(
        "agent_utilities.core.http_client.create_async_http_client", create
    )
    foreign = _ForeignTimeout(connect=30.0, read=15.0, write=30.0, pool=30.0)

    _httpx_client_factory(trust, 12.0)(timeout=foreign)

    kwargs = create.call_args.kwargs
    assert isinstance(kwargs["timeout"], httpx.Timeout)
    assert kwargs["timeout"].read == 15.0
