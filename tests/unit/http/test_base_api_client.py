"""Tests for agent_utilities.http.client (CONCEPT:ECO-4.35).

All transport via httpx.MockTransport — no live calls. Pins the dockerhub-api
envelope shape, rate-limit capture + bounded 429 backoff, error mapping,
auth injection (incl. 401 invalidate-and-retry), destructive gating,
pagination wiring, redaction of raised errors, and the async twin.
"""

from __future__ import annotations

import httpx
import pytest

from agent_utilities.core.exceptions import (
    ApiError,
    AuthError,
    ParameterError,
    UnauthorizedError,
)
from agent_utilities.http import (
    AsyncBaseApiClient,
    AuthHeaderInjector,
    BaseApiClient,
    BasicAuth,
    DestructiveOperationError,
    QueryApiKeyAuth,
    TokenAuth,
)

BASE = "https://api.unit.test"


def _client(handler, **kwargs) -> BaseApiClient:
    return BaseApiClient(BASE, transport=httpx.MockTransport(handler), **kwargs)


def _async_client(handler, **kwargs) -> AsyncBaseApiClient:
    return AsyncBaseApiClient(BASE, transport=httpx.MockTransport(handler), **kwargs)


# --------------------------------------------------------------------------- #
# envelope + base-URL joining
# --------------------------------------------------------------------------- #


def test_envelope_shape_and_base_url_joining():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE}/v2/items?q=x"
        return httpx.Response(200, json={"ok": 1})

    with _client(handler) as client:
        envelope = client.get("/v2/items", params={"q": "x", "skip_me": None})
    assert envelope == {"status_code": 200, "data": {"ok": 1}, "rate_limit": None}


def test_envelope_includes_headers_when_opted_in():
    handler = lambda request: httpx.Response(200, json=[], headers={"ETag": "abc"})  # noqa: E731
    with _client(handler, include_response_headers=True) as client:
        envelope = client.get("/v2/items")
    assert envelope["headers"]["etag"] == "abc"


def test_non_json_and_empty_bodies():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/text":
            return httpx.Response(
                200, text="plain words", headers={"Content-Type": "text/plain"}
            )
        return httpx.Response(204)

    with _client(handler) as client:
        assert client.get("/text")["data"] == "plain words"
        assert client.delete("/gone")["data"] is None


def test_absolute_urls_pass_through():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://other.test/abs"
        return httpx.Response(200, json={})

    with _client(handler) as client:
        assert client.get("https://other.test/abs")["status_code"] == 200


def test_default_headers_constructor_headers_and_per_call_merge():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "application/vnd.kafka.v2+json"
        assert request.headers["X-Fleet"] == "yes"
        assert request.headers["Content-Type"] == "application/json"
        return httpx.Response(200, json={})

    with _client(handler, headers={"X-Fleet": "yes"}) as client:
        client.post(
            "/topics",
            json={"a": 1},
            content_type="application/json",
            accept="application/vnd.kafka.v2+json",
        )


# --------------------------------------------------------------------------- #
# auth strategies
# --------------------------------------------------------------------------- #


def test_token_auth_header_injected():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tok-1"
        return httpx.Response(200, json={})

    with _client(handler, auth=TokenAuth("tok-1")) as client:
        client.get("/me")


def test_basic_auth_and_query_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/basic":
            assert request.headers["Authorization"].startswith("Basic ")
        else:
            assert request.url.params["apikey"] == "qk-1"
        return httpx.Response(200, json={})

    with _client(handler, auth=BasicAuth("u", "p")) as client:
        client.get("/basic")
    with _client(handler, auth=QueryApiKeyAuth("apikey", "qk-1")) as client:
        client.get("/query")


def test_401_triggers_single_invalidate_and_retry():
    class RefreshingAuth(AuthHeaderInjector):
        def __init__(self):
            self.generation = 0

        def headers(self):
            return {"Authorization": f"Bearer gen-{self.generation}"}

        def invalidate(self):
            self.generation += 1

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.headers["Authorization"] == "Bearer gen-0":
            return httpx.Response(401, json={"message": "expired"})
        return httpx.Response(200, json={"ok": 1})

    with _client(handler, auth=RefreshingAuth()) as client:
        envelope = client.get("/secured")
    assert envelope["data"] == {"ok": 1}
    assert calls["n"] == 2


def test_401_without_invalidate_raises_auth_error():
    handler = lambda request: httpx.Response(401, json={"message": "nope"})  # noqa: E731
    with _client(handler, auth=TokenAuth("static")) as client:
        with pytest.raises(AuthError):
            client.get("/secured")


# --------------------------------------------------------------------------- #
# rate-limit capture + 429 backoff
# --------------------------------------------------------------------------- #


def test_rate_limit_snapshot_attached_to_envelope():
    handler = lambda request: httpx.Response(  # noqa: E731
        200,
        json=[],
        headers={"X-RateLimit-Limit": "100", "X-RateLimit-Remaining": "97"},
    )
    with _client(handler) as client:
        envelope = client.get("/items")
        assert envelope["rate_limit"] == {"limit": 100, "remaining": 97}
        assert client.rate_limit.remaining == 97


def test_429_bounded_backoff_then_success(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("agent_utilities.http.client.time.sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"ok": 1})

    with _client(handler) as client:
        envelope = client.get("/limited")
    assert envelope["data"] == {"ok": 1}
    assert sleeps == [2.0, 2.0]


def test_429_exhaustion_raises_api_error_with_rate_context(monkeypatch):
    monkeypatch.setattr("agent_utilities.http.client.time.sleep", lambda s: None)
    handler = lambda request: httpx.Response(  # noqa: E731
        429,
        headers={
            "Retry-After": "1",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "30",
        },
    )
    with _client(handler, max_retries_429=2) as client:
        with pytest.raises(ApiError, match="rate limited after 2 retries"):
            client.get("/limited")


def test_retry_after_cap_bounds_each_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("agent_utilities.http.client.time.sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "9999"})
        return httpx.Response(200, json={})

    with _client(handler, retry_after_cap_s=5.0) as client:
        client.get("/limited")
    assert sleeps == [5.0]


# --------------------------------------------------------------------------- #
# error mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (400, ParameterError),
        (401, AuthError),
        (403, UnauthorizedError),
        (404, ParameterError),
        (500, ApiError),
    ],
)
def test_default_error_map(status, exc):
    handler = lambda request: httpx.Response(status, json={"message": "boom"})  # noqa: E731
    with _client(handler) as client:
        with pytest.raises(exc, match="boom"):
            client.get("/err")


def test_error_map_override_and_default_error():
    class TeapotError(Exception):
        """Marker for the override mapping."""

    handler = lambda request: httpx.Response(418, json={"message": "short and stout"})  # noqa: E731
    with _client(handler, error_map={418: TeapotError}) as client:
        with pytest.raises(TeapotError):
            client.get("/teapot")


def test_raise_for_status_false_returns_error_envelope():
    handler = lambda request: httpx.Response(404, json={"message": "missing"})  # noqa: E731
    with _client(handler) as client:
        envelope = client.get("/missing", raise_for_status=False)
    assert envelope["status_code"] == 404
    assert envelope["data"] == {"message": "missing"}


def test_error_messages_are_redacted():
    handler = lambda request: httpx.Response(  # noqa: E731
        400, json={"message": "bad token Bearer abcdef123456secret"}
    )
    with _client(handler) as client:
        with pytest.raises(ParameterError) as excinfo:
            client.get("/err")
    assert "abcdef123456secret" not in str(excinfo.value)


# --------------------------------------------------------------------------- #
# destructive gating
# --------------------------------------------------------------------------- #


def test_destructive_gate_blocks_by_default():
    handler = lambda request: httpx.Response(200, json={})  # noqa: E731
    with _client(handler) as client:
        with pytest.raises(DestructiveOperationError, match="delete_stack"):
            client.guard_destructive("delete_stack")
    with _client(handler, allow_destructive=True) as client:
        client.guard_destructive("delete_stack")  # no raise


# --------------------------------------------------------------------------- #
# pagination wiring
# --------------------------------------------------------------------------- #


def test_paginate_cursor_through_client():
    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(200, json={"items": [{"id": 1}], "next_cursor": "c2"})
        return httpx.Response(200, json={"items": [{"id": 2}], "next_cursor": None})

    with _client(handler) as client:
        items = list(client.paginate("/widgets", mode="cursor", items_path="items"))
    assert [r["id"] for r in items] == [1, 2]


def test_paginate_raises_mapped_error_on_failed_page():
    handler = lambda request: httpx.Response(403, json={"message": "denied"})  # noqa: E731
    with _client(handler) as client:
        with pytest.raises(UnauthorizedError):
            list(client.paginate("/widgets", mode="page"))


# --------------------------------------------------------------------------- #
# per-call timeout + verify default
# --------------------------------------------------------------------------- #


def test_per_call_timeout_reaches_transport():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions.get("timeout", {}))
        return httpx.Response(200, json={})

    with _client(handler) as client:
        client.get("/fast", timeout=3.5)
    assert seen["read"] == 3.5


def test_verify_defaults_true():
    with BaseApiClient(
        BASE, transport=httpx.MockTransport(lambda r: httpx.Response(200))
    ) as client:
        assert client.verify is True


# --------------------------------------------------------------------------- #
# async twin
# --------------------------------------------------------------------------- #


async def test_async_envelope_and_auth():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "SSWS ok-tok"
        return httpx.Response(
            200, json={"hello": "async"}, headers={"X-Rate-Limit-Remaining": "5"}
        )

    async with _async_client(
        handler, auth=TokenAuth("ok-tok", prefix="SSWS")
    ) as client:
        envelope = await client.get("/me")
    assert envelope["status_code"] == 200
    assert envelope["data"] == {"hello": "async"}
    assert envelope["rate_limit"] == {"remaining": 5}


async def test_async_429_backoff(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("agent_utilities.http.client.asyncio.sleep", fake_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "4"})
        return httpx.Response(200, json={})

    async with _async_client(handler) as client:
        envelope = await client.get("/limited")
    assert envelope["status_code"] == 200
    assert sleeps == [4.0]


async def test_async_error_mapping():
    handler = lambda request: httpx.Response(403, json={"message": "no"})  # noqa: E731
    async with _async_client(handler) as client:
        with pytest.raises(UnauthorizedError):
            await client.get("/denied")


async def test_async_pagination():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", 1))
        items = [{"id": page}] if page <= 2 else []
        return httpx.Response(200, json=items)

    async with _async_client(handler) as client:
        iterator = client.paginate("/rows", mode="page", page_size=1)
        items = [r async for r in iterator]
    assert [r["id"] for r in items] == [1, 2]
