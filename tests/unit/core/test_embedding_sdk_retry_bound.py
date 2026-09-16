"""CONCEPT:AU-KG.retrieval.embedding-fast-fail — bound the embedding HTTP retry loop.

D2 (GHSA-8mgp-746c-j5xp) removed llama-index: embedding requests are now a
plain HTTP POST (``_post_with_one_retry``) instead of going through
``llama_index.embeddings.openai.OpenAIEmbedding`` (which defaulted to
``max_retries=10`` with exponential backoff up to ~8s per retry when the
caller passed no explicit value). agent-utilities already owns a separate,
endpoint-aware circuit-breaker/backoff layer, so a second unbounded retry
loop only adds latency — every embedding request must retry a small, bounded
number of times, never silently inherit an SDK's larger default.
"""

from __future__ import annotations

import httpx
import pytest

from agent_utilities.core.embedding_utilities import (
    _EMBED_SDK_MAX_RETRIES,
    _post_with_one_retry,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def test_max_retries_is_small_and_bounded():
    """Sanity: we are actually bounding it, not inheriting a large SDK default."""
    assert 0 < _EMBED_SDK_MAX_RETRIES < 10


def test_persistent_5xx_retries_exactly_the_bound_then_fails():
    calls = {"n": 0}

    class _FakeClient:
        def post(self, url, json):  # noqa: A002 - matches httpx.Client.post signature
            calls["n"] += 1
            return _FakeResponse(503)

    with pytest.raises(ValueError, match="HTTP 503"):
        _post_with_one_retry(_FakeClient(), "/embeddings", {"model": "m", "input": []})

    # Exactly bound+1 attempts total — never silently retries more.
    assert calls["n"] == _EMBED_SDK_MAX_RETRIES + 1


def test_persistent_transport_error_retries_exactly_the_bound_then_raises():
    calls = {"n": 0}

    class _FakeClient:
        def post(self, url, json):  # noqa: A002
            calls["n"] += 1
            raise httpx.ConnectError("connection refused")

    with pytest.raises(httpx.ConnectError):
        _post_with_one_retry(_FakeClient(), "/embeddings", {"model": "m", "input": []})

    assert calls["n"] == _EMBED_SDK_MAX_RETRIES + 1


def test_success_after_one_transient_5xx_does_not_over_retry():
    calls = {"n": 0}

    class _FakeClient:
        def post(self, url, json):  # noqa: A002
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResponse(503)
            return _FakeResponse(200, {"data": []})

    response = _post_with_one_retry(
        _FakeClient(), "/embeddings", {"model": "m", "input": []}
    )
    assert response.status_code == 200
    assert calls["n"] == 2
