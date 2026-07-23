"""CONCEPT:AU-KG.retrieval.embedding-fast-fail — query-time embedding fast-fail.

Covers the three pieces of the fix:

* an already-OPEN embedding circuit breaker skips the network embed call
  entirely and falls back to keyword search immediately (no retry cost paid
  on every query once the endpoint is known-bad);
* a failed embed call records the outcome into that SAME breaker so it trips
  for subsequent calls;
* the fallback log surfaces the REAL root cause (an SDK-wrapped exception's
  chained ``__cause__``) instead of a generic wrapper message.
"""

from __future__ import annotations

import logging

import pytest

from agent_utilities.knowledge_graph.retrieval import hybrid_retriever as hr_module
from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import HybridRetriever


class _StubBreaker:
    """Minimal circuit-breaker stand-in — records every ``record()`` call."""

    def __init__(self, *, tripped: bool):
        self._tripped = tripped
        self.calls: list[dict] = []

    def is_tripped(self) -> bool:
        return self._tripped

    def record(self, *, ok: bool, status: int | None = None) -> None:
        self.calls.append({"ok": ok, "status": status})


class _StubEndpoint:
    model_key = "embedding"


class _StubEmbedModel:
    """Records whether/how many times ``get_text_embedding`` was invoked."""

    def __init__(self, *, error: Exception | None = None):
        self._error = error
        self.calls: list[str] = []

    def get_text_embedding(self, text: str):
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return [0.1, 0.2, 0.3]


class _StubBackend:
    pass


class _StubEngine:
    """The minimal engine surface ``retrieve_hybrid`` touches on the keyword-
    fallback path with ``skip_quality_gate=True`` and an empty result set."""

    def __init__(self):
        self.backend = _StubBackend()
        self.keyword_calls: list[tuple[str, int]] = []

    def _search_keyword(self, query: str, top_k: int):
        self.keyword_calls.append((query, top_k))
        return []


def _make_retriever(monkeypatch, *, breaker, embed_model) -> tuple[HybridRetriever, _StubEngine]:
    engine = _StubEngine()
    retriever = HybridRetriever(engine, enable_rerank=False)
    # Bypass the lazy ``embed_model`` property (which would otherwise call the
    # hermetic-test-blocked ``create_embedding_model``) with our stub directly.
    retriever._embed_model = embed_model
    retriever._embed_model_initialized = True
    monkeypatch.setattr(
        hr_module, "_query_embedding_circuit_breaker", lambda: (_StubEndpoint(), breaker)
    )
    return retriever, engine


@pytest.mark.concept(id="AU-KG.retrieval.embedding-fast-fail")
def test_open_breaker_skips_embedding_call_entirely(monkeypatch):
    """An already-tripped breaker means: no network embed attempt at all."""
    breaker = _StubBreaker(tripped=True)
    embed_model = _StubEmbedModel()
    retriever, engine = _make_retriever(monkeypatch, breaker=breaker, embed_model=embed_model)

    result = retriever.retrieve_hybrid("what is the deployment architecture?", skip_quality_gate=True)

    assert result == []
    assert embed_model.calls == []  # the network call was never attempted
    assert engine.keyword_calls == [("what is the deployment architecture?", 10)]
    assert breaker.calls == []  # nothing to record — we never attempted a call


@pytest.mark.concept(id="AU-KG.retrieval.embedding-fast-fail")
def test_embedding_failure_records_breaker_and_falls_back(monkeypatch):
    """A live embedding failure records into the breaker and still degrades
    to keyword search (unchanged external behaviour, now instrumented)."""
    breaker = _StubBreaker(tripped=False)
    boom = RuntimeError("boom")
    embed_model = _StubEmbedModel(error=boom)
    retriever, engine = _make_retriever(monkeypatch, breaker=breaker, embed_model=embed_model)

    result = retriever.retrieve_hybrid("what is the deployment architecture?", skip_quality_gate=True)

    assert result == []
    assert embed_model.calls == ["what is the deployment architecture?"]
    assert engine.keyword_calls == [("what is the deployment architecture?", 10)]
    assert breaker.calls == [{"ok": False, "status": None}]


@pytest.mark.concept(id="AU-KG.retrieval.embedding-fast-fail")
def test_successful_embedding_records_breaker_success(monkeypatch):
    """A successful query embed still records ``ok=True`` (keeps the
    breaker's CLOSED state fed by real retrieval traffic, not just bulk
    ingestion)."""
    breaker = _StubBreaker(tripped=False)
    embed_model = _StubEmbedModel()
    retriever, engine = _make_retriever(monkeypatch, breaker=breaker, embed_model=embed_model)
    # The stub engine has no ``.graph``, so ``_engine_vector_search`` returns no
    # candidates and retrieval degrades to keyword search via its OWN "no
    # semantic matches" branch (not the exception branch) — the embed call
    # itself still succeeded, which is what we're asserting here.
    retriever.retrieve_hybrid("hello", skip_quality_gate=True)

    assert embed_model.calls == ["hello"]
    assert breaker.calls == [{"ok": True, "status": None}]
    assert engine.keyword_calls == [("hello", 10)]


@pytest.mark.concept(id="AU-KG.retrieval.embedding-fast-fail")
def test_fallback_log_surfaces_real_root_cause(monkeypatch, caplog):
    """The fallback warning must include the REAL cause, not a generic
    SDK-wrapper message like openai's fixed 'Connection error.' string."""
    breaker = _StubBreaker(tripped=False)

    class _WrappedConnectionError(RuntimeError):
        pass

    try:
        try:
            raise PermissionError("outbound destination was rejected")
        except PermissionError as inner:
            raise _WrappedConnectionError("Connection error.") from inner
    except _WrappedConnectionError as wrapped:
        error = wrapped

    embed_model = _StubEmbedModel(error=error)
    retriever, _engine = _make_retriever(monkeypatch, breaker=breaker, embed_model=embed_model)

    with caplog.at_level(logging.WARNING, logger=hr_module.__name__):
        retriever.retrieve_hybrid("hello", skip_quality_gate=True)

    assert "Connection error." in caplog.text
    assert "outbound destination was rejected" in caplog.text


# ── pure helper functions ────────────────────────────────────────────────────


def test_describe_embedding_failure_surfaces_chained_cause():
    try:
        try:
            raise PermissionError("outbound destination was rejected")
        except PermissionError as inner:
            raise RuntimeError("Connection error.") from inner
    except RuntimeError as wrapped:
        described = hr_module._describe_embedding_failure(wrapped)

    assert "Connection error." in described
    assert "outbound destination was rejected" in described
    assert "PermissionError" in described


def test_describe_embedding_failure_no_cause_returns_plain_str():
    err = RuntimeError("plain failure")
    assert hr_module._describe_embedding_failure(err) == "plain failure"


def test_http_status_of_reads_status_code_attribute():
    class _Err(Exception):
        status_code = 503

    assert hr_module._http_status_of(_Err()) == 503


def test_http_status_of_reads_response_status_code():
    class _Resp:
        status_code = 429

    class _Err(Exception):
        response = _Resp()

    assert hr_module._http_status_of(_Err()) == 429


def test_http_status_of_returns_none_for_opaque_failure():
    assert hr_module._http_status_of(RuntimeError("connection error")) is None
