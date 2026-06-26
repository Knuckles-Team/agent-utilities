"""FanOut durable-mirror dialect + resilience regressions (CONCEPT:KG-2.74).

Covers two live mirror bugs (observed repeating 200+ times in graph-os logs) and the
poison-entry safety net that stops the runaway retry:

* Neo4j mirror got ``MATCH (n) WHERE n.id = $id RETURN label(n) as lbl`` — but
  ``label(n)`` is NOT valid Neo4j Cypher ("Unknown function 'label'"); the plural
  ``labels(n)`` (a list) is. The engine now picks the dialect from the backend's
  declared ``cypher_support`` capability (correct through a fan-out wrapper), so a
  full-openCypher mirror gets ``labels(n)[0]`` and a bounded-subset store keeps
  ``label(n)``.
* FalkorDB mirror got "Missing parameters" because an empty ``{}`` params map still
  made the client prepend a ``CYPHER `` header; the backend now binds ``params or
  None`` so a parameter-free query applies cleanly.
* A PERMANENT apply error (malformed/incompatible cypher) is dropped after a few
  confirmations instead of stalling the mirror and spamming the log forever.

All run against in-process fakes — no real Neo4j/FalkorDB server required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends import fanout_backend
from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.backends.fanout_backend import (
    FanOutBackend,
    _is_permanent_apply_error,
)
from agent_utilities.knowledge_graph.migration import _portable_writer


class _CapBackend(GraphBackend):
    """Recording backend with a configurable ``cypher_support`` capability."""

    def __init__(self, cypher_support: str = "full") -> None:
        self._cs = cypher_support
        self.queries: list[str] = []

    @property
    def cypher_support(self) -> str:
        return self._cs

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        # Answer the engine's label lookup so ``_upsert_edge`` proceeds to the MERGE.
        if "as lbl" in query:
            return [{"lbl": "Doc"}]
        return []

    def execute_batch(self, query: str, batch: list[dict[str, Any]]):
        return []

    def create_schema(self) -> None:
        pass

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        pass

    def semantic_search(self, query_embedding, n_results: int = 5):
        return []

    def prune(self, criteria: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


def _label_lookups(be: _CapBackend) -> list[str]:
    return [q for q in be.queries if "as lbl" in q]


def test_full_cypher_backend_emits_labels_plural() -> None:
    """A full-openCypher mirror (Neo4j/FalkorDB/AGE) MUST get ``labels(n)`` — the
    singular ``label(n)`` is rejected by Neo4j as 'Unknown function'."""
    be = _CapBackend("full")
    _portable_writer(be)._upsert_edge("a", "b", "RELATED_TO", {})

    lookups = _label_lookups(be)
    assert lookups, "expected the engine to issue a label-lookup query"
    for q in lookups:
        assert "labels(n)" in q, f"full-cypher backend must use labels(): {q!r}"
        # The invalid Neo4j form must never be emitted here.
        assert "RETURN label(n)" not in q, f"label(n) is invalid Neo4j Cypher: {q!r}"


def test_subset_cypher_backend_keeps_singular_label() -> None:
    """A bounded-subset store (epistemic-graph / pggraph transpiler) keeps the
    singular ``label(n)`` it understands — the fix must not break it."""
    be = _CapBackend("subset")
    _portable_writer(be)._upsert_edge("a", "b", "RELATED_TO", {})

    lookups = _label_lookups(be)
    assert lookups
    for q in lookups:
        assert "label(n)" in q and "labels(n)" not in q, q


# --------------------------------------------------------------------------- #
# FalkorDB param binding
# --------------------------------------------------------------------------- #


class _FakeResult:
    result_set: list[Any] = []
    header: list[Any] = []


class _FakeFalkorGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def query(self, q: str, params: Any = None) -> _FakeResult:
        self.calls.append((q, params))
        return _FakeResult()


def _falkor_with_fake_graph():
    from agent_utilities.knowledge_graph.backends.contrib import falkordb_backend

    be = falkordb_backend.FalkorDBBackend.__new__(falkordb_backend.FalkorDBBackend)
    be.graph = _FakeFalkorGraph()  # type: ignore[attr-defined]
    return be


def test_falkor_empty_params_bind_none_not_empty_map() -> None:
    """Empty params must reach the client as ``None`` (no spurious ``CYPHER `` header
    → no "Missing parameters")."""
    be = _falkor_with_fake_graph()
    be.execute("MATCH (n) RETURN n")
    assert be.graph.calls[-1][1] is None


def test_falkor_real_params_are_bound() -> None:
    """A parameterized query keeps its bound params."""
    be = _falkor_with_fake_graph()
    be.execute("MATCH (n) WHERE n.id = $id RETURN n", {"id": "x"})
    assert be.graph.calls[-1][1] == {"id": "x"}


# --------------------------------------------------------------------------- #
# Poison-entry drop (no infinite retry / log spam)
# --------------------------------------------------------------------------- #


def test_permanent_error_classifier() -> None:
    assert _is_permanent_apply_error(RuntimeError("Unknown function 'label'"))
    assert _is_permanent_apply_error(RuntimeError("Missing parameters"))
    assert _is_permanent_apply_error(RuntimeError("SyntaxError near ..."))
    # A transient outage is NOT permanent — it must keep retrying.
    assert not _is_permanent_apply_error(ConnectionError("connection refused"))
    assert not _is_permanent_apply_error(TimeoutError("timed out"))


class _AuthorityBackend(GraphBackend):
    def execute(self, query, params=None):
        return [{"ok": True}]

    def execute_batch(self, query, batch):
        return []

    def create_schema(self) -> None:
        pass

    def add_embedding(self, node_id, embedding) -> None:
        pass

    def semantic_search(self, query_embedding, n_results: int = 5):
        return []

    def prune(self, criteria) -> None:
        pass

    def close(self) -> None:
        pass


class _PoisonMirror(_AuthorityBackend):
    """A mirror whose every apply fails with a PERMANENT (un-retryable) error."""

    def execute(self, query, params=None):
        raise RuntimeError("Unknown function 'label'")


def test_poison_entry_is_dropped_not_retried_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed mirror write is skipped after a few confirmations so the mirror
    keeps draining (lag→0) instead of blocking + spamming the log 200+ times."""
    # Tiny backoff so the few permanent-error confirmations complete fast.
    monkeypatch.setattr(fanout_backend, "_BASE_BACKOFF_S", 0.01)
    monkeypatch.setattr(fanout_backend, "_MAX_BACKOFF_S", 0.02)

    fan = FanOutBackend(
        _AuthorityBackend(),
        {"poison": _PoisonMirror()},
        outbox_path=str(tmp_path / "outbox.db"),
    )
    try:
        fan.execute("CREATE (n:Doc {id:'1'})", is_write=True)
        fan.execute("CREATE (n:Doc {id:'2'})", is_write=True)
        # The poison entries can never apply, but the drainer drops them, so the
        # mirror fully drains rather than stalling forever.
        assert fan.flush_mirrors(timeout=15.0), "poison entries were never dropped"
        stats = fan.durability_stats()["mirrors"]["poison"]
        assert stats["lag"] == 0
        assert stats["dropped"] == 2
        assert stats["stalled"] is False
    finally:
        fan.close()
