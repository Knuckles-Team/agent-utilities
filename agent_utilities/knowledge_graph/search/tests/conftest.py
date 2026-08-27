"""Fixtures shared by the CA-24 OpenSearch search-tier tests.

``FakeOpenSearch`` implements the small subset of the real ``opensearch-py``
client surface :mod:`~..client` calls (``indices.exists/create/delete``,
``index``/``get``/``delete``/``search``/``count``/``delete_by_query``) purely
in-process, so :class:`~..client.OpenSearchClient`'s REAL code (not a bypass
of it) is exercised by every test in this package — the "fixture/mock
OpenSearch client" the lane doc calls for while `services/opensearch`
integration is exercised separately (see ``test_live_integration.py``,
marked ``@pytest.mark.live``).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator
from typing import Any

import pytest

from agent_utilities.knowledge_graph.ontology import permissioning
from agent_utilities.knowledge_graph.search.client import (
    OpenSearchClient,
    OpenSearchNotFoundError,
)

try:  # pragma: no cover - exercised only when the optional dep is installed
    from opensearchpy.exceptions import NotFoundError as _RealNotFoundError
except ImportError:  # pragma: no cover
    _RealNotFoundError = None

# The exception FakeOpenSearch raises for "not found" must be whichever type
# OpenSearchClient's own except-clauses are actually catching in THIS
# environment (real opensearchpy if installed, our OpenSearchNotFoundError
# fallback otherwise) — see client.py's own try/except ImportError shim.
_NOT_FOUND_EXC = _RealNotFoundError or OpenSearchNotFoundError


def _match_pattern(pattern: str, name: str) -> bool:
    if "," in pattern:
        return any(_match_pattern(p.strip(), name) for p in pattern.split(","))
    return fnmatch.fnmatchcase(name, pattern)


def _field_matches(field_value: Any, target: Any) -> bool:
    if isinstance(field_value, list):
        return target in field_value
    return field_value == target


def _match_query(query: dict[str, Any] | None, source: dict[str, Any]) -> bool:
    if not query or "match_all" in query:
        return True
    if "term" in query:
        ((field, value),) = query["term"].items()
        return _field_matches(source.get(field), value)
    if "terms" in query:
        ((field, values),) = query["terms"].items()
        return any(_field_matches(source.get(field), v) for v in values)
    if "bool" in query:
        clauses = query["bool"]
        for clause in clauses.get("must", []):
            if not _match_query(clause, source):
                return False
        for clause in clauses.get("filter", []):
            if not _match_query(clause, source):
                return False
        for clause in clauses.get("must_not", []):
            if _match_query(clause, source):
                return False
        return True
    raise NotImplementedError(f"FakeOpenSearch cannot evaluate query {query!r}")


class _FakeIndices:
    """Mirrors real OpenSearch semantics: ``exists``/``delete`` accept an
    index-pattern (wildcard) too, not only a literal name — a rebuild's
    ``tenant_wildcard`` drop (``kg-<tenant>-*``) must match/remove every
    concrete index under it, exactly like the real ``indices.delete``
    API does."""

    def __init__(self, store: dict[str, dict[str, dict[str, Any]]]) -> None:
        self._store = store

    def _matches(self, pattern: str) -> list[str]:
        return [name for name in self._store if _match_pattern(pattern, name)]

    def exists(self, index: str) -> bool:
        if index in self._store:
            return True
        return bool(self._matches(index))

    def create(self, index: str, body: Any = None) -> dict[str, Any]:
        self._store.setdefault(index, {})
        return {"acknowledged": True, "index": index}

    def delete(self, index: str) -> dict[str, Any]:
        for name in self._matches(index):
            self._store.pop(name, None)
        self._store.pop(index, None)
        return {"acknowledged": True}


class FakeOpenSearch:
    """In-process double for the ``opensearch-py`` client surface."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, dict[str, Any]]] = {}
        self.indices = _FakeIndices(self._store)

    def index(
        self, index: str, id: str, body: dict[str, Any], refresh: Any = False
    ) -> dict[str, Any]:  # noqa: A002
        self._store.setdefault(index, {})[id] = dict(body)
        return {"_index": index, "_id": id, "result": "created"}

    def get(self, index: str, id: str) -> dict[str, Any]:  # noqa: A002
        try:
            source = self._store[index][id]
        except KeyError:
            raise _NOT_FOUND_EXC(404, "not_found", {}) from None
        return {"_index": index, "_id": id, "_source": source, "found": True}

    def delete(self, index: str, id: str) -> dict[str, Any]:  # noqa: A002
        try:
            del self._store[index][id]
        except KeyError:
            raise _NOT_FOUND_EXC(404, "not_found", {}) from None
        return {"result": "deleted"}

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        query = body.get("query")
        size = body.get("size", 10)
        hits: list[dict[str, Any]] = []
        for idx_name, docs in self._store.items():
            if not _match_pattern(index, idx_name):
                continue
            for doc_id, source in docs.items():
                if _match_query(query, source):
                    hits.append({"_index": idx_name, "_id": doc_id, "_source": source})
        return {"hits": {"total": {"value": len(hits)}, "hits": hits[:size]}}

    def count(self, index: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        query = (body or {}).get("query")
        total = 0
        for idx_name, docs in self._store.items():
            if not _match_pattern(index, idx_name):
                continue
            total += sum(1 for source in docs.values() if _match_query(query, source))
        return {"count": total}

    def delete_by_query(
        self, index: str, body: dict[str, Any], ignore_unavailable: bool = True
    ) -> dict[str, Any]:
        query = body.get("query")
        deleted = 0
        for idx_name in list(self._store):
            if not _match_pattern(index, idx_name):
                continue
            for doc_id in list(self._store[idx_name]):
                if _match_query(query, self._store[idx_name][doc_id]):
                    del self._store[idx_name][doc_id]
                    deleted += 1
        return {"deleted": deleted}


def make_client(raw: FakeOpenSearch | None = None) -> OpenSearchClient:
    return OpenSearchClient(client=raw if raw is not None else FakeOpenSearch())


def require_doc(client: OpenSearchClient, index: str, doc_id: str) -> dict[str, Any]:
    """``get_document`` is typed ``dict[str, Any] | None`` (a real absent-doc
    case the production code must handle) -- tests that already know a doc
    was just written assert non-None once here rather than repeating an
    ``assert ... is not None`` (or an unchecked ``[...]``, which mypy
    correctly refuses) at every call site."""
    doc = client.get_document(index, doc_id)
    assert doc is not None, f"expected a document at {index}/{doc_id}, found none"
    return doc


class _FakeMarkingStore:
    """Minimal durable-store double for ``permissioning``'s hydrate/persist
    calls — an in-memory Cypher-shaped store is unnecessary; markings are
    populated directly via ``permissioning.apply_marking`` in tests, which
    round-trips through this store's ``execute`` without needing it to
    actually interpret the query text."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def execute(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        self.calls.append((query, params))
        if query.strip().upper().startswith("MATCH"):
            return []
        return []


@pytest.fixture
def marking_authority() -> Iterator[_FakeMarkingStore]:
    """Install an isolated, empty marking authority for one test — markings
    a test applies via ``permissioning.apply_marking(..., tenant=...)`` are
    visible to :func:`~..dls.markings_for_node` for the duration of the
    test, and every bit of registry/hydration state is restored afterward
    (``permissioning.use_marking_authority``'s own contract)."""
    store = _FakeMarkingStore()
    with permissioning.use_marking_authority(store):
        yield store
