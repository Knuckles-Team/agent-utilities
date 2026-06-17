"""Generic write-layer content-hash delta for ingest_external_batch (KG-2.9).

Every connector becomes incremental at the write layer: an entity whose stored
content_hash is unchanged is skipped (no MERGE, no re-reasoning) even on a full
fetch.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


class DeltaBackend(GraphBackend):
    """Stateful stand-in that records content_hashes and answers the prefetch."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.node_batches: list[list[dict]] = []

    def execute(self, query: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        if "RETURN n.id AS id, n.content_hash AS h" in query:
            return [
                {"id": i, "h": self.store[i]}
                for i in params.get("ids", [])
                if i in self.store
            ]
        return []

    def execute_batch(
        self, query: str, batch_params: list[dict]
    ) -> list[dict[str, Any]]:
        if "MERGE (n:" in query:
            self.node_batches.append(batch_params)
            for row in batch_params:
                if row.get("id") is not None and row.get("content_hash"):
                    self.store[str(row["id"])] = str(row["content_hash"])
        return []

    def add_embedding(self, *a, **k):  # pragma: no cover - unused
        ...

    def create_schema(self, *a, **k):  # pragma: no cover - unused
        ...

    def prune(self, *a, **k):  # pragma: no cover - unused
        ...

    def semantic_search(self, *a, **k):  # pragma: no cover - unused
        return []

    def close(self):  # pragma: no cover - unused
        ...


def _ents():
    return [
        {"id": "a:1", "type": "Thing", "name": "X"},
        {"id": "a:2", "type": "Thing", "name": "Y"},
    ]


def test_write_delta_skips_unchanged_then_writes_changed(monkeypatch):
    monkeypatch.delenv("KG_WRITE_DELTA", raising=False)
    eng = IntelligenceGraphEngine(backend=DeltaBackend())

    # 1) First ingest: nothing stored → both written.
    r1 = eng.ingest_external_batch("d", _ents())
    assert r1["nodes"] == 2
    assert r1["skipped_unchanged"] == 0

    # 2) Re-ingest identical data → both skipped, zero writes.
    r2 = eng.ingest_external_batch("d", _ents())
    assert r2["nodes"] == 0
    assert r2["skipped_unchanged"] == 2

    # 3) Change one entity → only the changed one is written.
    changed = [
        {"id": "a:1", "type": "Thing", "name": "X"},
        {"id": "a:2", "type": "Thing", "name": "CHANGED"},
    ]
    r3 = eng.ingest_external_batch("d", changed)
    assert r3["nodes"] == 1
    assert r3["skipped_unchanged"] == 1


def test_write_delta_disabled_writes_everything(monkeypatch):
    monkeypatch.setenv("KG_WRITE_DELTA", "0")
    eng = IntelligenceGraphEngine(backend=DeltaBackend())
    eng.ingest_external_batch("d", _ents())
    r2 = eng.ingest_external_batch("d", _ents())
    # No delta when disabled: unchanged rows are re-written, none skipped.
    assert r2["nodes"] == 2
    assert r2["skipped_unchanged"] == 0


def test_content_hash_is_stable_and_excludes_volatile():
    eng = IntelligenceGraphEngine(backend=DeltaBackend())
    a = {"id": "x", "name": "n", "observedAt": "t1"}
    b = {"id": "x", "name": "n", "observedAt": "t2"}
    # Volatile timestamp + id excluded → identical semantic hash.
    assert eng._content_hash(a) == eng._content_hash(b)
    c = {"id": "x", "name": "DIFFERENT", "observedAt": "t1"}
    assert eng._content_hash(a) != eng._content_hash(c)
