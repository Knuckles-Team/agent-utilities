"""Per-channel embedding backfill round-robin + capacity fan-out (CONCEPT:AU-KG.compute.per-channel-embedding-backfill).

A single ``WHERE embedding IS NULL LIMIT n`` FIFO lets one huge channel (822K
codebase ``Code`` chunks) starve a small url/doc crawl's chunks that share the
table. ``_collect_unembedded_rows`` instead round-robins across ``source_system``
channels so a starved channel makes progress every tick. The embed call is fanned
out to the embedding model's parallel capacity via ``make_embed_fn`` /
``map_concurrent_sync``.

These use a fake psycopg-style cursor (no live pgvector) to drive the SQL.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin


class _FakeCursor:
    """Minimal psycopg cursor over an in-memory table model.

    ``table`` is a list of dicts with keys id, text, source_system, embedded(bool).
    ``columns`` is the set of column names reported by information_schema.
    """

    def __init__(self, store):
        self.store = store
        self._result: list[tuple] = []

    def execute(self, sql, params=None):
        params = params or ()
        s = self.store
        if "information_schema.columns" in sql:
            self._result = [(c,) for c in sorted(s["columns"])]
            return
        if "SELECT DISTINCT source_system" in sql:
            chans = {r["source_system"] for r in s["rows"] if r["embedding"] is None}
            self._result = [(c,) for c in chans]
            return
        if sql.strip().startswith("SELECT id,"):
            # per-source (or plain) fetch of unembedded rows
            limit = params[-1]
            want_source = None
            source_is_null = "source_system IS NULL" in sql
            if "source_system = %s" in sql:
                want_source = params[0]
            out = []
            for r in s["rows"]:
                if r["embedding"] is not None:
                    continue
                if source_is_null and r["source_system"] is not None:
                    continue
                if want_source is not None and r["source_system"] != want_source:
                    continue
                out.append((r["id"], r["text"]))
                if len(out) >= limit:
                    break
            self._result = out
            return
        if sql.strip().startswith("UPDATE"):
            # UPDATE ... SET embedding = ... WHERE id = %s AND embedding IS NULL
            vec, nid = params
            for r in s["rows"]:
                if r["id"] == nid and r["embedding"] is None:
                    r["embedding"] = vec
            self._result = []
            return
        self._result = []

    def fetchall(self):
        return list(self._result)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, store):
        self.store = store
        self.committed = 0

    def cursor(self):
        return _FakeCursor(self.store)

    def commit(self):
        self.committed += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _conn_factory(store):
    def _f():
        return _FakeConn(store)

    return _f


class _Harness:
    _EMBED_TEXT_COLS = TaskManagerMixin._EMBED_TEXT_COLS
    _EMBED_SOURCE_CURSORS: dict[str, int] = {}
    _collect_unembedded_rows = TaskManagerMixin._collect_unembedded_rows

    def __init__(self):
        self._EMBED_SOURCE_CURSORS = {}


def _make_store(rows, columns=("id", "content", "embedding", "source_system")):
    return {
        "columns": set(columns),
        "rows": [
            {
                "id": rid,
                "text": txt,
                "source_system": src,
                "embedding": None,
            }
            for rid, txt, src in rows
        ],
    }


def test_round_robin_gives_starved_channel_progress():
    """codebase has 100 chunks, a tiny url crawl has 2 — one tick must still pull
    the url chunks, not just the head of the codebase FIFO."""
    rows = [(f"cb-{i}", f"code {i}", "codebase") for i in range(100)]
    rows += [("url-0", "page a", "webcrawl"), ("url-1", "page b", "webcrawl")]
    store = _make_store(rows)
    h = _Harness()

    items = h._collect_unembedded_rows(_conn_factory(store), "Code", take=10)
    ids = {nid for nid, _ in items}
    # The starved webcrawl channel made progress in the SAME tick.
    assert "url-0" in ids or "url-1" in ids
    # And codebase chunks still get a share too.
    assert any(str(nid).startswith("cb-") for nid in ids)
    assert len(items) <= 10


def test_rotation_cursor_advances_to_share_lead():
    """The lead channel rotates tick-to-tick so no channel permanently leads."""
    rows = [("a-0", "x", "alpha"), ("b-0", "y", "beta"), ("c-0", "z", "gamma")]
    store = _make_store(rows)
    h = _Harness()
    # per_channel = max(1, take//3); with take=3 each channel gets 1 slot.
    first = h._collect_unembedded_rows(_conn_factory(store), "Code", take=3)
    # cursor advanced — leading channel differs from a fresh-cursor harness.
    assert h._EMBED_SOURCE_CURSORS["Code"] == 1
    assert len(first) == 3  # all three channels covered


def test_falls_back_to_plain_scan_without_source_column():
    rows = [(f"n-{i}", f"t{i}", None) for i in range(5)]
    store = _make_store(rows, columns=("id", "content", "embedding"))
    h = _Harness()
    items = h._collect_unembedded_rows(_conn_factory(store), "Concept", take=3)
    assert len(items) == 3
    # No source column → no rotation cursor created.
    assert "Concept" not in h._EMBED_SOURCE_CURSORS


def test_single_channel_uses_plain_scan():
    rows = [(f"cb-{i}", f"c{i}", "codebase") for i in range(4)]
    store = _make_store(rows)
    h = _Harness()
    items = h._collect_unembedded_rows(_conn_factory(store), "Code", take=2)
    assert len(items) == 2
    # Single channel → no rotation needed.
    assert "Code" not in h._EMBED_SOURCE_CURSORS


def test_blank_text_rows_are_dropped():
    rows = [("a", "", "s1"), ("b", "real", "s1"), ("c", "  ", "s1")]
    store = _make_store(rows)
    h = _Harness()
    items = h._collect_unembedded_rows(_conn_factory(store), "Code", take=10)
    assert items == [("b", "real")]


# --- capacity fan-out of the embed call (CONCEPT:AU-KG.compute.per-channel-embedding-backfill / KG-2.143) --------
def test_backfill_embed_fn_fans_out_to_capacity(monkeypatch):
    """The backfill's ``make_embed_fn`` runs up to ``capacity`` embed BATCHES
    concurrently (scaling with vLLM instances), order-preserving."""
    import threading
    import time

    from agent_utilities.core import model_concurrency
    from agent_utilities.knowledge_graph.enrichment import semantic

    model_concurrency.reset_controllers()

    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    class _FakeEmbModel:
        def get_text_embedding_batch(self, texts):
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            time.sleep(0.02)
            with lock:
                state["current"] -= 1
            return [[float(len(t))] for t in texts]

    monkeypatch.setattr(
        "agent_utilities.core.embedding_utilities.create_embedding_model",
        lambda: _FakeEmbModel(),
    )
    monkeypatch.setattr(
        "agent_utilities.core.model_concurrency.resolve_capacity",
        lambda model=None, default=1: 4,
    )

    # batch_size=1 → one batch per text, so 12 texts = 12 batches fanned to cap 4.
    embed_fn = semantic.make_embed_fn(batch_size=1)
    texts = [f"t{i}" for i in range(12)]
    out = embed_fn(texts)

    assert len(out) == 12  # one vector per text, order preserved
    assert out[0] == [2.0] and out[11] == [3.0]  # len("t0")=2, len("t11")=3
    assert state["max"] == 4  # saturated to capacity
    model_concurrency.reset_controllers()
