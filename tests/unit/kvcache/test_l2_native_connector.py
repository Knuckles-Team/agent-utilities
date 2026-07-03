"""Unit tests for the LMCache ``native_plugin`` L2 connector (CONCEPT:KG-2.311).

Exercises :class:`EpistemicGraphL2Connector` — the native-client half of the
EG-187 L2 adapter — against a MOCK EG-187 ``/kv`` surface
(:class:`httpx.MockTransport`), so no live engine is needed. Covers the async
``native_plugin`` contract LMCache's :class:`NativeConnectorL2Adapter` drives:
``event_fd`` signalling, ``submit_batch_set`` / ``submit_batch_get`` /
``submit_batch_exists`` + ``drain_completions``, buffer fill on a hit, miss and
size-mismatch handling, and clean ``close``.
"""

from __future__ import annotations

import select
import time
from collections.abc import Iterator

import httpx
import pytest

from agent_utilities.core.http_client import create_http_client
from agent_utilities.kvcache import EpistemicGraphL2Connector, KvCacheConfig
from agent_utilities.kvcache.remote_backend import EpistemicGraphKVBackend

BASE = "http://kv.test"


class _FakeKvServer:
    """In-memory stand-in for the EG-187 ``/kv`` HTTP surface (KG-2.311)."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/kv/stats":
            return httpx.Response(
                200,
                json={
                    "unique_blocks": len(self.store),
                    "resident_bytes": sum(len(v) for v in self.store.values()),
                },
            )
        key = path[len("/kv/") :]
        if request.method == "PUT":
            new = key not in self.store
            self.store[key] = request.content
            return httpx.Response(201 if new else 200)
        if request.method == "HEAD":
            return httpx.Response(200 if key in self.store else 404)
        if request.method == "GET":
            if key in self.store:
                return httpx.Response(200, content=self.store[key])
            return httpx.Response(404)
        return httpx.Response(405)


@pytest.fixture
def connector(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[EpistemicGraphL2Connector]:
    """A connector whose backend is wired to a fresh in-memory EG-187 mock."""
    server = _FakeKvServer()

    def _make(cfg: KvCacheConfig) -> EpistemicGraphKVBackend:
        client = create_http_client(
            base_url=cfg.base_url, transport=httpx.MockTransport(server.handler)
        )
        return EpistemicGraphKVBackend(cfg, client=client)

    monkeypatch.setattr(
        "agent_utilities.kvcache.l2_native_connector.EpistemicGraphKVBackend",
        _make,
    )
    conn = EpistemicGraphL2Connector(base_url=BASE, num_workers=4)
    conn._server = server  # type: ignore[attr-defined]  # test-only handle
    yield conn
    conn.close()


def _await(conn: EpistemicGraphL2Connector, *future_ids: int, timeout: float = 5.0):
    """Poll the eventfd + drain until every ``future_id`` has completed."""
    pending = set(future_ids)
    results: dict[int, tuple[bool, str, list[bool] | None]] = {}
    poller = select.poll()
    poller.register(conn.event_fd(), select.POLLIN)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        poller.poll(200)
        for fid, ok, error, bools in conn.drain_completions():
            results[fid] = (ok, error, bools)
            pending.discard(fid)
    assert not pending, f"timed out waiting for {pending}"
    return results


def test_set_then_get_round_trip_kg_2_311(
    connector: EpistemicGraphL2Connector,
) -> None:
    """KG-2.311 — a stored block is fetched back byte-for-byte into the buffer."""
    key, blob = "tok-hash-1", b"\x00\x01paged-kv\xfe\xff"

    fid = connector.submit_batch_set([key], [memoryview(bytearray(blob))])
    _await(connector, fid)

    buf = bytearray(len(blob))
    gid = connector.submit_batch_get([key], [memoryview(buf)])
    res = _await(connector, gid)

    ok, _err, hits = res[gid]
    assert ok is True
    assert hits == [True]
    assert bytes(buf) == blob


def test_get_miss_leaves_buffer_and_reports_false_kg_2_311(
    connector: EpistemicGraphL2Connector,
) -> None:
    """KG-2.311 — an absent key is a miss; the pre-zeroed buffer is untouched."""
    buf = bytearray(8)
    gid = connector.submit_batch_get(["never-stored"], [memoryview(buf)])
    _ok, _err, hits = _await(connector, gid)[gid]
    assert hits == [False]
    assert bytes(buf) == bytes(8)


def test_exists_reflects_store_kg_2_311(
    connector: EpistemicGraphL2Connector,
) -> None:
    """KG-2.311 — batch exists probe flips once the key is stored."""
    eid = connector.submit_batch_exists(["k"])
    assert _await(connector, eid)[eid][2] == [False]

    sid = connector.submit_batch_set(["k"], [memoryview(bytearray(b"page"))])
    _await(connector, sid)

    eid2 = connector.submit_batch_exists(["k"])
    assert _await(connector, eid2)[eid2][2] == [True]


def test_batch_set_get_multiple_keys_kg_2_311(
    connector: EpistemicGraphL2Connector,
) -> None:
    """KG-2.311 — a mixed batch: two hits + one miss in one lookup."""
    blobs = {"a": b"aaaa", "b": b"bbbbbb"}
    sid = connector.submit_batch_set(
        list(blobs), [memoryview(bytearray(v)) for v in blobs.values()]
    )
    _await(connector, sid)

    keys = ["a", "b", "missing"]
    bufs = [bytearray(4), bytearray(6), bytearray(4)]
    gid = connector.submit_batch_get(keys, [memoryview(b) for b in bufs])
    _ok, _err, hits = _await(connector, gid)[gid]
    assert hits == [True, True, False]
    assert bytes(bufs[0]) == b"aaaa"
    assert bytes(bufs[1]) == b"bbbbbb"


def test_size_mismatch_is_miss_kg_2_311(
    connector: EpistemicGraphL2Connector,
) -> None:
    """KG-2.311 — a stored page that doesn't fit the load buffer is a safe miss."""
    connector._server.store["k"] = b"0123456789"  # type: ignore[attr-defined]
    buf = bytearray(4)  # smaller than the stored 10 bytes
    gid = connector.submit_batch_get(["k"], [memoryview(buf)])
    _ok, _err, hits = _await(connector, gid)[gid]
    assert hits == [False]
    assert bytes(buf) == bytes(4)  # buffer never partially written


def test_set_snapshots_bytes_synchronously_kg_2_311(
    connector: EpistemicGraphL2Connector,
) -> None:
    """KG-2.311 — mutating the source buffer after submit does not corrupt the store."""
    src = bytearray(b"original")
    fid = connector.submit_batch_set(["k"], [memoryview(src)])
    src[:] = b"MUTATED!"  # caller reuses the buffer immediately
    _await(connector, fid)

    buf = bytearray(len(b"original"))
    gid = connector.submit_batch_get(["k"], [memoryview(buf)])
    _await(connector, gid)
    assert bytes(buf) == b"original"


def test_close_is_idempotent_kg_2_311(
    connector: EpistemicGraphL2Connector,
) -> None:
    """KG-2.311 — close() may be called more than once without raising."""
    connector.close()
    connector.close()
