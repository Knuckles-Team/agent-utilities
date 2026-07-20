"""Unit + live-integration tests for the zero-copy KV snapshot→fork driver.

CONCEPT:EG-KG.memory.zero-copy-snapshot-fork — exercises the fork rung on
:class:`EpistemicGraphKVBackend` (``snapshot`` / ``fork`` / ``branch_get`` /
``branch_put`` / ``fork_stats``). Two layers:

* A MOCK-HTTP unit test (:class:`_FakeForkServer` over :class:`httpx.MockTransport`)
  that proves the snapshot → fork → branch_get roundtrip and copy-on-write
  ``branch_put`` without a live engine, plus the graceful-degradation posture
  (every error → a safe default, never a raise).
* A LIVE-integration test guarded by a reachability check against the engine's KV
  HTTP surface, so CI without a running engine still passes (it skips). It proves
  the same roundtrip AND that ``/kv/fork/stats`` reports SHARED (Arc'd, not copied)
  bytes — the zero-copy invariant.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import httpx
import pytest

from agent_utilities.core.http_client import create_http_client
from agent_utilities.kvcache import EpistemicGraphKVBackend, KvCacheConfig

BASE = "http://kv.test"

# Live engine KV HTTP surface (anonymous loopback). Overridable for a non-default bind.
LIVE_BASE = os.environ.get("EPISTEMIC_GRAPH_KVCACHE_URL", "http://127.0.0.1:9141")


class _FakeForkServer:
    """In-memory stand-in for the engine ``/kv`` + fork surface (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork).

    Models the zero-copy semantics faithfully enough to test the driver: a snapshot
    pins the CURRENT bytes of the requested keys; a fork reads through to the
    snapshot's pinned pages unless the branch has written its own copy-on-write
    override; ``/kv/fork/stats`` reports shared vs overlay bytes.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.snapshots: dict[int, dict[str, bytes]] = {}
        self.branches: dict[
            int, tuple[int, dict[str, bytes]]
        ] = {}  # bid -> (snap_id, overlay)
        self._next_snap = 1
        self._next_branch = 1

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if path == "/kv/snapshot" and method == "POST":
            keys = json.loads(request.content or b"{}").get("keys", [])
            pinned = {k: self.store[k] for k in keys if k in self.store}
            sid = self._next_snap
            self._next_snap += 1
            self.snapshots[sid] = pinned
            return httpx.Response(200, json={"snapshot": sid, "pages": len(pinned)})

        if (
            path.startswith("/kv/snapshot/")
            and path.endswith("/fork")
            and method == "POST"
        ):
            sid = int(path[len("/kv/snapshot/") : -len("/fork")])
            if sid not in self.snapshots:
                return httpx.Response(404)
            bid = self._next_branch
            self._next_branch += 1
            self.branches[bid] = (sid, {})
            return httpx.Response(200, json={"branch": bid})

        if path.startswith("/kv/branch/"):
            _, _, _, bid_s, key = path.split("/", 4)
            bid = int(bid_s)
            if bid not in self.branches:
                return httpx.Response(404)
            snap_id, overlay = self.branches[bid]
            if method == "PUT":
                overlay[key] = request.content  # copy-on-write, branch-local
                return httpx.Response(200)
            if method == "GET":
                if key in overlay:
                    return httpx.Response(200, content=overlay[key])
                pinned = self.snapshots.get(snap_id, {})
                if key in pinned:  # read through to the SHARED page
                    return httpx.Response(200, content=pinned[key])
                return httpx.Response(404)

        if path == "/kv/fork/stats" and method == "GET":
            shared = sum(
                len(v) for pins in self.snapshots.values() for v in pins.values()
            )
            overlay = sum(
                len(v) for _sid, ov in self.branches.values() for v in ov.values()
            )
            return httpx.Response(
                200,
                json={
                    "branches": len(self.branches),
                    "snapshots": len(self.snapshots),
                    "shared_bytes": shared,
                    "shared_pages": sum(len(p) for p in self.snapshots.values()),
                    "overlay_bytes": overlay,
                    "overlay_pages": sum(len(ov) for _s, ov in self.branches.values()),
                    "resident_fork_bytes": shared + overlay,
                },
            )

        # Plain /kv/<key> PUT so the test can seed pages the snapshot pins.
        if path.startswith("/kv/") and method == "PUT":
            key = path[len("/kv/") :]
            new = key not in self.store
            self.store[key] = request.content
            return httpx.Response(201 if new else 200)

        return httpx.Response(405)


def _backend(server: _FakeForkServer) -> EpistemicGraphKVBackend:
    client = create_http_client(
        base_url=BASE, transport=httpx.MockTransport(server.handler)
    )
    return EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE), client=client)


# ── mocked-HTTP unit tests (no live engine) ───────────────────────────────────
def test_snapshot_fork_branch_get_roundtrip_mock() -> None:
    """CONCEPT:EG-KG.memory.zero-copy-snapshot-fork — snapshot → fork → branch_get returns the pinned page."""
    server = _FakeForkServer()
    backend = _backend(server)

    backend.put("fk1", b"hello-page-1")
    backend.put("fk2", b"hello-page-2")

    snap = backend.snapshot(["fk1", "fk2"])
    assert isinstance(snap, int)

    bid = backend.fork(snap)
    assert isinstance(bid, int)

    # Branch reads through to the SHARED snapshot page (no copy yet).
    assert backend.branch_get(bid, "fk1") == b"hello-page-1"
    assert backend.branch_get(bid, "fk2") == b"hello-page-2"


def test_branch_put_is_copy_on_write_mock() -> None:
    """CONCEPT:EG-KG.memory.zero-copy-snapshot-fork — a branch_put is private to its branch; siblings keep the shared page."""
    server = _FakeForkServer()
    backend = _backend(server)
    backend.put("fk1", b"shared")

    snap = backend.snapshot(["fk1"])
    b1 = backend.fork(snap)
    b2 = backend.fork(snap)

    assert backend.branch_put(b1, "fk1", b"b1-override") is True
    # b1 sees its own override; b2 still reads the shared page — isolation.
    assert backend.branch_get(b1, "fk1") == b"b1-override"
    assert backend.branch_get(b2, "fk1") == b"shared"


def test_fork_stats_reports_shared_bytes_mock() -> None:
    """CONCEPT:EG-KG.memory.zero-copy-snapshot-fork — many branches off one snapshot keep shared_bytes flat."""
    server = _FakeForkServer()
    backend = _backend(server)
    backend.put("fk1", b"0123456789")  # 10 bytes

    snap = backend.snapshot(["fk1"])
    ids = [backend.fork(snap) for _ in range(4)]
    assert all(isinstance(i, int) for i in ids)

    stats = backend.fork_stats()
    assert stats["branches"] == 4
    # THE zero-copy proof: 4 branches, but the page is stored ONCE (shared, not ×4).
    assert stats["shared_bytes"] == 10
    assert stats["overlay_bytes"] == 0  # no branch has written its own copy yet


def test_fork_rung_transport_error_degrades_mock() -> None:
    """CONCEPT:EG-KG.memory.zero-copy-snapshot-fork — engine outage ⇒ safe defaults, never a raise."""

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("engine unreachable")

    client = create_http_client(base_url=BASE, transport=httpx.MockTransport(boom))
    backend = EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE), client=client)

    assert backend.snapshot(["fk1"]) is None
    assert backend.fork(1) is None
    assert backend.branch_get(1, "fk1") is None
    assert backend.branch_put(1, "fk1", b"x") is False
    assert backend.fork_stats() == {}


def test_fork_rung_bad_status_and_json_degrades_mock() -> None:
    """CONCEPT:EG-KG.memory.zero-copy-snapshot-fork — a 500 / malformed body degrades, not crashes."""

    def bad(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"not json")

    client = create_http_client(base_url=BASE, transport=httpx.MockTransport(bad))
    backend = EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE), client=client)
    assert backend.snapshot(["fk1"]) is None
    assert backend.fork(1) is None
    assert backend.fork_stats() == {}


def test_branch_get_miss_returns_none_mock() -> None:
    """CONCEPT:EG-KG.memory.zero-copy-snapshot-fork — a key absent from the snapshot is a None miss."""
    server = _FakeForkServer()
    backend = _backend(server)
    backend.put("fk1", b"present")
    snap = backend.snapshot(["fk1"])
    bid = backend.fork(snap)
    assert backend.branch_get(bid, "never-pinned") is None


# ── live-integration (skipped when the engine KV surface is unreachable) ───────
def _engine_reachable(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/kv/fork/stats", timeout=1.0) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


live_only = pytest.mark.skipif(
    not _engine_reachable(LIVE_BASE),
    reason=f"live engine KV surface not reachable at {LIVE_BASE}",
)


@live_only
def test_live_snapshot_fork_branch_get_and_shared_bytes() -> None:
    """CONCEPT:EG-KG.memory.zero-copy-snapshot-fork — against the LIVE engine: roundtrip + shared_bytes proof.

    Seeds a page, snapshots it, forks N branches, reads the shared page back through
    each branch, and asserts ``/kv/fork/stats`` reports positive ``shared_bytes`` — the
    branches read the SAME physical page by Arc rather than each holding a copy.
    """
    backend = EpistemicGraphKVBackend(KvCacheConfig(base_url=LIVE_BASE))
    try:
        key, blob = "kv-fork-test-page", b"\x00\x01live-paged-kv-block\xff" * 8
        assert backend.put(key, blob) is True

        snap = backend.snapshot([key])
        assert isinstance(snap, int), (
            "snapshot must return an id against the live engine"
        )

        branch_ids = [backend.fork(snap) for _ in range(3)]
        assert all(isinstance(b, int) for b in branch_ids), branch_ids

        # Roundtrip: every branch reads the SAME pinned page zero-copy.
        for bid in branch_ids:
            assert backend.branch_get(bid, key) == blob

        # Copy-on-write is branch-local: one branch overrides, the others are untouched.
        assert backend.branch_put(branch_ids[0], key, b"branch-0-override") is True
        assert backend.branch_get(branch_ids[0], key) == b"branch-0-override"
        assert backend.branch_get(branch_ids[1], key) == blob

        stats = backend.fork_stats()
        assert stats, "live /kv/fork/stats must return a populated dict"
        assert int(stats.get("branches", 0)) >= 3
        # THE zero-copy invariant: pages are SHARED (Arc'd), not copied per branch.
        assert int(stats.get("shared_bytes", 0)) > 0
    finally:
        backend.close()
