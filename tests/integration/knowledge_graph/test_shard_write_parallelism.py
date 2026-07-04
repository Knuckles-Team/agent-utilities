"""Durable cross-shard write-parallelism (CONCEPT:AU-KG.ingest.floor-codebase-admission-cap) — live engine proof.

The submission-side bottleneck this guards against: when ingestion writes ONE
graph at a time, every commit funnels onto the ONE redb shard writer that owns
that graph (EG-026 ``FNV-1a(name) % K``), leaving K-1 writers idle — the profiled
single-hot-``eg-redb`` symptom and the ``parallelism_factor`` ceiling. Routing
distinct sources to distinct per-repo graphs (KG-2.269) plus admitting enough
concurrent codebase ingests to fill the shard width (KG-2.279 ``codebase_cap``
shard floor) makes independent-graph writes fan across ALL K shard writers.

This test reproduces the symptom and the fix at the substrate: writing to K
graphs that hash to K DISTINCT shards activates more shard-writer threads than
writing the same volume to K graphs that all hash to ONE shard. It is skipped
unless a built ``epistemic-graph-server`` binary is discoverable (no engine in
CI by default) — it is a measurement/integration proof, not a unit gate.
"""

from __future__ import annotations

import asyncio
import glob
import os
import re
import shutil
import subprocess
import time

import pytest

pytestmark = pytest.mark.concept("AU-KG.ingest.floor-codebase-admission-cap")

_K = 4


def _find_engine_binary() -> str | None:
    """Locate a built epistemic-graph-server with the K-way sharded redb writer.

    Prefers an explicit override and a freshly-built sibling ``target/release``
    over a possibly-stale ``epistemic-graph-server`` on ``PATH`` (an old on-PATH
    build may predate the EG-026 sharded writer and run in snapshot/WAL mode)."""
    env = os.environ.get("EPISTEMIC_GRAPH_SERVER_BIN")
    if env and os.path.exists(env):
        return env
    # Walk up from this test file looking for a sibling epistemic-graph checkout
    # (handles both the canonical layout and a detached worktree).
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        for sub in ("epistemic-graph", "agent-packages/epistemic-graph"):
            cand = os.path.join(cur, sub, "target", "release", "epistemic-graph-server")
            if os.path.exists(cand):
                return cand
        cur = os.path.dirname(cur)
    # Canonical workspace location as a final concrete candidate.
    canonical = (
        "/home/apps/workspace/agent-packages/epistemic-graph/"
        "target/release/epistemic-graph-server"
    )
    if os.path.exists(canonical):
        return canonical
    # Last resort — an on-PATH build (may be stale; the test asserts sharding so a
    # non-sharded build simply fails loudly rather than silently passing).
    return shutil.which("epistemic-graph-server")


def _shard_of(name: str, k: int = _K) -> int:
    """Mirror the engine's EG-026 ``FNV-1a(sanitized_name) % K`` shard routing."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    h = 0xCBF29CE484222325
    for b in sanitized.encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % k


def _names_for_shards() -> tuple[list[str], list[str]]:
    """Return (distinct: one name per shard 0..K-1, same: K names all on shard 0)."""
    distinct: dict[int, str] = {}
    same: list[str] = []
    i = 0
    while len(distinct) < _K or len(same) < _K:
        g = f"bench:p{i}"
        s = _shard_of(g)
        distinct.setdefault(s, g)
        if s == 0 and len(same) < _K:
            same.append(g)
        i += 1
    return [distinct[s] for s in range(_K)], same


def _shard_commit_marks(persist: str) -> dict[str, tuple[int, float]]:
    """``(size, mtime)`` of each durable ``graph-<i>.redb`` shard file.

    Each shard file is owned by exactly one ``eg-redb-writer`` thread and is
    rewritten (new mtime, possibly larger) on every durable commit to a graph that
    routes to it. So the SET of shard files touched during a write phase IS the set
    of shard writers that did durable work — a deterministic, timing-independent
    proxy for "writers active" (the 10ms jiffie granularity is too coarse for fast
    commits, and redb reuses free pages so size alone can stay flat). mtime is the
    reliable per-commit signal."""
    out: dict[str, tuple[int, float]] = {}
    for f in glob.glob(os.path.join(persist, "graph-*.redb")):
        try:
            st = os.stat(f)
            out[os.path.basename(f)] = (st.st_size, st.st_mtime)
        except OSError:
            continue
    return out


def _touched(
    before: dict[str, tuple[int, float]], after: dict[str, tuple[int, float]]
) -> int:
    """Count shard files that were committed to (newly appeared, grew, or got a
    fresher mtime) between two snapshots."""
    n = 0
    for k in set(before) | set(after):
        b = before.get(k)
        a = after.get(k)
        if a is None:
            continue
        if b is None or a[0] > b[0] or a[1] > b[1]:
            n += 1
    return n


def test_cross_graph_writes_fan_across_shard_writers(tmp_path, monkeypatch):
    binary = _find_engine_binary()
    if binary is None:
        pytest.skip("no built epistemic-graph-server binary found")
    try:
        from epistemic_graph.client import EpistemicGraphClient
    except ImportError:  # pragma: no cover
        pytest.skip("epistemic_graph client not importable")

    # This test stands up its OWN throwaway engine; drop any session/engine-fixture
    # env so neither the engine subprocess nor the client picks up a foreign socket,
    # endpoint, or auth secret (which would make us talk to a different engine).
    for var in (
        "GRAPH_SERVICE_SOCKET",
        "GRAPH_SERVICE_AUTH_SECRET",
        "GRAPH_SERVICE_ENDPOINTS",
        "GRAPH_SERVICE_PERSIST_DIR",
        "EPISTEMIC_GRAPH_REDB_SHARDS",
    ):
        monkeypatch.delenv(var, raising=False)

    # Bind a free ephemeral port so a stray engine from a prior run can never be
    # mistaken for ours (the connect would otherwise succeed against the squatter
    # while OUR persist dir stays empty).
    import socket

    with socket.socket() as _s:
        _s.bind(("127.0.0.1", 0))
        port = _s.getsockname()[1]
    persist = tmp_path / "data"
    persist.mkdir()
    env = dict(os.environ)
    env["EPISTEMIC_GRAPH_REDB_SHARDS"] = str(_K)
    env["EPISTEMIC_GRAPH_ALLOW_INSECURE"] = "1"
    # Force the durable K-way sharded redb backend (the subject under test); the
    # test env may otherwise select the snapshot/WAL backend (per-graph ``.wal``
    # files, no ``graph-<i>.redb`` shards).
    env["EPISTEMIC_GRAPH_PERSIST_BACKEND"] = "redb"
    env["EPISTEMIC_GRAPH_REDB_AUTHORITATIVE"] = "1"
    # A stray host-engine UDS socket in the shared XDG runtime dir could collide;
    # give this throwaway engine its own socket. It must be a SHORT path — a UDS
    # path under pytest's long tmp basetemp blows the ~108-char ``SUN_LEN`` limit.
    sock_path = f"/tmp/eg-test-{port}.sock"
    env["GRAPH_SERVICE_SOCKET"] = sock_path
    # Route engine logs to a FILE, not a PIPE: the engine logs verbosely, and an
    # undrained ``subprocess.PIPE`` deadlocks it once the 64 KB buffer fills.
    log_path = tmp_path / "engine.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [
            binary,
            "--tcp-addr",
            f"127.0.0.1:{port}",
            "--persist-dir",
            str(persist),
            "--checkpoint-interval",
            "300",
            "--allow-insecure",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_fh,
    )
    try:
        ep = f"127.0.0.1:{port}"

        async def _conn(g: str):
            return await EpistemicGraphClient.connect(
                tcp_addr=ep, auth_secret="", graph_name=g
            )

        async def _wait_ready() -> None:
            last: Exception | None = None
            for _ in range(75):
                if proc.poll() is not None:  # engine exited (e.g. bind failure)
                    err = ""
                    try:
                        err = log_path.read_text(errors="replace")[-400:]
                    except OSError:
                        pass
                    pytest.skip(f"engine exited early (rc={proc.returncode}): {err}")
                try:
                    # The ephemeral free port guarantees this is OUR engine (a
                    # squatter can't hold a port we just allocated), so a successful
                    # connect is sufficient readiness. (Shard ``graph-<i>.redb`` files
                    # are created lazily on first write, so we must NOT gate on them.)
                    c = await _conn("__commons__")
                    await c.close()
                    return
                except Exception as e:  # noqa: BLE001 — engine still booting
                    last = e
                await asyncio.sleep(0.2)
            raise RuntimeError(f"engine did not become ready: {last!r}")

        async def _write_concurrent(graphs: list[str], n: int) -> int:
            # Snapshot BEFORE creating the per-graph tenants so a shard file that is
            # newly created by this phase counts as touched even if the filesystem
            # mtime granularity is too coarse to see the subsequent commit.
            before = _shard_commit_marks(str(persist))
            conns = []
            for g in graphs:
                c = await _conn(g)
                try:
                    await c.tenants.create(g)
                except Exception:  # noqa: BLE001 — already exists
                    pass
                conns.append(c)
            batches = [
                [
                    {
                        "op": "add_node",
                        "id": f"{g}:x{j}",
                        "properties": {"label": "B", "blob": "x" * 64},
                    }
                    for j in range(n)
                ]
                for g in graphs
            ]
            await asyncio.gather(
                *[
                    c.lifecycle.batch_update(b)
                    for c, b in zip(conns, batches, strict=True)
                ]
            )
            for c in conns:
                await c.close()
            # Give the authoritative writer a beat to land the commit on disk.
            time.sleep(1.1)
            after = _shard_commit_marks(str(persist))
            return _touched(before, after)

        async def _run() -> tuple[int, int]:
            await _wait_ready()
            distinct, same = _names_for_shards()
            cross_active = await _write_concurrent(distinct, 20000)
            same_active = await _write_concurrent(same, 20000)
            return cross_active, same_active

        cross_active, same_active = asyncio.run(_run())

        # The proof: independent-graph writes fan across ALL K shard files (each
        # owned by its own writer thread); same-shard writes pin ONE. Cross must
        # touch every shard while same-shard touches at most one — the exact gap
        # between K busy writers and the profiled single-hot-writer bottleneck.
        # (``same_active`` is asserted ``<= 1`` rather than ``== 1`` only to absorb
        # the filesystem mtime granularity on a tmpfs tmp_path; the load-bearing
        # claim is ``cross_active == K`` >> ``same_active``.)
        assert cross_active == _K, (
            f"cross-graph writes must fan across all {_K} shard writers, "
            f"got {cross_active}"
        )
        assert same_active <= 1, (
            f"same-shard writes must pin <=1 shard writer, got {same_active}"
        )
        assert cross_active > same_active, (
            f"cross-graph ({cross_active}) must write more shards than "
            f"same-shard ({same_active})"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        log_fh.close()
