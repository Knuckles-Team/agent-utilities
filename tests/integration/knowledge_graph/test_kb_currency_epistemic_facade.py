"""Seam 1 (CONCEPT:AU-KB-CURRENCY) — end-to-end proof that
``KnowledgeGraph.query(..., include_epistemic=True)`` carries the engine's
per-row epistemic envelope (score/confidence/valid+tx time/source_refs/
policy_labels) instead of flattening the result to a plain ``dict``.

This test stands up a REAL, ephemeral ``epistemic-graph-server`` (the sibling
``epistemic-graph`` checkout, which carries ``Method::ExplainProvenanceByIds`` /
the widened ``ExplainProvenanceRowWire``, CONCEPT:EG-KB-CURRENCY), seeds a Claim +
Evidence node pair with a real confidence + bitemporal window and a ``SUPPORTS``
edge directly over the raw engine client, then asserts the values the facade
returns via ``include_epistemic=True`` originated in the engine (the confidence,
the bitemporal window, AND the derived ``source_refs``/``policy_labels`` the
engine's belief-substrate resolution computes from the ``SUPPORTS`` edge — values
this test never computes itself).

Skips (does not fail) when no engine binary is discoverable — this is an
integration/measurement proof against a real database, not a unit gate (the same
convention ``test_shard_write_parallelism.py`` uses).
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def _find_engine_binary() -> str | None:
    """Locate a built ``epistemic-graph-server`` (CONCEPT:EG-KB-CURRENCY: needs
    ``Method::ExplainProvenanceByIds``, merged to the sibling checkout's ``main``).

    Same discovery convention as ``test_shard_write_parallelism.py``: an explicit
    override, then a sibling ``epistemic-graph`` checkout's ``target/release``
    (walking up from this file so a worktree layout resolves too), then the
    canonical workspace location, then whatever is on ``PATH``.
    """
    env = os.environ.get("EPISTEMIC_GRAPH_SERVER_BIN")
    if env and os.path.exists(env):
        return env
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
    canonical = (
        "/home/apps/workspace/agent-packages/epistemic-graph/"
        "target/release/epistemic-graph-server"
    )
    if os.path.exists(canonical):
        return canonical
    import shutil

    return shutil.which("epistemic-graph-server")


def _free_socket_path(root: Path) -> str:
    """A unique, short ephemeral UDS path under ``root`` (UDS paths are length-
    limited; keep the name short and rely on the unique ``root`` for isolation)."""
    return str(root / f"eg-{uuid.uuid4().hex[:8]}.sock")


def _wait_for_socket(proc: subprocess.Popen, sock_path: str, log_path: Path) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_bytes()[-4000:].decode("utf-8", "replace")
            raise RuntimeError(
                f"epistemic-graph-server exited early "
                f"(code {proc.returncode}) during startup:\n{tail}"
            )
        if os.path.exists(sock_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect(sock_path)
                return
            except OSError:
                pass
        time.sleep(0.1)
    raise RuntimeError("epistemic-graph-server did not become ready in time")


@pytest.fixture()
def kb_currency_engine(tmp_path, monkeypatch):
    """Start a discovered ``epistemic-graph-server`` on an isolated socket +
    persist dir, wire the AU engine resolver (``GRAPH_SERVICE_SOCKET``/
    ``..._AUTH_SECRET``) at THIS engine, and tear it down after. Yields
    ``(socket_path, auth_secret)``. Skips when no engine binary is discoverable, or
    when the discovered build predates ``Method::ExplainProvenanceByIds``.
    """
    binary = _find_engine_binary()
    if binary is None:
        pytest.skip(
            "no epistemic-graph-server binary discoverable "
            "(EPISTEMIC_GRAPH_SERVER_BIN, sibling checkout target/release, or PATH)"
        )

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    sock_path = _free_socket_path(tmp_path)
    auth_secret = "au-eg-kb-currency-test-secret"  # nosec B105 - test-only

    log_path = tmp_path / "engine.log"
    log_fh = open(log_path, "wb")  # noqa: SIM115 - closed in finally below
    env = dict(os.environ)
    env["GRAPH_SERVICE_AUTH_SECRET"] = auth_secret
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            str(binary),
            "--socket-path",
            sock_path,
            "--persist-dir",
            str(persist_dir),
            "--auth-secret",
            auth_secret,
            "--idle-shutdown-secs",
            "60",
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        _wait_for_socket(proc, sock_path, log_path)

        # This test manages its OWN throwaway engine — clear anything a
        # session-level fixture set (sharded/remote overrides) and point the
        # resolver at exactly this socket, mirroring tests/_test_engine.py's
        # EphemeralEngine wiring convention.
        for var in ("GRAPH_SERVICE_ENDPOINTS", "ENGINE_ENDPOINT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GRAPH_SERVICE_SOCKET", sock_path)
        monkeypatch.setenv("GRAPH_SERVICE_AUTH_SECRET", auth_secret)
        monkeypatch.setenv("EPISTEMIC_GRAPH_AUTOSTART", "0")
        yield sock_path, auth_secret
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_fh.close()


def test_facade_query_include_epistemic_carries_engine_confidence_and_evidence(
    kb_currency_engine, isolate_graph_compute_engine
):
    """Write a Claim+Evidence(+SUPPORTS) pair straight into the real engine, then
    prove ``KnowledgeGraph.query(..., include_epistemic=True)`` returns
    :class:`EpistemicRow` results carrying the confidence, the bitemporal window,
    AND the belief-substrate-derived ``source_refs``/``policy_labels`` — values
    the engine computed from the ``SUPPORTS`` edge, never fabricated AU-side.

    ``isolate_graph_compute_engine`` (autouse elsewhere in the suite) remaps a
    bare/``__commons__`` ``GraphComputeEngine(graph_name=...)`` to a per-test
    unique graph so tests never collide — requesting it here (autouse fixtures
    can still be explicitly depended on) gets its yielded per-test graph name so
    the raw seeding client below targets the SAME graph the facade's
    ``EpistemicGraphBackend`` resolves to.
    """
    sock_path, auth_secret = kb_currency_engine
    test_graph_name = isolate_graph_compute_engine

    try:
        from epistemic_graph.client import SyncEpistemicGraphClient
    except ImportError:  # pragma: no cover
        pytest.skip("epistemic_graph client not importable")

    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.epistemic_row import EpistemicRow
    from agent_utilities.knowledge_graph.facade import KnowledgeGraph

    claim_id = f"claim-{uuid.uuid4().hex[:8]}"
    evidence_id = f"evidence-{uuid.uuid4().hex[:8]}"

    # ── Seed directly over the raw engine client (the "writes a Claim+Evidence
    # into a real/ephemeral engine" step) — same graph the facade's backend below
    # resolves to (via the isolate_graph_compute_engine remap), so both sides see
    # the same data.
    raw = SyncEpistemicGraphClient.connect(
        socket_path=sock_path, auth_secret=auth_secret, graph_name=test_graph_name
    )
    try:
        try:
            raw.tenants.create(test_graph_name)
        except Exception:  # noqa: BLE001 - "already exists" is fine
            pass
        raw.nodes.add(
            claim_id,
            {
                "node_type": "Claim",
                "name": "kb-currency test claim",
                "confidence": 0.83,
                "valid_from": 1_700_000_000,
                "valid_until": 1_800_000_000,
                "tx_from": 1_650_000_000,
            },
        )
        raw.nodes.add(
            evidence_id,
            {"node_type": "Evidence", "confidence": 0.95},
        )
        raw.edges.add(evidence_id, claim_id, {"relationship_type": "SUPPORTS"})
    finally:
        raw.close()

    if not hasattr(SyncEpistemicGraphClient, "connect"):  # pragma: no cover
        pytest.skip("epistemic_graph client shape unexpected")

    # ── Read via the AU facade. Wire the facade's backend to the SAME engine
    # explicitly (bypassing the lazy `create_backend()` factory, which has no
    # `graph_name`/socket override) — this still exercises the real
    # `KnowledgeGraph.query`/`_attach_epistemic` code path under test.
    kg = KnowledgeGraph()
    kg._store = EpistemicGraphBackend()
    if not hasattr(kg.store.graph, "explain_provenance_by_ids"):  # pragma: no cover
        pytest.skip(
            "installed epistemic_graph client predates "
            "explain_provenance_by_ids (CONCEPT:EG-KB-CURRENCY)"
        )
    try:
        cypher = f"MATCH (n:Claim) WHERE n.id = '{claim_id}' RETURN n"

        # Default path — byte-for-byte unaffected: plain dict rows.
        plain_rows = kg.query(cypher)
        assert len(plain_rows) == 1
        assert isinstance(plain_rows[0], dict)
        assert plain_rows[0]["n"]["id"] == claim_id

        # Opt-in path — the Seam 1 currency upgrade.
        rows = kg.query(cypher, include_epistemic=True)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, EpistemicRow)
        assert row.id == claim_id
        assert row.kind == "Claim"

        # Confidence + bitemporal window: straight field copies off the engine's
        # KnowledgeRow for THIS exact node — proves the numbers originated
        # server-side, not just echoed from the write we issued (a
        # differently-computed/rounded value here would be a fabrication bug).
        assert row.confidence == pytest.approx(0.83)
        assert row.calibration == pytest.approx(0.83)
        assert row.valid_time == (1_700_000_000, 1_800_000_000)
        assert row.tx_time[0] == 1_650_000_000

        # Evidence provenance + policy label: DERIVED by the engine's belief-
        # substrate resolution from the SUPPORTS edge we wrote above — this is
        # NOT a stored property on the Claim node, so its presence here proves
        # real server-side epistemic resolution ran (CONCEPT:E2/E3/X1), not a
        # client-side echo of the write.
        assert evidence_id in row.source_refs
        assert row.policy_labels, "engine should classify a SUPPORTS-only claim"

        # Opting in never drops the plain properties a caller would have gotten
        # from the default path.
        assert row.properties.get("name") == "kb-currency test claim"
    finally:
        # Close the facade's own client explicitly, THEN null the tracked
        # `GraphComputeEngine._client` reference. The autouse
        # `isolate_graph_compute_engine` fixture's own (later-running, since
        # autouse fixtures tear down after explicitly-requested ones) finalizer
        # calls `engine._client.clear()`/`.tenants.delete()` UNCONDITIONALLY on
        # every engine it tracked — including this one — and by the time it
        # runs, `kb_currency_engine`'s finalizer has already killed this test's
        # engine subprocess. An RPC against that dead socket would otherwise
        # retry/block with no bounded timeout (`SyncEpistemicGraphClient.clear()`
        # calls `future.result()` with none). Nulling `_client` here makes that
        # finalizer's own `if hasattr(engine, "_client") and engine._client:`
        # guard skip it cleanly instead.
        graph_engine = getattr(kg.store, "graph", None)
        client = getattr(graph_engine, "_client", None)
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        if graph_engine is not None:
            graph_engine._client = None
