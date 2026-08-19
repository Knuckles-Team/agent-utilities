"""Seam 1 (CONCEPT:AU-KB-CURRENCY) — end-to-end proof that
``KnowledgeGraph.query(..., include_epistemic=True)`` carries the engine's
per-row epistemic envelope (score/confidence/valid+tx time/source_refs/
policy_labels) instead of flattening the result to a plain ``dict``.

This test stands up a REAL, ephemeral ``epistemic-graph-server`` (the sibling
``epistemic-graph`` checkout, which carries ``Method::ExplainProvenanceByIds`` and
the schema-bound ``EvidenceBundle`` projection, CONCEPT:EG-KB-CURRENCY), seeds a Claim +
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
from _test_engine import (
    TEST_AGENT_ID,
    TEST_AUDIENCE,
    TEST_POLICY_VERSION,
    TEST_SIGNER_KEY,
    TEST_TENANT,
    EngineBinaryIdentity,
    EngineUnavailable,
    bootstrap_context,
    request_context,
    resolve_engine_binary_identity,
    strict_server_env,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def _find_engine_binary() -> EngineBinaryIdentity | None:
    """Return the shared exact client/server artifact, or no artifact."""

    try:
        return resolve_engine_binary_identity()
    except EngineUnavailable:
        return None


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


class _ProcessServerHandle:
    """Lifecycle object registered before the auxiliary process is spawned."""

    def __init__(self, binary_identity: EngineBinaryIdentity) -> None:
        self.binary_identity = binary_identity
        self.proc: subprocess.Popen | None = None
        self.log_fh = None

    def binary_path_for_launch(self) -> str:
        """Recheck the retained artifact identity immediately before spawn."""

        self.binary_identity.verify_for_launch()
        return str(self.binary_identity.path)

    def stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        self.proc = None
        if self.log_fh is not None:
            self.log_fh.close()
            self.log_fh = None


@pytest.fixture()
def kb_currency_engine(tmp_path, monkeypatch, test_engine_lifecycle):
    """Start a discovered ``epistemic-graph-server`` on an isolated socket +
    persist dir, wire the AU engine resolver (``GRAPH_SERVICE_ENDPOINTS``/
    ``..._AUTH_SECRET``) at THIS engine, and tear it down after. Yields
    ``(socket_path, auth_secret)``. Skips when no engine binary is discoverable, or
    when the discovered build predates ``Method::ExplainProvenanceByIds``.
    """
    binary = _find_engine_binary()
    if binary is None:
        pytest.skip(
            "no exact epistemic-graph-server artifact available "
            "(set EPISTEMIC_GRAPH_SERVER_BIN or install epistemic-graph)"
        )

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    sock_path = _free_socket_path(tmp_path)

    # Register before spawning so setup failures also stop a partially-started
    # process.  The registry's stop order is graph delete -> transport close ->
    # process stop -> exact socket unlink.
    server = _ProcessServerHandle(binary)
    registration = test_engine_lifecycle.register_auxiliary_engine(
        server, socket_path=sock_path
    )
    try:
        auth_secret = "au-eg-" + "kb-currency-test-secret"  # nosec B105 - test-only

        log_path = tmp_path / "engine.log"
        server.log_fh = open(log_path, "wb")  # noqa: SIM115 - closed in stop()
        env = dict(os.environ)
        env.update(
            strict_server_env(str(tmp_path / "security"), auth_secret=auth_secret)
        )
        server.proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                server.binary_path_for_launch(),
                "--socket-path",
                sock_path,
                "--persist-dir",
                str(persist_dir),
                "--auth-secret",
                auth_secret,
                "--idle-shutdown-secs",
                "60",
            ],
            stdout=server.log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        proc = server.proc
        assert proc is not None
        _wait_for_socket(proc, sock_path, log_path)

        from epistemic_graph.client import SyncEpistemicGraphClient

        bootstrap = SyncEpistemicGraphClient.connect(
            socket_path=sock_path,
            auth_secret=auth_secret,
            verified_context=bootstrap_context(),
        )
        try:
            bootstrap.consensus.bootstrap_system_identity(
                agent_id=TEST_AGENT_ID,
                signer_id=TEST_AGENT_ID,
                signer_key=TEST_SIGNER_KEY,
            )
        finally:
            bootstrap.close()

        # This test manages its OWN throwaway engine — clear anything a
        # session-level fixture set (sharded/remote overrides) and point the
        # resolver at exactly this socket, mirroring tests/_test_engine.py's
        # EphemeralEngine wiring convention.
        monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", f"unix://{sock_path}")
        monkeypatch.setenv("GRAPH_SERVICE_AUTH_SECRET", auth_secret)
        yield sock_path, auth_secret
    finally:
        registration.stop()
        assert not Path(sock_path).exists(), (
            f"auxiliary engine socket survived lifecycle teardown: {sock_path}"
        )


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
    from agent_utilities.knowledge_graph.core.graph_compute import (
        GraphComputeEngine,
    )
    from agent_utilities.knowledge_graph.core.session import (
        GraphSession,
        reset_session,
        set_session,
    )
    from agent_utilities.knowledge_graph.facade import KnowledgeGraph
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    claim_id = f"claim-{uuid.uuid4().hex[:8]}"
    evidence_id = f"evidence-{uuid.uuid4().hex[:8]}"

    # ── Seed directly over the raw engine client (the "writes a Claim+Evidence
    # into a real/ephemeral engine" step) — same graph the facade's backend below
    # resolves to (via the isolate_graph_compute_engine remap), so both sides see
    # the same data.
    raw = SyncEpistemicGraphClient.connect(
        socket_path=sock_path,
        auth_secret=auth_secret,
        graph_name=test_graph_name,
        verified_context=request_context(),
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
                # secured_reads.scope() injects a mandatory
                # "n.tenant_id = <actor's tenant>" predicate (KG-2.6,
                # cross-org isolation) into every read below — the raw
                # engine-client seed here bypasses the guarded write path
                # that normally stamps this, so without it the node is
                # invisible to kg.query() regardless of the actor's
                # privileges (scope() has no privileged bypass; it is the
                # PRIMARY isolation boundary).
                "tenant_id": TEST_TENANT,
            },
        )
        raw.nodes.add(
            evidence_id,
            {
                "node_type": "Evidence",
                "confidence": 0.95,
                "tenant_id": TEST_TENANT,
            },
        )
        # "relationship" (not "relationship_type") is the canonical edge
        # property key the native engine's SUPPORTS-edge provenance scan
        # matches on (crates/eg-jobs/src/claim.rs: "two SUPPORTS edges
        # (relationship = 'SUPPORTS', the canonical key ...)"; every
        # Rust-side SUPPORTS edge construction uses this same key).
        raw.edges.add(evidence_id, claim_id, {"relationship": "SUPPORTS"})
    finally:
        raw.close()

    if not hasattr(SyncEpistemicGraphClient, "connect"):  # pragma: no cover
        pytest.skip("epistemic_graph client shape unexpected")

    # ── Read via the AU facade. Wire the facade's backend to the SAME engine
    # explicitly (bypassing the lazy `create_backend()` factory, which has no
    # `graph_name`/socket override) — this still exercises the real
    # `KnowledgeGraph.query`/`_attach_epistemic` code path under test.
    actor = ActorContext(
        actor_id=TEST_AGENT_ID,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=TEST_TENANT,
        scopes=frozenset({"kg:read", "kg:write", "kg:admin", "*"}),
        graph=test_graph_name,
        policy_version=TEST_POLICY_VERSION,
        audience=TEST_AUDIENCE,
    )
    token = set_session(session)
    kg: KnowledgeGraph | None = None
    try:
        # A bare EpistemicGraphBackend() independently resolves the ambient
        # ACTOR's tenant-routed default graph (resolve_routing_graph(None)
        # reads current_actor().tenant_id, NOT the GraphSession set just
        # above), not the explicit test_graph_name this test seeded through
        # the raw client. Construct the compute client bound to that exact
        # graph_name first and rebind the backend to it — the same idiom
        # already established for this defect elsewhere (D-OTR-2/D-OTR-3).
        # KnowledgeGraph.compute also requires an active process-owned
        # engine unless explicitly injected (see from_engine's docstring);
        # this test has no IntelligenceGraphEngine, so set kg._compute
        # directly too.
        compute = GraphComputeEngine(graph_name=test_graph_name, backend_type="rust")
        kg = KnowledgeGraph()
        kg._store = EpistemicGraphBackend()
        kg._store._graph = compute
        kg._compute = compute
        # ontology.permissioning's mandatory-marking lookup independently
        # constructs a BARE KnowledgeGraph() and reads its .store, which
        # requires IntelligenceGraphEngine.get_active() to be set (same
        # "active process-owned engine" requirement .compute has above) --
        # bind and activate a real engine over the SAME isolated backend so
        # that internal lookup resolves consistently instead of raising.
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        active_engine = IntelligenceGraphEngine(backend=kg._store)
        IntelligenceGraphEngine.set_active(active_engine)
        if not hasattr(kg.store.graph, "explain_provenance_by_ids"):  # pragma: no cover
            pytest.skip(
                "installed epistemic_graph client predates "
                "explain_provenance_by_ids (CONCEPT:EG-KB-CURRENCY)"
            )
        cypher = f"MATCH (n:Claim) WHERE n.id = '{claim_id}' RETURN n"

        # kg.query's read path also runs every row through
        # ontology.permissioning.enforce() (restricted_view ->
        # _acl_permits): a node with NO explicit ACL is denied to EVERY
        # actor, privileged or not ("No ACL defined — default deny") — the
        # raw engine-client seed above bypasses whatever write path would
        # normally register one. Register an explicit ACL granting this
        # test's actor read access via data_owner, the same helper
        # (build_acl) the permissioning module exports for exactly this.
        from agent_utilities.knowledge_graph.ontology.permissioning import build_acl

        build_acl(claim_id, data_owner=TEST_AGENT_ID)
        build_acl(evidence_id, data_owner=TEST_AGENT_ID)

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
        # The explicit test-engine lifecycle owns graph deletion and transport
        # close before its auxiliary server is stopped.  Only clear the active
        # facade authority here; closing the tracked client early would make the
        # lifecycle unable to perform its durable delete.
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine as _IGE,
        )

        _IGE.set_active(None)
        reset_session(token)
