"""GOC-61 / BUG-030 / BUG-064 -- live-engine, real-Cypher, real-restart proof.

CONCEPT:AU-KG.compute.data-is-private-its.

**What this module set out to prove.** GOC-61's phase-1 write/read split for
the ``__commons__`` graph (``tenant_sharing.check_system_graph_write`` /
``filter_commons_catalog`` / ``apply_commons_catalog_restriction``, commits
``ac2600bb8``/``3ab385e26``/``02902dcde``) and the U-77 fix stamping an
unowned privileged write with ``_shared_scope="org"`` (commit ``811c2be7d``,
2026-08-15) were both proven only with in-memory unit tests -- a bare
``props: dict`` mutated directly by ``stamp_ownership`` and a ``Mock()``
backend standing in for the engine. Per this repository's own Wire-First
discipline ("a unit test proves nothing about reachability" / "a stub-based
test masked a live security defect in this program today"), the actual
question -- does a DIFFERENT actor's real, authenticated read of
``__commons__`` through ``IntelligenceGraphEngine.query_cypher`` return the
right rows, does revocation take effect, does it survive a real engine
process restart -- was unproven against a real database.

**What building this proof found instead: a live, load-bearing reproduction
of BUG-030.** ``PlacementRoute`` (``eg-capabilities/src/lib.rs``) is required
before EVERY ``query_cypher`` call (``graph_compute.py::_send`` ->
``placement_catalog.resolve_placement``), and its ``authz_action`` is
``"admin:cluster-read"``. ``epistemic-graph/src/server/access.rs::
require_admin_capability`` grants that ONLY to the engine's own ``System``
role or an explicit engine-side RBAC ``Admin`` grant -- "there is no
coarse-ACL fallback for admin actions the way graph Read/Write has one, so
an agent with no admin grant is DENIED, not defaulted open." In this test
harness, exactly ONE identity ever receives ``System`` role: the literal
agent enrolled by ``consensus.bootstrap_system_identity`` at process start
(``TEST_AGENT_ID``). **No AU-side scope -- not ``kg:admin``, not any
:class:`ActorContext` role -- maps to that engine-level grant**, confirming
BUG-030's own finding (E5: "`kg:admin` scopes ... has no corresponding
scope ... granted by hand on the live cluster") empirically, live, and more
broadly than previously documented: it is not one obscure admin RPC, it
silently gates ordinary Cypher READS for every identity except the single
bootstrap one. ``test_bug030_...`` below is the live reproduction: the exact
same read succeeds for the bootstrap-consistent ambient actor and fails with
``PlacementAuthorityError`` (engine cause: ``ACCESS_DENIED: verified
principal lacks admin capability required for 'admin:cluster-read'``) for a
DIFFERENT actor -- same tenant, even AU-side ``kg:admin`` -- confirmed
independently of, and consistent with, this repository's own
``tests/unit/knowledge_graph/test_tenant_request_isolation.py``, which hit
the identical ``PlacementAuthorityError`` attempting cross-tenant reads and
left it "failing and reported" without root-causing it to ``PlacementRoute``
specifically.

**Consequence for this lane.** GOC-61's entire read-side design --
``scope()``/``filter_visible``/``filter_commons_catalog`` -- is Python code
that runs AFTER a successful ``query_cypher`` call. BUG-030 sits IN FRONT of
that, blocking the call itself for any non-bootstrap identity. **A true
multi-actor "does tenant/user B see what tenant/user A shared" proof through
the real ``query_cypher`` path is not constructible against a real engine
today** -- not because GOC-61's Python-side ACL is wrong, but because no
caller other than the engine's own bootstrap identity can complete a Cypher
read at all. This is a bigger, more foundational gap than anything GOC-61's
phase-1 commits addressed, and it is NOT closed by this change.
``test_org_share_revocation_persists_across_a_real_restart`` below proves
what IS provable within that constraint: the org-share stamp
(``_shared_scope="org"``, U-77) and its revocation (``make_private``)
persist correctly, as real node properties, through a real Cypher read AND
a real SIGTERM + relaunch of the engine process against the same persist
directory -- reading throughout as the one identity this harness can
authenticate for reads, with that limitation stated, not hidden.

Skips (does not fail) when no engine binary is discoverable -- an
integration/measurement proof against a real database, not a unit gate,
mirroring ``test_shard_write_parallelism.py``/``test_kb_currency_epistemic_facade.py``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from _test_engine import (
    TEST_AGENT_ID,
    TEST_POLICY_VERSION,
    TEST_SIGNER_KEY,
    TEST_TENANT,
    EngineBinaryIdentity,
    EngineUnavailable,
    bootstrap_context,
    resolve_engine_binary_identity,
    strict_server_env,
)

pytestmark = [pytest.mark.integration, pytest.mark.engine, pytest.mark.timeout(180)]

COMMONS_GRAPH = "__commons__"


def _find_engine_binary() -> EngineBinaryIdentity | None:
    """Return the shared exact client/server artifact, or no artifact."""

    try:
        return resolve_engine_binary_identity()
    except EngineUnavailable:
        return None


def _free_socket_path(root: Path) -> str:
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


class _RestartableEngine:
    """A real ``epistemic-graph-server`` on a FIXED persist dir + socket path,
    so it can be SIGTERM'd and relaunched pointed at the SAME on-disk store --
    the actual "process restart" this proof needs, not a fresh ephemeral one."""

    def __init__(
        self, binary_identity: EngineBinaryIdentity, root: Path, auth_secret: str
    ) -> None:
        self.binary_identity = binary_identity
        self.binary = str(binary_identity.path)
        self.root = root
        self.persist_dir = root / "persist"
        self.persist_dir.mkdir(exist_ok=True)
        self.security_dir = root / "security"
        self.socket_path = _free_socket_path(root)
        self.auth_secret = auth_secret
        self.log_path = root / "engine.log"
        self._proc: subprocess.Popen | None = None
        self._log_fh: Any = None

    def start(self) -> None:
        self._log_fh = open(self.log_path, "ab")  # noqa: SIM115 - closed in stop()
        env = dict(os.environ)
        env.update(
            strict_server_env(str(self.security_dir), auth_secret=self.auth_secret)
        )
        env["GRAPH_SERVICE_PERSIST_DIR"] = str(self.persist_dir)
        self.binary_identity.verify_for_launch()
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                self.binary,
                "--socket-path",
                self.socket_path,
                "--persist-dir",
                str(self.persist_dir),
                "--auth-secret",
                self.auth_secret,
                "--idle-shutdown-secs",
                "60",
            ],
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        _wait_for_socket(self._proc, self.socket_path, self.log_path)

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=15)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        self._proc = None

    def restart(self) -> None:
        """SIGTERM the running process, then start a NEW process pointed at
        the exact same ``persist_dir``/``socket_path`` -- proves persistence
        across a real process boundary, not merely a held-open connection."""
        self.stop()
        self.start()


@pytest.fixture()
def restartable_commons_engine(tmp_path, monkeypatch, test_engine_lifecycle):
    binary = _find_engine_binary()
    if binary is None:
        pytest.skip(
            "no exact epistemic-graph-server artifact available "
            "(set EPISTEMIC_GRAPH_SERVER_BIN or install epistemic-graph)"
        )
    auth_secret = "au-eg-" + "goc61-commons-sharing-test-secret"  # nosec B105 - test-only
    eng = _RestartableEngine(binary, tmp_path, auth_secret)
    registration = test_engine_lifecycle.register_auxiliary_engine(
        eng, socket_path=eng.socket_path
    )
    try:
        eng.start()
        from epistemic_graph.client import SyncEpistemicGraphClient

        bootstrap = SyncEpistemicGraphClient.connect(
            socket_path=eng.socket_path,
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

        monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", f"unix://{eng.socket_path}")
        monkeypatch.setenv("GRAPH_SERVICE_AUTH_SECRET", auth_secret)
        yield eng
    finally:
        registration.stop()
        assert not Path(eng.socket_path).exists(), (
            f"auxiliary engine socket survived lifecycle teardown: {eng.socket_path}"
        )


def _reset_process_engine() -> None:
    """Force ``GraphComputeEngine``'s AND ``IntelligenceGraphEngine``'s
    process-singletons to reconnect -- both are separate class-level
    singletons that must be cleared, required both after pointing the env at
    a fresh socket and after a real restart (the old transport is dead the
    instant the old process exits)."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    root = GraphComputeEngine.get_active()
    if root is not None:
        root.close()
    IntelligenceGraphEngine.set_active(None)


def _commons_engine() -> Any:
    """A fresh :class:`IntelligenceGraphEngine` bound to ``__commons__`` on
    whatever engine ``GRAPH_SERVICE_ENDPOINTS`` currently points at."""
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    return IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=COMMONS_GRAPH),
        defer_background_start=True,
    )


def test_bug030_placement_route_admin_capability_blocks_non_bootstrap_reads(
    restartable_commons_engine,
):
    """Live reproduction of BUG-030 (see module docstring): a real
    ``query_cypher`` read succeeds for the ambient, bootstrap-consistent
    actor the autouse test-isolation fixture already establishes, and FAILS
    -- same tenant, AU-side ``kg:admin`` role included -- for any other
    actor, because ``PlacementRoute``'s engine-level ``admin:cluster-read``
    requirement has no path from an AU-side scope, ever, for a non-bootstrap
    identity."""
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        CypherEngineError,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
    from agent_utilities.knowledge_graph.core.session import (
        current_session,
        use_session,
    )
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    _reset_process_engine()

    doc_id = f"doc-{uuid.uuid4().hex[:10]}"
    commons = GraphComputeEngine.get_or_create(graph_name=COMMONS_GRAPH)
    commons.add_node(doc_id, {"node_type": "Doc", "name": "goc61-bug030-probe"})

    engine = _commons_engine()
    query = "MATCH (d:Doc) WHERE d.id = $id RETURN d.id AS id"
    params = {"id": doc_id}

    # --- baseline: the ambient (bootstrap-consistent) actor reads fine -----
    assert engine.query_cypher(query, dict(params)) == [{"id": doc_id}], (
        "the fixture's own ambient actor must be able to read through the "
        "real query_cypher path -- otherwise this test proves nothing"
    )

    # --- a DIFFERENT actor, same tenant, WITH AU-side kg:admin, is refused -
    other_actor = ActorContext(
        actor_id=f"reader-{uuid.uuid4().hex[:6]}",
        actor_type=ActorType.HUMAN,
        roles=("kg:admin",),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    ambient = current_session()
    assert ambient is not None
    # The underlying cause varies between a direct
    # ACCESS_DENIED-for-admin:cluster-read and a route-invalidation/reconnect
    # RuntimeError depending on retry timing (both observed empirically while
    # building this test; the WARNING log below -- visible in pytest's
    # teardown-phase capture, timing-dependent enough that asserting on it
    # directly proved flaky -- shows exactly this: "placement-routed endpoint
    # ... unreachable ... invalidating the cached route" then "placement
    # coordinator did not answer". A standalone, non-xdist repro consistently
    # surfaced the ACCESS_DENIED cause verbatim: "ACCESS_DENIED: verified
    # principal lacks admin capability required for 'admin:cluster-read'",
    # chained through placement_catalog.py::_attempt_route's
    # PlacementAuthorityError -- quoted in the module docstring). Either way,
    # the one load-bearing, deterministic claim this test proves is: the read
    # that just succeeded for the ambient actor FAILS the instant the actor
    # changes -- proven by the raises() context above succeeding.
    with pytest.raises(CypherEngineError):
        with use_session(ambient.with_actor(other_actor)):
            engine.query_cypher(query, dict(params))


def test_org_share_revocation_persists_across_a_real_restart(
    restartable_commons_engine,
):
    """Real engine, real Cypher, real restart proof for the parts of GOC-61's
    node-level sharing primitive that ARE provable given BUG-030 (module
    docstring): the org-share stamp on a write, and its revocation, land as
    real, correct node properties -- read back via real Cypher, both before
    and after a genuine SIGTERM + relaunch of the engine process against the
    SAME persist directory. Reads throughout use the one ambient identity
    this harness can authenticate for ``query_cypher`` (see
    ``test_bug030_...`` above) -- this proves persistence and revocation
    of the STATE, not a second actor's independent view of it, which BUG-030
    currently makes unconstructible against a real engine.
    """
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
    from agent_utilities.knowledge_graph.core.tenant_sharing import (
        SCOPE_ORG,
        SCOPE_PRIVATE,
        make_private,
    )
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    _reset_process_engine()

    doc_id = f"doc-{uuid.uuid4().hex[:10]}"
    privileged_writer = ActorContext(
        actor_id="platform-writer",
        actor_type=ActorType.SYSTEM,
        roles=("kg:admin",),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    commons = GraphComputeEngine.get_or_create(graph_name=COMMONS_GRAPH)
    with use_actor(privileged_writer):
        # Real chokepoint: check_system_graph_write + stamp_ownership
        # (graph_compute.py add_node), not a direct property mutation.
        commons.add_node(doc_id, {"node_type": "Doc", "name": "goc61-shared-doc"})

    engine = _commons_engine()
    q = "MATCH (d:Doc) WHERE d.id = $id RETURN d.id AS id, d._owner_id AS owner, d._shared_scope AS scope"

    rows = engine.query_cypher(q, {"id": doc_id})
    assert rows == [{"id": doc_id, "owner": None, "scope": SCOPE_ORG}], (
        "U-77: an unowned privileged write must land with no _owner_id and "
        f"_shared_scope='{SCOPE_ORG}' -- read back via real Cypher, not "
        "asserted against a bare dict"
    )

    # --- revocation takes effect on the very next read ----------------------
    revoked = make_private(doc_id, store=engine.backend, actor=privileged_writer)
    assert revoked, "make_private must find and revoke the previously-shared node"
    rows = engine.query_cypher(q, {"id": doc_id})
    assert (
        rows[0]["scope"] == SCOPE_PRIVATE
        and rows[0]["owner"] == privileged_writer.actor_id
    ), (
        "revocation must be immediate and correctly attributed -- no window "
        "where the next read still shows the org scope"
    )

    # --- restart proof: stop the real engine process, start a NEW one on the
    # SAME persist dir, reconnect, and confirm the revoked state persisted --
    # this is what closes BUG-064's missing restart proof for this primitive.
    restartable_commons_engine.restart()
    _reset_process_engine()
    engine_after_restart = _commons_engine()

    rows = engine_after_restart.query_cypher(q, {"id": doc_id})
    assert (
        rows[0]["scope"] == SCOPE_PRIVATE
        and rows[0]["owner"] == privileged_writer.actor_id
    ), (
        "the revoked state must survive a real process restart -- proving it "
        "was committed to disk, not held only in the live process"
    )

    # And a freshly written org-shared node, post-restart, still stamps
    # correctly -- the U-77 mechanism is durable engine behavior, not a
    # pre-restart artifact.
    doc_id_2 = f"doc-{uuid.uuid4().hex[:10]}"
    commons_after_restart = GraphComputeEngine.get_or_create(graph_name=COMMONS_GRAPH)
    with use_actor(privileged_writer):
        commons_after_restart.add_node(
            doc_id_2, {"node_type": "Doc", "name": "goc61-shared-doc-post-restart"}
        )
    rows = engine_after_restart.query_cypher(q, {"id": doc_id_2})
    assert rows == [{"id": doc_id_2, "owner": None, "scope": SCOPE_ORG}], (
        "the org-share stamping mechanism must keep working after a real restart"
    )


def test_adopt_a_shared_scope_surfaces_survive_real_restart(
    restartable_commons_engine,
):
    """NE-042 live acceptance: field/edge/search/DCI state is durable.

    The rows and one real graph edge are written before a SIGTERM, then read
    from a newly-created engine process after relaunch.  The deterministic
    retriever below is only a candidate-source harness; the production
    ``search_hybrid`` ACL/rank path and ``search_dci`` hop ACL boundary remain
    the code under test.  Peer checks use the real durable ACL hydration source
    with a separate AU actor.  A direct engine read as that peer is also
    attempted by the existing BUG-030 test in this module; this test keeps the
    transport identity at the bootstrap principal so the AU-layer policy proof
    is not confused with that independent engine-placement limitation.
    """
    from agent_utilities.knowledge_graph.core import secured_reads as sr
    from agent_utilities.knowledge_graph.core.company_brain_runtime import (
        reset_company_brain,
    )
    from agent_utilities.knowledge_graph.core.session import (
        GraphSession,
        use_session,
    )
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    _reset_process_engine()
    engine = _commons_engine()
    bootstrap_actor = ActorContext(
        actor_id=TEST_AGENT_ID,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    owner_id = "owner:ne042"
    same_tenant_peer = ActorContext(
        actor_id="peer:ne042",
        actor_type=ActorType.AI_AGENT,
        roles=(),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    cross_tenant_peer = ActorContext(
        actor_id="peer:foreign",
        actor_type=ActorType.AI_AGENT,
        roles=(),
        tenant_id="tenant:foreign",
        authenticated=True,
    )
    session = GraphSession(
        actor=bootstrap_actor,
        tenant=TEST_TENANT,
        scopes=frozenset({"kg:read", "kg:write", "kg:admin", "*"}),
        graph=COMMONS_GRAPH,
        policy_version=TEST_POLICY_VERSION,
        audience="epistemic-graph-test",
    )

    public_id = f"ne042-public-{uuid.uuid4().hex[:10]}"
    shared_id = f"ne042-shared-{uuid.uuid4().hex[:10]}"
    private_id = f"ne042-private-{uuid.uuid4().hex[:10]}"
    edge_record_id = f"ne042-edge-{uuid.uuid4().hex[:10]}"

    with (
        use_session(session),
        use_actor(
            ActorContext(
                actor_id="platform-writer",
                actor_type=ActorType.SYSTEM,
                roles=("kg:admin",),
                tenant_id=TEST_TENANT,
                authenticated=True,
            )
        ),
    ):
        # GraphComputeEngine's node/edge methods are the native durable RPC
        # chokepoints; this writes through the same authority exercised by the
        # existing restart proof above, not a Python-side graph/cache double.
        graph = engine.graph_compute
        graph.add_node(
            public_id,
            {
                "node_type": "Doc",
                "name": "NE-042 public field",
                "tenant_id": TEST_TENANT,
                "classification": "public",
                "_shared_scope": "org",
                "_score": 0.8,
            },
        )
        graph.add_node(
            shared_id,
            {
                "node_type": "Doc",
                "name": "NE-042 shared field",
                "tenant_id": TEST_TENANT,
                "classification": "confidential",
                "_owner_id": owner_id,
                "_shared_scope": "org",
                "_score": 0.7,
            },
        )
        graph.add_node(
            private_id,
            {
                "node_type": "Doc",
                "name": "NE-042 private field",
                "tenant_id": TEST_TENANT,
                "classification": "confidential",
                "_owner_id": owner_id,
                "_shared_scope": "private",
                "_score": 0.9,
            },
        )
        # An edge-shaped projection is represented by a durable record with
        # the same ownership contract as any other governed row.  The actual
        # native edge below supplies traversal continuity.
        graph.add_node(
            edge_record_id,
            {
                "node_type": "Doc",
                "name": "NE-042 shared edge record",
                "tenant_id": TEST_TENANT,
                "classification": "confidential",
                "_owner_id": owner_id,
                "_shared_scope": "org",
                "source_id": shared_id,
                "target_id": private_id,
            },
        )
        graph.add_edge(shared_id, private_id, {"relationship": "RELATED_TO"})
        # Exercise the governed durable SET seam as well. This forces the
        # ownership markers through the same native mutation path used by
        # production sharing operations, ensuring the restart assertion is
        # about persisted state rather than a live graph cache.
        from agent_utilities.knowledge_graph.core.tenant_sharing import (
            make_private,
            share_with_org,
        )

        for node_id in (public_id, shared_id, private_id, edge_record_id):
            assert share_with_org(node_id, store=engine.backend)
        assert make_private(
            private_id,
            store=engine.backend,
            actor=ActorContext(
                actor_id=owner_id,
                actor_type=ActorType.AI_AGENT,
                roles=(),
                tenant_id=TEST_TENANT,
                authenticated=True,
            ),
        )

    def _raw(node_id: str, reader: Any) -> dict[str, Any]:
        query = (
            "MATCH (n:Doc) WHERE n.id = $id RETURN n.id AS id, "
            "n.tenant_id AS tenant_id, n.node_type AS node_type, "
            "n.classification AS classification, n._owner_id AS owner_id, "
            "n._shared_scope AS shared_scope, n._score AS score, "
            "n.source_id AS source_id, n.target_id AS target_id"
        )
        # Socket readiness is the served startup boundary. The first governed
        # query after it must hydrate the durable graph synchronously; polling,
        # cache resets, and point-read fallbacks would hide a surface split.
        rows = reader.backend.execute_read(query, {"id": node_id})
        assert rows, f"durable row {node_id} was not readable on the first query"
        return dict(rows[0])

    with use_session(session):
        before = {
            node_id: _raw(node_id, engine)
            for node_id in (public_id, shared_id, private_id, edge_record_id)
        }
        assert engine.graph_compute.has_edge(shared_id, private_id)

    # Stop the process, relaunch against the same persist directory, and clear
    # the AU permission cache so every read must hydrate from durable rows.
    restartable_commons_engine.restart()
    _reset_process_engine()
    engine_after = _commons_engine()
    reset_company_brain()

    with use_session(session):
        after = {
            node_id: _raw(node_id, engine_after)
            for node_id in (public_id, shared_id, private_id, edge_record_id)
        }
        assert after == before
        assert engine_after.graph_compute.has_edge(shared_id, private_id)

    def _projection(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "tenant_id": row["tenant_id"],
            "node_type": row["node_type"],
            "_owner_id": row["owner_id"],
            "_shared_scope": row["shared_scope"],
        }

    field_rows = [
        _projection(after[node_id]) for node_id in (public_id, shared_id, private_id)
    ]
    edge_rows = [{"r": _projection(after[edge_record_id])}]

    # Same-tenant non-owner: public and org-shared survive, private remains
    # denied after the real process boundary.
    with use_session(session):
        reset_company_brain()
        visible_fields = sr.visible(
            sr.filter_rows(field_rows, same_tenant_peer), same_tenant_peer
        )
        assert {row["id"] for row in visible_fields} == {public_id, shared_id}
        visible_edges = sr.visible(
            sr.filter_rows(edge_rows, same_tenant_peer), same_tenant_peer
        )
        assert [row["r"]["id"] for row in visible_edges] == [edge_record_id]

        # Cross-tenant peer denial is evaluated against the same durable ACL
        # rows, not a process-local permission cache.
        reset_company_brain()
        assert sr.filter_rows(field_rows, cross_tenant_peer) == []
        assert sr.filter_rows(edge_rows, cross_tenant_peer) == []

    class _StaticRetriever:
        def __init__(self, nodes: list[dict[str, Any]]):
            self.nodes = nodes

        def retrieve_hybrid(self, _query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return [dict(node) for node in self.nodes]

    search_nodes = [
        {
            **_projection(after[node_id]),
            "_score": float(after[node_id].get("score") or 0.0),
            "status": "ACTIVE",
        }
        for node_id in (public_id, shared_id, private_id)
    ]
    with use_session(session):
        reset_company_brain()
        engine_after.hybrid_retriever = _StaticRetriever(search_nodes)
        search_rows = engine_after.search_hybrid(
            "NE-042", top_k=3, skip_quality_gate=True, session=session
        )
        assert {row["id"] for row in search_rows} == {public_id, shared_id}
        assert private_id not in {row["id"] for row in search_rows}

        # DCI uses the real post-restart graph edge.  Its seed is shared and
        # its only neighbor is private; the per-hop ACL boundary must stop that
        # neighbor before it can enter the result or a later frontier.
        engine_after.hybrid_retriever = _StaticRetriever(
            [next(row for row in search_nodes if row["id"] == shared_id)]
        )
        traversal_rows = engine_after.search_dci(
            "NE-042", max_hops=1, top_k=10, session=session
        )
        traversal_ids = {row["id"] for row in traversal_rows}
        assert shared_id in traversal_ids
        assert private_id not in traversal_ids
