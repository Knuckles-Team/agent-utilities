"""EH-249 / PA-08 -- the Agent Library's served-Python surface, against a REAL engine.

EXIT-CRITERIA-MATRIX.md names this "the single largest concrete gap in the
whole matrix": E2E-CENSUS found EG's own Python test suite exercises
``AgentComponent`` only through a fake transport
(``tests/test_connector_pack_client.py``, ``pytest.mark.no_engine``), and AU's
production code has zero callers of the typed ``AgentComponent``/``AgentGraph``/
``AgentTemplate``/``AgentLibrary`` wire methods anywhere --
``agent_utilities/core/registry/kg_adapter.py``'s "AgentTemplate CRUD" is a
wholly separate, generic Cypher-node mechanism, not this typed family. Decide
(A5/A6) and pack import (PA4/PA6) both depend on this surface as their
substrate.

This module is the first real, engine-backed Python exercise of that surface:
it drives ``epistemic_graph.generated.storage.send_agent_component`` (and its
typed ``search``/``current`` wrappers) directly against the real
``engine_graph`` fixture, the same fixture and calling convention
``agent_utilities/knowledge_graph/core/graph_compute.py``'s own
``shacl_validate_ad_hoc`` uses in production (there is no AU-side wrapper for
Agent Library yet -- that product wiring is out of this lane's scope, tracked
separately; this closes the *test coverage* gap the census names).

Requires a real engine; every test skips cleanly (via the standard AU
engine-availability mechanism) when none is available in this environment.

``test_published_component_survives_a_real_engine_restart`` below covers
``survives_restart`` (PA-08's eighth named assertion) directly: it runs its
OWN dedicated ``epistemic-graph-server`` process on a fixed, test-owned
persist directory (the shared, session-scoped ``engine_graph``/``tiny_engine``
fixtures intentionally never expose this -- one engine for the whole test
session, restarting it would affect every other test sharing it), publishes
a component, SIGTERMs that process, launches a NEW process pointed at the
exact same on-disk store, and asserts the component and its provenance
(digests, revision, lifecycle) are unchanged. The restartable-engine helper
below is copied from the two existing real-restart proofs already in this
suite (``tests/integration/knowledge_graph/test_goc61_commons_sharing_live.py``,
``tests/integration/protocols/test_a2a_epistemic_live.py``) rather than
invented fresh, since ``tests/_test_engine.py``'s own ``EphemeralEngine``
does not expose a fixed/reusable persist directory (each ``start()`` mints a
new one) and both of those modules already solved exactly this problem the
same way.
"""

from __future__ import annotations

import asyncio as _asyncio
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
    TEST_AUDIENCE,
    TEST_POLICY_VERSION,
    TEST_SIGNER_KEY,
    TEST_TENANT,
    EngineBinaryIdentity,
    EngineUnavailable,
    bootstrap_context,
    resolve_engine_binary_identity,
    strict_server_env,
)

from agent_utilities.knowledge_graph.core.session import current_session

pytestmark = [pytest.mark.integration, pytest.mark.engine]


# --- restart harness (survives_restart) -------------------------------- #
# Copied/adapted from the two existing real-restart proofs in this suite
# (test_goc61_commons_sharing_live.py, test_a2a_epistemic_live.py): a fixed
# persist dir + socket path a process can be SIGTERM'd and relaunched
# against, which `tests/_test_engine.py`'s own `EphemeralEngine` does not
# expose (every `start()` mints its own throwaway persist dir).


def _free_socket_path(root: Path) -> str:
    return str(root / f"eg-{uuid.uuid4().hex[:8]}.sock")


def _wait_for_restart_socket(
    proc: subprocess.Popen[bytes], sock_path: str, log_path: Path
) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_bytes()[-4000:].decode("utf-8", "replace")
            raise RuntimeError(
                f"epistemic-graph-server exited early (code {proc.returncode}) "
                f"during startup:\n{tail}"
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
    the actual "process restart" ``survives_restart`` needs, not a fresh
    ephemeral one."""

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
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fh: Any = None

    def start(self) -> None:
        self._log_fh = open(self.log_path, "ab")  # noqa: SIM115 -- closed in stop()
        env = dict(os.environ)
        env.update(
            strict_server_env(str(self.security_dir), auth_secret=self.auth_secret)
        )
        env["GRAPH_SERVICE_PERSIST_DIR"] = str(self.persist_dir)
        self.binary_identity.verify_for_launch()
        self._proc = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
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
        _wait_for_restart_socket(self._proc, self.socket_path, self.log_path)

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
        the exact same ``persist_dir``/``socket_path`` (a fresh socket file
        under the same fixed root is fine -- only the on-disk store's
        persistence is under test). Deliberately does NOT re-run
        ``bootstrap_system_identity``: that call is a one-time enrollment
        (CONCEPT:AU-OS.identity.authenticated-identity-enforcement) whose
        durable record already survives on this same persist dir from the
        first ``start()`` -- re-issuing it against an already-bootstrapped
        store is exactly the destructive "REPLACES the role set" shape this
        program has already been burned by once (see memory:
        engine-rbac-register-replaces-roles). The identity used before the
        restart must simply still work after it.
        """
        self.stop()
        self.start()


def _hex_nonce() -> str:
    """A 64-lowercase-hex-char nonce (``AgentLibraryMutationContext.attempt_nonce``)."""
    return uuid.uuid4().hex + uuid.uuid4().hex


#: The exact, verbatim message the installed epistemic-graph wheel's client
#: raises for EVERY real request in this environment -- a pre-existing,
#: already-documented MAC/signing bug (the Python canonical-body signer bug),
#: fixed by EG train-3's eg-canon d4f310441, not yet restaged into this venv's
#: wheel. See EH-431's WRAPUP.md "Cannot run here" section for the full
#: cross-check (an untouched, pre-existing file reproduces the identical
#: failure in this same environment).
_KNOWN_WHEEL_MAC_BUG_MESSAGE = "Authentication failed"


def _result_or_fail_on_known_wheel_bug(future: Any) -> Any:
    """Unwrap a dispatched future, translating the one known pre-existing
    failure into a `pytest.fail` that names its root cause and fix up front.

    Split out of ``_call`` so each function stays within the KISS complexity
    caps on its own -- this is the only place that branches on the failure
    reason.
    """
    try:
        return future.result()
    except RuntimeError as exc:
        if str(exc) != _KNOWN_WHEEL_MAC_BUG_MESSAGE:
            raise
        # `pytest.fail` raises `Failed` (a `BaseException`, not an
        # `Exception`), so this is never mistaken for a genuine authorization
        # refusal by a test asserting `pytest.raises(Exception)`.
        pytest.fail(
            "blocked by the known pre-existing EG-wheel MAC/signing bug "
            "(RuntimeError: 'Authentication failed'), not a defect in this "
            "test: the installed epistemic-graph wheel predates EG "
            "train-3's eg-canon d4f310441 fix. Fix = restage this venv's "
            "epistemic-graph wheel from EG train-3 (or later) -- see "
            "EH-431's WRAPUP.md 'Cannot run here' section. Re-verify at "
            "Wave B once that restage has happened."
        )
        raise  # pragma: no cover -- pytest.fail never returns


def _call(engine: Any, coro_factory: Any) -> Any:
    """Drive one generated-sender coroutine on the engine's own event loop.

    Mirrors ``GraphComputeEngine.shacl_validate_ad_hoc``'s internal pattern
    exactly (the one proven, production-used way AU code drives a generated
    sender against a real engine from synchronous code) -- there is no public
    AU helper for this yet because no AU production code calls the Agent
    Library wire methods (that is exactly the gap this test closes).
    """
    loop = engine._engine_loop()
    if loop is None:
        pytest.skip("no engine loop available for the Agent Library surface")
    future = _asyncio.run_coroutine_threadsafe(coro_factory(), loop)
    return _result_or_fail_on_known_wheel_bug(future)


def _agent_component_client(engine: Any) -> Any:
    return engine._engine_async_client()


def _draft(
    *, component_id: str, tenant_id: str, version: str = "1.0.0"
) -> dict[str, Any]:
    """A minimal, valid ``AgentComponentDraft`` (opaque facts, native provenance --
    the two simplest variants of each discriminated union) for a synthetic test tool."""
    digest = "sha256:" + ("ab" * 32)
    return {
        "actor_scope": "scope:served-python-coverage",
        "component_id": component_id,
        "content_digest": digest,
        "facts": {"facts": "opaque"},
        "kind": "tool",
        "policy_digest": digest,
        "provenance": {"origin": "native"},
        "purpose_id": "purpose:served-python-coverage",
        "source_revision": "rev-1",
        "source_revision_digest": digest,
        "summary": "EH-249 served-Python Agent Library lifecycle proof component.",
        "tenant_id": tenant_id,
        "version": version,
    }


def _mutation_context(
    *, tenant_id: str, principal: str, expected_revision: int | None = None
) -> dict[str, Any]:
    """A plausible ``AgentLibraryMutationContext``.

    Shape mirrors ``tests/test_connector_pack_client.py::_context()`` in the
    epistemic-graph repo -- that test only ever exercises a FAKE transport, so
    these field values were never checked by a real engine; this module is
    what actually proves (or disproves) their plausibility.
    """
    digest = "sha256:" + ("22" * 32)
    return {
        "request_id": uuid.uuid4().int & 0xFFFFFFFF,
        "principal": principal,
        "caller_principal": principal,
        "attempt_nonce": _hex_nonce(),
        "tenant_id": tenant_id,
        "actor_scope": "scope:served-python-coverage",
        "purpose_id": "purpose:served-python-coverage",
        "policy_revision": "policy-1",
        "policy_digest": digest,
        "policy_decision_id": "decision-served-python-coverage",
        "idempotency_key": f"eh249-{uuid.uuid4().hex}",
        "expected_revision": expected_revision,
        "trace_id": f"trace-{uuid.uuid4().hex}",
        "created_at_ms": 1_700_000_000_000,
    }


def _publish(engine: Any, draft: dict[str, Any], context: dict[str, Any]) -> Any:
    from epistemic_graph.generated.storage import send_agent_component

    params = {
        "op": {"op": "publish", "request": {"component": draft, "context": context}}
    }
    client = _agent_component_client(engine)
    return _call(
        engine, lambda: send_agent_component(client, params, engine.graph_name)
    )


def _retire(engine: Any, component_id: str, context: dict[str, Any]) -> Any:
    from epistemic_graph.generated.storage import send_agent_component

    params = {
        "op": {
            "op": "retire",
            "request": {"component_id": component_id, "context": context},
        }
    }
    client = _agent_component_client(engine)
    return _call(
        engine, lambda: send_agent_component(client, params, engine.graph_name)
    )


def _current(engine: Any, component_id: str, tenant_id: str) -> Any:
    from epistemic_graph.generated.agent_component import AgentComponentOpCurrent
    from epistemic_graph.generated.storage import send_agent_component_current

    client = _agent_component_client(engine)
    request = AgentComponentOpCurrent(
        op="current", component_id=component_id, tenant_id=tenant_id
    )
    return _call(
        engine,
        lambda: send_agent_component_current(client, request, engine.graph_name),
    )


def _history(engine: Any, component_id: str, tenant_id: str) -> Any:
    from epistemic_graph.generated.storage import send_agent_component

    params = {
        "op": {"op": "history", "component_id": component_id, "tenant_id": tenant_id}
    }
    client = _agent_component_client(engine)
    return _call(
        engine, lambda: send_agent_component(client, params, engine.graph_name)
    )


def _search_by_kind(engine: Any, *, tenant_id: str, kind: str) -> Any:
    from epistemic_graph.generated.agent_component import AgentComponentSearchRequest
    from epistemic_graph.generated.storage import send_agent_component_search

    client = _agent_component_client(engine)
    request = AgentComponentSearchRequest(tenant_id=tenant_id, kinds=[kind])
    return _call(
        engine, lambda: send_agent_component_search(client, request, engine.graph_name)
    )


def _search_by_capability(engine: Any, *, tenant_id: str, capability: str) -> Any:
    """The typed ``send_agent_component_search`` wrapper refuses a capability
    query client-side ("requires kinds and no task/capabilities") -- it only
    covers pure kind search. Capability search has to go through the untyped
    generic sender with a hand-built request."""
    from epistemic_graph.generated.agent_component import AgentComponentSearchRequest
    from epistemic_graph.generated.storage import send_agent_component

    client = _agent_component_client(engine)
    request = AgentComponentSearchRequest(
        tenant_id=tenant_id, capabilities=[capability]
    )
    params = {
        "op": {
            "op": "search",
            "request": request.model_dump(mode="json", exclude_none=True),
        }
    }
    return _call(
        engine, lambda: send_agent_component(client, params, engine.graph_name)
    )


def _tenant_id() -> str:
    session = current_session()
    assert session is not None, "engine_graph must provide a verified GraphSession"
    return str(session.tenant)


def test_publish_new_component_is_searchable_and_has_a_queryable_current_version(
    engine_graph: Any,
) -> None:
    """publish_agent_definition + search_by_capability + served_python_coverage."""
    tenant_id = _tenant_id()
    component_id = f"test:eh249-{uuid.uuid4().hex[:12]}"
    draft = _draft(component_id=component_id, tenant_id=tenant_id)
    draft["declared_capabilities"] = ["eh249:capability/probe"]
    context = _mutation_context(
        tenant_id=tenant_id, principal="agent-library-lifecycle-test"
    )

    result = _publish(engine_graph, draft, context)
    assert result.method == "AgentComponent"
    assert isinstance(result.payload, dict), (
        f"AgentComponent.publish must return a decoded body; got {type(result.payload).__name__}"
    )

    current = _current(engine_graph, component_id, tenant_id)
    assert current is not None, (
        "a just-published component must have a queryable current version"
    )
    assert current.component_id == component_id
    assert current.version == "1.0.0"
    assert current.lifecycle.value == "published"

    page = _search_by_capability(
        engine_graph, tenant_id=tenant_id, capability="eh249:capability/probe"
    )
    found = [
        entry
        for entry in page.payload.get("entries", [])
        if entry.get("component_id") == component_id
    ]
    assert found, (
        f"search-by-capability must surface a just-published component declaring it; "
        f"page={page.payload!r}"
    )

    kind_page = _search_by_kind(engine_graph, tenant_id=tenant_id, kind="tool")
    assert any(entry.component_id == component_id for entry in kind_page.entries), (
        "search-by-kind must also surface the same just-published component"
    )


def test_republish_bumps_version_not_overwrites(engine_graph: Any) -> None:
    """version_bump_on_republish."""
    tenant_id = _tenant_id()
    component_id = f"test:eh249-{uuid.uuid4().hex[:12]}"
    draft_v1 = _draft(component_id=component_id, tenant_id=tenant_id, version="1.0.0")
    context = _mutation_context(
        tenant_id=tenant_id, principal="agent-library-lifecycle-test"
    )
    _publish(engine_graph, draft_v1, context)

    current_v1 = _current(engine_graph, component_id, tenant_id)
    assert current_v1 is not None
    v1_revision = current_v1.entry_revision

    draft_v2 = _draft(component_id=component_id, tenant_id=tenant_id, version="2.0.0")
    context_v2 = _mutation_context(
        tenant_id=tenant_id,
        principal="agent-library-lifecycle-test",
        expected_revision=v1_revision,
    )
    _publish(engine_graph, draft_v2, context_v2)

    current_v2 = _current(engine_graph, component_id, tenant_id)
    assert current_v2 is not None
    assert current_v2.version == "2.0.0"
    assert current_v2.entry_revision > v1_revision, (
        "a republish must create a NEW version, not overwrite the prior entry_revision in place"
    )


def test_retire_then_history_is_queryable(engine_graph: Any) -> None:
    """retire_definition + history_queryable."""
    tenant_id = _tenant_id()
    component_id = f"test:eh249-{uuid.uuid4().hex[:12]}"
    draft = _draft(component_id=component_id, tenant_id=tenant_id)
    publish_context = _mutation_context(
        tenant_id=tenant_id, principal="agent-library-lifecycle-test"
    )
    _publish(engine_graph, draft, publish_context)
    published = _current(engine_graph, component_id, tenant_id)
    assert published is not None

    retire_context = _mutation_context(
        tenant_id=tenant_id,
        principal="agent-library-lifecycle-test",
        expected_revision=published.entry_revision,
    )
    retire_result = _retire(engine_graph, component_id, retire_context)
    assert retire_result.method == "AgentComponent"

    history = _history(engine_graph, component_id, tenant_id)
    entries = (
        history.payload
        if isinstance(history.payload, list)
        else history.payload.get("entries", [])
    )
    assert entries, "a retired component's full version history must remain queryable"
    from epistemic_graph.generated.agent_component import AgentComponentEntry

    validated = [AgentComponentEntry.model_validate(entry) for entry in entries]
    assert all(entry.component_id == component_id for entry in validated)
    assert any(entry.lifecycle.value == "retired" for entry in validated), (
        "retiring must be visible in the version history, not just remove the definition"
    )


def test_publish_with_mismatched_tenant_context_is_refused(engine_graph: Any) -> None:
    """authorize_publish_gate: publishing is gated, not open to any caller.

    Concrete boundary: a mutation context whose ``tenant_id`` does not match
    the session's actual bound tenant must be refused, not silently accepted
    under the wrong tenant (the same confused-deputy shape EH-373/374 already
    established elsewhere in this program).

    In an environment blocked by the known pre-existing wheel MAC/signing bug
    (EH-431's finding), ``_call`` converts that specific failure into
    ``pytest.fail`` (a `Failed`, not an `Exception`) precisely so it is never
    mistaken for the tenant-authorization refusal this test asserts -- this
    test FAILS with the documented root cause in that environment rather than
    passing for the wrong reason. Re-verify after the wheel restage at
    landing: on a working engine, this must raise for a tenant/authorization
    reason specifically, not silently succeed.
    """
    real_tenant_id = _tenant_id()
    foreign_tenant_id = f"foreign-{uuid.uuid4().hex[:12]}"
    component_id = f"test:eh249-{uuid.uuid4().hex[:12]}"
    draft = _draft(component_id=component_id, tenant_id=real_tenant_id)
    context = _mutation_context(
        tenant_id=foreign_tenant_id, principal="agent-library-lifecycle-test"
    )

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 -- the engine's error type/message is not ours to pin here
        _publish(engine_graph, draft, context)
    message = str(excinfo.value)
    assert message, (
        "an authorization refusal must carry a reason, not a bare/blank failure"
    )


def test_published_component_survives_a_real_engine_restart(
    test_engine_lifecycle: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """survives_restart: the library's state (definitions, versions, history)
    survives a process restart -- a genuine SIGTERM + relaunch against the
    SAME on-disk store, not merely a held-open connection or a fresh
    ephemeral engine that never actually proves persistence.

    Runs its own dedicated engine process (never the shared session
    ``engine_graph``/``tiny_engine`` -- restarting THAT would break every
    other test sharing it); skips cleanly when no real engine binary is
    obtainable in this environment, exactly like every other engine-backed
    test here.
    """
    try:
        binary_identity = resolve_engine_binary_identity()
    except EngineUnavailable as exc:
        pytest.skip(f"no real epistemic-graph-server artifact available: {exc}")

    auth_secret = "au-eg-" + "agent-library-restart-test-secret"  # nosec B105 -- test-only
    eng = _RestartableEngine(binary_identity, tmp_path, auth_secret)
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

        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )
        from agent_utilities.knowledge_graph.core.session import (
            GraphSession,
            use_session,
        )
        from agent_utilities.security.actor_identity import ActorType
        from agent_utilities.security.brain_context import ActorContext

        tenant_id = TEST_TENANT
        actor = ActorContext(
            actor_id=TEST_AGENT_ID,
            actor_type=ActorType.AUTOMATED_SERVICE,
            roles=("test",),
            tenant_id=tenant_id,
            authenticated=True,
        )
        session = GraphSession(
            actor=actor,
            tenant=tenant_id,
            scopes=frozenset({"kg:read", "kg:write", "kg:admin", "*"}),
            graph=tenant_id,
            policy_version=TEST_POLICY_VERSION,
            audience=TEST_AUDIENCE,
        )
        component_id = f"test:eh249-restart-{uuid.uuid4().hex[:12]}"

        with use_session(session):
            compute = GraphComputeEngine(graph_name=tenant_id)
            draft = _draft(component_id=component_id, tenant_id=tenant_id)
            context = _mutation_context(tenant_id=tenant_id, principal=TEST_AGENT_ID)
            _publish(compute, draft, context)
            before = _current(compute, component_id, tenant_id)
            assert before is not None, (
                "the component must be queryable before the restart -- otherwise "
                "this test proves nothing about what survives it"
            )
            compute.close()

        # --- the actual restart proof: SIGTERM the running process, then start
        # a NEW process pointed at the exact same on-disk persist directory.
        eng.restart()

        with use_session(session):
            compute_after = GraphComputeEngine(graph_name=tenant_id)
            after = _current(compute_after, component_id, tenant_id)
            assert after is not None, (
                "the published component must still be queryable after a real "
                "process restart against the same on-disk store"
            )
            assert after.component_id == before.component_id
            assert after.version == before.version
            assert after.entry_revision == before.entry_revision, (
                "a restart must not fabricate a new revision or lose the old one"
            )
            assert after.definition_digest == before.definition_digest
            assert after.content_digest == before.content_digest
            assert after.lifecycle == before.lifecycle
            assert after.created_at_ms == before.created_at_ms, (
                "provenance (when the version was actually created) must survive "
                "the restart unchanged, not be re-stamped"
            )

            history = _history(compute_after, component_id, tenant_id)
            entries = (
                history.payload
                if isinstance(history.payload, list)
                else history.payload.get("entries", [])
            )
            assert entries, "the version history must also survive the restart"
            compute_after.close()
    finally:
        registration.stop()
