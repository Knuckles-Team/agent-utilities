"""Engine-resolution matrix (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision).

One resolver, every entrypoint inherits it. The resolver decides — by ONE
precedence, remote → share-running-local → autostart-shared-supervised — how a
process reaches the ONE engine authority. These tests assert:

* (a) **remote configured** → ``mode="remote"``, autostart NEVER permitted (no
  spawn);
* (b) **a running engine** → ``mode="shared"`` (reuse it, single PID, no 2nd
  process);
* (c) **nothing reachable** → ``mode="autostart"``; a 2nd ``resolve_engine``
  shares the just-started engine (no redb-lock collision across concurrent
  resolves), and the supervised idle-shutdown grace is plumbed through;
* the **persistent** lifecycle (idle_shutdown_secs<=0 / engine_lifecycle=persistent)
  is long-living (no ``--idle-shutdown-secs`` flag).

The autostart legs use a fake ``epistemic-graph-server`` shell binary so the
matrix runs with no real engine and asserts *spawn count* (the single-PID /
no-collision contract) without depending on the Rust wheel.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.knowledge_graph.core import engine_resolver as er
from agent_utilities.knowledge_graph.core import graph_compute as gc

_REAL_LOCAL_ENCRYPTION_KEY_LOADER = gc._load_or_create_engine_encryption_key


@pytest.fixture(autouse=True)
def _hermetic_local_encryption_key(monkeypatch):
    """Keep launcher tests away from the operator's real XDG key store."""

    monkeypatch.setattr(
        gc,
        "_load_or_create_engine_encryption_key",
        lambda: "test-engine-encryption-key-material-0001",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_unix_sock(tmp_path: Path) -> str:
    return f"unix://{tmp_path / 'eg-test.sock'}"


class _UDSServer:
    """A trivial UDS listener so ``probe_endpoint`` sees an engine 'running'."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(8)
        self._stop = False
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
                conn.close()
            except OSError:
                continue

    def close(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(self.path)


# ---------------------------------------------------------------------------
# (a) remote configured → mode=remote, NEVER autostart
# ---------------------------------------------------------------------------


def test_remote_endpoint_resolves_remote_no_autostart(monkeypatch):
    """Configured topology is connect-only; autostart is impossible."""
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "tcp://engine.internal:9100")
    cfg = AgentConfig()

    resolved = er.resolve_engine(cfg, "__commons__")

    assert resolved.mode == "remote"
    assert resolved.endpoint == "tcp://engine.internal:9100"
    assert resolved.autostart_allowed is False
    # Remote is inherently persistent — the resolver never passes idle-shutdown.
    assert resolved.idle_shutdown_secs == 0


def test_sharded_endpoints_resolve_remote_no_autostart(monkeypatch):
    """2+ endpoints (sharding) → remote contract, fail-loud, never autostart."""
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "tcp://a:1,tcp://b:2")
    cfg = AgentConfig()

    resolved = er.resolve_engine(cfg, "__commons__")

    assert resolved.mode == "remote"
    assert resolved.autostart_allowed is False
    assert resolved.endpoint == "tcp://a:1"


def test_resolver_selects_a_contact_without_guessing_graph_placement(monkeypatch):
    monkeypatch.setenv(
        "GRAPH_SERVICE_ENDPOINTS",
        "tcp://coordinator-a:9100,tcp://coordinator-b:9100",
    )
    monkeypatch.setattr(
        er,
        "resolve_endpoints",
        lambda _cfg: ["tcp://coordinator-a:9100", "tcp://coordinator-b:9100"],
    )
    resolved = er.resolve_engine(AgentConfig(), "tenant:graph")
    assert resolved.endpoint == "tcp://coordinator-a:9100"
    assert not hasattr(resolved, "placement_group")


def test_configured_topology_disables_autostart_helper(monkeypatch):
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "tcp://x:1")
    assert er.setting_autostart(AgentConfig()) is False


def test_configured_local_contact_is_still_connect_only(monkeypatch):
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "tcp://127.0.0.1:8765")
    resolved = er.resolve_engine(AgentConfig(), "__commons__")
    assert resolved.mode == "remote"
    assert resolved.autostart_allowed is False


# ---------------------------------------------------------------------------
# (b) a running engine → mode=shared, no spawn
# ---------------------------------------------------------------------------


def test_running_local_engine_is_shared(monkeypatch, tmp_path):
    """A local endpoint already serving → mode=shared (reuse; spawn nothing)."""
    sock_path = str(tmp_path / "eg-running.sock")
    server = _UDSServer(sock_path)
    try:
        monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
        monkeypatch.setattr(
            er, "resolve_endpoints", lambda _cfg: [f"unix://{sock_path}"]
        )
        cfg = AgentConfig()

        resolved = er.resolve_engine(cfg, "__commons__")

        assert resolved.mode == "shared", "an already-running engine must be reused"
        assert resolved.endpoint == f"unix://{sock_path}"
    finally:
        server.close()


def test_no_running_engine_resolves_autostart(monkeypatch, tmp_path):
    """Nothing serving on a local endpoint → mode=autostart (caller will spawn)."""
    sock_path = str(tmp_path / "eg-absent.sock")
    monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
    monkeypatch.setattr(er, "resolve_endpoints", lambda _cfg: [f"unix://{sock_path}"])
    cfg = AgentConfig()

    resolved = er.resolve_engine(cfg, "__commons__")

    assert resolved.mode == "autostart"
    assert resolved.autostart_allowed is True
    assert resolved.idle_shutdown_secs == 60  # default refcounted grace


def test_loopback_tcp_resolves_local_autostart(monkeypatch):
    """Windows' loopback-TCP default is a local, autostartable transport."""
    monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
    monkeypatch.setattr(er, "resolve_endpoints", lambda _cfg: ["tcp://127.0.0.1:8765"])
    monkeypatch.setattr(er, "probe_endpoint", lambda *_a, **_k: False)

    resolved = er.resolve_engine(AgentConfig(), "__commons__")

    assert resolved.endpoint == "tcp://127.0.0.1:8765"
    assert resolved.mode == "autostart"
    assert resolved.autostart_allowed is True


def test_autostart_passes_loopback_tcp_address(monkeypatch):
    """The spawned server must bind the same loopback address as its client."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import graph_compute as gc

    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []
    retained_environments: list[dict[str, str]] = []
    monkeypatch.setenv("UNRELATED_PROVIDER_API_KEY", "must-not-reach-engine")
    monkeypatch.setenv("EPISTEMIC_GRAPH_ENCRYPTION_KEY", "ambient-raw-must-not-win")
    monkeypatch.setenv(
        "EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF", "env://AMBIENT_REF_MUST_NOT_CROSS"
    )
    fake_child = SimpleNamespace(poll=lambda: None)
    fake_subprocess = SimpleNamespace(
        DEVNULL=-1,
        Popen=lambda cmd, **kwargs: (
            commands.append(list(cmd)),
            environments.append(dict(kwargs["env"])),
            retained_environments.append(kwargs["env"]),
            fake_child,
        )[-1],
    )
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(lambda **_kwargs: "connected"),
    )

    engine = object.__new__(gc.GraphComputeEngine)
    result = engine._autostart_engine(
        {
            "tcp_addr": "127.0.0.1:8765",
            "graph_name": "__commons__",
            "verified_context": gc._transport_only_verified_context(),
        },
        None,
        "test-secret",
        SimpleNamespace(
            epistemic_graph_max_resident_graphs=1024,
            epistemic_graph_lazy_open_page_size=4096,
            epistemic_graph_max_nodes_per_graph=250_000,
        ),
        fake_subprocess,
        sys,
        SimpleNamespace(sleep=lambda _seconds: None, monotonic=time.monotonic),
        Path,
        coupled=False,
        idle_shutdown_secs=45,
    )

    assert result == "connected"
    assert commands
    tcp_index = commands[0].index("--tcp-addr")
    assert commands[0][tcp_index : tcp_index + 2] == [
        "--tcp-addr",
        "127.0.0.1:8765",
    ]
    idle_index = commands[0].index("--idle-shutdown-secs")
    assert commands[0][idle_index : idle_index + 2] == [
        "--idle-shutdown-secs",
        "45",
    ]
    assert "--checkpoint-interval" not in commands[0]
    assert environments[0]["EPISTEMIC_GRAPH_LAZY_STARTUP"] == "1"
    assert environments[0]["GRAPH_SERVICE_AUTH_SECRET"] == "test-secret"
    assert environments[0]["EPISTEMIC_GRAPH_MAX_RESIDENT_GRAPHS"] == "1024"
    assert environments[0]["EPISTEMIC_GRAPH_LAZY_OPEN_PAGE_SIZE"] == "4096"
    assert environments[0]["EPISTEMIC_GRAPH_MAX_NODES_PER_GRAPH"] == "250000"
    assert environments[0]["EPISTEMIC_GRAPH_MAX_MSGPACK_ITEMS"] == "1000000"
    assert (
        environments[0]["EPISTEMIC_GRAPH_ENCRYPTION_KEY"]
        == "test-engine-encryption-key-material-0001"
    )
    assert "EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF" not in environments[0]
    assert "EPISTEMIC_GRAPH_ENCRYPTION_KEY" not in retained_environments[0]
    assert "UNRELATED_PROVIDER_API_KEY" not in environments[0]


def test_engine_encryption_reference_resolves_only_for_child(monkeypatch):
    from agent_utilities.security import cli_secrets

    monkeypatch.setattr(
        cli_secrets,
        "resolve_runtime_secret_reference",
        lambda _reference: "resolved-engine-encryption-key-0001",
    )
    config = SimpleNamespace(
        epistemic_graph_encryption_key_ref="env://TEST_ENGINE_DATA_KEY",
        app_profile="production",
        deployment_profile="enterprise",
    )

    assert (
        gc._resolve_engine_encryption_key(config)
        == "resolved-engine-encryption-key-0001"
    )


@pytest.mark.parametrize(
    "material",
    ["short", "x" * 31, "x" * 4097, "x" * 32 + "\n"],
)
def test_engine_encryption_material_is_bounded_and_control_free(material):
    with pytest.raises(RuntimeError, match="material is invalid"):
        gc._validate_engine_encryption_material(material)


def test_engine_encryption_missing_reference_is_tiny_dev_only():
    assert (
        gc._resolve_engine_encryption_key(
            SimpleNamespace(app_profile="dev", deployment_profile="tiny")
        )
        == "test-engine-encryption-key-material-0001"
    )
    with pytest.raises(RuntimeError, match="reference is required"):
        gc._resolve_engine_encryption_key(
            SimpleNamespace(app_profile="production", deployment_profile="tiny")
        )
    with pytest.raises(RuntimeError, match="reference is required"):
        gc._resolve_engine_encryption_key(
            SimpleNamespace(app_profile="dev", deployment_profile="enterprise")
        )


def test_local_engine_encryption_key_is_stable_private_and_path_redacted(
    monkeypatch, tmp_path
):
    from agent_utilities.core import paths

    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    first = _REAL_LOCAL_ENCRYPTION_KEY_LOADER()
    second = _REAL_LOCAL_ENCRYPTION_KEY_LOADER()
    private_directory = tmp_path / "engine-private"
    key_file = private_directory / "encryption_key"

    assert first == second
    assert len(first.encode("utf-8")) >= 32
    if os.name == "posix":
        assert private_directory.stat().st_mode & 0o777 == 0o700
        assert key_file.stat().st_mode & 0o777 == 0o600
        key_file.chmod(0o644)
        with pytest.raises(RuntimeError) as exc:
            _REAL_LOCAL_ENCRYPTION_KEY_LOADER()
        assert str(key_file) not in str(exc.value)


def test_local_engine_encryption_key_rejects_broad_private_directory(
    monkeypatch, tmp_path
):
    from agent_utilities.core import paths

    private_directory = tmp_path / "engine-private"
    private_directory.mkdir(mode=0o755)
    if os.name == "posix":
        private_directory.chmod(0o755)
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)

    if os.name == "posix":
        with pytest.raises(RuntimeError) as exc:
            _REAL_LOCAL_ENCRYPTION_KEY_LOADER()
        assert str(private_directory) not in str(exc.value)


def test_autostart_resolves_private_feature_roots_at_runtime(monkeypatch):
    """Durable config carries refs; only resolved values reach the local engine."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import graph_compute as gc

    environments: list[dict[str, str]] = []
    fake_subprocess = SimpleNamespace(
        DEVNULL=-1,
        Popen=lambda _cmd, **kwargs: (
            environments.append(dict(kwargs["env"])),
            SimpleNamespace(poll=lambda: None),
        )[-1],
    )
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(lambda **_kwargs: "connected"),
    )
    monkeypatch.setattr(
        gc, "_resolve_engine_path_ref", lambda ref: f"resolved-{ref[-1]}"
    )

    engine = object.__new__(gc.GraphComputeEngine)
    engine._autostart_engine(
        {
            "tcp_addr": "127.0.0.1:8765",
            "graph_name": "__commons__",
            "verified_context": gc._transport_only_verified_context(),
        },
        None,
        "test-secret",
        SimpleNamespace(
            epistemic_graph_max_resident_graphs=256,
            epistemic_graph_lazy_open_page_size=4096,
            epistemic_graph_max_nodes_per_graph=250_000,
            epistemic_graph_sqlite_transfer_root_ref="secret://sqlite-root",
            epistemic_graph_backup_root_ref="secret://backup-root",
        ),
        fake_subprocess,
        sys,
        SimpleNamespace(sleep=lambda _seconds: None, monotonic=time.monotonic),
        Path,
        coupled=False,
        idle_shutdown_secs=0,
    )

    assert environments[0]["EPISTEMIC_GRAPH_SQLITE_TRANSFER_ROOT"] == "resolved-t"
    assert environments[0]["EPISTEMIC_GRAPH_BACKUP_ROOT"] == "resolved-t"
    assert "EPISTEMIC_GRAPH_SQLITE_TRANSFER_ROOT_REF" not in environments[0]
    assert "EPISTEMIC_GRAPH_BACKUP_ROOT_REF" not in environments[0]


def test_autostart_polls_until_delayed_engine_is_ready(monkeypatch):
    """A healthy cold start is retried instead of failing after one second."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import graph_compute as gc

    attempts = 0

    def _connect(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise ConnectionRefusedError("listener not ready")
        return "connected"

    child = SimpleNamespace(poll=lambda: None)
    fake_subprocess = SimpleNamespace(
        DEVNULL=-1,
        Popen=lambda _cmd, **_kwargs: child,
    )
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(_connect),
    )

    engine = object.__new__(gc.GraphComputeEngine)
    result = engine._autostart_engine(
        {
            "tcp_addr": "127.0.0.1:8765",
            "graph_name": "__commons__",
            "verified_context": gc._transport_only_verified_context(),
        },
        None,
        "test-secret",
        SimpleNamespace(
            epistemic_graph_max_resident_graphs=256,
            epistemic_graph_lazy_open_page_size=4096,
            epistemic_graph_max_nodes_per_graph=250_000,
        ),
        fake_subprocess,
        sys,
        SimpleNamespace(sleep=lambda _seconds: None, monotonic=time.monotonic),
        Path,
        coupled=False,
        idle_shutdown_secs=0,
    )

    assert result == "connected"
    assert attempts == 4


def test_autostart_rejects_signer_registry_without_verified_subject(monkeypatch):
    """An explicit registry may not silently mint a different local authority."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import graph_compute as gc
    from agent_utilities.knowledge_graph.core import session as session_module

    context = {
        "principal": "subject:verified",
        "tenant": "tenant-verified",
        "audience": "graph-audience",
        "agent_id": "subject:verified",
        "roles": ["admin"],
        "scopes": ["kg:admin", "kg:read", "kg:write"],
        "policy_version": "policy:current",
        "delegation": [],
    }
    monkeypatch.setattr(
        session_module,
        "current_session",
        lambda: SimpleNamespace(engine_verified_context=lambda: dict(context)),
    )
    monkeypatch.setenv(
        "EPISTEMIC_GRAPH_SIGNER_KEYS_JSON",
        '{"subject:other":"0123456789abcdef0123456789abcdef"}',
    )
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(lambda **_kwargs: "must-not-connect"),
    )
    spawned = 0

    def _popen(*_args, **_kwargs):
        nonlocal spawned
        spawned += 1
        raise AssertionError("invalid signer registry reached process spawn")

    engine = object.__new__(gc.GraphComputeEngine)
    with pytest.raises(RuntimeError, match="absent from the engine signer registry"):
        engine._autostart_engine(
            {
                "tcp_addr": "127.0.0.1:8765",
                "graph_name": "tenant:graph",
                "verified_context": gc._transport_only_verified_context(),
            },
            None,
            "test-secret",
            SimpleNamespace(
                epistemic_graph_max_resident_graphs=256,
                epistemic_graph_lazy_open_page_size=4096,
                epistemic_graph_max_nodes_per_graph=250_000,
            ),
            SimpleNamespace(DEVNULL=-1, Popen=_popen),
            sys,
            SimpleNamespace(sleep=lambda _seconds: None, monotonic=time.monotonic),
            Path,
            coupled=False,
            idle_shutdown_secs=0,
        )

    assert spawned == 0


def test_fresh_local_engine_bootstraps_exact_verified_identity(monkeypatch):
    """A fresh store gets one signer-bound System identity, then must route."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import graph_compute as gc

    verified_context = {
        "principal": "subject:verified",
        "tenant": "tenant-verified",
        "audience": "graph-audience",
        "agent_id": "subject:verified",
        "roles": ["admin"],
        "scopes": ["kg:admin", "kg:read", "kg:write"],
        "policy_version": "policy:current",
        "delegation": [],
    }
    route_contexts: list[dict[str, object]] = []
    route_calls = 0

    class _Placement:
        def route(self, tenant, sub_key, *, client_epoch):
            nonlocal route_calls
            route_calls += 1
            assert (tenant, sub_key, client_epoch) == (
                "tenant-verified",
                "graph",
                0,
            )
            if route_calls == 1:
                raise PermissionError("empty identity policy")
            return {
                "schema_version": "1",
                "route_id": "route:opaque",
                "tenant_ref": tenant,
                "partition_ref": sub_key,
                "authoritative": True,
                "placed": False,
                "group": 0,
                "epoch": 0,
                "fencing_token": 0,
                "leader_ref": None,
            }

    class _MainClient:
        placement = _Placement()

        @contextlib.contextmanager
        def use_verified_context(self, context):
            route_contexts.append(dict(context))
            yield self

    bootstrap_calls: list[dict[str, str]] = []
    closed = 0

    class _Consensus:
        def bootstrap_system_identity(self, **kwargs):
            bootstrap_calls.append(dict(kwargs))
            return "registered"

    class _BootstrapClient:
        consensus = _Consensus()

        def close(self):
            nonlocal closed
            closed += 1

    connect_calls: list[dict[str, object]] = []

    def _connect(**kwargs):
        connect_calls.append(dict(kwargs))
        return _BootstrapClient()

    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(_connect),
    )
    engine = object.__new__(gc.GraphComputeEngine)
    engine._local_bootstrap_identity = (
        "subject:verified",
        "0123456789abcdef0123456789abcdef",
        dict(verified_context),
    )

    engine._bootstrap_autostarted_engine_identity(
        _MainClient(),
        {
            "tcp_addr": "127.0.0.1:8765",
            "graph_name": "tenant-verified:graph",
            "verified_context": gc._transport_only_verified_context(),
        },
    )

    assert route_calls == 2
    assert route_contexts == [verified_context, verified_context]
    assert len(connect_calls) == 1
    assert connect_calls[0]["graph_name"] == "__commons__"
    assert connect_calls[0]["verified_context"] == {
        **verified_context,
        "roles": [],
        "scopes": ["security:bootstrap"],
        "delegation": [],
    }
    assert bootstrap_calls == [
        {
            "agent_id": "subject:verified",
            "signer_id": "subject:verified",
            "signer_key": "0123456789abcdef0123456789abcdef",
        }
    ]
    assert closed == 1


def _local_graph_session(context, *, admin=True):
    class _Session:
        def require_scope(self, scope):
            assert scope == "kg:admin"
            if not admin:
                raise PermissionError("admin authority required")

        def engine_verified_context(self):
            return dict(context)

    return _Session()


def test_local_graph_readiness_materializes_missing_verified_graph():
    """A routed local tenant graph is explicitly lifecycle-created once."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    graph_name = "tenant-verified:graph"
    context = {"principal": "subject:verified"}
    routed: list[tuple[str, str, int]] = []
    created: list[tuple[str, str]] = []
    contexts: list[dict[str, str]] = []

    class _Placement:
        def route(self, tenant, sub_key, *, client_epoch):
            routed.append((tenant, sub_key, client_epoch))

    class _Tenants:
        @staticmethod
        def list():
            return []

        @staticmethod
        def create(name, graph_type):
            created.append((name, graph_type))

    class _Client:
        placement = _Placement()
        tenants = _Tenants()

        @contextlib.contextmanager
        def use_verified_context(self, supplied):
            contexts.append(dict(supplied))
            yield self

    GraphComputeEngine._ensure_local_session_graph(
        _Client(),
        graph_name,
        _local_graph_session(context),
    )

    assert routed == [("tenant-verified", "graph", 0)]
    assert created == [(graph_name, "Agent")]
    assert contexts == [context]


def test_local_graph_readiness_reuses_existing_graph():
    """An already-running local engine does not recreate its tenant graph."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    graph_name = "tenant-verified:graph"
    created = 0

    class _Placement:
        @staticmethod
        def route(_tenant, _sub_key, *, client_epoch):
            assert client_epoch == 0

    class _Tenants:
        @staticmethod
        def list():
            return [{"name": graph_name}]

        @staticmethod
        def create(_name, _graph_type):
            nonlocal created
            created += 1

    class _Client:
        placement = _Placement()
        tenants = _Tenants()

        @contextlib.contextmanager
        def use_verified_context(self, _context):
            yield self

    GraphComputeEngine._ensure_local_session_graph(
        _Client(),
        graph_name,
        _local_graph_session({"principal": "subject:verified"}),
    )

    assert created == 0


def test_local_graph_readiness_skips_commons_without_lifecycle_authority():
    """The engine-owned shared graph never enters tenant lifecycle handling."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    class _ForbiddenSession:
        @staticmethod
        def require_scope(_scope):
            raise AssertionError("commons requested lifecycle authority")

    GraphComputeEngine._ensure_local_session_graph(
        object(),
        "__commons__",
        _ForbiddenSession(),
    )


def test_local_graph_readiness_reconciles_only_a_confirmed_create_race():
    """A concurrent create is accepted only after an authoritative re-list."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    graph_name = "tenant-verified:graph"
    listings = iter([[], [{"name": graph_name}]])

    class _Placement:
        @staticmethod
        def route(_tenant, _sub_key, *, client_epoch):
            assert client_epoch == 0

    class _Tenants:
        @staticmethod
        def list():
            return next(listings)

        @staticmethod
        def create(_name, _graph_type):
            raise RuntimeError("concurrent create")

    class _Client:
        placement = _Placement()
        tenants = _Tenants()

        @contextlib.contextmanager
        def use_verified_context(self, _context):
            yield self

    GraphComputeEngine._ensure_local_session_graph(
        _Client(),
        graph_name,
        _local_graph_session({"principal": "subject:verified"}),
    )


def test_local_graph_readiness_fails_closed_when_create_race_is_unconfirmed():
    """A create error is preserved when the graph is still absent."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    class _Placement:
        @staticmethod
        def route(_tenant, _sub_key, *, client_epoch):
            assert client_epoch == 0

    class _Tenants:
        @staticmethod
        def list():
            return []

        @staticmethod
        def create(_name, _graph_type):
            raise RuntimeError("create failed")

    class _Client:
        placement = _Placement()
        tenants = _Tenants()

        @contextlib.contextmanager
        def use_verified_context(self, _context):
            yield self

    with pytest.raises(RuntimeError, match="create failed"):
        GraphComputeEngine._ensure_local_session_graph(
            _Client(),
            "tenant-verified:graph",
            _local_graph_session({"principal": "subject:verified"}),
        )


def test_local_graph_readiness_requires_admin_before_engine_lifecycle_calls():
    """A non-admin process cannot implicitly provision a local graph."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    with pytest.raises(PermissionError, match="admin authority required"):
        GraphComputeEngine._ensure_local_session_graph(
            object(),
            "tenant-verified:graph",
            _local_graph_session({"principal": "subject:verified"}, admin=False),
        )


def test_autostart_surfaces_early_child_exit_without_sensitive_details(monkeypatch):
    """An exited child reports only its status instead of ConnectionRefused."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core import graph_compute as gc

    child = SimpleNamespace(poll=lambda: 17)
    fake_subprocess = SimpleNamespace(
        DEVNULL=-1,
        Popen=lambda _cmd, **_kwargs: child,
    )
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(lambda **_kwargs: (_ for _ in ()).throw(ConnectionRefusedError())),
    )

    engine = object.__new__(gc.GraphComputeEngine)
    with pytest.raises(ConnectionError, match=r"exited during startup \(status 17\)"):
        engine._autostart_engine(
            {
                "tcp_addr": "127.0.0.1:8765",
                "graph_name": "__commons__",
                "verified_context": gc._transport_only_verified_context(),
            },
            None,
            "test-secret",
            SimpleNamespace(
                epistemic_graph_max_resident_graphs=256,
                epistemic_graph_lazy_open_page_size=4096,
                epistemic_graph_max_nodes_per_graph=250_000,
            ),
            fake_subprocess,
            sys,
            SimpleNamespace(sleep=lambda _seconds: None, monotonic=time.monotonic),
            Path,
            coupled=False,
            idle_shutdown_secs=0,
        )


def test_guard_timeout_never_spawns_a_competing_engine(monkeypatch):
    """A process that did not acquire the spawn guard must fail without spawning."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.core import config as config_module
    from agent_utilities.knowledge_graph.core import (
        engine_breaker,
        engine_lock,
        session,
        shard_topology,
    )
    from agent_utilities.knowledge_graph.core import graph_compute as gc

    fake_config = SimpleNamespace(
        graph_service_endpoints=None,
        kg_default_graph="__commons__",
    )
    resolved = er.ResolvedEngine(
        endpoint="unix://engine.sock",
        auth_secret="test-secret",
        mode="autostart",
        autostart_allowed=True,
        idle_shutdown_secs=60,
    )

    @contextlib.contextmanager
    def _not_owner(_key):
        yield False

    spawned = 0

    def _unexpected_spawn(*_args, **_kwargs):
        nonlocal spawned
        spawned += 1
        raise AssertionError("non-owner attempted to spawn")

    monkeypatch.setattr(config_module, "AgentConfig", lambda: fake_config)
    monkeypatch.setattr(er, "resolve_engine", lambda *_args: resolved)
    monkeypatch.setattr(session, "graph_session_required", lambda: False)
    monkeypatch.setattr(
        shard_topology,
        "resolve_endpoints",
        lambda _config: [resolved.endpoint],
    )
    monkeypatch.setattr(shard_topology, "record_shard_connect", lambda *_args: None)
    monkeypatch.setattr(
        engine_breaker,
        "get_breaker",
        lambda _endpoint: SimpleNamespace(
            before_call=lambda: None,
            record_failure=lambda: None,
        ),
    )
    monkeypatch.setattr(engine_lock, "engine_spawn_guard", _not_owner)
    monkeypatch.setattr(
        SyncEpistemicGraphClient,
        "connect",
        staticmethod(lambda **_kwargs: (_ for _ in ()).throw(ConnectionRefusedError())),
    )
    monkeypatch.setattr(gc.GraphComputeEngine, "_autostart_engine", _unexpected_spawn)

    with pytest.raises(ConnectionError, match="startup owner"):
        gc.GraphComputeEngine(graph_name="__commons__")

    assert spawned == 0


# ---------------------------------------------------------------------------
# idle-shutdown / lifecycle resolution
# ---------------------------------------------------------------------------


def test_refcounted_lifecycle_passes_grace(monkeypatch):
    monkeypatch.setenv("ENGINE_LIFECYCLE", "refcounted")
    monkeypatch.setenv("ENGINE_IDLE_SHUTDOWN_SECS", "45")
    assert er.engine_idle_shutdown_secs(AgentConfig()) == 45


def test_persistent_lifecycle_is_long_living(monkeypatch):
    """engine_lifecycle=persistent → 0 (never auto-stop), even with a grace set."""
    monkeypatch.setenv("ENGINE_LIFECYCLE", "persistent")
    monkeypatch.setenv("ENGINE_IDLE_SHUTDOWN_SECS", "60")
    assert er.engine_idle_shutdown_secs(AgentConfig()) == 0


def test_nonpositive_grace_is_persistent(monkeypatch):
    monkeypatch.setenv("ENGINE_IDLE_SHUTDOWN_SECS", "0")
    assert er.engine_idle_shutdown_secs(AgentConfig()) == 0
    monkeypatch.setenv("ENGINE_IDLE_SHUTDOWN_SECS", "-5")
    assert er.engine_idle_shutdown_secs(AgentConfig()) == 0


# ---------------------------------------------------------------------------
# (c) autostart spawn contract — single PID, no redb-lock collision across
#     concurrent resolves (uses a fake engine binary, no real engine needed)
# ---------------------------------------------------------------------------


_FAKE_ENGINE = """#!/usr/bin/env bash
# Fake epistemic-graph-server: --help advertises the flag; otherwise bind the
# socket and idle so a probe sees it 'running' and a spawn-count file proves how
# many were started.
if [ "$1" = "--help" ]; then
  echo "usage: epistemic-graph-server [--socket-path P] [--idle-shutdown-secs N]"
  exit 0
fi
sock=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --socket-path) sock="$2"; shift 2;;
    *) shift;;
  esac
done
echo "$$" >> "$SPAWN_COUNT_FILE"
python3 -c "
import socket, sys, time, os
p = sys.argv[1]
try: os.unlink(p)
except OSError: pass
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(p); s.listen(8); s.settimeout(0.2)
end = time.time() + 8
while time.time() < end:
    try:
        c,_ = s.accept(); c.close()
    except OSError:
        pass
" "$sock"
"""


def _install_fake_engine(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    """Install a fake engine binary. Returns (binary_path, spawn_count_file)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "epistemic-graph-server"
    binary.write_text(_FAKE_ENGINE)
    binary.chmod(0o755)
    count_file = tmp_path / "spawn_count"
    count_file.write_text("")
    monkeypatch.setenv("SPAWN_COUNT_FILE", str(count_file))
    return str(binary), count_file


@pytest.mark.skipif(
    not Path("/bin/bash").exists() and not Path("/usr/bin/bash").exists(),
    reason="spawn matrix needs bash + a writable runtime",
)
def test_concurrent_autostart_shares_one_engine(monkeypatch, tmp_path):
    """Two concurrent spawners on the same socket spawn exactly ONE engine.

    Exercises the per-socket ``engine_spawn_guard`` (double-checked) the resolver's
    autostart leg uses: the second spawner shares the first's engine instead of
    starting a second on the same --persist-dir (the redb-lock collision this
    prevents). Uses a fake engine binary so no real engine is required.
    """
    import subprocess

    from agent_utilities.knowledge_graph.core.engine_lock import engine_spawn_guard

    binary, count_file = _install_fake_engine(tmp_path, monkeypatch)
    sock = str(tmp_path / "eg-spawn.sock")

    def _resolve_and_spawn() -> None:
        # Mirror the resolver's autostart leg: probe, then guard + double-check.
        if er.probe_endpoint(f"unix://{sock}", timeout=0.3):
            return
        with engine_spawn_guard(sock):
            if er.probe_endpoint(f"unix://{sock}", timeout=0.3):
                return
            subprocess.Popen(  # noqa: S603
                [binary, "--socket-path", sock],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=dict(os.environ),
            )
            # Wait until the fake engine is actually listening before releasing
            # the guard, so the peer's double-check sees it up.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if er.probe_endpoint(f"unix://{sock}", timeout=0.2):
                    break
                time.sleep(0.1)

    t1 = threading.Thread(target=_resolve_and_spawn)
    t2 = threading.Thread(target=_resolve_and_spawn)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    spawned = [ln for ln in count_file.read_text().splitlines() if ln.strip()]
    assert len(spawned) == 1, f"expected ONE engine spawned, got {len(spawned)}"
