# CONCEPT:AU-KG.compute.graph-compute-engine - High-Performance Graph Compute Engine
# CONCEPT:AU-ORCH.sandbox.compiled-orchestration-kernel - Compiled Orchestration Kernel
# CONCEPT:AU-KG.compute.tokio-service-layer - Tokio Service Layer (Tokio-first)
# CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw - Engine-authoritative tenant partition placement

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Linux prctl(2) option to deliver a signal to the calling thread when its
# parent dies — used to lifecycle-couple an embedded engine to its spawner.
_PR_SET_PDEATHSIG = 1

# A full-feature engine may need more than one second to open its durable store
# and bind its listener on a constrained host. Keep the spawn guard held while
# polling so another process cannot mistake a healthy cold start for a dead
# daemon and launch a competing writer.
_ENGINE_STARTUP_TIMEOUT_SECS = 30.0
_ENGINE_STARTUP_POLL_SECS = 0.1

# Children spawned in *coupled* mode (the embedded/tiny path) so the embedded
# engine dies with this process. The parent-death signal (Linux) is the primary
# guard; this registry backs an atexit + SIGTERM/SIGINT handler so the child is
# reaped on a clean interpreter exit too. Detached daemons (graph-os-host /
# enterprise shards) are NOT tracked here — they intentionally outlive us.
_coupled_children: list[Any] = []
_coupled_handlers_installed = False


_CYPHER_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MAX_LEDGER_ENTRY_BYTES = 8 * 1024 * 1024
_OPAQUE_PROGRAM_REF = re.compile(
    r"^eg:[a-z0-9_-]{1,32}(?::[a-z0-9_-]{1,32}){0,3}:[0-9a-f]{16,128}$"
)
_PROGRAM_RESULT_FIELDS = frozenset(
    {
        "id",
        "kind",
        "confidence",
        "evidence_refs",
        "source_refs",
        "proof_ids",
        "contradiction_ids",
        "program_ref",
        "optimizer",
        "execution",
        "candidate_role",
        "demonstration_refs",
        "artifact_refs",
        "composition_refs",
        "instruction_ref",
        "tool_policy_ref",
        "model_profile_ref",
        "modalities",
        "plan_ref",
        "plan_step_kinds",
        "plan_executors",
        "plan_input_refs",
        "plan_output_refs",
        "plan_depends_on",
        "max_operations",
        "selected",
    }
)
_PROGRAM_RESULT_SCHEMA = (
    ("id", "string", False),
    ("kind", "string", False),
    ("confidence", "float64", False),
    ("evidence_refs", "list<string>", False),
    ("source_refs", "list<string>", False),
    ("proof_ids", "list<string>", False),
    ("contradiction_ids", "list<string>", False),
    ("program_ref", "string", False),
    ("optimizer", "string", False),
    ("execution", "string", False),
    ("candidate_role", "string", True),
    ("demonstration_refs", "list<string>", False),
    ("artifact_refs", "list<string>", False),
    ("composition_refs", "list<string>", False),
    ("instruction_ref", "string", True),
    ("tool_policy_ref", "string", True),
    ("model_profile_ref", "string", True),
    ("modalities", "list<string>", False),
    ("plan_ref", "string", True),
    ("plan_step_kinds", "list<string>", False),
    ("plan_executors", "list<string>", False),
    ("plan_input_refs", "list<string>", False),
    ("plan_output_refs", "list<string>", False),
    ("plan_depends_on", "list<string>", False),
    ("max_operations", "uint64", True),
    ("selected", "bool", False),
)
_PROGRAM_MODALITIES = frozenset(
    {
        "text",
        "document",
        "image",
        "audio",
        "video",
        "graph",
        "table",
        "time_series",
        "vector",
        "spatial",
        "tensor",
        "code",
        "trace",
        "binary",
    }
)
_PROGRAM_OPTIMIZER_EXECUTIONS = {
    "labeled_few_shot": "native_kernel",
    "bootstrap_few_shot": "native_kernel",
    "bootstrap_few_shot_with_random_search": "native_kernel",
    "avatar": "model_transport_plan",
    "copro": "model_transport_plan",
    "mipro_v2": "model_transport_plan",
    "simba": "model_transport_plan",
    "gepa": "model_transport_plan",
    "better_together": "composite_plan",
    "bootstrap_finetune": "trainer_plan",
    "knn_few_shot": "graph_kernel_plan",
    "ensemble": "native_kernel",
    "infer_rules": "model_transport_plan",
}
_PROGRAM_CANDIDATE_ROLES = frozenset({"proposal", "ensemble_member", "ensemble"})
_PROGRAM_PLAN_STEP_KINDS = frozenset(
    {
        "query_similarity",
        "propose_instruction",
        "compare_tool_use",
        "propose_rules",
        "reflect_on_trace",
        "pareto_reflect",
        "compose_programs",
        "train_weights",
        "evaluate_candidates",
    }
)
_PROGRAM_PLAN_EXECUTORS = frozenset(
    {"native_kernel", "graph_similarity", "model_transport", "evaluator", "trainer"}
)

# A native client validates authority at construction even though opening the
# socket sends no request.  The process transport therefore starts with one
# deliberately powerless, non-personal context.  Routed operations MUST replace
# it with the current GraphSession before dispatch (enforced in
# ``_SessionRoutedAsyncClient._send``); no role, scope, workstation name, path,
# or deployment identity is encoded here.
_TRANSPORT_PRINCIPAL = (
    "service:sha256:"
    + hashlib.sha256(b"agent-utilities:epistemic-graph:transport-only:v1").hexdigest()
)


def _transport_only_verified_context() -> dict[str, Any]:
    """Return a fresh, zero-authority context used only to open the socket."""

    return {
        "principal": _TRANSPORT_PRINCIPAL,
        "tenant": "tenant:transport-only",
        "audience": "epistemic-graph",
        "agent_id": _TRANSPORT_PRINCIPAL,
        "roles": [],
        "scopes": [],
        "policy_version": "policy:transport-only",
        "delegation": [],
    }


def _engine_child_environment() -> dict[str, str]:
    """Return the minimal inherited environment for the trusted Rust engine.

    Autostart previously copied the agent's complete environment, needlessly
    disclosing unrelated provider/API credentials to the child. The engine now
    receives only operating-system essentials, transport trust settings, proxy
    settings, diagnostics, and its own documented namespaces.
    """

    exact = {
        "ALL_PROXY",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "NO_PROXY",
        "OPENSSL_CONF",
        "PATH",
        "PATHEXT",
        "RUST_BACKTRACE",
        "RUST_LOG",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "USERPROFILE",
        "WINDIR",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    }
    prefixes = ("EG_", "EPISTEMIC_GRAPH_", "GRAPH_SERVICE_", "LC_")
    inherited = {
        name: value
        for name, value in os.environ.items()
        if name in exact or name.startswith(prefixes)
    }
    # These capabilities accept paths only after runtime secret resolution;
    # direct inherited paths and the references themselves never cross over.
    for name in (
        "EPISTEMIC_GRAPH_BACKUP_ROOT",
        "EPISTEMIC_GRAPH_BACKUP_ROOT_REF",
        "EPISTEMIC_GRAPH_ENCRYPTION_KEY",
        "EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF",
        "EPISTEMIC_GRAPH_SQLITE_TRANSFER_ROOT",
        "EPISTEMIC_GRAPH_SQLITE_TRANSFER_ROOT_REF",
        "GRAPH_SERVICE_AUTH_SECRET",
    ):
        inherited.pop(name, None)
    return inherited


def _resolve_engine_path_ref(reference: str) -> str:
    """Resolve one runtime-only directory reference without logging its value."""

    try:
        from agent_utilities.security.secrets_client import create_secrets_client

        value = create_secrets_client().resolve_ref(reference)
        if isinstance(value, bytes):
            rendered = value.decode("utf-8")
        else:
            rendered = str(value or "")
    except Exception as exc:
        raise RuntimeError("engine runtime directory reference is unavailable") from exc
    if (
        not rendered
        or len(rendered.encode("utf-8")) > 4_096
        or any(ord(character) < 32 for character in rendered)
    ):
        raise RuntimeError("engine runtime directory reference is invalid")
    return rendered


# Every graph-scoped facade below reuses this one namespace list from the
# generated epistemic-graph client.  Re-instantiating a namespace is cheap: it
# stores only a reference to the routed async view and opens no connection.
_CLIENT_NAMESPACES: tuple[str, ...] = (
    "nodes",
    "work_items",
    "changes",
    "edges",
    "graph",
    "analytics",
    "lifecycle",
    "reasoning",
    "ledger",
    "channels",
    "tenants",
    "resharding",
    "placement",
    "consensus",
    "finance",
    "datascience",
    "mining",
    "graphlearn",
    "query",
    "txn",
    "timeseries",
    "rdf",
    "streaming",
    "blob",
    "broker",
    "rbac",
    "admin",
    "jobs",
)


class _SessionRoutedAsyncClient:
    """A zero-connection view over one async engine transport.

    Every namespace call lands on :meth:`_send`, where the current immutable
    :class:`GraphSession` is both (a) bound into the engine's v2 signed request
    context and (b) used as the default named graph.  A fixed graph creates a
    scoped view for ingestion fan-out without mutating the shared transport.
    """

    def __init__(
        self,
        base: Any,
        *,
        fixed_graph: str | None = None,
        route_config: Any = None,
        route_endpoints: tuple[str, ...] = (),
        transport_endpoint: str | None = None,
    ) -> None:
        self._base = base
        self._graph_name = fixed_graph or str(getattr(base, "_graph_name", ""))
        self._fixed_graph = fixed_graph
        self._route_config = route_config
        self._route_endpoints = route_endpoints
        self._transport_endpoint = transport_endpoint
        self._server_ops: set[str] | None = None
        for name in _CLIENT_NAMESPACES:
            namespace = getattr(base, name)
            setattr(self, name, type(namespace)(self))

    @staticmethod
    def _route_bound_params(
        method: str, params: dict[str, Any] | None, route: Any
    ) -> dict[str, Any] | None:
        """Bind the current catalog fence into an engine-native mutation."""
        if method != "ApplyChangeEnvelope" or params is None:
            return params
        import copy

        bound = copy.deepcopy(params)
        mutation = bound.get("envelope", {}).get("mutation")
        if isinstance(mutation, dict):
            epoch = int(route.epoch)
            fencing_token = int(route.fencing_token or 0)
            mutation["placement_epoch"] = epoch
            if epoch == 0:
                # The placement catalog uses (epoch=0, group=0) for the
                # authoritative local/unplaced route. ``fencing_token`` is an
                # optional current-wire field, so a zero/null sentinel must be
                # absent rather than injected after generated-client
                # canonicalization; otherwise the signed Python and Rust
                # method bodies differ and authentication fails closed.
                mutation.pop("fencing_token", None)
            elif fencing_token > 0:
                mutation["fencing_token"] = fencing_token
            else:
                raise RuntimeError("placed route is missing a fencing token")
        return bound

    async def _invoke_at(
        self,
        endpoint: str,
        method: str,
        params: dict[str, Any] | None,
        target: str,
        idempotency_key: str | None,
        session: Any,
    ) -> Any:
        """Invoke on the process transport or a bounded redirect connection."""
        if not endpoint or endpoint == self._transport_endpoint:
            with self._base.use_verified_context(session.engine_verified_context()):
                return await self._base._send(
                    method, params, graph=target, idempotency_key=idempotency_key
                )

        from epistemic_graph.client import EpistemicGraphClient

        kwargs: dict[str, Any] = {
            "auth_secret": str(getattr(self._base, "_auth_secret", "") or ""),
            "graph_name": target,
            "verified_context": session.engine_verified_context(),
        }
        if endpoint.startswith(("tcp://", "tls://")):
            from .engine_transport import (
                engine_client_transport_kwargs,
                native_endpoint_address,
            )

            kwargs["tcp_addr"] = native_endpoint_address(endpoint)[0]
            kwargs.update(
                engine_client_transport_kwargs(endpoint, config=self._route_config)
            )
        elif endpoint.startswith("unix://"):
            kwargs["socket_path"] = endpoint[7:]
        else:
            kwargs["socket_path"] = endpoint
        redirected = await EpistemicGraphClient.connect(**kwargs)
        try:
            with redirected.use_verified_context(session.engine_verified_context()):
                return await redirected._send(
                    method, params, graph=target, idempotency_key=idempotency_key
                )
        finally:
            await redirected.close()

    async def _send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        graph: str | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        from .session import SessionRequiredError, current_session, resolve_session

        session = current_session()
        # The socket was opened with a fixed, zero-authority transport context.
        # It is never a request identity. Every operation must inherit the
        # authentication boundary's task-local GraphSession and replace that
        # context before the native client signs or writes a frame.
        if session is None or not getattr(session.actor, "authenticated", False):
            raise SessionRequiredError(
                "A task-local verified GraphSession is required for every engine operation"
            )
        session = resolve_session(session)

        target = graph or self._fixed_graph
        if self._fixed_graph and session.graph != self._fixed_graph:
            raise PermissionError(
                "A graph-scoped view cannot retarget the verified GraphSession"
            )
        if graph and session.graph and graph != session.graph:
            raise PermissionError(
                "An explicit graph cannot retarget the verified GraphSession"
            )
        target = target or session.graph

        target = target or self._graph_name

        # Service-level operations are connection-scoped, not graph-routed.
        unrouted = {
            "Ping",
            "Health",
            "PlacementRoute",
            "Shutdown",
            "Checkpoint",
            "ResourceStats",
            "CancelRequest",
        }
        if (
            method in unrouted
            or self._route_config is None
            or not self._route_endpoints
        ):
            return await self._invoke_at(
                self._transport_endpoint or "",
                method,
                params,
                target,
                idempotency_key,
                session,
            )

        import asyncio

        from epistemic_graph.client import StaleRouteError

        from .placement_catalog import resolve_placement

        route = await asyncio.to_thread(
            resolve_placement,
            target,
            self._route_endpoints,
            self._route_config,
        )
        routed_session = session.with_route(
            endpoint=route.endpoint,
            placement_group=(int(route.group) if int(route.group or 0) > 0 else None),
            catalog_epoch=int(route.epoch),
        )
        routed_params = self._route_bound_params(method, params, route)
        try:
            return await self._invoke_at(
                route.endpoint,
                method,
                routed_params,
                target,
                idempotency_key,
                routed_session,
            )
        except StaleRouteError:
            # A stale response is guaranteed to be pre-commit. Refresh the
            # authoritative catalog and retry exactly once with the same
            # idempotency key and the new placement fence.
            fresh = await asyncio.to_thread(
                resolve_placement,
                target,
                self._route_endpoints,
                self._route_config,
                force_refresh=True,
            )
            fresh_session = session.with_route(
                endpoint=fresh.endpoint,
                placement_group=(
                    int(fresh.group) if int(fresh.group or 0) > 0 else None
                ),
                catalog_epoch=int(fresh.epoch),
            )
            return await self._invoke_at(
                fresh.endpoint,
                method,
                self._route_bound_params(method, params, fresh),
                target,
                idempotency_key,
                fresh_session,
            )

    def _verified_tenant(self) -> str:
        """Return the tenant from the current verified graph authority.

        Generated ChangeEnvelope read helpers bind their tenant parameter
        before calling :meth:`_send`. Resolve that value from the task-local
        ``GraphSession`` instead of delegating to the process transport's
        deliberately powerless bootstrap context.
        """
        from .session import SessionRequiredError, current_session, resolve_session

        session = current_session()
        if session is None or not getattr(session.actor, "authenticated", False):
            raise SessionRequiredError(
                "A task-local verified GraphSession is required for every engine operation"
            )
        return str(resolve_session(session).tenant)

    # Service-level methods live on the root client rather than a namespace;
    # spell them out so they also pass through the routed _send above.
    async def ping(self) -> str:
        return await self._send("Ping")

    async def health(self) -> dict[str, Any]:
        return await self._send("Health")

    async def cancel_request(self, target_req_id: int) -> bool:
        return await self._send("CancelRequest", {"target_req_id": target_req_id})

    async def resource_stats(self) -> dict[str, Any]:
        return await self._send("ResourceStats")

    async def supports(self, op: str) -> bool:
        if self._server_ops is None:
            try:
                health = await self.health()
                self._server_ops = (
                    set(health.get("ops", []) or [])
                    if isinstance(health, dict)
                    else set()
                )
            except Exception:
                self._server_ops = set()
        return op in self._server_ops

    async def checkpoint(self) -> str:
        return await self._send("Checkpoint")

    async def reconcile(self, graph_name: str, json_str: str) -> str:
        return await self._send(
            "Reconcile", {"graph_name": graph_name, "json_str": json_str}
        )

    async def shutdown(self) -> str:
        return await self._send("Shutdown")

    async def apply_mutation(self, event_type: str, query: str) -> str:
        return await self._send(
            "ApplyMutation", {"event_type": event_type, "query": query}
        )

    async def close(self) -> None:
        """A view never owns or closes the shared process transport."""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class _AsyncFromSyncView:
    """Awaitable facade over a non-owning sync client/namespace view."""

    def __init__(self, sync_target: Any) -> None:
        self._sync_target = sync_target

    def __getattr__(self, name: str) -> Any:
        import asyncio

        attr = getattr(self._sync_target, name)
        if not callable(attr):
            return attr

        async def call(*args: Any, **kwargs: Any) -> Any:
            # to_thread propagates ContextVars, including GraphSession, into
            # the sync scheduling call. The generated client's own event-loop
            # thread and socket remain the single process authority.
            return await asyncio.to_thread(attr, *args, **kwargs)

        return call


class _AsyncProcessClientView(_AsyncFromSyncView):
    """Async-compatible namespace facade without another engine connection."""

    def __init__(self, sync_client: Any) -> None:
        super().__init__(sync_client)
        for name in _CLIENT_NAMESPACES:
            setattr(self, name, _AsyncFromSyncView(getattr(sync_client, name)))

    async def close(self) -> None:
        """A process client view never owns the shared transport."""


def _sync_client_view(sync_client: Any, *, graph: str | None = None) -> Any:
    """Create a sync graph/session view while reusing transport, loop and thread."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    async_client = sync_client._client
    if isinstance(async_client, _SessionRoutedAsyncClient):
        async_client = async_client._base
    route_config = getattr(sync_client, "_au_route_config", None)
    route_endpoints = tuple(getattr(sync_client, "_au_route_endpoints", ()) or ())
    transport_endpoint = getattr(sync_client, "_au_route_endpoint", None)
    async_view = _SessionRoutedAsyncClient(
        async_client,
        fixed_graph=graph,
        route_config=route_config,
        route_endpoints=route_endpoints,
        transport_endpoint=transport_endpoint,
    )
    view = SyncEpistemicGraphClient(async_view, sync_client._loop, sync_client._thread)
    view._au_route_config = route_config
    view._au_route_endpoints = route_endpoints
    view._au_route_endpoint = transport_endpoint
    # A scoped view never owns the shared loop/thread. SyncEpistemicGraphClient
    # normally stops both from close()/context-manager exit, so shadow close on
    # this instance with an explicit no-op.
    view.close = lambda: None
    return view


def _set_pdeathsig() -> None:
    """preexec_fn: ask the kernel to SIGTERM this child when its parent dies.

    Best-effort and Linux-only (``prctl`` via ``ctypes``). On any other platform
    — or if libc/prctl is unavailable — this is a no-op and the atexit/signal
    handler remains the coupling mechanism. Runs in the forked child between
    fork and exec, so it must stay tiny and dependency-free.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        import signal as _signal

        libc.prctl(_PR_SET_PDEATHSIG, _signal.SIGTERM, 0, 0, 0)
    except Exception:  # noqa: BLE001 — coupling is best-effort
        pass


def _terminate_coupled_children() -> None:
    """Terminate every coupled embedded engine we spawned. Idempotent."""
    while _coupled_children:
        child = _coupled_children.pop()
        try:
            if child.poll() is None:
                child.terminate()
        except Exception:  # noqa: BLE001 — teardown must never raise
            pass


def _install_coupled_handlers() -> None:
    """Register the atexit + SIGTERM/SIGINT teardown for coupled children once.

    The parent-death signal handles the hard-kill case on Linux; this covers the
    clean-exit and signalled-shutdown cases (and is the only mechanism on
    non-Linux). Existing signal handlers are chained, not clobbered.
    """
    global _coupled_handlers_installed
    if _coupled_handlers_installed:
        return
    import atexit
    import signal

    atexit.register(_terminate_coupled_children)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(_sig)

            def _handler(signum: int, frame: Any, _prev: Any = prev) -> None:
                _terminate_coupled_children()
                if callable(_prev) and _prev not in (
                    signal.SIG_IGN,
                    signal.SIG_DFL,
                ):
                    _prev(signum, frame)

            signal.signal(_sig, _handler)
        except (ValueError, OSError, RuntimeError):
            # Not on the main thread (e.g. inside a server worker) — the atexit
            # hook still covers clean shutdown; skip the signal handler.
            pass
    _coupled_handlers_installed = True


def _load_or_create_engine_secret() -> str:
    """Load (or generate once) the per-install engine HMAC secret.

    CONCEPT:AU-OS.identity.authenticated-identity-enforcement — Authenticated Identity Enforcement. The secret lives at
    ``data_dir()/engine_secret`` with mode 0600 so every local process — and
    every engine this launcher spawns — shares it. Creation is race-safe
    (``O_EXCL``; the loser re-reads the winner's secret). If the data dir is
    unwritable a process-local secret is used (warned: siblings won't share it).
    """
    import secrets as _secrets

    from agent_utilities.core.paths import data_dir

    path = data_dir() / "engine_secret"
    try:
        if path.exists():
            existing = path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        secret = _secrets.token_hex(32)
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return path.read_text(encoding="utf-8").strip() or secret
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(secret)
        return secret
    except OSError as exc:
        logger.warning(
            "Could not persist engine secret (%s); using a process-local secret — "
            "sibling processes will not share it.",
            type(exc).__name__,
        )
        return _secrets.token_hex(32)


def _load_or_create_engine_bootstrap_signer_key() -> str:
    """Return the private per-install key for one local identity bootstrap.

    A packaged engine starts with an empty, fail-closed identity store.  The
    first externally verified process identity therefore needs one signer-backed
    ``RegisterIdentity(System)`` operation before administrative placement can
    be queried.  This key is generated outside source/configuration, persisted
    privately beside the engine secret, and used only for that one local
    initialization path.  Remote engines remain operator-provisioned.
    """
    import secrets as _secrets

    from agent_utilities.core.paths import data_dir

    path = data_dir() / "engine_bootstrap_signer_key"
    try:
        if path.exists():
            existing = path.read_text(encoding="utf-8").strip()
            if len(existing.encode("utf-8")) >= 32:
                return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        key = _secrets.token_urlsafe(48)
        try:
            descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = path.read_text(encoding="utf-8").strip()
            if len(existing.encode("utf-8")) < 32:
                raise RuntimeError("local engine bootstrap signer is invalid") from None
            return existing
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(key)
        return key
    except OSError as exc:
        raise RuntimeError("local engine bootstrap signer is unavailable") from exc


_ENGINE_ENCRYPTION_KEY_MIN_BYTES = 32
_ENGINE_ENCRYPTION_KEY_MAX_BYTES = 4_096


def _validate_engine_encryption_material(value: Any) -> str:
    """Return bounded, environment-safe engine data-key material.

    Diagnostics deliberately describe only the policy class. The material,
    reference, and filesystem location never enter an exception or log record.
    """

    import unicodedata

    try:
        rendered = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        encoded = rendered.encode("utf-8")
    except (UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeError("engine encryption key material is invalid") from exc
    if not (
        _ENGINE_ENCRYPTION_KEY_MIN_BYTES
        <= len(encoded)
        <= _ENGINE_ENCRYPTION_KEY_MAX_BYTES
    ) or any(unicodedata.category(character).startswith("C") for character in rendered):
        raise RuntimeError("engine encryption key material is invalid")
    return rendered


def _read_private_engine_encryption_key(path: Any) -> str:
    """Read one stable local data key through a no-follow private descriptor."""

    import stat

    descriptor = -1
    try:
        before_open = path.lstat()
        if stat.S_ISLNK(before_open.st_mode) or not stat.S_ISREG(before_open.st_mode):
            raise PermissionError("unsafe local key source")
        if os.name == "posix":
            if before_open.st_uid not in {0, os.geteuid()}:
                raise PermissionError("untrusted local key owner")
            if stat.S_IMODE(before_open.st_mode) != 0o600:
                raise PermissionError("local key permissions are not private")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(str(path), flags)
        opened = os.fstat(descriptor)
        if (
            before_open.st_dev,
            before_open.st_ino,
        ) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise PermissionError("local key source changed during open")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(_ENGINE_ENCRYPTION_KEY_MAX_BYTES + 1)
            after_read = os.fstat(handle.fileno())
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            after_read.st_dev,
            after_read.st_ino,
            after_read.st_size,
            after_read.st_mtime_ns,
        ) or len(payload) != opened.st_size:
            raise PermissionError("local key source changed during read")
        return _validate_engine_encryption_material(payload)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError("local engine encryption key is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_or_create_engine_encryption_key() -> str:
    """Load or atomically create the stable private key for local tiny mode.

    Unlike an authentication-only process-local fallback, encryption-at-rest
    must remain decryptable after restart. Failure to persist or privately read
    this key therefore fails closed.
    """

    import secrets as _secrets
    import stat

    from agent_utilities.core.paths import data_dir

    private_directory = data_dir() / "engine-private"
    try:
        private_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = private_directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError("unsafe local key directory")
        if os.name == "posix":
            if metadata.st_uid not in {0, os.geteuid()}:
                raise PermissionError("untrusted local key directory owner")
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                raise PermissionError("local key directory is not private")
    except Exception as exc:
        raise RuntimeError("local engine encryption key is unavailable") from exc

    path = private_directory / "encryption_key"
    try:
        return _read_private_engine_encryption_key(path)
    except FileNotFoundError:
        pass

    material = _secrets.token_urlsafe(48)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError:
        return _read_private_engine_encryption_key(path)
    except OSError as exc:
        raise RuntimeError("local engine encryption key is unavailable") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(material.encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            os.chmod(path, 0o600)
    except Exception as exc:
        with contextlib.suppress(OSError):
            path.unlink()
        raise RuntimeError("local engine encryption key is unavailable") from exc
    return _read_private_engine_encryption_key(path)


def _resolve_engine_encryption_key(config: Any) -> str:
    """Resolve one key for a locally spawned Rust engine without exposing it."""

    reference = str(
        getattr(config, "epistemic_graph_encryption_key_ref", "") or ""
    ).strip()
    if reference:
        if not reference.startswith(("env://", "vault://")):
            raise RuntimeError("engine encryption key reference is invalid")
        try:
            from agent_utilities.security.cli_secrets import (
                resolve_runtime_secret_reference,
            )

            material = resolve_runtime_secret_reference(reference)
        except Exception as exc:
            raise RuntimeError(
                "engine encryption key reference is unavailable"
            ) from exc
        return _validate_engine_encryption_material(material)

    from agent_utilities.core.profile_guard import is_production_profile

    profile = str(getattr(config, "app_profile", "dev") or "dev")
    deployment = str(getattr(config, "deployment_profile", "tiny") or "tiny")
    if is_production_profile(profile) or deployment != "tiny":
        raise RuntimeError("local engine encryption key reference is required")
    return _validate_engine_encryption_material(_load_or_create_engine_encryption_key())


def engine_encryption_readiness(config: Any, *, remote: bool = False) -> dict[str, Any]:
    """Return privacy-safe durable-encryption readiness for doctor output."""

    if remote:
        return {
            "ready": True,
            "source": "remote_managed",
            "material_exposed": False,
        }
    source = (
        "explicit_reference"
        if getattr(config, "epistemic_graph_encryption_key_ref", None)
        else "local_private_key"
    )
    try:
        material = _resolve_engine_encryption_key(config)
        ready = bool(material)
    except Exception:  # noqa: BLE001 - status is intentionally category-only
        ready = False
    finally:
        material = ""
    return {
        "ready": ready,
        "source": source,
        "material_exposed": False,
    }


def resolve_engine_auth(config: Any) -> str:
    """Resolve the mandatory engine HMAC authentication secret.

    CONCEPT:AU-OS.identity.authenticated-identity-enforcement — current-only:

    * ``GRAPH_SERVICE_AUTH_SECRET`` set → use it verbatim.
    * Otherwise → the persisted per-install secret
      (:func:`_load_or_create_engine_secret`).

    There is no unauthenticated or legacy-envelope branch. Development uses the
    same authenticated current protocol as every served deployment.
    """
    configured = str(getattr(config, "graph_service_auth_secret", "") or "").strip()
    return configured or _load_or_create_engine_secret()


def ensure_local_engine() -> Any | None:
    """Scan-or-spawn a coupled embedded engine for an embedded/tiny deployment.

    CONCEPT:AU-OS.deployment.embedded-auto-provision — embedded auto-provision. For a server that has no remote
    engine configured (the resolved endpoint is local per
    :func:`shard_topology.is_local_endpoint`) this provisions the ONE engine
    authority as a lifecycle-COUPLED child via the EXISTING scan-or-spawn
    machinery — instantiating a :class:`GraphComputeEngine` with ``coupled=True``,
    whose ``__init__`` runs the lock-guarded (``engine_lock.engine_spawn_guard``)
    scan-or-spawn (the resolver's autostart leg, CONCEPT:AU-OS.deployment.engine-resolver-auto-provision). The explicit
    ``coupled=True`` is the true single-process embedded case: the engine dies
    with this process (vs the resolver's DEFAULT detached+supervised engine that
    other entrypoints share). Host-reuse via ``host_lock``/``engine_lock`` means
    co-located servers share the ONE engine; this adds no new locking.

    Active only for the packaged-local path (``GRAPH_SERVICE_ENDPOINTS`` unset)
    and a no-op for every explicit connect-only topology.
    Returns the engine handle (also used to key teardown) or ``None`` when it
    did not provision anything. Best-effort: never raises.
    """
    from agent_utilities.core.config import AgentConfig

    from .shard_topology import is_local_endpoint, resolve_endpoints

    try:
        config = AgentConfig()
        if config.graph_service_endpoints:
            return None
        from .engine_resolver import setting_autostart

        if not setting_autostart(config):
            return None
        endpoints = resolve_endpoints(config)
        # Packaged-local only: a single local endpoint. Explicit coordinator
        # topology returned above before this path can provision anything.
        if len(endpoints) != 1 or not is_local_endpoint(endpoints[0]):
            return None
        # Triggers the resolver's scan-or-spawn; coupled=True selects the
        # lifetime-bound child (vs the default detached engine).
        return GraphComputeEngine.get_or_create(coupled=True)
    except Exception as exc:  # noqa: BLE001 — provisioning is best-effort
        logger.warning(
            "ensure_local_engine could not provision embedded engine (%s)",
            type(exc).__name__,
        )
        return None


def teardown_local_engine(_engine: Any | None = None) -> None:
    """Tear down any coupled embedded engine this process spawned.

    Pairs with :func:`ensure_local_engine` for server shutdown. The coupled
    children are also reaped by the atexit/SIGTERM handler, so this is the
    explicit early-teardown hook for a graceful server stop. Idempotent.
    """
    _terminate_coupled_children()


def connect_external_read_transport(
    *,
    endpoint: str,
    graph_name: str,
    auth_secret: str,
    verified_context: Mapping[str, Any],
    tls_profile: str | None = None,
    tls_profile_ref: str | None = None,
    tls_server_name: str | None = None,
) -> Any:
    """Open a bounded, non-authoritative transport to a registered foreign engine.

    Operational graph access uses the one process-owned :class:`GraphComputeEngine`
    transport. External epistemic-graph instances are read-only connector sources,
    but their sockets still belong at this transport bootstrap boundary so feature
    modules cannot create unmanaged engine clients or bypass the shared TLS policy.
    The caller owns and must close the returned transport.
    """

    from epistemic_graph.client import SyncEpistemicGraphClient

    target = str(endpoint or "").strip()
    if not target:
        raise ValueError("external engine endpoint is missing")
    connect: dict[str, Any] = {
        "auth_secret": auth_secret,
        "graph_name": str(graph_name or "default"),
        "verified_context": dict(verified_context),
    }
    if target.startswith(("tcp://", "tls://")):
        from .engine_transport import (
            engine_client_transport_kwargs,
            native_endpoint_address,
        )

        connect["tcp_addr"] = native_endpoint_address(target)[0]
        connect.update(
            engine_client_transport_kwargs(
                target,
                profile_name=tls_profile,
                profile_ref=tls_profile_ref,
                server_hostname=tls_server_name,
            )
        )
    elif target.startswith("unix://"):
        connect["socket_path"] = target[7:]
    else:
        raise ValueError("external engine endpoint must use tls://, tcp://, or unix://")
    return SyncEpistemicGraphClient.connect(**connect)


class GraphComputeEngine:
    """Graph compute engine backed by the epistemic-graph Tokio service.

    All graph operations route through the Tokio service layer via UDS/TCP
    (length-prefixed MessagePack, HMAC-authenticated). There is **no PyO3 /
    in-process mode**. With no explicit coordinator topology the resolver
    shares or starts the packaged local service; configured
    ``GRAPH_SERVICE_ENDPOINTS`` is connect-only and must already be serving.
    """

    # The native client accepts the complete MessagePack property domain through
    # ``add_node``/``add_edge``.  Declare that contract explicitly so governance
    # proxies and ``IntelligenceGraphEngine`` retain the typed write path instead
    # of compiling mutations into the deliberately bounded Cypher surface.
    typed_mutation_support = "native"
    cypher_support = "native"

    _PROCESS_ENGINE: "GraphComputeEngine | None" = None
    _PROCESS_ENGINE_LOCK = threading.RLock()

    @classmethod
    def get_or_create(
        cls, graph_name: str | None = None, **kwargs: Any
    ) -> "GraphComputeEngine":
        """Acquire the one process transport, optionally as a graph-scoped view."""
        root = cls._PROCESS_ENGINE
        if root is None:
            with cls._PROCESS_ENGINE_LOCK:
                root = cls._PROCESS_ENGINE
                if root is None:
                    root = cls(graph_name=graph_name, **kwargs)
                    cls._PROCESS_ENGINE = root
        if graph_name and graph_name != root.graph_name:
            return root.for_graph(graph_name)
        return root

    @classmethod
    def get_active(cls) -> "GraphComputeEngine | None":
        return cls._PROCESS_ENGINE

    @property
    def client(self) -> Any:
        """Non-owning generated-client view over the process graph transport.

        The view is GraphSession-routed and its ``close`` is a no-op, allowing
        existing namespace consumers to use a context manager without stopping
        the process authority's event loop.
        """
        return self._client

    @property
    def async_client(self) -> Any:
        """Awaitable, non-owning facade over :attr:`client`.

        This is for async consumers that historically opened a second native
        client. Calls are scheduled through the process sync client while
        preserving the current GraphSession ContextVar.
        """
        return _AsyncProcessClientView(self._client)

    def for_graph(self, graph_name: str) -> "GraphComputeEngine":
        """Return a no-connection named-graph view over this process transport."""
        target = str(graph_name or self.graph_name)
        if target == self.graph_name:
            return self
        from .engine_breaker import unwrap_client, wrap_client_with_breaker

        raw = unwrap_client(self._client)
        # A view is routing metadata only. In particular, constructing one for
        # a request-supplied graph must not perform an implicit privileged
        # TenantsCreate before GraphSession scope/tenant checks run. Graph
        # provisioning is an explicit kg:admin operation and is never performed
        # by transport construction.
        view = object.__new__(type(self))
        view.__dict__ = self.__dict__.copy()
        view.graph_name = target
        view.graph = {}
        view._process_root = getattr(self, "_process_root", self)
        scoped = _sync_client_view(raw, graph=target)
        breaker = getattr(self._client, "_breaker", None)
        view._client = (
            wrap_client_with_breaker(scoped, breaker) if breaker is not None else scoped
        )
        return view

    def __init__(self, graph_name: str | None = None, **kwargs: Any) -> None:
        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.core.config import AgentConfig

        from .engine_resolver import resolve_engine
        from .session import graph_session_required
        from .shard_topology import (
            record_shard_connect,
            resolve_endpoints,
            resolve_routing_graph,
        )

        if graph_session_required() and self._PROCESS_ENGINE is not None:
            raise RuntimeError(
                "A process graph transport already exists; use "
                "GraphComputeEngine.get_or_create()/for_graph()"
            )

        self.graph: dict[str, Any] = {}
        self._process_root = self
        # SyncEpistemicGraphClient wrapped in a BreakerClientProxy
        # — attribute-transparent; raw client at
        # ``self._client.__wrapped__``. (CONCEPT:AU-OS.observability.no-op-without-metrics)
        self._client: Any
        self._mode: str = "service"

        if "endpoint" in kwargs:
            raise TypeError(
                "per-instance engine endpoints are not supported; configure "
                "GRAPH_SERVICE_ENDPOINTS"
            )

        config = AgentConfig()
        endpoints = resolve_endpoints(config)
        sharded = len(endpoints) > 1
        if sharded:
            # Select the tenant-scoped named graph submitted to the authenticated
            # placement authority. Endpoint selection happens per request.
            graph_name = resolve_routing_graph(graph_name, config)
        elif graph_name is None:
            # Per-tenant named-graph isolation must NOT require multiple shards
            # (CONCEPT:AU-KG.compute.data-is-private-its): with enforcement on, route the ambient tenant to
            # its own named graph even on a single endpoint. With enforcement off this retains the
            # configured default-graph behavior.
            from .company_brain_runtime import brain_enforcement_enabled

            if brain_enforcement_enabled():
                graph_name = resolve_routing_graph(None, config)
            else:
                graph_name = config.kg_default_graph
        # Retained so downstream consumers (e.g. the delta-ingestion manifest)
        # can key state by tenant graph. (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
        self.graph_name = graph_name

        # Since GraphComputeEngine is synchronous and often long-lived, its
        # process transport connects to one configured coordinator. Its one
        # process-owned transport natively pipelines request ids; it must not be
        # replaced by authority-caching connection pools.
        # ONE engine resolver (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision): remote → share-running-local →
        # autostart-shared-supervised. The resolver owns coordinator selection,
        # the local-vs-remote classification, the share-probe, the auth secret,
        # and whether autostart is permitted — so this chokepoint carries no
        # inline autostart sequence or per-instance topology override.
        resolved = resolve_engine(config, graph_name)
        endpoint = resolved.endpoint
        self.endpoint = endpoint
        auth_secret = resolved.auth_secret
        idle_shutdown_secs = resolved.idle_shutdown_secs
        connect_kwargs = {
            "auth_secret": auth_secret,
            "graph_name": graph_name,
            "verified_context": _transport_only_verified_context(),
        }
        if endpoint.startswith(("tcp://", "tls://")):
            from .engine_transport import (
                engine_client_transport_kwargs,
                native_endpoint_address,
            )

            connect_kwargs["tcp_addr"] = native_endpoint_address(endpoint)[0]
            connect_kwargs.update(
                engine_client_transport_kwargs(endpoint, config=config)
            )
        elif endpoint.startswith("unix://"):
            connect_kwargs["socket_path"] = endpoint[7:]
        else:
            connect_kwargs["socket_path"] = endpoint

        # Circuit breaker — ONE shared breaker per endpoint (CONCEPT:AU-OS.observability.no-op-without-metrics).
        # When the engine is down, N consecutive connect/timeout failures open
        # the circuit and every caller fails fast with the typed
        # EngineCircuitOpenError (a ConnectionError) instead of hammering a
        # dead socket; a half-open probe after the cooldown heals it.
        from agent_utilities.knowledge_graph.core.engine_breaker import (
            get_breaker,
            wrap_client_with_breaker,
        )

        breaker = get_breaker(endpoint)
        breaker.before_call()  # fast-fail BEFORE attempting a connect when open

        # The resolver already gated this to a LOCAL endpoint the process may
        # spawn (never a remote/sharded shard — that stays fail-loud below).
        autostart_allowed = resolved.autostart_allowed

        try:
            self._client = SyncEpistemicGraphClient.connect(**connect_kwargs)
        except Exception as initial_e:
            if isinstance(initial_e, OSError | EOFError):
                breaker.record_failure()
            if autostart_allowed:
                import subprocess
                import sys
                import time
                from pathlib import Path

                from .engine_lock import engine_spawn_guard

                sock = connect_kwargs.get("socket_path")
                # The single-instance guard is transport-keyed. Windows uses
                # loopback TCP instead of AF_UNIX, so retain the endpoint as
                # the guard key when there is no socket path.
                spawn_key = sock or endpoint
                try:
                    # Single-instance spawn (CONCEPT:EG-KG.storage.nonblocking-checkpoint / OS-5.9): serialize all
                    # autostart spawners for this socket behind a flock and
                    # double-check connectivity before spawning. Without this, two
                    # connects racing — or a client spawning while a displaced engine
                    # still holds the socket — produce a split-brain (two engines on
                    # one socket, clobbering the same --persist-dir). The guard is
                    # held across spawn+wait so a concurrent spawner finds the engine
                    # already up on re-check instead of spawning a second one.
                    with engine_spawn_guard(spawn_key) as owns_spawn_guard:
                        try:
                            # Double check: a peer may have brought it up while we
                            # waited for the guard.
                            self._client = SyncEpistemicGraphClient.connect(
                                **connect_kwargs
                            )
                        except Exception:  # noqa: BLE001 - still down; we spawn
                            if not owns_spawn_guard:
                                # A peer still owns the spawn right. Never start a
                                # competing writer after the bounded guard wait.
                                raise ConnectionError(
                                    "Timed out waiting for the local epistemic-graph "
                                    "startup owner; no competing engine was spawned."
                                ) from initial_e
                            # Detached + supervised (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision): the engine
                            # survives this spawner so OTHER entrypoints on the
                            # host share it (NOT coupled=pdeathsig), and it
                            # self-terminates ``idle_shutdown_secs`` after its last
                            # client disconnects (0 = persistent, never auto-stop).
                            self._client = self._autostart_engine(
                                connect_kwargs,
                                sock,
                                auth_secret,
                                config,
                                subprocess,
                                sys,
                                time,
                                Path,
                                coupled=bool(kwargs.get("coupled", False)),
                                idle_shutdown_secs=idle_shutdown_secs,
                            )
                except ConnectionError:
                    record_shard_connect(endpoint, False)
                    raise
                except Exception as retry_e:
                    if isinstance(retry_e, OSError | EOFError):
                        breaker.record_failure()
                    record_shard_connect(endpoint, False)
                    raise ConnectionError(
                        "Cannot connect to epistemic-graph service after auto-start "
                        f"({type(retry_e).__name__}); ensure the engine daemon is running."
                    ) from retry_e
            elif sharded:
                # Multiple configured contacts are remote and never auto-started.
                # Placement cannot be guessed when the authority is unreachable.
                record_shard_connect(endpoint, False)
                raise ConnectionError(
                    f"Configured engine shard {endpoint!r} for graph {graph_name!r} "
                    f"is unreachable ({type(initial_e).__name__}): {initial_e}. "
                    "Repair the coordinator topology — start that shard's "
                    "epistemic-graph-server, or remove it from "
                    "GRAPH_SERVICE_ENDPOINTS (moving a graph between shards "
                    "requires a manual snapshot export/import). Autostart applies "
                    "only to the local unix:// endpoint; remote contacts/shards "
                    "are never auto-started."
                ) from initial_e
            else:
                record_shard_connect(endpoint, False)
                raise ConnectionError(
                    "Cannot connect to epistemic-graph service "
                    f"({type(initial_e).__name__}); ensure the engine daemon is "
                    "running or enable local autostart in AgentConfig."
                ) from initial_e

        # A packaged local engine can authoritatively route a tenant partition
        # before that partition's graph has been materialized.  Establish the
        # process session's one configured graph while the raw client is still
        # available, regardless of whether this process connected to an already
        # running local engine or started a fresh one.  Remote/sharded engines
        # retain lifecycle authority and are never provisioned here.
        local_graph_name = str(graph_name or "__commons__")
        if autostart_allowed and local_graph_name != "__commons__":
            from .session import current_session

            session = current_session()
            if session is None:
                raise RuntimeError(
                    "local engine graph readiness requires verified process authority"
                )
            try:
                self._ensure_local_session_graph(
                    self._client,
                    local_graph_name,
                    session,
                )
            except Exception:
                with contextlib.suppress(Exception):
                    self._client.close()
                raise

        # Connected: close/reset the breaker and guard every subsequent call
        # with it. The proxy is attribute-transparent, and the raw client
        # stays reachable via ``self._client.__wrapped__``. (CONCEPT:AU-OS.observability.no-op-without-metrics)
        breaker.record_success()
        record_shard_connect(endpoint, True)

        # Keep the connected transport private and expose only a routed view.
        # This adds no socket/thread: it reconstructs namespace wrappers around
        # the same async client and event loop so concurrent GraphSessions can
        # safely target different tenant graphs.
        transport_client = self._client
        transport_client._au_route_config = config
        transport_client._au_route_endpoints = tuple(endpoints)
        transport_client._au_route_endpoint = endpoint

        self._client = wrap_client_with_breaker(
            _sync_client_view(transport_client), breaker
        )

        logger.info("Connected to configured epistemic-graph service")

        # Bridging local events to the rust service when kafka isn't running
        if (
            setting("KAFKA_BOOTSTRAP_SERVERS") is None
            or setting("KAFKA_BOOTSTRAP_SERVERS") == ""
        ):
            self._start_event_bridge()

        with self._PROCESS_ENGINE_LOCK:
            active = self._PROCESS_ENGINE
            if active is None:
                type(self)._PROCESS_ENGINE = self
            elif active is not self and graph_session_required():
                raise RuntimeError("Concurrent duplicate graph transport rejected")

    def _autostart_engine(
        self,
        connect_kwargs: dict[str, Any],
        sock: str | None,
        auth_secret: str,
        config: Any,
        subprocess: Any,
        sys: Any,
        time: Any,
        Path: Any,
        coupled: bool = True,
        idle_shutdown_secs: int = 0,
    ) -> Any:
        """Spawn the local epistemic-graph engine and return a connected client.

        Called ONLY while holding the per-socket spawn guard (see
        :func:`engine_lock.engine_spawn_guard`) and only after a double-checked
        connect confirmed the engine is still down — so this is the sole spawner
        for ``sock``. Mirrors the prior inline autostart: durable ``--persist-dir``
        and the same auth secret the client uses (CONCEPT:AU-OS.identity.authenticated-identity-enforcement).

        ``coupled`` (CONCEPT:AU-OS.deployment.embedded-auto-provision — embedded auto-provision) selects the child
        lifecycle. ``coupled=True`` (the ``ensure_local_engine`` / true
        single-process case) lifecycle-couples the engine to its spawner — it
        gets a parent-death signal (Linux ``prctl``) plus an atexit/SIGTERM/SIGINT
        teardown so it dies with the process that needs it. ``coupled=False`` (the
        resolver's autostart leg, CONCEPT:AU-OS.deployment.engine-resolver-auto-provision, and the graph-os-host /
        enterprise shard) detaches the child into its own session so it OUTLIVES
        the launcher and OTHER entrypoints on the host share it.
        ``KG_ENGINE_DETACHED=1`` forces detached for the autostart path too.

        ``idle_shutdown_secs`` (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision — supervised, reference-counted)
        is passed to a detached engine as ``--idle-shutdown-secs <secs>`` so it
        self-terminates that many seconds after its LAST client disconnects
        (robust to client crashes). ``0`` = persistent: NO flag passed, the engine
        runs forever like a local service. A coupled engine ignores this (its
        spawner already bounds its lifetime). The packaged current engine binary
        is the sole supported launch contract.
        """
        if setting("KG_ENGINE_DETACHED", "") == "1":
            coupled = False
        from epistemic_graph.client import SyncEpistemicGraphClient

        logger.info(
            "epistemic-graph Tokio service not running. Auto-starting daemon (single-instance guard held)..."
        )
        self._local_bootstrap_identity = None
        # The maturin wheel installs the binary next to the interpreter; on Windows
        # it carries a `.exe` suffix (Scripts/epistemic-graph-server.exe).
        _server_exe = (
            "epistemic-graph-server.exe"
            if os.name == "nt"
            else "epistemic-graph-server"
        )
        server_path = str(Path(sys.executable).parent / _server_exe)
        cmd = [server_path]
        if sock:
            cmd += ["--socket-path", str(sock)]
        elif connect_kwargs.get("tcp_addr"):
            # Windows' zero-config transport is loopback TCP. Passing the
            # address explicitly also makes an operator-selected local
            # loopback port deterministic on every platform.
            cmd += ["--tcp-addr", str(connect_kwargs["tcp_addr"])]
        # Durable by default (CONCEPT:EG-KG.storage.nonblocking-checkpoint / OS-5.9): snapshot the graphs to disk
        # so an auto-spawned engine warm-restarts from the last checkpoint instead
        # of starting empty. pggraph stays the durable system-of-record; this is
        # the fast local cache.
        persist_dir = setting("GRAPH_SERVICE_PERSIST_DIR")
        if persist_dir is None:
            try:
                from agent_utilities.core.paths import data_dir

                persist_dir = str(data_dir() / "graph_snapshots")
            except Exception:
                persist_dir = None
        if persist_dir:
            cmd += ["--persist-dir", persist_dir]
        # Reference-counted supervision (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision): a DETACHED engine that
        # outlives its spawner needs a self-shutdown so a crashed/exited fleet of
        # clients doesn't leave it running forever. >0 → arm idle shutdown;
        # 0 → persistent (omit the flag, engine runs forever like a service). A
        # coupled engine is already lifetime-bound to its spawner, so skip it.
        if not coupled and idle_shutdown_secs > 0:
            cmd += ["--idle-shutdown-secs", str(idle_shutdown_secs)]
        # Engine auth (CONCEPT:AU-OS.identity.authenticated-identity-enforcement):
        # the spawned engine gets the SAME mandatory secret this client uses.
        child_env = _engine_child_environment()
        # A packaged-local engine is a private child of this verified process.
        # Project the externally verified session's tenant/audience/policy into
        # the engine before it accepts a request, and provide a private signer
        # only for the empty-store System identity bootstrap.  An explicit
        # signer registry is fail-closed: it must already contain this subject.
        from agent_utilities.knowledge_graph.core.session import current_session

        bootstrap_session = current_session()
        if bootstrap_session is not None:
            bootstrap_context = bootstrap_session.engine_verified_context()
            authority_projection = {
                "EPISTEMIC_GRAPH_AUDIENCE": str(bootstrap_context["audience"]),
                "EPISTEMIC_GRAPH_TENANT": str(bootstrap_context["tenant"]),
                "EPISTEMIC_GRAPH_POLICY_VERSION": str(
                    bootstrap_context["policy_version"]
                ),
            }
            for environment_name, expected in authority_projection.items():
                configured = str(child_env.get(environment_name, "") or "").strip()
                if configured and configured != expected:
                    raise RuntimeError(
                        "local engine authority policy does not match the verified process"
                    )
                child_env[environment_name] = expected

            actor_id = str(bootstrap_context["agent_id"])
            raw_registry = str(
                child_env.get("EPISTEMIC_GRAPH_SIGNER_KEYS_JSON", "") or ""
            ).strip()
            if raw_registry:
                try:
                    registry = json.loads(raw_registry)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        "local engine signer registry is invalid"
                    ) from exc
                if not isinstance(registry, dict):
                    raise RuntimeError("local engine signer registry is invalid")
                signer_key = str(registry.get(actor_id, "") or "")
                if len(signer_key.encode("utf-8")) < 32:
                    raise RuntimeError(
                        "verified process identity is absent from the engine signer registry"
                    )
            else:
                signer_key = _load_or_create_engine_bootstrap_signer_key()
                child_env["EPISTEMIC_GRAPH_SIGNER_KEYS_JSON"] = json.dumps(
                    {actor_id: signer_key}, separators=(",", ":")
                )
            self._local_bootstrap_identity = (
                actor_id,
                signer_key,
                bootstrap_context,
            )
        child_env.update(
            {
                "EPISTEMIC_GRAPH_LAZY_STARTUP": "1",
                "EPISTEMIC_GRAPH_MAX_RESIDENT_GRAPHS": str(
                    config.epistemic_graph_max_resident_graphs
                ),
                "EPISTEMIC_GRAPH_LAZY_OPEN_PAGE_SIZE": str(
                    config.epistemic_graph_lazy_open_page_size
                ),
                "EPISTEMIC_GRAPH_MAX_NODES_PER_GRAPH": str(
                    config.epistemic_graph_max_nodes_per_graph
                ),
                "EPISTEMIC_GRAPH_MAX_REQUEST_BYTES": str(
                    getattr(
                        config, "epistemic_graph_max_request_bytes", 64 * 1024 * 1024
                    )
                ),
                "EPISTEMIC_GRAPH_MAX_RESPONSE_BYTES": str(
                    getattr(
                        config, "epistemic_graph_max_response_bytes", 64 * 1024 * 1024
                    )
                ),
                "EPISTEMIC_GRAPH_MAX_MSGPACK_ITEMS": str(
                    getattr(config, "epistemic_graph_max_msgpack_items", 1_000_000)
                ),
                "EPISTEMIC_GRAPH_CONNECTION_IO_TIMEOUT_SECS": str(
                    getattr(config, "epistemic_graph_connection_io_timeout_secs", 120)
                ),
                "EPISTEMIC_GRAPH_TLS_HANDSHAKE_TIMEOUT_SECS": str(
                    getattr(config, "epistemic_graph_tls_handshake_timeout_secs", 10)
                ),
                "EPISTEMIC_GRAPH_AST_MAX_FILES": str(
                    getattr(config, "epistemic_graph_ast_max_files", 4_096)
                ),
                "EPISTEMIC_GRAPH_AST_MAX_SOURCE_BYTES": str(
                    getattr(
                        config, "epistemic_graph_ast_max_source_bytes", 4 * 1024 * 1024
                    )
                ),
                "EPISTEMIC_GRAPH_AST_MAX_TOTAL_BYTES": str(
                    getattr(
                        config, "epistemic_graph_ast_max_total_bytes", 32 * 1024 * 1024
                    )
                ),
                "EPISTEMIC_GRAPH_MODALITY_MAX_BUNDLE_BYTES": str(
                    getattr(
                        config,
                        "epistemic_graph_modality_max_bundle_bytes",
                        4 * 1024 * 1024,
                    )
                ),
                "EPISTEMIC_GRAPH_MODALITY_MAX_SOURCE_BYTES": str(
                    getattr(
                        config,
                        "epistemic_graph_modality_max_source_bytes",
                        16 * 1024 * 1024,
                    )
                ),
                "EPISTEMIC_GRAPH_SQLITE_MAX_BYTES": str(
                    getattr(
                        config, "epistemic_graph_sqlite_max_bytes", 256 * 1024 * 1024
                    )
                ),
                "EPISTEMIC_GRAPH_SQLITE_MAX_ROWS": str(
                    getattr(config, "epistemic_graph_sqlite_max_rows", 1_000_000)
                ),
            }
        )
        runtime_roots = (
            (
                "EPISTEMIC_GRAPH_SQLITE_TRANSFER_ROOT",
                getattr(config, "epistemic_graph_sqlite_transfer_root_ref", None),
            ),
            (
                "EPISTEMIC_GRAPH_BACKUP_ROOT",
                getattr(config, "epistemic_graph_backup_root_ref", None),
            ),
        )
        for environment_name, reference in runtime_roots:
            if reference:
                child_env[environment_name] = _resolve_engine_path_ref(str(reference))
        # Encryption material exists only in this private child environment.
        # Ambient raw/ref variables were removed by ``_engine_child_environment``;
        # AgentConfig persists only the reference, and the Rust process receives
        # the validated value immediately before spawn.
        child_env["EPISTEMIC_GRAPH_ENCRYPTION_KEY"] = _resolve_engine_encryption_key(
            config
        )
        child_env["GRAPH_SERVICE_AUTH_SECRET"] = auth_secret
        if coupled:
            # Embedded/tiny path: the engine's lifetime is tied to ours. Do NOT
            # start a new session (that would detach it); instead arm the
            # parent-death signal in the child and track it for atexit/SIGTERM
            # teardown so the embedded engine never outlives its spawner.
            #
            # `preexec_fn` is POSIX-only — on Windows subprocess.Popen REFUSES it
            # (ValueError). The parent-death signal is Linux-only anyway, so on
            # non-POSIX we pass no preexec_fn and rely entirely on the
            # atexit/SIGTERM/SIGINT teardown registered by _install_coupled_handlers
            # (the documented cross-platform coupling mechanism).
            coupled_kwargs: dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "env": child_env,
            }
            if os.name == "posix":
                coupled_kwargs["start_new_session"] = False
                coupled_kwargs["preexec_fn"] = _set_pdeathsig
            try:
                child = subprocess.Popen(cmd, **coupled_kwargs)  # nosec B603
            finally:
                child_env.pop("EPISTEMIC_GRAPH_ENCRYPTION_KEY", None)
                coupled_kwargs.pop("env", None)
            _coupled_children.append(child)
            _install_coupled_handlers()
        else:
            # Explicit long-lived daemon (graph-os-host / enterprise shard):
            # detach so it survives this launcher. On POSIX that's a new session;
            # on Windows there is no setsid — DETACHED_PROCESS + a new process group
            # detaches the child from the launcher's console/job instead.
            detach_kwargs: dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "env": child_env,
            }
            if os.name == "posix":
                detach_kwargs["start_new_session"] = True
            else:  # Windows
                detach_kwargs["creationflags"] = (
                    subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
                    | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                )
            try:
                child = subprocess.Popen(cmd, **detach_kwargs)  # nosec B603
            finally:
                child_env.pop("EPISTEMIC_GRAPH_ENCRYPTION_KEY", None)
                detach_kwargs.pop("env", None)

        # Cold startup is readiness-driven, not a fixed one-second guess. The
        # full engine can legitimately spend several seconds opening a durable
        # store on a resource-constrained host. Retain the child handle so an
        # early exit is distinguishable from a listener that is merely not ready,
        # and tear down only the child we just spawned on timeout so it cannot
        # become an unreachable orphan holding the persistence lock.
        deadline = time.monotonic() + _ENGINE_STARTUP_TIMEOUT_SECS
        last_error: Exception | None = None
        while True:
            status = child.poll()
            if status is not None:
                if coupled:
                    with contextlib.suppress(ValueError):
                        _coupled_children.remove(child)
                raise ConnectionError(
                    "The local epistemic-graph process exited during startup "
                    f"(status {status})."
                )
            try:
                connected = SyncEpistemicGraphClient.connect(**connect_kwargs)
                self._bootstrap_autostarted_engine_identity(
                    connected,
                    connect_kwargs,
                )
                return connected
            except Exception as exc:  # noqa: BLE001 - bounded readiness probe
                last_error = exc
            if time.monotonic() >= deadline:
                with contextlib.suppress(Exception):
                    child.terminate()
                with contextlib.suppress(Exception):
                    child.wait(timeout=3.0)
                if child.poll() is None:
                    with contextlib.suppress(Exception):
                        child.kill()
                if coupled:
                    with contextlib.suppress(ValueError):
                        _coupled_children.remove(child)
                raise ConnectionError(
                    "The local epistemic-graph process did not become ready "
                    f"within {_ENGINE_STARTUP_TIMEOUT_SECS:g} seconds."
                ) from last_error
            time.sleep(_ENGINE_STARTUP_POLL_SECS)

    @staticmethod
    def _ensure_local_session_graph(
        client: Any,
        graph_name: str,
        session: Any,
    ) -> None:
        """Materialize the verified process graph on one packaged local engine.

        Placement routing proves authority but intentionally does not create a
        graph.  This local-only bootstrap seam closes that lifecycle gap without
        allowing request-scoped graph views or configured remote coordinators to
        provision graphs implicitly.
        """
        if graph_name == "__commons__":
            return
        session.require_scope("kg:admin")
        verified_context = session.engine_verified_context()
        from agent_utilities.knowledge_graph.core.placement_catalog import (
            split_tenant_key,
        )

        tenant, sub_key = split_tenant_key(graph_name)

        def _listed_names() -> set[str]:
            entries = client.tenants.list() or []
            return {
                str(entry.get("name") if isinstance(entry, Mapping) else entry)
                for entry in entries
            }

        with client.use_verified_context(verified_context):
            client.placement.route(tenant, sub_key, client_epoch=0)
            if graph_name in _listed_names():
                return
            try:
                client.tenants.create(graph_name, "Agent")
            except Exception:
                # Another authorized local process may win the create race.  A
                # fresh authoritative list is the only accepted reconciliation;
                # otherwise preserve the original fail-closed exception.
                if graph_name in _listed_names():
                    return
                raise

    def _bootstrap_autostarted_engine_identity(
        self,
        client: Any,
        connect_kwargs: Mapping[str, Any],
    ) -> None:
        """Initialize one fresh packaged engine from verified process authority.

        Existing engines are probed first.  If the process identity already has
        administrative placement access, no bootstrap request is sent.  On an
        empty identity store, the engine accepts exactly one signer-backed
        ``security:bootstrap`` System registration on ``__commons__``; any other
        initialized/unauthorized state remains fail-closed.
        """
        prepared = getattr(self, "_local_bootstrap_identity", None)
        if prepared is None or not hasattr(client, "placement"):
            return
        actor_id, signer_key, verified_context = prepared
        from agent_utilities.knowledge_graph.core.placement_catalog import (
            split_tenant_key,
        )

        graph_name = str(connect_kwargs.get("graph_name") or "__commons__")
        tenant, sub_key = split_tenant_key(graph_name)
        try:
            with client.use_verified_context(verified_context):
                client.placement.route(tenant, sub_key, client_epoch=0)
            return
        except Exception:  # noqa: BLE001 - empty store is established below
            pass

        bootstrap_context = dict(verified_context)
        bootstrap_context.update(
            {
                "principal": actor_id,
                "agent_id": actor_id,
                "roles": [],
                "scopes": ["security:bootstrap"],
                "delegation": [],
            }
        )
        from epistemic_graph.client import SyncEpistemicGraphClient

        bootstrap_connect = dict(connect_kwargs)
        bootstrap_connect["graph_name"] = "__commons__"
        bootstrap_connect["verified_context"] = bootstrap_context
        bootstrap_client = SyncEpistemicGraphClient.connect(**bootstrap_connect)
        try:
            bootstrap_client.consensus.bootstrap_system_identity(
                agent_id=actor_id,
                signer_id=actor_id,
                signer_key=signer_key,
            )
        finally:
            bootstrap_client.close()

        # Do not declare the engine ready until the actual process authority can
        # read its authoritative route under the newly durable policy state.
        with client.use_verified_context(verified_context):
            client.placement.route(tenant, sub_key, client_epoch=0)

    def _start_event_bridge(self) -> None:
        """Starts a background bridge to forward local EventBus events to the Rust service."""
        import asyncio
        import threading

        from agent_utilities.knowledge_graph.core.event_backend import get_event_backend

        def bridge_worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            eb = get_event_backend()

            async def handle_mutation(topic: str, payload: dict) -> None:
                if "event_type" in payload and "query" in payload:
                    try:
                        if self._client:
                            self._client.apply_mutation(
                                payload["event_type"], payload["query"]
                            )
                    except Exception as exc:
                        logger.error(
                            "Failed to forward mutation to epistemic-graph (%s)",
                            type(exc).__name__,
                        )

            async def run_subscriber() -> None:
                await eb.subscribe("kg.mutations", "epistemic-bridge", handle_mutation)
                # Keep loop alive to process events
                while True:
                    await asyncio.sleep(3600)

            try:
                loop.run_until_complete(run_subscriber())
            except Exception as e:
                logger.error("Event bridge worker failed (%s)", type(e).__name__)

        t = threading.Thread(
            target=bridge_worker, daemon=True, name="EventBridgeWorker"
        )
        t.start()
        logger.info("Started Local-First EventBus bridge to epistemic-graph")

    # ── Node CRUD ────────────────────────────────────────────────────────

    def add_node(self, node_id: str, properties: Any = None, **kwargs: Any) -> None:
        """Add a node with properties to the graph.

        Supports both explicit dict and NX-style kwargs::

            engine.add_node("n1", {"node_type": "Agent"})
            engine.add_node("n1", node_type="Agent", name="foo")
        """

        def clean_props(d: Mapping[str, Any]) -> dict[str, Any]:
            import datetime

            from pydantic import BaseModel

            def serialize(val: Any) -> Any:
                if hasattr(val, "model_dump"):
                    try:
                        return val.model_dump(mode="json")
                    except Exception:
                        pass
                if isinstance(val, BaseModel):
                    return val.model_dump(mode="json")
                if isinstance(val, dict):
                    return {k: serialize(v) for k, v in val.items()}
                if isinstance(val, list | tuple | set):
                    return [serialize(v) for v in val]
                if isinstance(val, datetime.datetime):
                    return val.isoformat()
                return val

            return {k: serialize(v) for k, v in d.items()}

        props = dict(properties or {})
        props.update(kwargs)
        if "type" in props:
            raise ValueError(
                "node property 'type' is not supported; use canonical 'node_type'"
            )
        declared_id = props.get("id")
        if declared_id is not None and str(declared_id) != str(node_id):
            raise ValueError("node property 'id' must match the native node identity")
        # Native Cypher property predicates intentionally use one stored `id`
        # field. Stamping it on every typed write removes the Python-side
        # virtual-id interpreter and keeps point predicates inside the engine.
        props["id"] = str(node_id)
        props = clean_props(props)
        self._client.nodes.add(node_id, props)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        properties: Any = None,
        **kwargs: Any,
    ) -> None:
        """Add a directed edge between two nodes with properties.

        Supports both explicit dict and NX-style kwargs::

            engine.add_edge("a", "b", {"relationship": "DEPENDS_ON"})
            engine.add_edge("a", "b", relationship="DEPENDS_ON")
        """

        def clean_props(d: dict[str, Any]) -> dict[str, Any]:
            import datetime

            from pydantic import BaseModel

            def serialize(val: Any) -> Any:
                if hasattr(val, "model_dump"):
                    try:
                        return val.model_dump(mode="json")
                    except Exception:
                        pass
                if isinstance(val, BaseModel):
                    return val.model_dump(mode="json")
                if isinstance(val, dict):
                    return {k: serialize(v) for k, v in val.items()}
                if isinstance(val, list | tuple | set):
                    return [serialize(v) for v in val]
                if isinstance(val, datetime.datetime):
                    return val.isoformat()
                return val

            return {k: serialize(v) for k, v in d.items()}

        props = dict(properties or {})
        props.update(kwargs)
        aliases = {"type", "rel_type", "relationship_type", "relation"}.intersection(
            props
        )
        if aliases:
            names = ", ".join(sorted(aliases))
            raise ValueError(
                f"edge relationship aliases are not supported ({names}); "
                "use canonical 'relationship'"
            )
        if not props.get("relationship"):
            raise ValueError("edge property 'relationship' is required")
        props = clean_props(props)

        if self.has_edge(source_id, target_id):
            self.remove_edge(source_id, target_id)

        try:
            self._client.edges.add(source_id, target_id, props)
        except Exception:
            # Ensure nodes exist without overwriting their existing properties
            if not self.has_node(source_id):
                self._client.nodes.add(source_id, {})
            if not self.has_node(target_id):
                self._client.nodes.add(target_id, {})
            self._client.edges.add(source_id, target_id, props)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all of its associated edges."""
        self._client.nodes.remove(node_id)

    def remove_edge(self, source_id: str, target_id: str, key: Any = None) -> None:
        """Remove a directed edge between source and target."""
        self._client.edges.remove(source_id, target_id)

    def has_node(self, node_id: str) -> bool:
        """Check if node_id exists in the graph."""
        return self._client.nodes.has(node_id)

    def has_batch(self, node_ids: list[str]) -> dict[str, bool]:
        """Existence of many nodes in ONE round-trip → ``{id: exists}`` (KG-2.147).

        The engine is a separate process behind a MessagePack/UDS socket, so each
        call is a round-trip — N per-element ``has_node`` calls = N round-trips.
        This surfaces the engine client's existing one-round-trip existence op
        (CONCEPT:EG-KG.compute.graph-compute-engine) through the facade so orchestration code (ingestion dedup)
        can check a whole batch natively instead of looping. (Bulk writes already
        flow through :meth:`batch_update`.)
        """
        ids = list(node_ids)
        if not ids:
            return {}
        return dict(self._client.nodes.has_batch(ids))

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if a directed edge exists between source and target."""
        return self._client.edges.has(source_id, target_id)

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        """Atomic engine-side compare-and-set on a node's top-level fields.

        Under the engine's graph write lock: if for every ``(field, expected)``
        in ``conditions`` the node's current top-level property equals
        ``expected`` (a missing field reads as null), merge every
        ``(field, value)`` in ``updates`` into the node and return ``True``;
        otherwise return ``False`` with no mutation. This is the authoritative,
        backend-agnostic primitive for atomic graph-field coordination
        (CONCEPT:AU-KG.compute.user-override-prompt-library). Mirrors ``add_node``: the call goes straight to
        ``self._client.nodes.*``; the SyncEpistemicGraphClient wrapper handles
        the run-in-loop dispatch.
        """
        return bool(self._client.nodes.compare_and_set(node_id, conditions, updates))

    def claim_work_item(self, request: Any) -> Any:
        """Atomically select and lease one WorkItem in the engine."""
        from agent_utilities.protocols.epistemic_operations import (
            ClaimWorkItemRequest,
            ClaimWorkItemResult,
        )

        namespace = getattr(self._client, "work_items", None)
        method = getattr(namespace, "claim", None)
        if not callable(method):
            raise NotImplementedError("connected engine has no work_items.claim")
        try:
            operation = ClaimWorkItemRequest.model_validate(request)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid ClaimWorkItem request") from exc
        return ClaimWorkItemResult.model_validate(
            method(operation.model_dump(mode="json"))
        )

    def renew_work_item_lease(self, request: dict[str, Any]) -> Any:
        """Renew a WorkItem lease under its epoch/fencing token."""
        namespace = getattr(self._client, "work_items", None)
        method = getattr(namespace, "renew", None)
        if not callable(method):
            raise NotImplementedError("connected engine has no work_items.renew")
        return method(
            tenant=str(request.get("tenant") or ""),
            work_item_id=str(request.get("work_item_id") or ""),
            worker_id=str(request.get("worker_ref") or ""),
            lease_epoch=int(request.get("expected_epoch") or 0),
            fencing_token=int(request.get("fencing_token") or 0),
            now_ms=int(float(request.get("now_unix") or 0) * 1000),
            lease_ms=max(1, int(float(request.get("lease_ttl") or 0) * 1000)),
        )

    def commit_work_item_result(self, request: dict[str, Any]) -> Any:
        """Atomically commit a fenced WorkItem result/retry/outbox outcome."""
        namespace = getattr(self._client, "work_items", None)
        method = getattr(namespace, "commit_result", None)
        if not callable(method):
            raise NotImplementedError(
                "connected engine has no work_items.commit_result"
            )
        return method(
            tenant=str(request.get("tenant") or ""),
            work_item_id=str(request.get("work_item_id") or ""),
            worker_id=str(request.get("worker_ref") or ""),
            lease_epoch=int(request.get("expected_epoch") or 0),
            fencing_token=int(request.get("fencing_token") or 0),
            idempotency_key=str(request.get("idempotency_key") or ""),
            outcome=str(request.get("outcome") or ""),
            now_ms=int(float(request.get("now_unix") or 0) * 1000),
            result_ref=request.get("result_ref"),
            error_ref=request.get("error_ref"),
            retryable=bool(request.get("retryable", False)),
        )

    def cancel_work_item(self, request: dict[str, Any]) -> Any:
        """Cancel an unleased WorkItem without manufacturing ownership."""
        namespace = getattr(self._client, "work_items", None)
        method = getattr(namespace, "cancel", None)
        if not callable(method):
            raise NotImplementedError("connected engine has no work_items.cancel")
        return method(
            tenant=str(request.get("tenant") or ""),
            work_item_id=str(request.get("work_item_id") or ""),
            idempotency_key=str(request.get("idempotency_key") or ""),
            reason_ref=request.get("reason_ref"),
            now_ms=int(float(request.get("now_unix") or 0) * 1000),
        )

    def defer_work_item(self, request: dict[str, Any]) -> Any:
        """Release a fenced lease until ``next_retry_at`` without an attempt."""
        namespace = getattr(self._client, "work_items", None)
        method = getattr(namespace, "defer", None)
        if not callable(method):
            raise NotImplementedError("connected engine has no work_items.defer")
        return method(
            tenant=str(request.get("tenant") or ""),
            work_item_id=str(request.get("work_item_id") or ""),
            worker_id=str(request.get("worker_ref") or ""),
            lease_epoch=int(request.get("expected_epoch") or 0),
            fencing_token=int(request.get("fencing_token") or 0),
            idempotency_key=str(request.get("idempotency_key") or ""),
            next_retry_at_ms=int(float(request.get("next_retry_at") or 0) * 1000),
            reason_ref=request.get("reason_ref"),
            now_ms=int(float(request.get("now_unix") or 0) * 1000),
        )

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return self._client.nodes.count()

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return self._client.edges.count()

    # ── Graph Algorithms ─────────────────────────────────────────────────

    def topological_sort(self) -> list[str]:
        """Perform topological sort across the graph.

        Raises:
            ValueError: If the graph contains dependency cycles.
        """
        try:
            return self._client.graph.topological_sort()
        except Exception as e:
            raise ValueError("Graph contains cycles") from e

    def find_cycle(self) -> list[str] | None:
        """Detect and return any cycles found within the graph."""
        return self._client.graph.find_cycle()

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Index an embedding in the engine's native HNSW (CONCEPT:AU-KG.query.object-graph-mapper).

        Distinct from storing an ``embedding`` node property: this registers the
        vector in the engine's SemanticStore so ``semantic_search`` is O(log N),
        instead of an O(N) cosine scan in Python.
        """
        self._client.graph.add_embedding(node_id, embedding)

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[tuple[str, float]]:
        """Native HNSW vector search via the engine — returns (node_id, score)."""
        return self._client.graph.semantic_search(query_embedding, n_results) or []

    def query_unified(
        self,
        plan: list[dict[str, Any]],
        *,
        reorder_filter_selectivity: float | None = None,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Run ONE cross-modal unified plan in a single costed round-trip (CONCEPT:AU-KG.compute.kg-2).

        ``plan`` is the engine's closed algebra over a shared ``RowSet`` — an
        ordered list of externally-tagged ``Op`` dicts (``Scan``/``Filter``/
        ``Traverse``/``Rank``/``RankText``/``FuseRrf``/``AsOf``/``Limit``) the
        engine sequences over ONE off-lock snapshot (filter via DataFusion,
        traverse via petgraph BFS, rank via the native ANN, RRF fusion in-plan).
        This is the engine doing filter+traverse+vector+rerank itself instead of
        the old hand-orchestrated Python pipeline of siloed round-trips
        (CONCEPT:AU-KG.compute.vector/214/215). Returns ``[{"id": str, "score": float|None}]``
        in the plan's final (post-``Rank``) order.

        Requires an engine built with the ``query`` feature; on a build without it
        the call raises a clear engine error — there is no O(N) Python fallback.

        Args:
            include_epistemic: Opt-in (CONCEPT:AU-KB-CURRENCY, Seam 1 — the
                ``KnowledgeBatch`` currency, extended from ``KnowledgeGraph.query``
                to this cross-modal surface). Default ``False`` — byte-for-byte the
                same ``[{"id", "score"}]`` rows as before this parameter existed.
                When ``True``, currency-upgrades the SAME ids (in the SAME
                post-``Rank`` order) via ``explain_provenance_by_ids`` into
                :class:`~.epistemic_row.EpistemicRow` results carrying the
                engine's confidence, bitemporal window, evidence provenance, and
                policy labels — never fabricated, resolved server-side. See
                ``docs/architecture/epistemic-columns-currency.md``.
        """
        rows = (
            self._client.query.unified(
                plan, reorder_filter_selectivity=reorder_filter_selectivity
            )
            or []
        )
        if include_epistemic:
            from .epistemic_row import attach_epistemic_rows

            return attach_epistemic_rows(rows, self.explain_provenance_by_ids)  # type: ignore[return-value]
        # Light epistemic layer (CONCEPT:AU-KB-CURRENCY, Native by default) —
        # see `KnowledgeGraph.query`'s identical wiring for the full rationale.
        from agent_utilities.core.config import config as _app_config

        from .epistemic_row import (
            attach_epistemic_columns,
            should_attach_epistemic_columns,
        )

        if should_attach_epistemic_columns(
            rows, default=_app_config.epistemic_light_default
        ):
            rows = attach_epistemic_columns(rows, self.explain_provenance_by_ids)
        return rows

    def query_cypher(self, query: str) -> list[dict[str, Any]]:
        """Run a native Cypher-subset ``query`` server-side and return row dicts (CONCEPT:AU-KG.query.object-graph-mapper).

        This is the engine's OWN Cypher executor (``eg-query::cypher``, compiled
        to the label index / VF2 / BFS — no Python-side regex interpretation),
        reached via ``client.query.cypher_read``. It requires an engine built with the
        ``cypher`` feature; on a build without it, or on a query the parser
        rejects, the call raises a clear engine error naming the problem — there
        is NO client-side fallback that silently narrows or drops the query.

        The wire protocol carries the literal query plus an explicit read mode
        (no separate params map). The engine parses the statement independently
        and rejects a mode mismatch before execution. Callers must inline every
        ``$param`` as a Cypher literal before calling this method (see
        ``EpistemicGraphBackend._inline_cypher_params``).
        """
        return self._client.query.cypher_read(query) or []

    def query_cypher_write(self, query: str) -> list[dict[str, Any]]:
        """Run one explicitly authorized native Cypher mutation.

        The Rust server reparses the complete statement, rejects a read/write
        mode mismatch, and commits graph state through its durable MutationBatch
        gateway before publishing the result. Public raw-query surfaces never
        call this method; their contract is read-only.
        """
        return self._client.query.cypher_write(query) or []

    def submit_program_optimization(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Submit one governed native ``ProgramOptimize`` analytics job.

        ``request`` is the exact versioned ``eg-program`` mapping. Raw prompts,
        outputs, identities, endpoints, and paths are intentionally absent; the
        engine authority-rebinds policy before persisting the job.
        """
        import msgpack

        if not isinstance(request, Mapping) or request.get("schema_version") != 1:
            raise ValueError("program optimization request is invalid")
        payload = msgpack.packb(dict(request), use_bin_type=True)
        return self._client.jobs.submit_program_optimization(
            self.graph_name,
            payload,
            purpose="program-optimization",
            max_attempts=1,
        )

    def program_optimization_status(self, job_id: str) -> dict[str, Any]:
        """Return the authoritative durable state for one submitted program job."""
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("program optimization job id is invalid")
        status = self._client.jobs.status(job_id)
        if not isinstance(status, dict) or status.get("job_id") != job_id:
            raise RuntimeError("program optimization status response is invalid")
        return status

    @staticmethod
    def program_optimization_result(job: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and return a succeeded job's typed program result rows."""
        state = job.get("state")
        if not isinstance(state, Mapping) or set(state) != {"Succeeded"}:
            raise RuntimeError("program optimization job did not succeed")
        succeeded = state.get("Succeeded")
        if not isinstance(succeeded, Mapping):
            raise RuntimeError("program optimization success state is invalid")
        result_ref = succeeded.get("result_ref")
        if not isinstance(result_ref, str) or not _OPAQUE_PROGRAM_REF.fullmatch(
            result_ref
        ):
            raise RuntimeError("program optimization result reference is invalid")

        output = job.get("output")
        if not isinstance(output, Mapping):
            raise RuntimeError("program optimization output is missing")
        schema = output.get("schema")
        if (
            not isinstance(schema, list)
            or tuple(
                (
                    column.get("name"),
                    column.get("logical_type"),
                    column.get("nullable"),
                )
                if isinstance(column, Mapping)
                else (None, None, None)
                for column in schema
            )
            != _PROGRAM_RESULT_SCHEMA
        ):
            raise RuntimeError("program optimization result schema is invalid")
        rows = output.get("rows")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("program optimization rows are missing")

        list_fields = {
            "evidence_refs",
            "source_refs",
            "proof_ids",
            "contradiction_ids",
            "demonstration_refs",
            "artifact_refs",
            "composition_refs",
            "modalities",
            "plan_step_kinds",
            "plan_executors",
            "plan_input_refs",
            "plan_output_refs",
            "plan_depends_on",
        }
        optional_refs = {
            "instruction_ref",
            "tool_policy_ref",
            "model_profile_ref",
            "plan_ref",
        }
        ref_lists = {
            "evidence_refs",
            "source_refs",
            "proof_ids",
            "contradiction_ids",
            "demonstration_refs",
            "artifact_refs",
            "composition_refs",
            "plan_input_refs",
            "plan_output_refs",
            "plan_depends_on",
        }
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != _PROGRAM_RESULT_FIELDS:
                raise RuntimeError("program optimization row schema is invalid")
            kind = row.get("kind")
            if kind not in {
                "program_candidate",
                "program_optimization_plan_step",
            }:
                raise RuntimeError("program optimization row kind is invalid")
            confidence = row.get("confidence")
            if (
                not isinstance(confidence, int | float)
                or isinstance(confidence, bool)
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise RuntimeError("program optimization confidence is invalid")
            for field in ("id", "program_ref"):
                value = row.get(field)
                if not isinstance(value, str) or not _OPAQUE_PROGRAM_REF.fullmatch(
                    value
                ):
                    raise RuntimeError("program optimization reference is invalid")
            for field in optional_refs:
                value = row.get(field)
                if value is not None and (
                    not isinstance(value, str)
                    or not _OPAQUE_PROGRAM_REF.fullmatch(value)
                ):
                    raise RuntimeError(
                        "program optimization optional reference is invalid"
                    )
            for field in list_fields:
                values = row.get(field)
                if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise RuntimeError("program optimization list field is invalid")
                if field in ref_lists and not all(
                    _OPAQUE_PROGRAM_REF.fullmatch(value) for value in values
                ):
                    raise RuntimeError("program optimization reference list is invalid")
            if not row.get("evidence_refs") or not row.get("source_refs"):
                raise RuntimeError(
                    "program optimization lineage references are missing"
                )
            if not isinstance(row.get("selected"), bool):
                raise RuntimeError("program optimization selection flag is invalid")
            optimizer = row.get("optimizer")
            execution = row.get("execution")
            if (
                not isinstance(optimizer, str)
                or not isinstance(execution, str)
                or _PROGRAM_OPTIMIZER_EXECUTIONS.get(optimizer) != execution
            ):
                raise RuntimeError("program optimization lineage is invalid")
            candidate_role = row.get("candidate_role")
            if candidate_role is not None and not isinstance(candidate_role, str):
                raise RuntimeError("program optimization candidate role is invalid")
            modalities = row.get("modalities")
            if not modalities or not set(modalities) <= _PROGRAM_MODALITIES:
                raise RuntimeError("program optimization modalities are invalid")
            max_operations = row.get("max_operations")
            if max_operations is not None and (
                not isinstance(max_operations, int)
                or isinstance(max_operations, bool)
                or max_operations <= 0
            ):
                raise RuntimeError("program optimization operation bound is invalid")
            if kind == "program_candidate":
                tool_policy_ref = row.get("tool_policy_ref")
                valid_tool_policy_binding = (
                    optimizer == "avatar"
                    and tool_policy_ref is not None
                    and tool_policy_ref in row.get("artifact_refs", [])
                    and row.get("instruction_ref") is None
                ) or (optimizer != "avatar" and tool_policy_ref is None)
                if (
                    candidate_role not in _PROGRAM_CANDIDATE_ROLES
                    or not valid_tool_policy_binding
                    or row.get("plan_ref") is not None
                    or not row.get("demonstration_refs")
                    or any(
                        row.get(field)
                        for field in (
                            "plan_step_kinds",
                            "plan_executors",
                            "plan_input_refs",
                            "plan_output_refs",
                            "plan_depends_on",
                        )
                    )
                    or max_operations is not None
                ):
                    raise RuntimeError(
                        "program optimization candidate shape is invalid"
                    )
            elif (
                candidate_role is not None
                or row.get("instruction_ref") is not None
                or row.get("tool_policy_ref") is not None
                or row.get("model_profile_ref") is not None
                or row.get("plan_ref") is None
                or row.get("selected") is not False
                or len(row.get("plan_step_kinds")) != 1
                or row["plan_step_kinds"][0] not in _PROGRAM_PLAN_STEP_KINDS
                or len(row.get("plan_executors")) != 1
                or row["plan_executors"][0] not in _PROGRAM_PLAN_EXECUTORS
                or not row.get("plan_input_refs")
                or not row.get("plan_output_refs")
                or max_operations is None
            ):
                raise RuntimeError("program optimization plan shape is invalid")
            normalized.append(dict(row))
        return {"result_ref": result_ref, "rows": normalized}

    def optimize_program(
        self,
        request: Mapping[str, Any],
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
    ) -> dict[str, Any]:
        """Submit, boundedly poll, and validate one native optimization job."""
        timeout = min(max(float(timeout_seconds), 0.1), 300.0)
        interval = min(max(float(poll_interval_seconds), 0.01), 1.0)
        submitted = self.submit_program_optimization(request)
        job_id = submitted.get("job_id") if isinstance(submitted, Mapping) else None
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("program optimization submission is invalid")

        deadline = time.monotonic() + timeout
        current = dict(submitted)
        while time.monotonic() < deadline:
            state = current.get("state")
            if isinstance(state, Mapping):
                if set(state) == {"Succeeded"}:
                    result = self.program_optimization_result(current)
                    return {
                        "status": "proposed",
                        "result": {"job_id": job_id, **result},
                    }
                if set(state) in ({"Failed"}, {"Cancelled"}):
                    raise RuntimeError("program optimization job terminated")
                if set(state) not in ({"Running"}, {"Publishing"}):
                    raise RuntimeError("program optimization state is invalid")
            elif state != "Submitted":
                raise RuntimeError("program optimization state is invalid")
            time.sleep(interval)
            current = self.program_optimization_status(job_id)
        raise TimeoutError("program optimization job timed out")

    def explain_provenance_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Resolve and project the engine's protocol EvidenceBundle for ``ids``
        (CONCEPT:EG-KB-CURRENCY — Seam 1, the ``KnowledgeBatch`` currency).

        Reaches ``Method::ExplainProvenanceByIds`` via
        ``client.query.explain_provenance_by_ids`` — the ID-seeded sibling of
        ``ExplainProvenance`` that needs no ``Op`` plan. Each returned row dict is
        projected as ``{"id", "kind", "score", "confidence", "valid_time",
        "tx_time", "source_refs", "policy_labels", "evidence_spans"}`` — straight
        field copies off the engine's schema-generated evidence claims (its
        ``EvidenceBundle``/``KnowledgeSet``), never fabricated client-side. This
        is the primitive
        :meth:`KnowledgeGraph.query`'s ``include_epistemic=True`` path builds
        :class:`~agent_utilities.knowledge_graph.core.epistemic_row.EpistemicRow`
        results from. Requires an engine built with the ``query`` feature; an id
        absent from the graph is silently skipped (never fabricated).
        """
        if not ids:
            return []
        from agent_utilities.protocols.epistemic_operations import EvidenceBundle

        bundle = EvidenceBundle.model_validate(
            self._client.query.explain_provenance_by_ids(list(ids))
        )
        return [
            {
                "id": claim.claim_ref,
                "kind": claim.kind,
                "score": claim.score,
                "confidence": claim.confidence,
                "valid_time": [claim.valid_time.start_ms, claim.valid_time.end_ms],
                "tx_time": [
                    claim.transaction_time.start_ms,
                    claim.transaction_time.end_ms,
                ],
                "source_refs": list(claim.source_refs),
                "evidence_spans": [],
                "evidence_locus_refs": list(claim.evidence_locus_refs),
                "policy_labels": list(claim.policy_labels),
                "contradiction_ids": list(claim.contradiction_refs),
                "proof_ids": list(claim.proof_refs),
            }
            for claim in bundle.claims
        ]

    def register_materialization(self, derived_id: str) -> dict[str, Any]:
        """Register ``derived_id`` as a live engine-side TruthMaintenance
        materialization (CONCEPT:EG-KG.epistemic.truth-maintenance, Seam 3 — X-6
        across the storage boundary).

        Reaches ``Method::RegisterMaterialization`` via
        ``client.query.register_materialization`` — the engine reads
        ``derived_id``'s OWN already-stored provenance (its ``invalidation_deps``
        property plus any outgoing ``:DerivedFrom``/``:GeneratedBy`` edge) into a
        dependency set and tracks it on its process-global truth-maintenance index.
        Call this ONCE, right after writing a derived node (a mined claim, a
        computed capability index entry, ...) plus its provenance edges — from then
        on, ANY committed change to a dependency (through the normal write path)
        automatically marks this materialization stale, with no further action
        needed here. Returns ``{"id", "depends_on", "generating_activity"}`` — the
        dependency set the engine actually resolved, never fabricated client-side;
        an empty ``depends_on`` means the node carried no recognized provenance
        (a legitimate answer, not an error). Requires an engine built with the
        ``epistemic-tms`` feature (opt-in, not part of ``full``); on a build
        without it the call raises a clear engine error — callers writing a
        derived artifact should treat this as best-effort, the same posture as
        every other write-path audit overlay in this codebase.
        """
        return self._client.query.register_materialization(derived_id) or {}

    def materialization_status(self, id: str) -> str | None:
        """Current status (``"Fresh"``/``"Stale"``/``"Retracted"``, or ``None`` if
        never registered) of a materialization tracked on the SAME index
        :meth:`register_materialization` writes to (CONCEPT:EG-KG.epistemic.truth-maintenance, Seam 3).
        Read-only — does not itself recompute anything. Requires an engine built
        with the ``epistemic-tms`` feature (opt-in, not part of ``full``)."""
        result = self._client.query.materialization_status(id) or {}
        return result.get("status") if isinstance(result, dict) else None

    def match_ontology_terms(self, query: str) -> list[dict[str, Any]]:
        """Embedding-free lexical capability gate via the engine (CONCEPT:EG-ORCH.routing.lexical-capability-escalation).

        Returns the capability-node terms (Tool/Skill/MCPServer names+synonyms)
        that appear as whole words in ``query``, each ``{term, node_type, label,
        score}``. A non-empty result means the turn names a real fleet capability
        — the "free" (~µs) tier between structural routing and ``semantic_search``.
        """
        return self._client.graph.match_ontology_terms(query) or []

    def discover(
        self,
        keywords: list[str],
        query_embedding: list[float] | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Engine-side hybrid keyword+semantic discovery in ONE round-trip.

        Ranks nodes by lexical keyword overlap (over ``name``/``description``/``type``)
        AND semantic similarity to ``query_embedding``, server-side, returning the
        top-``k`` hydrated as ``[{id, name, description, type, score}, ...]``. Pass an
        empty/omitted embedding for a keyword-only ranking (the engine degrades to a
        bounded keyword scan). This is the engine-scalable keyword primitive — it
        replaces the O(N) ``MATCH (n) WHERE … CONTAINS`` full-node scan that the engine
        does not filter server-side (per the dependency edict: keyword/vector ranking
        belongs on the engine, never an O(N) Python loop).
        """
        return self._client.graph.discover(keywords, query_embedding or [], k) or []

    def get_nodes_by_label(
        self, label: str, limit: int = 0
    ) -> list[tuple[str, dict[str, Any]]]:
        """Engine-side labeled fetch (CONCEPT:EG-KG.txn.per-graph-write-isolation) — at most ``limit`` nodes of
        ``label`` (``limit=0`` ⇒ no cap), bounding the wire payload so a
        ``MATCH (n:Label) … LIMIT k`` never materializes the whole graph."""
        return self._client.nodes.list_by_label(label, limit) or []

    def get_shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Get the shortest path between source and target nodes."""
        return self._client.graph.shortest_path(source_id, target_id)

    @staticmethod
    def _bfs_collect(
        start: Any,
        max_depth: int,
        neighbors_fn: Any,
        node_info_fn: Any,
    ) -> list[dict[str, Any]]:
        """Backend-agnostic BFS traversal collecting blast radius results.

        Args:
            start: Starting node identifier (backend-specific).
            max_depth: Maximum traversal depth.
            neighbors_fn: Callable(node) -> iterable of neighbor nodes.
            node_info_fn: Callable(node) -> dict with 'id' and 'type' keys.
        """
        visited: set = {start}
        queue: list[tuple[Any, int]] = [(start, 0)]
        results: list[dict[str, Any]] = []
        while queue:
            curr, depth = queue.pop(0)
            if curr != start:
                info = node_info_fn(curr)
                info["depth"] = depth
                results.append(info)
            if depth < max_depth:
                for neighbor in neighbors_fn(curr):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return results

    def get_blast_radius(self, node_id: str, max_depth: int) -> list[dict[str, Any]]:
        """Compute the blast radius dependencies from a starting node.

        Returns a list of dicts: [{'id': str, 'type': str, 'depth': int}]
        """
        nodes = self._client.graph.blast_radius(node_id, max_depth)
        res = []
        for i, nid in enumerate(nodes, start=1):
            res.append({"id": nid, "type": "Node", "depth": min(i, max_depth)})
        return res

    def parse_file(self, file_path: str, source: bytes) -> dict[str, Any]:
        """Parse one source file's AST natively via the Rust engine.

        Returns the ``ParseFile`` result (symbols + native test-quality metrics
        for Python). The compute layer — not Python — does the AST work.
        """
        return self._client.graph.parse_file(file_path, source)

    def parse_files(self, files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        """Batch-parse many files in ONE engine round-trip (CONCEPT:EG-KG.compute.graph-compute-engine).

        Returns one ``ParseFile``-shaped result per input file, in input order.
        Collapses a per-file parse storm into a single RPC. ``ParseFiles`` is
        mandatory in the current engine contract.
        """
        return self._client.graph.parse_files(files)

    def index_repository(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """Parse a batch AND resolve cross-file edges in ONE round-trip (CONCEPT:EG-KG.compute.turn-each-project).

        Treats ``files`` as one resolution scope (a repository, or a delta set):
        unlike :meth:`parse_files` (one raw result per file), this returns a SINGLE
        merged ``IndexResult`` dict whose ``calls`` (symbol→symbol) and
        ``depends_on`` (file→file) edges point at real node ids — the cross-file
        binding the GitLab code graph and impact analysis are built on.
        ``IndexRepository`` is mandatory in the exact current engine contract.
        """
        return self._client.graph.index_repository(files)

    def observe_screen(
        self,
        png: bytes,
        *,
        session_id: str,
        frame_seq: int = 0,
        prev_frame_id: str = "",
        prev_hash: int = 0,
        elements: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Materialise a captured desktop frame as durable graph entities in ONE
        round-trip (CONCEPT:AU-KG.ontology.owl-screen-bridge).

        Turns the screenshot ``png`` (only its dimensions + content hash persist) plus
        the AT-SPI ``elements`` (``[{role,name,x,y,w,h}, ...]``) into a
        ``ComputerUseSession`` + ``ScreenObservation`` frame + one ``UIElement`` per
        accessible, returning a ``ScreenObservationResult`` (nodes/edges + frame_id,
        hash, changed). ``ObserveScreen`` is mandatory in the exact current engine
        contract.
        """
        return self._client.graph.observe_screen(
            png,
            session_id=session_id,
            frame_seq=frame_seq,
            prev_frame_id=prev_frame_id,
            prev_hash=prev_hash,
            elements=elements or [],
        )

    def vf2_subgraph_match(self, pattern: "GraphComputeEngine") -> list[dict[str, str]]:
        """Find all subgraph isomorphism matches from pattern to target graph."""
        from agent_utilities.knowledge_graph.core.engine_breaker import unwrap_client

        # The wire call needs the RAW client of the pattern engine, not its
        # breaker proxy (CONCEPT:AU-OS.observability.no-op-without-metrics).
        return self._client.graph.vf2_subgraph_match(unwrap_client(pattern._client))

    # ── Ledger Operations ────────────────────────────────────────────────

    def get_ledger(self) -> list[str]:
        """Retrieve the mutation transaction ledger log."""
        return self._client.ledger.get()

    def clear_ledger(self) -> None:
        """Clear the mutation transaction ledger log."""
        self._client.ledger.clear()

    @staticmethod
    def _parse_ledger_entry(tx: str) -> tuple[str, list[str]]:
        """Parse a ledger transaction string into (operation, args).

        Shared parser to ensure Rust and Python ledger formats stay in sync.
        """
        if not isinstance(tx, str) or not tx:
            return ("", [])
        if len(tx.encode("utf-8")) > _MAX_LEDGER_ENTRY_BYTES or "\x00" in tx:
            raise ValueError("mutation ledger entry exceeds the safe input contract")
        operation, separator, payload = tx.partition("|")
        if not separator:
            return (operation, [])
        # JSON property values may legitimately contain ``|``. Split only the
        # fixed routing prefix and preserve the final JSON document verbatim.
        if operation == "AddNode":
            return (operation, payload.split("|", 1))
        if operation == "AddEdge":
            return (operation, payload.split("|", 2))
        return (operation, [payload])

    def apply_ledger(self, transactions: list[str]) -> None:
        """Replay mutations from a transaction ledger log."""
        self._client.ledger.apply(transactions)

    def audit_verify(self) -> dict[str, Any]:
        """Cryptographically verify this graph's tamper-evident hash-chained audit log.

        CONCEPT:AU-KG.audit.hash-chain-verify — routes to the engine's native
        ``Method::AuditVerify`` (``epistemic-graph/src/audit.rs``): every durable
        mutation already chains into a per-graph SHA-256 hash chain (the redb
        ``AUDIT`` table, keyed ``(graph, seq)``); this walks it and reports whether
        every entry's stored hash still matches its recomputed link — i.e. whether
        the durable write history has been tampered with post-hoc. The engine
        client (``epistemic_graph.client``) does not yet wrap ``AuditVerify`` as a
        typed ``LedgerClient`` method, so this uses the raw wire escape hatch
        used by :meth:`remove_triples` (``_send_wire``) rather than
        waiting on that package to add one — no Rust rebuild is required, the wire
        ``Method`` already exists and is dispatched by any engine built with the
        ``security`` feature (part of the default ``full`` build).

        Returns the engine's ``AuditReport``: ``{"graph", "ok", "entries",
        "first_broken_seq", "detail"}``. Raises if the engine build/config doesn't
        support it (no ``security`` feature, or no durable redb persist dir
        configured) so callers can degrade cleanly instead of silently trusting an
        unverified log.
        """
        return dict(self._send_wire("AuditVerify"))

    def flush_ledger_to_backend(self, backend: Any) -> int:
        """Flush the epistemic-graph mutation ledger to a persistent backend.

        Args:
            backend: A GraphBackend instance (e.g., LadybugBackend)

        Returns:
            int: The number of transactions flushed.
        """
        txs = self.get_ledger()
        if not txs:
            return 0

        count = 0
        for tx in txs:
            op, args = self._parse_ledger_entry(tx)
            if op == "AddNode":
                if len(args) < 2:
                    raise ValueError("incomplete node mutation in ledger")
                node_id = args[0]
                props_str = args[1]
                try:
                    props = json.loads(props_str)
                except Exception as exc:
                    raise ValueError("invalid node payload in mutation ledger") from exc
                if not isinstance(props, dict):
                    raise ValueError(
                        "node payload in mutation ledger must be an object"
                    )

                node_type = props.get("node_type", "Entity")
                if not isinstance(node_type, str) or not _CYPHER_IDENTIFIER.fullmatch(
                    node_type
                ):
                    raise ValueError("unsafe node type in mutation ledger")
                if node_type == "SYMBOL":
                    symbol_type = props.get("symbol_type", "Unknown")
                    file_path = props.get("file_path", "")
                    ast_hash = props.get("ast_hash", "")
                    name = props.get("name", node_id)
                    metadata_str = json.dumps(props)

                    query = (
                        "MERGE (n:Symbol {id: $id}) "
                        "SET n.node_type = 'SYMBOL', n.name = $name, "
                        "n.symbol_type = $sym_type, n.file_path = $fp, "
                        "n.ast_hash = $ast_hash, n.metadata = $meta"
                    )
                    try:
                        backend.execute_write(
                            query,
                            parameters={
                                "id": node_id,
                                "name": name,
                                "sym_type": symbol_type,
                                "fp": file_path,
                                "ast_hash": ast_hash,
                                "meta": metadata_str,
                            },
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "mutation ledger backend write failed "
                            f"({type(exc).__name__})"
                        ) from exc
                else:
                    # Generic node fallback
                    query = f"MERGE (n:{node_type} {{id: $id}}) SET n.metadata = $meta"
                    try:
                        backend.execute_write(
                            query,
                            parameters={
                                "id": node_id,
                                "meta": props_str,
                            },
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "mutation ledger backend write failed "
                            f"({type(exc).__name__})"
                        ) from exc
                count += 1
            elif op == "AddEdge":
                if len(args) < 3:
                    raise ValueError("incomplete edge mutation in ledger")
                src = args[0]
                tgt = args[1]
                props_str = args[2]
                try:
                    props = json.loads(props_str)
                except Exception as exc:
                    raise ValueError("invalid edge payload in mutation ledger") from exc
                if not isinstance(props, dict):
                    raise ValueError(
                        "edge payload in mutation ledger must be an object"
                    )

                edge_type = props.get("relationship") or "RELATED_TO"
                if not isinstance(edge_type, str):
                    raise ValueError("edge type in mutation ledger must be a string")
                edge_type = edge_type.replace(" ", "_").upper()
                if not _CYPHER_IDENTIFIER.fullmatch(edge_type):
                    raise ValueError("unsafe edge type in mutation ledger")
                query = (
                    f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                    f"MERGE (a)-[r:{edge_type}]->(b) "
                    "SET r.metadata = $meta"
                )
                try:
                    backend.execute_write(
                        query,
                        parameters={
                            "src": src,
                            "tgt": tgt,
                            "meta": props_str,
                        },
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"mutation ledger backend write failed ({type(exc).__name__})"
                    ) from exc
                count += 1
            else:
                raise ValueError("unsupported mutation operation in ledger")

        self.clear_ledger()
        return count

    # ── Serialization ────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize the graph to a JSON string representation."""
        nodes = []
        for nid in self._get_all_nodes():
            props = self._get_node_properties(nid)
            nodes.append({"id": nid, "properties": props})

        edges = []
        for src, tgt in self._get_all_edges():
            props = self._get_edge_properties(src, tgt)
            edges.append({"source": src, "target": tgt, "properties": props})

        return json.dumps({"nodes": nodes, "edges": edges}, default=str)

    def from_json(self, json_str: str) -> None:
        """Deserialize and rebuild the graph from a JSON string."""
        data = json.loads(json_str)
        # Clear existing graph nodes/edges via client if possible or just rebuild
        for nid in self._get_all_nodes():
            try:
                self.remove_node(nid)
            except Exception:
                pass

        # Re-add nodes
        for node_data in data.get("nodes", []):
            nid = node_data["id"]
            props = node_data.get("properties", {})
            self.add_node(nid, props)

        # Re-add edges
        for edge_data in data.get("edges", []):
            src = edge_data["source"]
            tgt = edge_data["target"]
            props = edge_data.get("properties", {})
            self.add_edge(src, tgt, props)

    def drop_graph(self) -> bool:
        """Unload this engine's named graph from the engine authority (free memory).

        The engine-side per-graph unload behind the KG-2.62 pool eviction hook:
        deletes the tenant's named graph from the engine process. **Lossy unless
        an explicit durable mirror holds the data**, so the pool only
        calls this when ``KG_ENGINE_POOL_DROP_ON_EVICT`` is set. Returns True on
        success. Never raises — eviction must not crash a request.
        """
        try:
            self._client.tenants.delete(self.graph_name)
            return True
        except Exception as exc:  # noqa: BLE001 — best-effort unload
            logger.debug("Configured graph unload failed (%s)", type(exc).__name__)
            return False

    def to_msgpack(self) -> bytes:
        """Serialize graph to MsgPack binary representation."""
        return self._client.lifecycle.to_msgpack()

    def from_msgpack(self, msgpack_bytes: bytes) -> None:
        """Deserialize graph from MsgPack binary representation."""
        self._client.lifecycle.from_msgpack(msgpack_bytes)

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _get_all_nodes(self) -> list[str]:
        return [nid for nid, _ in self._client.nodes.list()]

    def _get_all_nodes_with_properties(self) -> list[tuple[str, dict[str, Any]]]:
        """Return every ``(node_id, properties)`` pair in a SINGLE round-trip.

        ``nodes.list()`` already returns the full properties alongside each id, so
        a full-graph scan must consume them here rather than issuing one
        ``_get_node_properties`` round-trip per node (an N+1 that cost ~45s on a
        40K-node graph and held the GIL, starving foreground ingestion).
        (CONCEPT:EG-KG.storage.nonblocking-checkpoint ingestion throughput)
        """
        out: list[tuple[str, dict[str, Any]]] = []
        for nid, props in self._client.nodes.list():
            if isinstance(props, dict):
                out.append((nid, props))
            elif isinstance(props, str):
                try:
                    parsed = json.loads(props)
                    out.append((nid, parsed if isinstance(parsed, dict) else {}))
                except Exception:
                    out.append((nid, {}))
            else:
                out.append((nid, {}))
        return out

    def _get_node_properties(self, node_id: str) -> dict[str, Any]:
        props = self._client.nodes.properties(node_id)
        if isinstance(props, dict):
            return props
        if isinstance(props, str):
            try:
                parsed = json.loads(props)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _get_all_edges(self) -> list[tuple[str, str]]:
        return [(src, tgt) for src, tgt, _ in self._client.edges.list()]

    def _get_all_edges_with_properties(
        self,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Return every ``(src, tgt, properties)`` triple in a SINGLE round-trip.

        ``edges.list()`` already ships each edge's (msgpack) properties alongside
        its endpoints, so a full-graph edge scan must decode them locally rather
        than issuing one ``_get_edge_properties`` round-trip per edge — the same
        N+1 the bulk node scan avoids. On a 67K-edge graph the per-edge path cost
        ~100s/scan and was re-run once per assimilation stage.
        (CONCEPT:EG-KG.storage.nonblocking-checkpoint throughput)
        """
        import msgpack

        out: list[tuple[str, str, dict[str, Any]]] = []
        for src, tgt, raw in self._client.edges.list():
            props: Any = raw
            if isinstance(raw, bytes | bytearray | list):
                try:
                    props = msgpack.unpackb(bytes(raw), raw=False)
                except Exception:
                    props = {}
            out.append((src, tgt, props if isinstance(props, dict) else {}))
        return out

    # ── Rust-native API wrappers ─────────────────────────────────────────

    def in_degree(self, node_id: str) -> int:
        """Return the in-degree of a node."""
        try:
            return self._client.nodes.in_degree(node_id)
        except Exception:
            return 0

    def out_degree(self, node_id: str) -> int:
        """Return the out-degree of a node."""
        try:
            return self._client.nodes.out_degree(node_id)
        except Exception:
            return 0

    def get_predecessors(self, node_id: str) -> list[str]:
        """Return predecessor node IDs."""
        try:
            return self._client.nodes.predecessors(node_id)
        except Exception:
            return []

    def get_successors(self, node_id: str) -> list[str]:
        """Return successor node IDs."""
        try:
            return self._client.nodes.successors(node_id)
        except Exception:
            return []

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return all neighbor node IDs (predecessors + successors, deduplicated)."""
        try:
            return self._client.nodes.neighbors(node_id)
        except Exception:
            return []

    def node_ids(self) -> list[str]:
        """Return all node IDs in the graph."""
        return self._client.nodes.ids()

    def get_rdf(self) -> str:
        """Serialize the live graph as datatype/lang-faithful N-Triples.

        This is the engine's canonical ``GetRdf`` surface. Consumers that require
        structured triples parse the returned N-Triples document with an RDF parser;
        no lossy property-graph triple-list representation is exposed.
        """
        return str(self._client.rdf.get_rdf() or "")

    def sparql(
        self,
        query: str,
        base_iri: str = "",
        type_convention: str = "",
    ) -> list[dict[str, str | None]]:
        """Run a SPARQL 1.1 query over the LIVE engine graph (one round-trip).

        CONCEPT:AU-KG.compute.native-sparql-owl-shacl — Engine-native SPARQL/OWL/SHACL: the Python semantic-web stack
        (rdflib/owlready2/pyshacl) is demoted to the engine's native RDF surface. This
        routes to the engine's native ``client.rdf.sparql`` so the
        query executes against the engine's RDF projection of the live property graph
        (resource object -> typed edge, literal -> typed property cell, ``rdf:type`` ->
        the engine ``type`` label) -- NOT a rdflib materialization. Returns one dict per
        result row keyed by projected variable. Raises if the engine/op is unavailable
        (e.g. a server built without the ``sparql`` feature) so callers can fall back to
        the rdflib path.

        ``base_iri`` + ``type_convention`` select the engine's LPG→RDF projection
        vocabulary (engine concept KG-2.240): empty ⇒ the identity projection (verbatim keys);
        an ontology namespace + ``"camel"`` makes the engine emit ``<node> rdf:type
        <base + CamelCase(type)>`` and ``<node> <base + prop> <v>``. ``owl_bridge``
        passes its ``au:`` namespace so by-class queries resolve engine-native.
        """
        return list(
            self._client.rdf.sparql(
                query, base_iri=base_iri, type_convention=type_convention
            )
        )

    def owl_reason(
        self, ontology: str | None = None, target_class: str | None = None
    ) -> dict[str, Any]:
        """Run the engine's native OWL 2 (EL+/RL) reasoner over the live graph.

        CONCEPT:AU-KG.compute.native-sparql-owl-shacl — routes to ``client.rdf.owl_reason``: classifies the OWL
        axioms in the graph (plus any extra ``ontology`` Turtle) and materializes
        entailments, returning ``{"subclasses", "instances", "consistent",
        "unsatisfiable"}`` (confidence/decay-weighted, read-only -- does NOT mutate the
        graph). ``target_class`` restricts ``instances`` to that class's inferred
        members. Raises if the engine/op is unavailable (server built without the
        ``owl`` feature) so callers can fall back to the Python/owlready2 path.
        """
        return dict(
            self._client.rdf.owl_reason(ontology=ontology, target_class=target_class)
        )

    def add_triples(
        self, turtle: str | None = None, ntriples: str | None = None
    ) -> dict[str, int]:
        """Load Turtle or N-Triples into the engine's RDF dataset (one round-trip).

        CONCEPT:AU-KG.compute.native-sparql-owl-shacl — routes to ``client.rdf.add_triples``. Used to seed OWL
        axioms / RDF facts the native reasoner and SPARQL surface operate over. Raises
        if the engine/op is unavailable.
        """
        return dict(self._client.rdf.add_triples(turtle=turtle, ntriples=ntriples))

    def _send_wire(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        """Invoke a raw engine wire op by name (one round-trip).

        Escape hatch for wire ops the typed Python ``client.rdf``/``client.query``
        namespaces don't yet wrap (e.g. the engine's ``RemoveTriples`` /
        ``DropNamedGraph`` retract ops). It unwraps the
        circuit-breaker proxy, reaches the underlying async client + loop, and runs the
        coroutine on it. Raises if no sync engine client/loop is available.
        """
        import asyncio

        sc = getattr(self._client, "__wrapped__", self._client)  # unwrap breaker
        async_client = getattr(sc, "_client", None)
        loop = getattr(sc, "_loop", None)
        if async_client is None or loop is None:
            raise RuntimeError(f"{method} unavailable (no sync engine client/loop)")
        coro = (
            async_client._send(method, payload)
            if payload
            else async_client._send(method)
        )
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def remove_triples(
        self, turtle: str | None = None, ntriples: str | None = None
    ) -> dict[str, int]:
        """Physically retract Turtle / N-Triples from the engine's RDF dataset (KG-2.266).

        The retract counterpart to :meth:`add_triples` — used by the ontology
        lifecycle (KG-2.265) to drop an unloaded ontology's axioms from the engine so
        they stop being reasoned over, not just deactivated in a registry record.
        Prefers a typed ``client.rdf.remove_triples`` wrapper when the installed
        engine client exposes one, else falls back to the raw ``RemoveTriples`` wire op
        (the engine ships the op even where the Python client lacks the wrapper).
        Raises if the engine/op is unavailable so callers can report the gap honestly.
        """
        rdf = getattr(self._client, "rdf", None)
        fn = getattr(rdf, "remove_triples", None)
        if callable(fn):
            return dict(fn(turtle=turtle, ntriples=ntriples))
        return dict(
            self._send_wire("RemoveTriples", {"turtle": turtle, "ntriples": ntriples})
            or {}
        )

    def drop_named_graph(self, graph: str) -> dict[str, Any]:
        """Drop an entire named RDF graph from the engine (KG-2.266).

        Retracts every triple in the ``graph`` named-graph IRI in one op — the
        coarse-grained retract used when an ontology owns a dedicated named graph.
        Prefers a typed ``client.rdf.drop_named_graph`` wrapper, else falls back to the
        raw ``DropNamedGraph`` wire op. Raises if the engine/op is unavailable.
        """
        rdf = getattr(self._client, "rdf", None)
        fn = getattr(rdf, "drop_named_graph", None)
        if callable(fn):
            return dict(fn(graph) or {})
        return dict(self._send_wire("DropNamedGraph", {"graph": graph}) or {})

    def sql_exec(self, statement: str) -> Any:
        """Execute a write-capable SQL statement (DDL/DML) on the engine (KG-2.266).

        The write sibling of the read-only ``client.query.sql`` surface: lets us
        ``CREATE TABLE`` / ``INSERT`` / ``DROP TABLE`` against the engine's native
        DataFusion user-table store so connector + ETL data can be mirrored into
        engine SQL tables. Routes through ``client.query.sql`` (the same ``Sql`` wire
        op) — the engine, not this client, enforces what statements its user-table
        surface accepts. Raises if no engine query surface is available.
        """
        query_ns = getattr(self._client, "query", None)
        sql_fn = getattr(query_ns, "sql", None)
        if not callable(sql_fn):
            raise RuntimeError(
                "The active backend has no epistemic-graph SQL surface; "
                "engine SQL tables require the engine backend (build with the "
                "'query' feature)."
            )
        return sql_fn(statement)

    def degree_centrality_all(self) -> list[tuple[str, float]]:
        """Compute degree centrality for all nodes."""
        return self._client.analytics.degree_centrality_all()

    def pagerank(
        self, damping: float = 0.85, iterations: int = 100
    ) -> list[tuple[str, float]]:
        """Compute PageRank scores for all nodes."""
        return self._client.analytics.pagerank(damping, iterations)

    def connected_components(self) -> list[list[str]]:
        """Return weakly connected components as lists of node IDs."""
        return self._client.graph.connected_components()

    def strongly_connected_components(self) -> list[list[str]]:
        """Return strongly connected components via Tarjan's algorithm.

        CONCEPT:AU-KG.compute.tokio-service-layer — Tarjan's SCC via Tokio service (GIL-free).
        """
        return self._client.graph.strongly_connected_components()

    def minimum_spanning_tree(self) -> list[tuple[str, str, float]]:
        """Return the minimum spanning tree as (source, target, weight) edges.

        CONCEPT:AU-KG.compute.tokio-service-layer — Kruskal's MST via Tokio service (GIL-free).
        """
        return self._client.graph.minimum_spanning_tree()

    def community_detection(self, resolution: float = 1.0) -> list[list[str]]:
        """Detect communities using label propagation."""
        return self._client.graph.community_detection(resolution)

    def community_detect_ephemeral(
        self,
        node_ids: list[str],
        edges: list[tuple[str, str]],
        resolution: float = 1.0,
    ) -> list[list[str]]:
        """Stateless community detection over an inline call graph (KG-2.58).

        Runs detection on the passed nodes/edges in an in-memory throwaway graph on
        the engine — no tenant load, no persistence. Eliminates the bulk-load
        round-trip + comm-tenant churn of the load-then-detect pattern.
        """
        return self._client.graph.community_detect_ephemeral(
            node_ids, edges, resolution
        )

    def betweenness_centrality(self) -> list[tuple[str, float]]:
        """Compute betweenness centrality via Brandes' algorithm."""
        return self._client.analytics.betweenness_centrality()

    def graph_coloring(self) -> list[tuple[str, int]]:
        """Greedy graph coloring — assigns colors so no adjacent nodes share a color."""
        return self._client.graph.graph_coloring()

    def compute_similarity_edges(
        self, threshold: float = 0.8
    ) -> list[tuple[str, str, float]]:
        """Compute similarity edges between nodes with embeddings."""
        return self._client.graph.compute_similarity_edges(threshold)

    def resolve_candidates(
        self,
        sim_threshold: float = 0.8,
        merge_threshold: float = 0.92,
        node_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Native entity-resolution candidate generation (CONCEPT:AU-KG.compute.when-exposes-native).

        Returns merge proposals (``{canonical, members, score, kind}``;
        ``kind`` = ``same_as`` | ``extends``) — the server-side escalation tier the
        dedup ladder routes its residual through. Read/propose only; apply via
        ``batch_update``.
        """
        return (
            self._client.graph.resolve_candidates(
                sim_threshold, merge_threshold, node_type
            )
            or []
        )

    def prune_by_lifecycle(
        self, max_age_secs: int = 0, min_score: float = 0.0
    ) -> dict[str, Any]:
        """Lifecycle-aware pruning: remove nodes past max_age or below min_score."""
        result_json = self._client.lifecycle.prune(max_age_secs, min_score)
        return json.loads(result_json)

    def get_context_view(self, agent_id: str, max_tokens: int = 8000) -> dict[str, Any]:
        """Get an optimized context view for an agent within a token budget."""
        result_json = self._client.lifecycle.get_context_view(agent_id, max_tokens)
        return json.loads(result_json)

    def batch_update(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch update: apply multiple operations in a single service call."""
        result = self._client.lifecycle.batch_update(operations)
        if not isinstance(result, dict):
            raise RuntimeError("current BatchUpdate contract returned an invalid shape")
        return result

    def multi_graph_batch_update(
        self, batches: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Batched CROSS-GRAPH write in ONE round-trip (CONCEPT:AU-KG.ingest.batched-cross-graph-writer).

        ``batches`` maps ``graph_name → operations`` (each op-list has the same
        shape as :meth:`batch_update`). When the engine supports the
        ``MultiGraphBatchUpdate`` op (CONCEPT:EG-KG.storage.multi-graph-batch-write) the whole
        map ships in ONE round-trip and the engine applies each graph's sub-batch
        CONCURRENTLY across the K redb shard writers — the write stage then scales
        with the number of DISTINCT destination graphs (the per-shard content-keyed
        fanout, CONCEPT:AU-KG.ingest.unified-query-routing) instead of pinning one lock.

        Returns ``{"results": {graph: <batch_result>}, "errors": {graph: msg}}``.
        """
        if not batches:
            return {"results": {}, "errors": {}}

        lifecycle = getattr(self._client, "lifecycle", None)
        fn = getattr(lifecycle, "multi_graph_batch_update", None)
        if not callable(fn):
            raise RuntimeError("current engine contract requires MultiGraphBatchUpdate")
        result = fn(batches)
        if not isinstance(result, dict):
            raise RuntimeError(
                "current MultiGraphBatchUpdate contract returned an invalid shape"
            )
        self._register_multi_graph_targets(batches)
        return result

    @staticmethod
    def _register_multi_graph_targets(
        batches: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Register any routed content graphs touched by a multi-graph write so the
        unified read path fans across them (CONCEPT:AU-KG.ingest.unified-query-routing)."""
        try:
            from .ingest_routing import register_content_graph

            for graph in batches:
                register_content_graph(graph)
        except Exception:  # noqa: BLE001 — registration is best-effort
            pass

    def metrics(self) -> dict[str, Any]:
        """Runtime metrics for monitoring and observability."""
        result_json = self._client.lifecycle.metrics()
        return json.loads(result_json)

    def personalized_pagerank(
        self,
        seed_nodes: dict[str, float] | None = None,
        damping: float = 0.85,
        iterations: int = 100,
    ) -> dict[str, float]:
        """Personalized PageRank with seed teleport nodes."""
        seeds = list((seed_nodes or {}).items())
        result = self._client.analytics.personalized_pagerank(
            seeds, damping, iterations
        )
        return dict(result)

    # ── Batch Operations ─────────────────────────────────────────────────

    def bulk_mutate(self, operations: list[dict[str, Any]]) -> Any:
        """Send a batch of mutations in a single service call.

        Each operation dict should have a ``method`` key and any required
        parameters.  Example::

            engine.bulk_mutate([
                {"method": "AddNode", "node_id": "A", "properties_json": "{}"},
                {"method": "AddEdge", "source_id": "A", "target_id": "B", ...},
            ])
        """
        return self._client.lifecycle.batch_update(operations)

    def evict_lru(self, max_nodes: int = 50_000) -> int:
        """Evict oldest nodes to enforce an in-memory cap.

        Returns the number of evicted nodes.
        """
        return self._client.lifecycle.evict_lru(max_nodes)

    # ── Native multiplexed engine transport (CONCEPT:AU-KG.compute.when-exposes) ────────────
    # The current client demultiplexes many in-flight requests by request id on
    # the ONE authenticated process transport. An auxiliary connection pool is
    # redundant and would pin one caller's authority into long-lived clients.

    def _engine_loop(self) -> Any:
        """The background asyncio loop the sync client runs on, or ``None``.

        The native socket, demultiplexer, and write lock are bound to this loop,
        so concurrent submissions are driven there via
        ``run_coroutine_threadsafe``. Reached through the breaker proxy.
        """
        sc = getattr(self._client, "__wrapped__", self._client)
        return getattr(sc, "_loop", None)

    def _engine_async_client(self) -> Any:
        """Return the routed async view behind the process sync client."""

        sync_client = getattr(self._client, "__wrapped__", self._client)
        client = getattr(sync_client, "_client", None)
        if client is None:
            raise RuntimeError("no routed engine client is available")
        return client

    def batch_update_concurrent(
        self,
        batches: list[list[dict[str, Any]]],
        graph: str | None = None,
    ) -> list[dict[str, Any]]:
        """Apply independent batches through native request-id pipelining.

        Each entry of ``batches`` is a ``batch_update`` op-list (same shape as
        :meth:`batch_update`). The batches must be independent of one another; ops
        WITHIN a batch keep their order because each batch is one RPC. The RPCs
        share the process transport but remain concurrently in flight and are
        demultiplexed by request id. Results preserve input order. Keep a
        node-before-edge sequence inside one batch.
        """
        if not batches:
            return []
        loop = self._engine_loop()
        if loop is None:
            raise RuntimeError("no engine loop available for concurrent batches")
        if graph is not None and graph != self.graph_name:
            raise PermissionError("concurrent batches cannot retarget the graph view")
        import asyncio as _asyncio

        async def _drive() -> list[Any]:
            client = self._engine_async_client()
            return await _asyncio.gather(
                *(client.lifecycle.batch_update(batch) for batch in batches)
            )

        fut = _asyncio.run_coroutine_threadsafe(_drive(), loop)
        results = fut.result()
        out: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, str | bytes | bytearray):
                out.append(json.loads(result))
            else:
                out.append(result)
        return out

    async def map_concurrent(
        self,
        ops: list[Any],
        graph: str | None = None,
    ) -> list[Any]:
        """Run independent engine operations through native pipelining.

        Every task receives the same routed async view. The native client safely
        demultiplexes in-flight RPCs while its ContextVar binds the calling task's
        verified GraphSession. Results preserve input order.
        """
        if not ops:
            return []
        import asyncio as _asyncio

        loop = self._engine_loop()
        if loop is None:
            raise RuntimeError("no engine loop available for map_concurrent")
        if graph is not None and graph != self.graph_name:
            raise PermissionError(
                "concurrent operations cannot retarget the graph view"
            )

        async def _drive() -> list[Any]:
            client = self._engine_async_client()
            return await _asyncio.gather(*(op(client) for op in ops))

        coro = _drive()
        fut = _asyncio.run_coroutine_threadsafe(coro, loop)
        return await _asyncio.wrap_future(fut)

    # ── Graph Traversal API ──────────────────────────────────────────────
    # These provide the standard graph traversal interface used across
    # the codebase (owl_bridge, graph_validator, memory_retriever, etc).
    # All hot paths route to the Rust Tokio service; these are thin wrappers.

    @property
    def nodes(self) -> "_NodeView":
        """NX-compatible node view.  Supports iteration, ``in``, and ``[id]``."""
        return _NodeView(self)

    @property
    def edges(self) -> "_EdgeView":
        """NX-compatible edge view.  Supports iteration and ``data=True``."""
        return _EdgeView(self)

    def number_of_nodes(self) -> int:
        """Alias for ``node_count()``."""
        return self.node_count()

    def number_of_edges(self) -> int:
        """Alias for ``edge_count()``."""
        return self.edge_count()

    def degree(self, node_id: str) -> int:
        """Total degree (in + out) of *node_id*."""
        return self.in_degree(node_id) + self.out_degree(node_id)

    def successors(self, node_id: str) -> list[str]:
        """Return successors of *node_id*."""
        return self.get_successors(node_id)

    def predecessors(self, node_id: str) -> list[str]:
        """Return predecessors of *node_id*."""
        return self.get_predecessors(node_id)

    def neighbors(self, node_id: str) -> list[str]:
        """Return neighbors (successors + predecessors) of *node_id*."""
        return self.get_neighbors(node_id)

    def get_edge_data(self, source_id: str, target_id: str, default: Any = None) -> Any:
        """NX-compatible edge data lookup."""
        props = self._get_edge_properties(source_id, target_id)
        if not props:
            return default if not self.has_edge(source_id, target_id) else {0: {}}

        class MultiDiGraphCompatDict(dict):
            def __init__(self, p: dict[str, Any]):
                super().__init__({0: p})
                self._props = p

            def __getitem__(self, key: Any) -> Any:
                if key == 0:
                    return self._props
                return self._props[key]

            def get(self, key: Any, default_val: Any = None) -> Any:
                if key == 0:
                    return self._props
                return self._props.get(key, default_val)

        return MultiDiGraphCompatDict(props)

    def out_edges(self, node_id: str, data: bool = False) -> list:
        """Return outgoing edges from *node_id*.

        When *data* is True, returns ``(src, tgt, props)`` triples.
        """
        succs = self.get_successors(node_id)
        if data:
            return [(node_id, s, self._get_edge_properties(node_id, s)) for s in succs]
        return [(node_id, s) for s in succs]

    def in_edges(self, node_id: str, data: bool = False) -> list:
        """Return incoming edges to *node_id*.

        When *data* is True, returns ``(src, tgt, props)`` triples.
        """
        preds = self.get_predecessors(node_id)
        if data:
            return [(p, node_id, self._get_edge_properties(p, node_id)) for p in preds]
        return [(p, node_id) for p in preds]

    def _get_edge_properties(self, source_id: str, target_id: str) -> dict[str, Any]:
        """Retrieve edge properties between two nodes."""
        props_list = self._client.edges.properties(source_id, target_id)
        if props_list:
            props = props_list[0]
            if isinstance(props, dict):
                return props
            if isinstance(props, str):
                try:
                    parsed = json.loads(props)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
        return {}

    def __contains__(self, node_id: str) -> bool:
        """Support ``node_id in engine`` syntax."""
        return self.has_node(node_id)

    def __getitem__(self, node_id: str) -> dict[str, Any]:
        """Support ``engine[node_id]`` to get node properties."""
        return self._get_node_properties(node_id)


class _NodePropertiesProxy(dict):
    def __init__(
        self, engine: GraphComputeEngine, node_id: str, properties: dict[str, Any]
    ):
        super().__init__(properties)
        self._engine = engine
        self._node_id = node_id

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._engine.add_node(self._node_id, properties=dict(self))

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._engine.add_node(self._node_id, properties=dict(self))

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._engine.add_node(self._node_id, properties=dict(self))


class _EdgePropertiesProxy(dict):
    def __init__(
        self,
        engine: GraphComputeEngine,
        source_id: str,
        target_id: str,
        properties: dict[str, Any],
    ):
        super().__init__(properties)
        self._engine = engine
        self._source_id = source_id
        self._target_id = target_id

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))


class _NodeView:
    """Lightweight proxy providing NX-style ``graph.nodes`` access."""

    __slots__ = ("_engine",)

    def __init__(self, engine: GraphComputeEngine) -> None:
        self._engine = engine

    def __iter__(self):
        return iter(self._engine.node_ids())

    def __len__(self) -> int:
        return self._engine.node_count()

    def __contains__(self, node_id: str) -> bool:
        return self._engine.has_node(node_id)

    def __getitem__(self, node_id: str) -> dict[str, Any]:
        props = self._engine._get_node_properties(node_id)
        return _NodePropertiesProxy(self._engine, node_id, props)

    def get(self, node_id: str, default: Any = None) -> Any:
        """Support ``graph.nodes.get(id, default)`` pattern."""
        if self._engine.has_node(node_id):
            props = self._engine._get_node_properties(node_id)
            return _NodePropertiesProxy(self._engine, node_id, props)
        return default

    def __call__(self, data: bool = False):
        """Support ``graph.nodes(data=True)`` iteration."""
        if data:
            # One bulk round-trip (props ship with the node list) instead of an
            # ``_get_node_properties`` round-trip per node. (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
            return [
                (nid, _NodePropertiesProxy(self._engine, nid, props))
                for nid, props in self._engine._get_all_nodes_with_properties()
            ]
        return self._engine.node_ids()


class _EdgeView:
    """Lightweight proxy providing NX-style ``graph.edges`` access."""

    __slots__ = ("_engine",)

    def __init__(self, engine: GraphComputeEngine) -> None:
        self._engine = engine

    def __iter__(self):
        return iter(self._engine._get_all_edges())

    def __len__(self) -> int:
        return self._engine.edge_count()

    def __call__(
        self, data: bool = False, keys: bool = False, default: Any = None, **kwargs: Any
    ):
        """Support ``graph.edges(data=True, keys=True)`` iteration."""
        result: list[Any] = []
        # One bulk round-trip (props ship with the edge list, decoded locally)
        # instead of an ``_get_edge_properties`` round-trip per edge. (KG-2.8)
        for src, tgt, props in self._engine._get_all_edges_with_properties():
            proxy = _EdgePropertiesProxy(self._engine, src, tgt, props)
            if data and keys:
                result.append((src, tgt, 0, proxy))
            elif data:
                result.append((src, tgt, proxy))
            elif keys:
                result.append((src, tgt, 0))
            else:
                result.append((src, tgt))
        return result

    def __getitem__(self, key: Any) -> Any:
        """Support edge properties lookup by tuple key."""
        if not isinstance(key, tuple) or len(key) < 2:
            raise KeyError(key)
        src, tgt = key[0], key[1]
        props = self._engine._get_edge_properties(src, tgt)
        if props is None:
            raise KeyError(key)

        proxy = _EdgePropertiesProxy(self._engine, src, tgt, props)
        if len(key) >= 3:
            return proxy

        class MultiDiGraphCompatEdgeDict(dict):
            def __getitem__(self, k):
                return proxy

            def get(self, k, default=None):
                return proxy

        return MultiDiGraphCompatEdgeDict({0: proxy})
