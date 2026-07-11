# CONCEPT:AU-OS.deployment.engine-resolver-auto-provision - One engine resolver auto-provisioning every entrypoint by precedence remote then share-running-local then autostart-shared-supervised
"""ONE engine resolver — the single chokepoint provisions an engine for *every* entrypoint.

CONCEPT:AU-OS.deployment.engine-resolver-auto-provision — auto-bundled engine. Every entrypoint (graph-os MCP, the
gateway/host daemon, :class:`IntelligenceGraphEngine`, the facade,
:class:`EpistemicGraphBackend`, the tenant engine pool, messaging, agent/serving)
funnels through :class:`~.graph_compute.GraphComputeEngine.__init__`, which calls
:func:`resolve_engine` here. The resolver decides — by ONE precedence, with NO
per-entrypoint code — how the process reaches its engine:

    remote  →  share-running-local  →  autostart-shared-supervised

* **remote** — ``GRAPH_SERVICE_ENDPOINTS`` / ``GRAPH_SERVICE_TCP_ADDR`` set, or
  ``engine_mode=remote`` with an endpoint. "I deployed the engine in Docker on
  another host." The resolver returns that endpoint and NEVER autostarts — an
  unreachable configured remote stays fail-loud (the contract preserved in
  ``graph_compute``'s sharded/remote branch).
* **shared** — the default/local endpoint is already serving (a cheap connect
  probe succeeds, or a spawn-lock holder is recorded *and* a probe verifies it).
  Reuse it; spawn nothing. This is how co-located entrypoints on one host share
  the ONE engine.
* **autostart** — nothing reachable. Under the per-socket
  :func:`~.engine_lock.engine_spawn_guard` (first-one-wins flock), a
  double-checked probe re-shares a peer's just-started engine; otherwise spawn a
  **detached, supervised** engine via the existing
  :meth:`GraphComputeEngine._autostart_engine`. Detached = it survives the
  spawning process so OTHER entrypoints on the host share it (distinct from the
  ``coupled`` pdeathsig mode, kept for a true single-process case). Supervised =
  reference-counted idle shutdown: the engine self-terminates ``grace`` seconds
  after its LAST client disconnects (robust to client crashes) — unless the
  operator chose a **persistent** lifecycle, in which case it runs forever like a
  local service.

The resolver REUSES the existing building blocks — it invents no new locking,
probing, auth, or topology logic:

* :func:`~.placement_catalog.resolve_placement` — engine placement-catalog
  lookup (epoch-cached, redirect-aware), falling back to
  :func:`~.shard_topology.resolve_endpoints` / :func:`~.shard_topology.shard_endpoint_for`
  — endpoint list + the static HRW shard ring — only when no catalog is
  reachable/advertised.
* :func:`~.shard_topology.is_local_endpoint` / :func:`~.shard_topology.probe_endpoint`
  — local-vs-remote classification + the cheap connect probe.
* :func:`~.graph_compute.resolve_engine_auth` — the HMAC secret / insecure flag.
* :func:`~.engine_lock.engine_lock_holder` — recorded spawner identity.

So ``GraphComputeEngine.__init__`` no longer carries an inline autostart sequence;
it asks the resolver for a :class:`ResolvedEngine` and connects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .placement_catalog import resolve_placement
from .shard_topology import (
    is_local_endpoint,
    probe_endpoint,
    resolve_endpoints,
)

logger = logging.getLogger(__name__)

#: How long the cheap share-probe waits for a connect before declaring an
#: endpoint unreachable. A named constant (config discipline): short enough that
#: a cold start doesn't stall, long enough to span a busy local accept queue.
_PROBE_TIMEOUT_S = 0.5

__all__ = [
    "ResolvedEngine",
    "client_connect_kwargs",
    "engine_idle_shutdown_secs",
    "resolve_engine",
]


@dataclass(frozen=True)
class ResolvedEngine:
    """The resolved engine target for this process.

    * ``endpoint`` — the verbatim ``unix://``/``tcp://`` endpoint to connect to
      (already HRW-placed for the routing graph).
    * ``auth_secret`` — the HMAC secret to authenticate with (``None`` when
      insecure).
    * ``insecure`` — True when engine auth is disabled (dev).
    * ``mode`` — ``"remote"`` | ``"shared"`` | ``"autostart"``: which precedence
      leg won. ``remote`` and ``shared`` never spawn; only ``autostart`` may.
    * ``autostart_allowed`` — True only when this is a local endpoint the process
      is permitted to spawn (never a configured remote shard).
    * ``idle_shutdown_secs`` — reference-counted idle grace to pass the spawned
      engine (``> 0``), or ``0`` for a **persistent** engine that never
      self-stops. Only meaningful for ``mode="autostart"``.
    """

    endpoint: str
    auth_secret: str | None
    insecure: bool
    mode: str
    autostart_allowed: bool
    idle_shutdown_secs: int


def engine_idle_shutdown_secs(config: Any) -> int:
    """Resolve the reference-counted idle-shutdown grace for an autostarted engine.

    CONCEPT:AU-OS.deployment.engine-resolver-auto-provision — lifecycle choice, no env-sprawl (both reads are typed
    :class:`AgentConfig` fields):

    * ``engine_lifecycle == "persistent"`` → ``0`` (never self-stop; runs forever
      like a local service). This wins regardless of ``engine_idle_shutdown_secs``.
    * otherwise (``"refcounted"``, the default) → ``engine_idle_shutdown_secs``
      when ``> 0``, else ``0`` (a non-positive grace is itself a persistent
      choice).

    A return of ``0`` means "pass NO ``--idle-shutdown-secs`` flag" — the engine
    is long-living. A positive return is the grace in seconds after the last
    client disconnects.
    """
    lifecycle = (
        (getattr(config, "engine_lifecycle", "refcounted") or "refcounted")
        .strip()
        .lower()
    )
    if lifecycle == "persistent":
        return 0
    secs = int(getattr(config, "engine_idle_shutdown_secs", 60) or 0)
    return secs if secs > 0 else 0


def resolve_engine(
    config: Any, graph_name: str, *, endpoint_override: str | None = None
) -> ResolvedEngine:
    """Resolve how THIS process reaches its engine, by ONE precedence.

    ``graph_name`` is the already-routed graph (HRW key); ``endpoint_override``
    pins placement for a dedicated engine (the ingest path) and bypasses HRW.

    Returns a :class:`ResolvedEngine`. This function performs NO connect of its
    own beyond the cheap share-probe — the caller
    (:class:`GraphComputeEngine`) owns the real authenticated connect, the
    circuit breaker, and (for ``mode="autostart"``) the guarded spawn — so the
    resolver stays a pure decision over the existing building blocks.

    Precedence (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision):

    1. **remote** — multiple endpoints (sharding), an explicit ``tcp://`` target,
       or ``engine_mode=remote``. ``autostart_allowed`` is True only for the rare
       local case; a remote endpoint is never autostarted (fail-loud preserved).
    2. **shared** — a local endpoint that is already serving: the connect probe
       succeeds, OR a spawn-lock holder is recorded AND the probe verifies it.
    3. **autostart** — a local endpoint with nothing listening: the caller spawns
       a detached, supervised engine (reference-counted unless persistent).
    """
    # Auth is independent of the leg — resolve it once via the existing helper.
    from .graph_compute import resolve_engine_auth

    auth_secret, insecure = resolve_engine_auth(config)

    endpoints = resolve_endpoints(config)
    sharded = len(endpoints) > 1
    if endpoint_override:
        endpoint = str(endpoint_override)
    else:
        # Engine placement catalog first (DIST-P2-2b, CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw);
        # the static HRW ring is only the bootstrap/fallback when no catalog
        # is reachable/advertised — see ``placement_catalog`` module docstring.
        endpoint = resolve_placement(graph_name, endpoints, config).endpoint

    mode_setting = (getattr(config, "engine_mode", "auto") or "auto").strip().lower()
    local = is_local_endpoint(endpoint)

    # ── remote leg ───────────────────────────────────────────────────────
    # A configured remote shard, an explicit tcp target, or engine_mode=remote
    # is a hard contract: connect to it, never auto-spawn a local stand-in
    # (auto-starting one silently splits the keyspace into invisible islands —
    # the fail-loud convention preserved by graph_compute's sharded/remote
    # branch). Autostart is permitted ONLY for a local endpoint.
    autostart_allowed = bool(setting_autostart(config)) and (not sharded) and local
    if mode_setting == "remote" or sharded or not local:
        return ResolvedEngine(
            endpoint=endpoint,
            auth_secret=auth_secret,
            insecure=insecure,
            mode="remote",
            autostart_allowed=False,
            idle_shutdown_secs=0,
        )

    # ── shared leg ───────────────────────────────────────────────────────
    # A local endpoint that is already serving: reuse it, spawn nothing. The
    # cheap connect probe is authoritative; a recorded spawn-lock holder alone
    # is NOT (the holder could be stale), so we still require a probe to verify.
    if _local_engine_running(endpoint):
        return ResolvedEngine(
            endpoint=endpoint,
            auth_secret=auth_secret,
            insecure=insecure,
            mode="shared",
            # Already up — no spawn needed; but keep autostart permitted so a
            # race (it dies between probe and connect) can still self-heal.
            autostart_allowed=autostart_allowed,
            idle_shutdown_secs=engine_idle_shutdown_secs(config),
        )

    # ── autostart leg ────────────────────────────────────────────────────
    # Nothing reachable on a local endpoint. The caller spawns under the
    # per-socket guard (double-checked) — detached + supervised.
    return ResolvedEngine(
        endpoint=endpoint,
        auth_secret=auth_secret,
        insecure=insecure,
        mode="autostart",
        autostart_allowed=autostart_allowed,
        idle_shutdown_secs=engine_idle_shutdown_secs(config),
    )


def setting_autostart(config: Any) -> bool:
    """Whether local autostart is enabled for this process (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision).

    The auto-bundled-engine default: ``engine_mode`` in {``auto``, ``embedded``,
    ``shared``} enables local autostart so a local endpoint with nothing serving
    is provisioned on demand — every entrypoint gets an engine with no
    per-entrypoint code. ``remote`` disables it (a configured remote is never
    autostarted; it stays fail-loud). The legacy ``EPISTEMIC_GRAPH_AUTOSTART=0``
    opt-OUT is still honoured (read via the sanctioned :func:`config.setting`
    accessor — no bare ``os.environ``) so an operator can force a connect-only
    process.
    """
    from agent_utilities.core.config import setting

    mode = (getattr(config, "engine_mode", "auto") or "auto").strip().lower()
    if mode == "remote":
        return False
    if mode not in {"embedded", "shared", "auto"}:
        return False
    # Never autostart inside the unit suite — it pins the in-memory backend and
    # must not spawn a real engine process (and a forced opt-in there is honoured
    # for the resolver's own integration tests).
    if (
        setting("AGENT_UTILITIES_TESTING", "false").strip().lower()
        in {
            "1",
            "true",
            "yes",
        }
        and setting("EPISTEMIC_GRAPH_AUTOSTART", "") != "1"
    ):
        return False
    # auto/embedded/shared all want a local engine when none is configured
    # remote. Default ON (auto-bundle); an explicit EPISTEMIC_GRAPH_AUTOSTART=0
    # opt-out forces a connect-only process.
    return setting("EPISTEMIC_GRAPH_AUTOSTART", "1") != "0"


def client_connect_kwargs(
    config: Any | None = None, graph_name: str | None = None
) -> dict[str, Any]:
    """Build ``SyncEpistemicGraphClient.connect`` kwargs via the ONE resolver.

    CONCEPT:AU-OS.deployment.engine-resolver-auto-provision — the centralized path for the few DIRECT-client callers
    (``domains/finance/*``, ``core/ingest_engine``) that connect outside
    :class:`GraphComputeEngine`. Resolves the same endpoint (HRW-placed) and auth
    secret the chokepoint uses, so a remote/sharded/insecure deployment is honoured
    everywhere instead of relying on the engine's bare env defaults. These callers
    intentionally do NOT autostart — they degrade gracefully when the engine is
    down — so this returns connect kwargs only, never spawns.
    """
    from agent_utilities.core.config import AgentConfig

    from .shard_topology import default_graph_name

    cfg = config if config is not None else AgentConfig()
    graph = graph_name or default_graph_name(cfg)
    resolved = resolve_engine(cfg, graph)
    kwargs: dict[str, Any] = {
        "auth_secret": resolved.auth_secret,
        "graph_name": graph,
    }
    ep = resolved.endpoint
    if ep.startswith("tcp://"):
        kwargs["tcp_addr"] = ep[6:]
    elif ep.startswith("unix://"):
        kwargs["socket_path"] = ep[7:]
    else:
        kwargs["socket_path"] = ep
    return kwargs


def _local_engine_running(endpoint: str) -> bool:
    """Cheap "is a local engine already serving here?" check (no auth handshake).

    A transport-level connect probe (:func:`shard_topology.probe_endpoint`) is
    authoritative. A recorded spawn-lock holder (:func:`engine_lock_holder`) is
    used only as a hint for logging — never to declare the engine up without a
    probe, since the holder record can outlive a crashed engine.
    """
    up = probe_endpoint(endpoint, timeout=_PROBE_TIMEOUT_S)
    if up:
        return True
    # Not reachable — log if a stale spawn-lock holder is recorded (diagnostic).
    if endpoint.startswith("unix://") or endpoint.startswith("/"):
        try:
            from .engine_lock import engine_lock_holder

            sock = endpoint[7:] if endpoint.startswith("unix://") else endpoint
            holder = engine_lock_holder(sock)
            if holder:
                logger.debug(
                    "engine spawn-lock holder pid=%s recorded for %s but the "
                    "endpoint is not serving (stale or starting).",
                    holder.get("pid", "?"),
                    endpoint,
                )
        except Exception:  # noqa: BLE001 — diagnostics must never raise
            pass
    return False
