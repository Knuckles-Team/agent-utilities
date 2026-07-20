# CONCEPT:AU-OS.deployment.engine-resolver-auto-provision - One engine resolver auto-provisioning every entrypoint by precedence remote then share-running-local then autostart-shared-supervised
"""ONE engine resolver — the single chokepoint provisions an engine for *every* entrypoint.

CONCEPT:AU-OS.deployment.engine-resolver-auto-provision — auto-bundled engine. Every entrypoint (graph-os MCP, the
gateway/host daemon, :class:`IntelligenceGraphEngine`, the facade,
:class:`EpistemicGraphBackend`, the tenant engine pool, messaging, agent/serving)
funnels through :class:`~.graph_compute.GraphComputeEngine.__init__`, which calls
:func:`resolve_engine` here. The resolver decides — by ONE precedence, with NO
per-entrypoint code — how the process reaches its engine:

    remote  →  share-running-local  →  autostart-shared-supervised

* **remote** — ``GRAPH_SERVICE_ENDPOINTS`` is configured. The resolver returns
  its first coordinator contact and NEVER autostarts — an
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

* :func:`~.shard_topology.resolve_endpoints` — coordinator contact topology.
  Per-graph placement is resolved later, under the request's authenticated
  :class:`GraphSession`, by :func:`~.placement_catalog.resolve_placement`.
* :func:`~.shard_topology.is_local_endpoint` / :func:`~.shard_topology.probe_endpoint`
  — local-vs-remote classification + the cheap connect probe.
* :func:`~.graph_compute.resolve_engine_auth` — the mandatory HMAC secret.
* :func:`~.engine_lock.engine_lock_holder` — recorded spawner identity.

So ``GraphComputeEngine.__init__`` no longer carries an inline autostart sequence;
it asks the resolver for a :class:`ResolvedEngine` and connects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .shard_topology import is_local_endpoint, probe_endpoint, resolve_endpoints

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

    * ``endpoint`` — the verbatim ``unix://``/``tcp://`` coordinator contact.
      Authenticated request routing resolves placement after construction.
    * ``auth_secret`` — the non-empty HMAC secret used to authenticate every
      current-protocol request. There is no unauthenticated engine mode.
    * ``mode`` — ``"remote"`` | ``"shared"`` | ``"autostart"``: which precedence
      leg won. ``remote`` and ``shared`` never spawn; only ``autostart`` may.
    * ``autostart_allowed`` — True only when this is a local endpoint the process
      is permitted to spawn (never a configured remote shard).
    * ``idle_shutdown_secs`` — reference-counted idle grace to pass the spawned
      engine (``> 0``), or ``0`` for a **persistent** engine that never
      self-stops. Only meaningful for ``mode="autostart"``.
    """

    endpoint: str
    auth_secret: str
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


def resolve_engine(config: Any, graph_name: str) -> ResolvedEngine:
    """Resolve how THIS process reaches its engine, by ONE precedence.

    ``graph_name`` identifies the future request route but is not resolved here:
    construction has no authenticated request authority.

    Returns a :class:`ResolvedEngine`. This function performs NO connect of its
    own beyond the cheap share-probe — the caller
    (:class:`GraphComputeEngine`) owns the real authenticated connect, the
    circuit breaker, and (for ``mode="autostart"``) the guarded spawn — so the
    resolver stays a pure decision over the existing building blocks.

    Precedence (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision):

    1. **remote** — ``GRAPH_SERVICE_ENDPOINTS`` is present. Every configured
       topology is connect-only, including a single loopback/Unix contact.
    2. **shared** — a local endpoint that is already serving: the connect probe
       succeeds, OR a spawn-lock holder is recorded AND the probe verifies it.
    3. **autostart** — a local endpoint with nothing listening: the caller spawns
       a detached, supervised engine (reference-counted unless persistent).
    """
    # Auth is independent of the leg — resolve it once via the existing helper.
    from .graph_compute import resolve_engine_auth

    auth_secret = resolve_engine_auth(config)

    endpoints = resolve_endpoints(config)
    external_topology = bool(getattr(config, "graph_service_endpoints", None))
    # This is a coordinator contact, not a placement decision. The routed
    # client asks the engine for the graph's authoritative group only after
    # middleware has supplied a verified GraphSession.
    endpoint = endpoints[0]
    _ = graph_name

    local = is_local_endpoint(endpoint)

    # ── remote leg ───────────────────────────────────────────────────────
    # A configured topology is a hard contract: connect to it, never auto-spawn
    # a local stand-in. This is true even for loopback TCP or a Unix contact.
    # (auto-starting one silently splits the keyspace into invisible islands —
    # the fail-loud convention preserved by graph_compute's sharded/remote
    # branch). Autostart is permitted ONLY for a local endpoint.
    autostart_allowed = bool(setting_autostart(config)) and local
    if external_topology:
        return ResolvedEngine(
            endpoint=endpoint,
            auth_secret=auth_secret,
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
        mode="autostart",
        autostart_allowed=autostart_allowed,
        idle_shutdown_secs=engine_idle_shutdown_secs(config),
    )


def setting_autostart(config: Any) -> bool:
    """Whether local autostart is enabled for this process (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision).

    ``GRAPH_SERVICE_ENDPOINTS`` present means connect-only and disables local
    autostart. When absent, the packaged local engine is provisioned on demand.
    The test-suite guard remains an internal harness concern, not a runtime
    topology setting.
    """
    from agent_utilities.core.config import setting

    if getattr(config, "graph_service_endpoints", None):
        return False
    # Never autostart inside the unit suite — it pins the in-memory backend and
    # must not spawn a real engine process. Resolver tests explicitly remove
    # this harness-only flag when exercising the packaged lifecycle.
    if (
        setting("AGENT_UTILITIES_TESTING", "false").strip().lower()
        in {
            "1",
            "true",
            "yes",
        }
    ):
        return False
    return True


def client_connect_kwargs(
    config: Any | None = None,
    graph_name: str | None = None,
    *,
    verified_context: dict[str, Any],
) -> dict[str, Any]:
    """Build low-level client kwargs for bootstrap and diagnostic callers.

    Served code acquires :class:`GraphComputeEngine` instead of calling this
    helper directly. It remains for resolver tests and external diagnostics that
    explicitly own their process client. Callers must supply the complete current
    request authority; this boundary never invents identity or derives a
    request envelope from transport inputs.
    """
    from agent_utilities.core.config import AgentConfig

    from .shard_topology import default_graph_name

    cfg = config if config is not None else AgentConfig()
    graph = graph_name or default_graph_name(cfg)
    resolved = resolve_engine(cfg, graph)
    kwargs: dict[str, Any] = {
        "auth_secret": resolved.auth_secret,
        "graph_name": graph,
        "verified_context": verified_context,
    }
    ep = resolved.endpoint
    if ep.startswith(("tcp://", "tls://")):
        from .engine_transport import (
            engine_client_transport_kwargs,
            native_endpoint_address,
        )

        kwargs["tcp_addr"] = native_endpoint_address(ep)[0]
        kwargs.update(engine_client_transport_kwargs(ep, config=cfg))
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
                    "An engine spawn-lock holder is recorded, but its transport "
                    "is not serving (stale or starting)."
                )
        except Exception:  # noqa: BLE001 — diagnostics must never raise
            pass
    return False
