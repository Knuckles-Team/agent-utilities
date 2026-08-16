# CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw - Engine-authoritative tenant partition placement over configured coordinator contacts
# CONCEPT:AU-OS.scaling.shard-topology-visibility-per - Shard topology visibility with per-shard reachability status surfaces and per-endpoint engine gauges and counters
"""Tenant naming and engine coordinator-contact topology.

``GRAPH_SERVICE_ENDPOINTS`` is an ordered list of contacts, not a placement
ring. A request uses the authenticated engine ``PlacementRoute`` RPC, validates
its epoch and numeric group fence, then maps that group through
``GRAPH_RAFT_GROUP_ENDPOINTS`` (or the sole stable coordinator). No caller-side
hash, fallback, or topology guess is permitted.

Tenant isolation still selects a named graph with :func:`tenant_graph_name`.
The engine placement catalog owns where that graph lives and performs governed
movement. Local autostart is available only when this explicit topology is
unset and the packaged-local transport resolves to one local endpoint.

Topology visibility (CONCEPT:AU-OS.scaling.shard-topology-visibility-per — Shard Topology Visibility):
:func:`shard_topology_status` powers the unified daemon status and the
gateway's ``/daemon/shards`` route, and exports the per-shard
``agent_utilities_engine_shard_up{endpoint}`` gauge.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CONTROL_GRAPH_NAME",
    "DEFAULT_GRAPH",
    "DEFAULT_LOCAL_ENDPOINT",
    "is_local_endpoint",
    "is_system_graph",
    "probe_endpoint",
    "resolve_endpoints",
    "resolve_routing_graph",
    "shard_topology_status",
    "sharding_active",
    "tenant_graph_name",
]

#: Fallback default graph when no config object is reachable. The runtime
#: default lives on ``AgentConfig.kg_default_graph`` (KG_DEFAULT_GRAPH).
DEFAULT_GRAPH = "__commons__"

#: The one fixed, tenant-shared control-plane graph (U-16/BUG-113): the sole
#: WorkItem/:Schedule authority (``engine.py::_build_control_backend``,
#: ``ingest_profile.py``'s ``EpistemicGraphBackend().for_graph("__control__")``
#: pattern). A system ``__…__`` graph per ``is_system_graph`` -- readable by
#: every tenant by design, never content/document/codebase data. Named here
#: as ONE constant so every control-plane binding site (currently
#: ``engine.py`` and ``ingest_profile.py``) agrees on the literal instead of
#: re-deriving/hardcoding it independently.
CONTROL_GRAPH_NAME = "__control__"


def _platform_default_endpoint() -> str:
    """The per-platform default local endpoint (CONCEPT:AU-OS.deployment.cross-platform-locks-plus).

    Mirrors the engine's own per-platform default transport (``src/main.rs``):

    * **Unix:** a UDS at ``$XDG_RUNTIME_DIR/epistemic-graph.sock`` (when the dir
      exists) else ``/tmp/epistemic-graph.sock`` — ``unix://`` scheme.
    * **Windows:** Tokio has no AF_UNIX listener, so the engine's default
      transport is TCP loopback (``127.0.0.1:8765``, overridable via
      ``GRAPH_SERVICE_TCP_FALLBACK_ADDR`` engine-side). The client therefore
      defaults to the same ``tcp://127.0.0.1:8765`` so a zero-config Windows
      install reaches an autostarted engine.
    """
    import os as _os
    import sys as _sys

    from agent_utilities.core.config import setting

    if _os.name == "nt" or _sys.platform.startswith("win"):
        return "tcp://127.0.0.1:8765"
    xdg = setting("XDG_RUNTIME_DIR")  # sanctioned accessor (no bare os.environ)
    if xdg and _os.path.isdir(str(xdg)):
        return f"unix://{_os.path.join(str(xdg), 'epistemic-graph.sock')}"
    return "unix:///tmp/epistemic-graph.sock"  # nosec B108


#: The single-endpoint default (matches the engine's own per-platform fallback).
#: Computed once at import — on Windows this is a ``tcp://`` loopback (no UDS),
#: on Unix a ``unix://`` socket. Kept as a module constant for the many callers
#: that reference it; the resolver also re-derives it lazily where config is None.
DEFAULT_LOCAL_ENDPOINT = _platform_default_endpoint()

# Tenant ids come from JWT claims (org_id/tid/...) and may contain arbitrary
# characters; graph names should stay shell/file/metric friendly.
_TENANT_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _config(config: Any = None) -> Any:
    if config is not None:
        return config
    from agent_utilities.core.config import AgentConfig

    return AgentConfig()


def default_graph_name(config: Any = None) -> str:
    """The configured default graph (``KG_DEFAULT_GRAPH``, default ``__commons__``)."""
    cfg = _config(config)
    return getattr(cfg, "kg_default_graph", DEFAULT_GRAPH) or DEFAULT_GRAPH


def is_system_graph(name: str | None) -> bool:
    """True for a dunder-named system-scoped graph (GOC-61 §2.5/§6, PHASE-1 STOPGAP).

    Matches the informal ``__…__`` convention already documented in this
    codebase (``security/secrets_client.py``: "a system ``__…__`` graph like
    ``__control__`` / ``__commons__`` / ``__secrets__``") — the naming family
    that is readable/reachable by every tenant by design, as opposed to a
    per-tenant (``tenant__<slug>__<base>``) or per-source content graph
    (``code:``/``src:``/``chat:``/``research:`` — see
    :func:`~agent_utilities.knowledge_graph.core.ingest_routing.is_content_graph`,
    this predicate's sibling for the OTHER system-graph family).

    **THIS IS A PHASE-1 STOPGAP, NOT A PERMANENT DESIGN.** GOC-61 §2.5 point 4
    replaces this hardcoded naming check with the durable ``:GraphNamespace``
    registry in phase 2, so a new system-shared graph never needs a code
    change. Do not grow this into a hardcoded name list — if a new system
    graph shows up that this predicate misses, that is the signal phase 2 is
    overdue, not a reason to special-case it here.
    """
    if not name:
        return False
    n = str(name).strip()
    return len(n) >= 5 and n.startswith("__") and n.endswith("__")


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def resolve_endpoints(config: Any = None) -> list[str]:
    """Resolve ordered engine coordinator contacts.

    ``GRAPH_SERVICE_ENDPOINTS`` is the sole explicit external/coordinator
    topology. Any configured list is returned verbatim and is connect-only.
    When it is absent, the packaged engine's platform default is used.
    Contact order is deterministic, but it never decides placement.
    """
    cfg = _config(config)
    eps = [
        str(e).strip() for e in (cfg.graph_service_endpoints or []) if str(e).strip()
    ]
    if eps:
        return eps
    return [DEFAULT_LOCAL_ENDPOINT]


def sharding_active(config: Any = None) -> bool:
    """True when multiple coordinator contacts are configured."""
    return len(resolve_endpoints(config)) > 1


def is_local_endpoint(endpoint: str) -> bool:
    """True for endpoints that are local-by-construction.

    Unix sockets and TCP loopback addresses are local. Treating loopback TCP as
    remote breaks the zero-configuration Windows installation because the
    engine and client deliberately use ``127.0.0.1:8765`` when AF_UNIX is not
    available. This function classifies transport locality only: any contact
    supplied through ``GRAPH_SERVICE_ENDPOINTS`` remains connect-only even when
    it is loopback or a Unix socket.
    """
    ep = endpoint.strip()
    if ep.startswith("unix://") or "://" not in ep:
        return True
    # A TLS endpoint is an externally provisioned listener even when its
    # authority is loopback: autostart cannot safely invent the server identity.
    if ep.startswith("tls://"):
        return False
    if not ep.startswith("tcp://"):
        return False

    # ``urlsplit`` handles bracketed IPv6. Keep localhost independent of DNS.
    from urllib.parse import urlsplit

    try:
        host = urlsplit(ep).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Tenant → graph naming discipline
# ---------------------------------------------------------------------------


def tenant_graph_name(tenant: str | None, base: str = DEFAULT_GRAPH) -> str:
    """Map a tenant id onto its per-tenant named graph: ``tenant__<t>__<base>``.

    The single naming rule for tenant-scoped graph placement: facade, backends
    and the engine client all use this helper so a tenant's data consistently
    lands on one named graph whose placement is engine-authoritative. An empty
    or unset tenant returns ``base`` unchanged — single-tenant deployments are
    byte-for-byte unaffected.
    """
    if not tenant:
        return base
    slug = _TENANT_SLUG_RE.sub("_", tenant.strip()).strip("_").lower()
    if not slug:
        return base
    return f"tenant__{slug}__{base}"


def resolve_routing_graph(graph_name: str | None, config: Any = None) -> str:
    """Resolve the named graph submitted to the placement authority.

    Resolution order:

    1. An explicit, non-empty ``graph_name`` — the operation targets a named
       graph; use it verbatim. This includes the literal default/commons
       name itself: an EXPLICIT request for ``__commons__`` is honored
       exactly like an explicit request for any other name.
    2. ``graph_name is None`` (the caller asked for nothing specific) and the
       ambient :class:`ActorContext` carries a tenant — the default graph is
       mapped to the tenant's graph via :func:`tenant_graph_name`.
    3. ``graph_name is None`` and no tenant — the configured default graph
       (``KG_DEFAULT_GRAPH``).

    BUG-020 / GOC-61 phase 1 (fixed): the prior ``graph_name != default``
    clause conflated two different intents — "the caller explicitly asked
    for the graph literally named ``__commons__``" and "the caller passed
    nothing and wants their ambient default" — because both produced
    ``graph_name == default_graph_name()``. Only ``graph_name is None``
    triggers tenant mapping now; any other explicit, non-empty name
    (including ``"__commons__"`` itself) is honored verbatim. See
    ``plans/graph-os-completion-program/decisions/GOC-61-native-graph-sharing.md``
    §6 for the full root-cause analysis and the 5-site call-site audit that
    cleared this change.
    """
    default = default_graph_name(config)
    if graph_name:
        return graph_name
    from agent_utilities.security.brain_context import current_actor

    tenant = current_actor().tenant_id
    if tenant:
        return tenant_graph_name(tenant, base=default)
    return graph_name or default


# ---------------------------------------------------------------------------
# Topology visibility (CONCEPT:AU-OS.scaling.shard-topology-visibility-per)
# ---------------------------------------------------------------------------


def probe_endpoint(endpoint: str, timeout: float = 0.5) -> bool:
    """Transport-level reachability probe (raw connect; no auth handshake).

    Cheap by design: a TCP/UDS connect proves the engine process is listening
    without spending an authenticated RPC per scrape.
    """
    ep = endpoint.strip()
    try:
        if ep.startswith(("tcp://", "tls://")):
            from .engine_transport import native_endpoint_address

            authority = native_endpoint_address(ep)[0]
            if authority.startswith("[") and "]:" in authority:
                host, port = authority[1:].split("]:", 1)
            else:
                host, separator, port = authority.rpartition(":")
                if not separator:
                    return False
            with socket.create_connection(
                (host or "127.0.0.1", int(port)), timeout=timeout
            ):
                return True
        path = ep[7:] if ep.startswith("unix://") else ep
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(path)
            return True
        finally:
            sock.close()
    except (OSError, ValueError):
        return False


def record_shard_connect(endpoint: str, up: bool) -> None:
    """Export shard reachability observed on a real client connect attempt."""
    try:
        from agent_utilities.observability.gateway_metrics import ENGINE_SHARD_UP

        ENGINE_SHARD_UP.labels(endpoint=endpoint).set(1.0 if up else 0.0)
    except Exception:  # pragma: no cover - metrics must never break clients
        logger.debug("Could not export shard-up metric")


def shard_topology_status(
    config: Any = None, probe: bool = True, timeout: float = 0.5
) -> dict[str, Any]:
    """Shard topology + per-shard reachability for status/health surfaces.

    Returns::

        {
          "mode": "single" | "sharded",
          "default_graph": "...",
          "endpoints": [
            {"endpoint": "...", "local": bool, "reachable": bool, "breaker": "..."},
            ...
          ],
        }

    Also refreshes ``agent_utilities_engine_shard_up{endpoint}`` when probing.
    The host flock governs only the LOCAL engine; remote shards are reported,
    never managed, from here.
    """
    cfg = _config(config)
    endpoints = resolve_endpoints(cfg)
    entries: list[dict[str, Any]] = []
    for ep in endpoints:
        entry: dict[str, Any] = {"endpoint": ep, "local": is_local_endpoint(ep)}
        if probe:
            up = probe_endpoint(ep, timeout=timeout)
            entry["reachable"] = up
            record_shard_connect(ep, up)
        try:
            from agent_utilities.knowledge_graph.core.engine_breaker import get_breaker

            entry["breaker"] = get_breaker(ep).state
        except Exception:  # pragma: no cover - status must stay best-effort
            pass
        entries.append(entry)
    return {
        "mode": "sharded" if len(endpoints) > 1 else "single",
        "default_graph": default_graph_name(cfg),
        "endpoints": entries,
    }
