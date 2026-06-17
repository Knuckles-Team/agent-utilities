"""Materialize a registered source extractor INTO the graph (CONCEPT:KG-2.9).

The self-registering source extractors (``extractors/camunda.py``,
``extractors/aris.py``, ``extractors/egeria.py``, …) turn an injected vendor
client into a uniform :class:`~agent_utilities.knowledge_graph.enrichment.models.ExtractionBatch`.
:func:`~agent_utilities.knowledge_graph.orchestration.engine_federation.GraphEngineFederation.register_rest_source`
already runs them at *query time* (TTL-cached virtualization), but — as its own
docstring notes — virtualization reasons only over the fetched slice. To let OWL
reasoning extrapolate across the **whole** cross-vendor crosswalk (a Camunda
``BusinessProcess`` ``ALIGNED_WITH`` its ARIS/Egeria twin, transitive
``FLOWS_TO`` lineage, ``governedBy`` policy attachment) the records must be
**persisted**.

This module is that materializing twin: it runs an extractor by category and
persists its batch through the one generic writer (``registry.write_batch``) —
no new extraction logic, no second writer. The MCP/REST surface
(``graph_ingest action=materialize_source``) calls :func:`materialize_source`
and then runs one reasoning cycle so the new process structure is reasoned over
natively.

The vendor client is **injected**. :func:`resolve_source_client` is a
best-effort in-process resolver that imports the connector package's
``auth.get_client()`` facade — exactly the duck-typed surface each extractor
probes (the camunda extractor wants ``Api.v7``; the egeria/aris extractors want
the bare API facade). It never raises: a missing/unconfigured connector yields
``None`` so the caller can report "no client" instead of crashing.
"""

from __future__ import annotations

import logging
from typing import Any

from .registry import discover_extractors, get_source, write_batch

logger = logging.getLogger(__name__)


def materialize_source(
    backend: Any,
    category: str,
    client: Any,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Run the registered ``category`` extractor over ``client`` and persist it.

    Returns ``(nodes_written, edges_written)``. Reuses ``registry.get_source``
    (auto-discovering extractor modules if the registry is cold) and the single
    generic ``registry.write_batch`` writer, so this adds no source-specific
    persistence code. A ``None`` backend is a clean no-op — the extractor still
    runs (useful for a dry count) but nothing is written.
    """
    src = get_source(category)
    if src is None:
        discover_extractors()  # lazy-import extractor modules, then retry
        src = get_source(category)
    if src is None:
        raise ValueError(f"unknown source extractor category {category!r}")

    batch = src.extract({"client": client, **(config or {})})
    if backend is None:
        return (0, 0)
    # Stamp the shared provenance contract (source_system + domain) so materialized
    # sources are uniform with the hydration path — they route into their own
    # urn:source:<category> named graph on a SPARQL mirror.
    n, e = write_batch(backend, batch, source=category)
    logger.info("materialized source %s: %d nodes, %d edges", category, n, e)
    return n, e


# Connector package -> the in-process call that returns the duck-typed client
# the matching extractor consumes. Kept as a small table (not env-driven) so a
# new connector is one line; each ``get_client`` reads its own package's
# environment in its own repo (Configuration discipline lives there, not here).
# category → connector package whose ``auth.get_client()`` yields the vendor client.
# Adding a new in-process source is one line here (plus an extractor + sink).
_CLIENT_MODULES: dict[str, str] = {
    "aris": "aris_mcp",
    "egeria": "egeria_mcp",
    "ciso_assistant": "ciso_assistant_api",
    "nextcloud": "nextcloud_agent",
    "okta": "okta_agent",
    "keycloak": "keycloak_agent",
    "salesforce": "salesforce_agent",
    "ansible": "ansible_tower_mcp",
    "homeassistant": "home_assistant_agent",
    "technitium_dns": "technitium_dns_mcp",
    "caddy": "caddy_mcp",
    "uptime_kuma": "uptime_kuma_agent",
    "portainer": "portainer_agent",
    "kafka": "kafka_mcp",
    "lgtm": "lgtm_mcp",
    "twenty": "twenty_mcp",
    "archimate": "archimate_mcp",
    "wger": "wger_agent",
    "mealie": "mealie_mcp",
}


def resolve_source_client(category: str) -> Any | None:
    """Best-effort: build the vendor client for ``category`` in-process, or ``None``.

    Imports the connector package's ``auth.get_client()`` facade. Any failure
    (package absent, creds unset, import error) returns ``None`` rather than
    raising — the surface reports "no client for <category>" cleanly.
    """
    try:
        if category == "camunda":
            from camunda_mcp.auth import get_client

            return get_client().v7
        if category == "servicenow":
            from servicenow_api.auth import get_client

            from .source_adapters import ServiceNowSourceClient

            return ServiceNowSourceClient(get_client())
        if category == "erpnext":
            from erpnext_agent.auth import get_client

            from .source_adapters import ErpNextSourceClient

            return ErpNextSourceClient(get_client())
        if category == "emerald":
            from emerald_exchange.backends import TradingMode, create_backend

            backend = create_backend(name="paper", config={}, mode=TradingMode.PAPER)
            connect = getattr(backend, "connect", None)
            if callable(connect):
                connect()
            return backend
        if category == "microsoft":
            from microsoft_agent.auth import get_client

            from .source_adapters import MicrosoftGraphSourceClient, run_sync

            # get_client() is async (msgraph) — bridge to sync, then wrap.
            api = run_sync(lambda: get_client())
            return MicrosoftGraphSourceClient(api)
        module = _CLIENT_MODULES.get(category)
        if module:
            mod = __import__(f"{module}.auth", fromlist=["get_client"])
            return mod.get_client()
    except Exception as exc:  # noqa: BLE001 - missing/unconfigured connector → no client
        logger.debug("no source client for %s: %s", category, exc)
        return None
    return None


# Categories whose ingest substrate is the materialize path (extractor over an
# in-process vendor client + reasoning cycle), as opposed to the CAPABILITY_REGISTRY
# hydration path. Used by the unified ``source_sync`` entrypoint to route correctly.
MATERIALIZE_SOURCES: frozenset[str] = frozenset(
    {
        "camunda",
        "aris",
        "egeria",
        "ciso_assistant",
        "servicenow",
        "erpnext",
        "nextcloud",
        "okta",
        "keycloak",
        "salesforce",
        "ansible",
        "homeassistant",
        "technitium_dns",
        "caddy",
        "uptime_kuma",
        "portainer",
        "kafka",
        "lgtm",
        "twenty",
        "archimate",
        "wger",
        "mealie",
        "emerald",
        "microsoft",
    }
)


def run_materialize_source(
    engine: Any, category: str, *, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Materialize a source extractor + run one reasoning cycle (the shared core).

    The single implementation both ``graph_ingest action=materialize_source`` and
    the unified ``source_sync`` entrypoint call, so the two never drift. Resolves
    the in-process vendor client, persists the extractor batch, then extrapolates
    OWL inferences so the new structure folds into the cross-vendor crosswalk.
    """
    category = (category or "").strip()
    if not category:
        return {"status": "error", "error": "materialize requires a source category"}
    client = resolve_source_client(category)
    if client is None:
        return {
            "status": "skipped",
            "reason": f"no source client for {category!r} (connector absent or creds unset)",
            "source": category,
        }
    backend = getattr(engine, "backend", None)
    nodes, edges = materialize_source(backend, category, client, config=config)
    inferred = 0
    new_topics = 0
    try:
        from ..research.ara.reasoning_driver import OntologyReasoningDriver

        harvest = OntologyReasoningDriver(engine).extrapolate(persist=True)
        inferred = len(getattr(harvest, "inferred_edges", []) or [])
        new_topics = len(getattr(harvest, "new_topics", []) or [])
    except Exception:  # noqa: BLE001 - reasoning is best-effort post-persist
        logger.debug("reasoning cycle after materialize failed", exc_info=True)
    return {
        "status": "materialized",
        "source": category,
        "category": category,
        "nodes": nodes,
        "edges": edges,
        "inferred_edges": inferred,
        "new_topics": new_topics,
    }
