#!/usr/bin/python
"""Database environment provisioner — Stardog + pg-age, from credentials.

This module is the **single source of truth** behind the `setup-databases` CLI,
the ``graph_configure`` MCP action ``setup_databases``/``verify_databases``, and
the ``database-environment-setup`` skill. It does no graph work of its own — it
composes existing capabilities:

- **Postgres extension check** — :class:`PostgreSQLBackend` extension probes
  (``pggraph_available`` / ``pgvector_available`` / ``paradedb_available``).
- **Projection setup** — persists the runtime connection-profile reference,
  native AGE mode, and the external mirror declaration.
- **Ontology distribution (KG-2.6)** — :class:`OntologyPublisher` push to Stardog
  (prod) or Jena Fuseki (dev), with the built-in ``/api/sparql`` endpoint already
  serving the dev case with zero infra.
- **Durable backfill (KG-2.7)** — authority-to-mirror reconciliation.

Two environment shapes are supported:

- ``profile="prod"`` — push the ontology to **Stardog** and consume via Stardog's
  SPARQL endpoint.
- ``profile="dev"`` — host SPARQL **locally** (built-in ``/api/sparql`` by default,
  optional Jena Fuseki) with no Stardog.

And two Postgres modes (an operator may use both across environments):

- ``postgres_mode="managed_image"`` — a Postgres we control (the combined
  ``docker/pg-age-full`` image) where AGE + pgvector + pg_search are guaranteed.
- ``postgres_mode="existing"`` — an externally-managed Postgres we only connect to;
  extensions that need superuser + ``shared_preload_libraries`` may be absent, which
  this module reports honestly rather than failing silently.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# The three extensions a "full" pg-age tier carries.
_REQUIRED_EXTENSIONS = ("age", "vector", "pg_search")


# ──────────────────────────────────────────────────────────────────────────
# Config persistence — write the existing keys; never invent a new flag.
# ──────────────────────────────────────────────────────────────────────────
def _config_json_path() -> Path:
    """Resolve the XDG ``config.json`` agent-utilities reads at startup."""
    override = setting("AGENT_UTILITIES_CONFIG_DIR", "")
    if override:
        cfg_dir = Path(override).expanduser()
    else:
        import platformdirs

        cfg_dir = Path(
            platformdirs.user_config_path("agent-utilities", "knuckles-team")
        )
    return cfg_dir / "config.json"


def _persist_settings(values: dict[str, str]) -> str:
    """Apply values live and persist only non-secret selection settings.

    Credentials and connection strings are runtime secret material. They are
    never copied into ``config.json``; the deployment must reinject them after a
    restart. Non-secret mode and mirror-selection keys remain durable.
    """
    cfg_path = _config_json_path()
    data: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
        except Exception as exc:  # noqa: BLE001 — a corrupt file shouldn't block setup
            logger.warning(
                "config.json unreadable (%s); recreating", type(exc).__name__
            )
            data = {}
    secret_keys = {"STATE_DB_URI"}
    for key in secret_keys:
        data.pop(key, None)
    for key, val in values.items():
        os.environ[key] = val
        if key not in secret_keys:
            data[key] = val
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(data, indent=4))

    # Re-parse the typed config so in-process readers see the new values.
    try:
        from agent_utilities.core.config import config as _cfg

        _cfg.reload()
    except Exception as exc:  # noqa: BLE001 — reload is best-effort; env is already set
        logger.debug("config reload after persist failed: %s", type(exc).__name__)
    return "xdg://agent-utilities/config.json"


def _resolve_dsn(connection_profile_ref: str | None) -> str:
    """Resolve a required Postgres DSN from the runtime connection profile."""

    reference = connection_profile_ref or setting(
        "GRAPH_DB_CONNECTION_PROFILE_REF", ""
    )
    if not reference:
        raise ValueError(
            "Postgres connection profile is not configured; inject "
            "GRAPH_DB_CONNECTION_PROFILE_REF"
        )
    from agent_utilities.knowledge_graph.backends import _resolve_connection_profile

    profile = _resolve_connection_profile(str(reference))
    resolved = str(profile.get("uri") or "")
    if not resolved:
        raise ValueError("graph connection profile does not contain a Postgres URI")
    return resolved


# ──────────────────────────────────────────────────────────────────────────
# Step 1 — verify Postgres extensions
# ──────────────────────────────────────────────────────────────────────────
def verify_postgres(connection_profile_ref: str | None = None) -> dict[str, Any]:
    """Probe a Postgres for the AGE / pgvector / pg_search extensions.

    Returns a report with per-extension availability and a ``ready`` flag (all
    three present). Connection failures are returned as ``status='error'`` rather
    than raised, so the caller can surface a clear remediation message.
    """
    try:
        resolved = _resolve_dsn(connection_profile_ref)
    except ValueError:
        return {
            "status": "error",
            "dsn_configured": False,
            "error": "Postgres DSN is not configured",
            "hint": "Inject GRAPH_DB_CONNECTION_PROFILE_REF.",
        }
    try:
        from agent_utilities.knowledge_graph.backends.postgresql_backend import (
            PostgreSQLBackend,
        )

        backend = PostgreSQLBackend(dsn=resolved)
        extensions = {
            "age": bool(backend.pggraph_available),
            "vector": bool(backend.pgvector_available),
            "pg_search": bool(backend.paradedb_available),
        }
    except Exception as exc:  # noqa: BLE001 — connection/driver problems are expected
        return {
            "status": "error",
            "dsn_configured": True,
            "error": "Postgres verification failed",
            "error_type": type(exc).__name__,
            "hint": "Check the DSN, that Postgres is reachable, and psycopg is installed.",
        }

    missing = [name for name in _REQUIRED_EXTENSIONS if not extensions[name]]
    report: dict[str, Any] = {
        "status": "success",
        "dsn_configured": True,
        "extensions": extensions,
        "ready": not missing,
        "missing": missing,
    }
    if missing:
        report["hint"] = (
            "Missing "
            + ", ".join(missing)
            + ". 'age'/'pg_search' need superuser + shared_preload_libraries — use the "
            "combined docker/pg-age-full image, or point AGE/full-text at a Postgres "
            "you control."
        )
    return report


# ──────────────────────────────────────────────────────────────────────────
# Step 2 — wire the backend so writes backfill into AGE
# ──────────────────────────────────────────────────────────────────────────
def configure_backend(
    connection_profile_ref: str | None = None,
    *,
    enable_age: bool = True,
    mirror_targets: list[str] | None = None,
) -> dict[str, Any]:
    """Wire pg-age as a projection of the fixed engine authority.

    Persists ``GRAPH_DB_CONNECTION_PROFILE_REF``, ``GRAPH_PG_AGE`` (native
    openCypher on AGE), and ``GRAPH_MIRROR_TARGETS`` (KG-2.74). A non-empty
    mirror set automatically enables projection fan-out; there is no operational
    backend or authority selector. The active backend is reset so the next
    :func:`create_backend` rebuilds against the new config.
    """
    try:
        _resolve_dsn(connection_profile_ref)
    except ValueError:
        return {
            "status": "error",
            "dsn_configured": False,
            "error": "Postgres DSN is not configured",
        }
    values: dict[str, str] = {
        "GRAPH_DB_CONNECTION_PROFILE_REF": str(
            connection_profile_ref
            or setting("GRAPH_DB_CONNECTION_PROFILE_REF", "")
        ),
    }
    if enable_age:
        values["GRAPH_PG_AGE"] = "1"
    # Default to pg-age so the resolved connection is actually projected to.
    if not mirror_targets:
        mirror_targets = ["age"]
    if mirror_targets:
        values["GRAPH_MIRROR_TARGETS"] = json.dumps(mirror_targets)

    cfg_path = _persist_settings(values)

    # Force the next backend build to pick up the new selection.
    try:
        from agent_utilities.knowledge_graph.backends import set_active_backend

        set_active_backend(None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not reset active backend: %s", type(exc).__name__)

    return {
        "status": "success",
        "config_path": cfg_path,
        "applied_settings": sorted(values),
        "dsn_configured": True,
        "note": "The connection profile is resolved only at runtime.",
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 3 — publish the ontology to the chosen SPARQL host
# ──────────────────────────────────────────────────────────────────────────
def publish_ontology(
    target: str = "builtin",
    *,
    endpoint: str | None = None,
    database: str | None = None,
    dataset: str = "agent_kg",
    named_graph: str | None = None,
) -> dict[str, Any]:
    """Distribute the bundled ontology to the SPARQL host (KG-2.6).

    ``target``:
      - ``"stardog"`` — push to Stardog (prod). Endpoint/credentials default to the
        existing ``STARDOG_*`` settings.
      - ``"fuseki"`` — push to a local Apache Jena Fuseki triple store (dev upgrade).
      - ``"builtin"`` — no push needed; the gateway already serves the live graph at
        ``/api/sparql`` (zero infra). Returns the triple count for confirmation.
    """
    from agent_utilities.knowledge_graph.core.ontology_publisher import (
        OntologyPublisher,
        collect_bundled_ontology_graph,
    )

    try:
        graph = collect_bundled_ontology_graph()
    except ImportError:
        return {
            "status": "error",
            "error": "rdflib not installed (pip install agent-utilities[owl]).",
        }

    triple_count = len(graph)
    publisher = OntologyPublisher()

    if target == "stardog":
        result = publisher.push_to_stardog(
            graph,
            endpoint=endpoint,
            database=database,
            named_graph=named_graph,
        )
        result.setdefault("target", "stardog")
        return result
    if target == "fuseki":
        result = publisher.push_to_jena_fuseki(
            graph, endpoint=endpoint, dataset=dataset, named_graph=named_graph
        )
        result.setdefault("target", "fuseki")
        return result
    # builtin — nothing to push; the endpoint materializes from the live graph.
    return {
        "status": "success",
        "target": "builtin",
        "triple_count": triple_count,
        "note": "Consume the live ontology at the gateway's GET/POST /api/sparql.",
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 3b — register Stardog as a live DATA mirror (instance data, not just TBox)
# ──────────────────────────────────────────────────────────────────────────
def register_stardog_mirror(
    name: str = "stardog",
    *,
    endpoint_ref: str = "env://STARDOG_ENDPOINT",
    database_ref: str = "env://STARDOG_DATABASE",
    username_ref: str = "env://STARDOG_USER",
    password_ref: str = "env://STARDOG_PASSWORD",
) -> dict[str, Any]:
    """Register Stardog as a ``role="mirror"`` graph connection (CONCEPT:AU-KG.backend.connection-registry).

    Once registered, ``_build_mirror_set`` auto-includes it, so every governed KG
    write replicates into Stardog via the durable fan-out outbox. Endpoint,
    database, and identity material remain
    behind runtime secret references; no literal connection material crosses into
    ``config.json``. Pair with :func:`backfill_to_age`'s reconcile to backfill the
    existing graph into a freshly added mirror.
    """
    spec: dict[str, Any] = {
        "backend": "stardog",
        "role": "mirror",
        "endpoint": endpoint_ref,
        "database": database_ref,
        "user": username_ref,
        "password": password_ref,
    }
    try:
        from agent_utilities.core.config import save_config_item
        from agent_utilities.mcp import kg_server

        registry = kg_server.get_connection_registry()
        registered = registry.register(name, spec)
        save_config_item("kg_connections", registry.export_specs())
    except Exception as exc:  # noqa: BLE001 — source-safe diagnostic only
        return {
            "status": "error",
            "error": "mirror registration failed",
            "error_type": type(exc).__name__,
        }

    # Force the next backend build to include the new mirror.
    try:
        from agent_utilities.knowledge_graph.backends import set_active_backend

        set_active_backend(None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not reset active backend: %s", type(exc).__name__)

    return {
        "status": "success",
        "connection": registered,
        "role": "mirror",
        "persisted": True,
        "note": "KG writes now fan out to the Stardog projection. "
        "Run 'reconcile' (or backfill_to_age) to backfill existing data.",
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 4 — backfill the engine authority into the pg-age mirror
# ──────────────────────────────────────────────────────────────────────────
def backfill_to_age() -> dict[str, Any]:
    """Reconcile the engine authority into the pg-age mirror (KG-2.7).

    Resolves a fanout backend (the active one, or a freshly built one if config
    now selects pg-age), runs the idempotent MERGE reconcile, and returns the
    drift report plus mirror counters. ``nodes_missing == 0`` means the mirror
    matches the engine authority.
    """
    backend = _resolve_reconcilable_backend()
    if backend is None:
        return {
            "status": "error",
            "error": "No fanout (engine + pg-age mirror) backend active. Run configure_backend first.",
        }
    try:
        summary = backend.reconcile()
        stats = backend.durability_stats()
    except Exception as exc:  # noqa: BLE001 — source-safe diagnostic only
        return {
            "status": "error",
            "error": "mirror reconciliation failed",
            "error_type": type(exc).__name__,
        }
    return {
        "status": "success",
        "reconcile": summary,
        "durability": stats,
        "consistent": bool(summary)
        and all(
            isinstance(report, dict)
            and "error" not in report
            and report.get("nodes_missing", 0) == 0
            and report.get("edges_missing", 0) == 0
            and report.get("errors", 0) == 0
            for report in summary.values()
        ),
    }


def _resolve_reconcilable_backend() -> Any:
    """Resolve the active ``FanOutBackend`` after unwrapping policy proxies."""
    from agent_utilities.knowledge_graph.backends import (
        create_backend,
        get_active_backend,
        set_active_backend,
    )
    from agent_utilities.knowledge_graph.backends.fanout_backend import FanOutBackend

    backend = get_active_backend()
    if backend is None:
        try:
            backend = create_backend()
            set_active_backend(backend)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "create_backend during backfill failed: %s", type(exc).__name__
            )
            return None
    cand = getattr(backend, "inner", backend)  # unwrap a BrainGuarded proxy
    if isinstance(cand, FanOutBackend):
        return cand
    return None


# ──────────────────────────────────────────────────────────────────────────
# Step 5 — prove SPARQL consumption end-to-end
# ──────────────────────────────────────────────────────────────────────────
_SMOKE_QUERY = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"


def verify_sparql(
    kind: str = "builtin",
    *,
    endpoint: str | None = None,
    database: str | None = None,
    dataset: str = "agent_kg",
    query: str | None = None,
) -> dict[str, Any]:
    """Run a smoke ``SELECT`` against the chosen SPARQL host to prove consumption.

    ``kind`` is ``"builtin"`` (in-process gateway endpoint), ``"stardog"`` or
    ``"fuseki"`` (HTTP). Returns ``status`` and ``rows`` (or an error).
    """
    q = query or _SMOKE_QUERY

    if kind == "builtin":
        try:
            from agent_utilities.gateway.graph_api import _get_sparql_bridge
            from agent_utilities.knowledge_graph.core.sparql_http import SPARQLEndpoint

            bridge = _get_sparql_bridge()
            if bridge is None:
                return {
                    "status": "error",
                    "error": "SPARQL bridge unavailable (need agent-utilities[owl]).",
                }
            result = SPARQLEndpoint(bridge).execute(q)
            if "error" in result:
                return {"status": "error", "error": "SPARQL query failed"}
            rows = len(result.get("results", {}).get("bindings", []))
            return {
                "status": "success",
                "kind": "builtin",
                "rows": rows,
                "url": "/api/sparql",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": "built-in SPARQL verification failed",
                "error_type": type(exc).__name__,
            }

    if kind == "stardog":
        url = str(endpoint or setting("STARDOG_ENDPOINT", "") or "").rstrip("/")
        username = str(setting("STARDOG_USER", "") or "")
        password = str(setting("STARDOG_PASSWORD", "") or "")
        if not url or not username or not password:
            return {
                "status": "error",
                "error": "Stardog endpoint and credentials are not configured",
            }
        db = database or setting("STARDOG_DATABASE", "agent_kg")
        query_url = f"{url}/{db}/query"
        auth = (username, password)
    elif kind == "fuseki":
        if endpoint is None:
            from agent_utilities.core.config import config as _cfg

            endpoint = _cfg.kg_fuseki_endpoint
        if not endpoint:
            return {"status": "error", "error": "Fuseki endpoint is not configured"}
        url = endpoint.rstrip("/")
        query_url = f"{url}/{dataset}/query"
        auth = None
    else:
        return {"status": "error", "error": f"Unknown SPARQL kind: {kind}"}

    try:
        from agent_utilities.core.http_client import create_http_client
        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )

        trust = resolve_configured_tls_profile(kind)
        try:
            with create_http_client(
                timeout=30,
                headers={"Accept": "application/sparql-results+json"},
                auth=auth,
                **trust.httpx_kwargs(),
            ) as client:
                resp = client.get(query_url, params={"query": q})
                resp.raise_for_status()
                data = resp.json()
        finally:
            trust.cleanup()
        rows = len(data.get("results", {}).get("bindings", []))
        return {"status": "success", "kind": kind, "rows": rows}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": "SPARQL verification failed",
            "error_type": type(exc).__name__,
        }


# ──────────────────────────────────────────────────────────────────────────
# Top-level driver — the one call the CLI / MCP / skill make
# ──────────────────────────────────────────────────────────────────────────
def setup_environment(
    profile: str = "dev",
    *,
    postgres_mode: str = "managed_image",
    connection_profile_ref: str | None = None,
    sparql_target: str | None = None,
    mirror_targets: list[str] | None = None,
    do_backfill: bool = True,
    mirror_data_to_stardog: bool | None = None,
) -> dict[str, Any]:
    """Provision a complete database environment and return a step-by-step report.

    Args:
        profile: ``"prod"`` (Stardog) or ``"dev"`` (local SPARQL).
        postgres_mode: ``"managed_image"`` (combined pg-age-full image) or
            ``"existing"`` (connect-only; report missing extensions honestly).
        connection_profile_ref: Runtime secret reference for the Postgres profile.
        sparql_target: override the publish/verify host
            (``stardog``/``fuseki``/``builtin``); defaults from ``profile``.
        mirror_targets: optional fanout mirror connection names (KG-2.74).
        do_backfill: run the durable backfill after wiring (default on).
        mirror_data_to_stardog: register Stardog as a live data mirror so instance
            data (not just the TBox) replicates continuously. Defaults to ON for the
            Stardog (prod) target; set False to publish only the ontology.

    The driver never raises on a sub-step failure — each step's report carries its
    own ``status`` so the operator sees exactly where to intervene.
    """
    target = sparql_target or ("stardog" if profile == "prod" else "builtin")
    report: dict[str, Any] = {
        "profile": profile,
        "postgres_mode": postgres_mode,
        "sparql_target": target,
        "steps": {},
    }

    # 1. Postgres extensions.
    pg = verify_postgres(connection_profile_ref)
    report["steps"]["verify_postgres"] = pg
    if postgres_mode == "existing" and pg.get("missing"):
        report["warnings"] = [
            "Existing Postgres is missing "
            + ", ".join(pg["missing"])
            + " — graph backfill into AGE / BM25 search will be unavailable until "
            "installed (needs superuser + shared_preload_libraries)."
        ]

    # 2. Backend wiring (only meaningful when AGE is present, but record the choice).
    report["steps"]["configure_backend"] = configure_backend(
        connection_profile_ref,
        enable_age=pg.get("extensions", {}).get("age", True),
        mirror_targets=mirror_targets,
    )

    # 3. Ontology distribution (TBox).
    report["steps"]["publish_ontology"] = publish_ontology(target)

    # 3b. Live instance-data mirror into Stardog (default on for the Stardog target).
    if mirror_data_to_stardog is None:
        mirror_data_to_stardog = target == "stardog"
    if mirror_data_to_stardog:
        report["steps"]["register_stardog_mirror"] = register_stardog_mirror()

    # 4. Durable backfill (also backfills a freshly registered Stardog mirror).
    if do_backfill:
        report["steps"]["backfill_to_age"] = backfill_to_age()

    # 5. Consumption smoke test.
    report["steps"]["verify_sparql"] = verify_sparql(target)

    report["status"] = (
        "success"
        if all(
            s.get("status") == "success"
            for s in report["steps"].values()
            if isinstance(s, dict)
        )
        else "partial"
    )
    return report
