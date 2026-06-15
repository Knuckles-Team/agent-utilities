#!/usr/bin/python
"""Database environment provisioner — Stardog + pg-age, from credentials.

This module is the **single source of truth** behind the `setup-databases` CLI,
the ``graph_configure`` MCP action ``setup_databases``/``verify_databases``, and
the ``database-environment-setup`` skill. It does no graph work of its own — it
composes existing capabilities:

- **Postgres extension check** — :class:`PostgreSQLBackend` extension probes
  (``pggraph_available`` / ``pgvector_available`` / ``paradedb_available``).
- **Backend selection** — writes the existing ``GRAPH_DB_URI`` / ``GRAPH_PG_AGE``
  / ``GRAPH_BACKEND`` keys (no new env flags) so the graph durably lands in AGE.
- **Ontology distribution (KG-2.6)** — :class:`OntologyPublisher` push to Stardog
  (prod) or Jena Fuseki (dev), with the built-in ``/api/sparql`` endpoint already
  serving the dev case with zero infra.
- **Durable backfill (KG-2.7)** — :meth:`TieredGraphBackend.reconcile_to_durable`.

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
    """Merge ``values`` into config.json and the live process env.

    config.json keeps the choice across restarts (the gateway/daemon reads it at
    boot); the ``os.environ`` write makes it take effect for the current process
    (an env *write* for cross-process signalling, which configuration discipline
    permits). Keys are the canonical env var names (e.g. ``GRAPH_DB_URI``).
    """
    cfg_path = _config_json_path()
    data: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
        except Exception as exc:  # noqa: BLE001 — a corrupt file shouldn't block setup
            logger.warning("config.json unreadable (%s); recreating", exc)
            data = {}
    for key, val in values.items():
        data[key] = val
        os.environ[key] = val
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(data, indent=4))

    # Re-parse the typed config so in-process readers see the new values.
    try:
        from agent_utilities.core.config import config as _cfg

        _cfg.reload()
    except Exception as exc:  # noqa: BLE001 — reload is best-effort; env is already set
        logger.debug("config reload after persist failed: %s", exc)
    return str(cfg_path)


def _resolve_dsn(dsn: str | None) -> str:
    """The Postgres DSN, falling back to the existing GRAPH_DB_URI / PGGRAPH_DSN."""
    return (
        dsn
        or setting("GRAPH_DB_URI", "")
        or setting("PGGRAPH_DSN", "")
        or "postgresql://agent:agent@localhost:5432/agent_kg"
    )


# ──────────────────────────────────────────────────────────────────────────
# Step 1 — verify Postgres extensions
# ──────────────────────────────────────────────────────────────────────────
def verify_postgres(dsn: str | None = None) -> dict[str, Any]:
    """Probe a Postgres for the AGE / pgvector / pg_search extensions.

    Returns a report with per-extension availability and a ``ready`` flag (all
    three present). Connection failures are returned as ``status='error'`` rather
    than raised, so the caller can surface a clear remediation message.
    """
    resolved = _resolve_dsn(dsn)
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
            "dsn": _redact(resolved),
            "error": str(exc),
            "hint": "Check the DSN, that Postgres is reachable, and psycopg is installed.",
        }

    missing = [name for name in _REQUIRED_EXTENSIONS if not extensions[name]]
    report: dict[str, Any] = {
        "status": "success",
        "dsn": _redact(resolved),
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
    dsn: str | None = None,
    *,
    enable_age: bool = True,
    backend: str = "tiered",
    mirror_targets: list[str] | None = None,
) -> dict[str, Any]:
    """Point the durable tier at pg-age by persisting the existing graph keys.

    Sets ``GRAPH_DB_URI`` (the durable L3 DSN), ``GRAPH_PG_AGE`` (native
    openCypher on AGE) and ``GRAPH_BACKEND`` (``tiered`` by default — L1
    epistemic-graph working store + pg-age L3). Optionally records
    ``GRAPH_MIRROR_TARGETS`` for fanout (KG-2.74). The active backend is reset so
    the next :func:`create_backend` rebuilds against the new config; a running
    gateway/daemon applies it on restart.
    """
    resolved = _resolve_dsn(dsn)
    values: dict[str, str] = {
        "GRAPH_DB_URI": resolved,
        "GRAPH_BACKEND": backend,
    }
    if enable_age:
        values["GRAPH_PG_AGE"] = "1"
    if mirror_targets:
        values["GRAPH_MIRROR_TARGETS"] = json.dumps(mirror_targets)
        if backend == "tiered":
            # tiered + mirror targets tees durable writes through a fanout L3.
            values["GRAPH_BACKEND"] = "tiered"

    cfg_path = _persist_settings(values)

    # Force the next backend build to pick up the new selection.
    try:
        from agent_utilities.knowledge_graph.backends import set_active_backend

        set_active_backend(None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not reset active backend: %s", exc)

    return {
        "status": "success",
        "config_path": cfg_path,
        "applied": {**values, "GRAPH_DB_URI": _redact(resolved)},
        "note": "A running gateway/daemon picks this up on restart.",
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
    endpoint: str | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    """Register Stardog as a ``role="mirror"`` graph connection (CONCEPT:KG-2.89).

    Once registered, ``_build_mirror_set`` auto-includes it, so under
    ``GRAPH_BACKEND=tiered``/``fanout`` every KG write (incl. LeanIX/ServiceNow
    ingests) replicates into Stardog via the durable fan-out outbox. Credentials
    default to the existing ``STARDOG_*`` settings. The connection is persisted to
    config.json so it survives restart. Pair with :func:`backfill_to_age`'s
    reconcile to backfill the existing graph into a freshly added mirror.
    """
    spec: dict[str, Any] = {
        "backend": "stardog",
        "role": "mirror",
        "endpoint": endpoint or setting("STARDOG_ENDPOINT", "http://localhost:5820"),
        "database": database or setting("STARDOG_DATABASE", "agent_kg"),
        "user": username or setting("STARDOG_USER", "admin"),
        "password": password or setting("STARDOG_PASSWORD", "admin"),
    }
    try:
        from agent_utilities.core.config import save_config_item
        from agent_utilities.mcp import kg_server

        registry = kg_server.get_connection_registry()
        registered = registry.register(name, spec)
        save_config_item("kg_connections", registry.export_specs())
    except Exception as exc:  # noqa: BLE001 — surface a clear failure to the operator
        return {"status": "error", "error": str(exc), "connection": name}

    # Force the next backend build to include the new mirror.
    try:
        from agent_utilities.knowledge_graph.backends import set_active_backend

        set_active_backend(None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not reset active backend: %s", exc)

    return {
        "status": "success",
        "connection": registered,
        "role": "mirror",
        "endpoint": spec["endpoint"],
        "database": spec["database"],
        "persisted": True,
        "note": "KG writes now fan out to Stardog (GRAPH_BACKEND=tiered/fanout). "
        "Run 'reconcile' (or backfill_to_age) to backfill existing data.",
    }


# ──────────────────────────────────────────────────────────────────────────
# Step 4 — backfill the working graph into the durable AGE tier
# ──────────────────────────────────────────────────────────────────────────
def backfill_to_age() -> dict[str, Any]:
    """Reconcile the L1 working graph into the durable AGE tier (KG-2.7).

    Resolves a tiered backend (the active one, or a freshly built one if config
    now selects pg-age), runs the idempotent MERGE reconcile, and returns the
    drift report plus durability counters. ``nodes_missing == 0`` means the
    durable copy matches L1.
    """
    backend = _resolve_tiered_backend()
    if backend is None:
        return {
            "status": "error",
            "error": "No durable (tiered/pg-age) backend active. Run configure_backend first.",
        }
    try:
        summary = backend.reconcile_to_durable()
        stats = backend.durability_stats()
    except Exception as exc:  # noqa: BLE001 — surface a clear failure to the operator
        return {"status": "error", "error": str(exc)}
    return {
        "status": "success",
        "reconcile": summary,
        "durability": stats,
        "consistent": summary.get("nodes_missing", 0) == 0
        and summary.get("edges_missing", 0) == 0,
    }


def _resolve_tiered_backend() -> Any:
    """Find an active backend exposing ``reconcile_to_durable`` (unwrap proxies)."""
    from agent_utilities.knowledge_graph.backends import (
        create_backend,
        get_active_backend,
        set_active_backend,
    )

    backend = get_active_backend()
    if backend is None:
        try:
            backend = create_backend()
            set_active_backend(backend)
        except Exception as exc:  # noqa: BLE001
            logger.warning("create_backend during backfill failed: %s", exc)
            return None
    cand = getattr(backend, "inner", backend)  # unwrap a BrainGuarded proxy
    if hasattr(cand, "reconcile_to_durable"):
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
                return {"status": "error", "error": result["error"]}
            rows = len(result.get("results", {}).get("bindings", []))
            return {
                "status": "success",
                "kind": "builtin",
                "rows": rows,
                "url": "/api/sparql",
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}

    # HTTP triple stores (Stardog / Fuseki).
    try:
        import requests
    except ImportError:
        return {"status": "error", "error": "requests not installed."}

    if kind == "stardog":
        url = (endpoint or setting("STARDOG_ENDPOINT", "http://localhost:5820")).rstrip(
            "/"
        )
        db = database or setting("STARDOG_DATABASE", "agent_kg")
        query_url = f"{url}/{db}/query"
        auth = (
            setting("STARDOG_USER", "admin"),
            setting("STARDOG_PASSWORD", "admin"),
        )
    elif kind == "fuseki":
        url = (endpoint or setting("FUSEKI_ENDPOINT", "http://localhost:3030")).rstrip(
            "/"
        )
        query_url = f"{url}/{dataset}/query"
        auth = None
    else:
        return {"status": "error", "error": f"Unknown SPARQL kind: {kind}"}

    try:
        resp = requests.get(
            query_url,
            params={"query": q},
            headers={"Accept": "application/sparql-results+json"},
            auth=auth,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = len(data.get("results", {}).get("bindings", []))
        return {"status": "success", "kind": kind, "rows": rows, "url": query_url}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc), "url": query_url}


# ──────────────────────────────────────────────────────────────────────────
# Top-level driver — the one call the CLI / MCP / skill make
# ──────────────────────────────────────────────────────────────────────────
def setup_environment(
    profile: str = "dev",
    *,
    postgres_mode: str = "managed_image",
    dsn: str | None = None,
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
        dsn: Postgres DSN; falls back to ``GRAPH_DB_URI`` / ``PGGRAPH_DSN``.
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
    pg = verify_postgres(dsn)
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
        dsn,
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


def _redact(dsn: str) -> str:
    """Hide the password in a DSN for safe logging/reporting."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    creds, _, host = rest.partition("@")
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return dsn
