#!/usr/bin/python
"""Graph Database Backends.

Provides the `GraphBackend` ABC and concrete primary implementations
(EpistemicGraph/file, PostgreSQL + pgvector). Demoted "contrib" backends
(LadybugDB, FalkorDB, Neo4j) live under ``backends.contrib`` and are opt-in:
they are never imported eagerly, so default backend selection works without
their optional driver packages. Use `create_backend()` to instantiate the
correct backend from configuration or environment variables.

Architecture (Tiered Graph Engine):
    The engine uses a two-tier architecture:
    - **Tier 1 (Source of Truth)**: A persistent Cypher-capable backend.
      PostgreSQL + pgvector is the **primary/default** durable tier; the
      Rust-native EpistemicGraph (``file``/``memory``) is the zero-config
      tier. Demoted contrib backends (LadybugDB/Neo4j/FalkorDB) remain fully
      supported via opt-in import.
    - **Tier 2 (Compute Scratchpad)**: GraphComputeEngine is loaded on-demand via
      ``load_subgraph()`` for graph algorithms (PageRank, VF2, spectral
      clustering) that databases cannot perform natively.

    The ``memory``/``file`` backends (Rust-native EpistemicGraph) are available
    for testing/CI and edge use where no external server is needed. ``file``
    matches the ``GRAPH_PERSISTENCE_TYPE=file`` config default.

Opt-in (contrib) access::

    from agent_utilities.knowledge_graph.backends.contrib.neo4j_backend import (
        Neo4jBackend,
    )
    # Back-compat shim — old import path still resolves lazily:
    from agent_utilities.knowledge_graph.backends import Neo4jBackend

Environment Variables:
    GRAPH_BACKEND: Backend type. Bare default: "tiered" — L1 epistemic_graph +
        L2 LadybugDB (embedded, no external server). Supported: "tiered",
        "memory", "file", "epistemic_graph", "postgresql" (primary), plus
        opt-in contrib: "ladybug", "falkordb", "neo4j".
    GRAPH_BACKEND_L1: L1 working store for "tiered". Default: "epistemic_graph".
    GRAPH_BACKEND_L2: L2 durable store for "tiered". Default: "ladybug"
        (or "postgresql" when a DB URI is configured).
    GRAPH_DB_PATH: File path for LadybugDB. Default: "knowledge_graph.db".
    GRAPH_DB_HOST: Host for FalkorDB/Neo4j. Default: "localhost".
    GRAPH_DB_PORT: Port for FalkorDB (6379) or Neo4j (7687).
    GRAPH_DB_URI:  Full URI for Neo4j or PostgreSQL.
    GRAPH_DB_USER: Username for Neo4j/PostgreSQL. Default: "neo4j".
    GRAPH_DB_PASSWORD: Password for Neo4j/PostgreSQL. Default: "password".
    GRAPH_DB_NAME: Database name for FalkorDB/PostgreSQL. Default: "agent_graph".
"""

import logging
import os
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Postgres connection-pool sizing (config discipline): sensible bounded defaults
# that work everywhere — a named constant, not a per-deploy env knob.
_PG_POOL_MIN = 2
_PG_POOL_MAX = 10

_ACTIVE_BACKEND: Any = None

# Sentinel parked in ``_ACTIVE_BACKEND`` while a composite backend (fanout) builds
# its members, so a recursively-built member never claims the global active slot —
# the composite itself claims it once, at the end of the outer call.
_BUILDING: Any = object()

__all__ = [
    "GraphBackend",
    "EpistemicGraphBackend",
    "LadybugBackend",
    "FalkorDBBackend",
    "Neo4jBackend",
    "PostgreSQLBackend",
    "JenaFusekiBackend",
    "LADYBUG_AVAILABLE",
    "create_backend",
    "get_active_backend",
    "set_active_backend",
]


def __getattr__(name: str):
    if name == "GraphBackend":
        from .base import GraphBackend

        return GraphBackend
    if name == "FalkorDBBackend":
        from .contrib.falkordb_backend import FalkorDBBackend

        return FalkorDBBackend
    if name in ("LadybugBackend", "LADYBUG_AVAILABLE"):
        from .contrib.ladybug_backend import LADYBUG_AVAILABLE, LadybugBackend

        if name == "LadybugBackend":
            return LadybugBackend
        return LADYBUG_AVAILABLE
    if name == "EpistemicGraphBackend":
        from .epistemic_graph_backend import EpistemicGraphBackend

        return EpistemicGraphBackend
    if name == "Neo4jBackend":
        from .contrib.neo4j_backend import Neo4jBackend

        return Neo4jBackend
    if name == "PostgreSQLBackend":
        from .postgresql_backend import PostgreSQLBackend

        return PostgreSQLBackend
    if name == "JenaFusekiBackend":
        from .sparql.jena_fuseki_backend import JenaFusekiBackend

        return JenaFusekiBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_member(spec: dict[str, Any]):
    """Build a composite member backend WITHOUT claiming the active slot.

    Used by the ``fanout`` composite (CONCEPT:KG-2.74): a member is a full
    ``create_backend`` call (so it gets schema init + driver resolution), but it
    must not register itself as ``_ACTIVE_BACKEND`` — only the composite does, at
    the end of the outer call. We park the ``_BUILDING`` sentinel in the slot so
    the inner call's ``is None`` claim check is a no-op, then restore.
    """
    global _ACTIVE_BACKEND
    saved = _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = _BUILDING
    try:
        return create_backend(**spec)
    finally:
        _ACTIVE_BACKEND = saved


def _build_mirror_set(skip_names: tuple[str, ...] = ()) -> dict[str, Any]:
    """Build ``{name: backend}`` for ``GRAPH_MIRROR_TARGETS`` (CONCEPT:KG-2.74),
    resolved against ``kg_connections``. Returns ``{}`` when none are configured.

    Shared by the ``fanout`` backend and the ``tiered`` durable tier (where it
    tees every write that lands in the L3 authority — e.g. pg-age — out to the
    named mirrors like neo4j / falkordb).
    """
    from agent_utilities.core.config import config as _cfg

    targets = setting("GRAPH_MIRROR_TARGETS") or _cfg.graph_mirror_targets or []
    if isinstance(targets, str):
        # config.json injects list values into the env as a JSON string; also
        # accept a plain comma list. (A naive comma-split would mangle the JSON.)
        s = targets.strip()
        if s.startswith("["):
            import json as _json

            try:
                parsed = _json.loads(s)
                targets = parsed if isinstance(parsed, list) else []
            except Exception:
                targets = []
        else:
            targets = [t.strip() for t in s.split(",") if t.strip()]
    if not targets:
        return {}
    conn_specs: dict[str, dict[str, Any]] = {}
    for spec in _cfg.kg_connections or []:
        d = dict(spec)
        nm = str(d.pop("name", "")).strip()
        if "backend" in d and "backend_type" not in d:
            d["backend_type"] = d.pop("backend")
        if nm:
            conn_specs[nm] = d
    mirrors: dict[str, Any] = {}
    for name in targets:
        if name in skip_names:
            continue
        member = _build_member(dict(conn_specs.get(name) or {"backend_type": name}))
        if member is not None:
            mirrors[name] = member
        else:
            logger.warning(
                "mirror '%s' unavailable (missing driver / unreachable); skipping.",
                name,
            )
    return mirrors


def get_active_backend():
    """Retrieve the currently active graph backend instance."""
    return _ACTIVE_BACKEND


def set_active_backend(backend):
    """Explicitly set the active graph backend instance."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def create_backend(
    backend_type: str | None = None,
    db_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    **kwargs,
):
    """Factory function to create the appropriate graph backend.

    Resolves configuration from explicit arguments first, then falls back to
    environment variables, then to sensible defaults. The bare default is the
    self-contained ``tiered`` backend (L1 epistemic_graph + L2 LadybugDB) — no
    external server required. PostgreSQL + pgvector is the durable production
    tier and is selected automatically for ``tiered`` whenever a DB URI is
    configured. Contrib backends (ladybug/falkordb/neo4j) are imported only
    when explicitly requested.

    Args:
        backend_type: One of "tiered" (default), "memory", "file",
            "epistemic_graph", "postgresql" (primary), or the opt-in contrib
            values "ladybug", "falkordb", "neo4j". Falls back to
            ``GRAPH_BACKEND`` env var, then "tiered" (zero-infra: epistemic_graph
            + LadybugDB; configure GRAPH_DB_URI for a PostgreSQL L2 in prod).
        db_path: File path for LadybugDB. Falls back to ``GRAPH_DB_PATH``.
        host: Host for FalkorDB/Neo4j. Falls back to ``GRAPH_DB_HOST``.
        port: Port for FalkorDB/Neo4j. Falls back to ``GRAPH_DB_PORT``.
        uri: Full URI for Neo4j/PostgreSQL. Falls back to ``GRAPH_DB_URI``.
        user: Username for Neo4j/PostgreSQL. Falls back to ``GRAPH_DB_USER``.
        password: Password for Neo4j/PostgreSQL. Falls back to ``GRAPH_DB_PASSWORD``.
        db_name: Database name for FalkorDB/PostgreSQL. Falls back to ``GRAPH_DB_NAME``.

    Returns:
        A configured ``GraphBackend`` instance, or ``None`` if the requested
        backend is not available (e.g., ladybug package not installed).
    """
    global _ACTIVE_BACKEND

    # Bare fallback is the self-contained "tiered" backend: L1 epistemic_graph
    # (always included) + L2 LadybugDB (embedded, no server). This runs as a
    # single binary with NO external system dependencies. PostgreSQL stays the
    # PRODUCTION durable tier and is selected automatically whenever a DB URI is
    # configured (GRAPH_DB_URI/PGGRAPH_DSN) or explicitly via
    # GRAPH_BACKEND_L2=postgresql; the prod-profile guard enforces it for prod.
    # The unit suite pins GRAPH_BACKEND=memory (see tests/conftest.py) to stay
    # purely ephemeral.
    backend_type = (
        (backend_type or setting("GRAPH_BACKEND") or "tiered").lower().strip()
    )

    from .base import GraphBackend

    backend: GraphBackend | None = None

    # Primary / default tiers — resolved WITHOUT importing any contrib backend.
    # "memory"/"file"/"epistemic_graph" all map to the zero-dependency,
    # Rust-native EpistemicGraphBackend (the config default ``GRAPH_PERSISTENCE_TYPE=file``
    # resolves here). For "file", an optional JSON path enables persistence.
    if backend_type in ("memory", "file", "epistemic_graph"):
        from .epistemic_graph_backend import EpistemicGraphBackend

        backend = EpistemicGraphBackend()
        if backend_type == "file":
            resolved_path = (
                db_path or setting("GRAPH_DB_PATH") or kwargs.get("json_path")
            )
            if resolved_path and os.path.exists(resolved_path):
                try:
                    backend.load_from_json(resolved_path)
                except Exception as e:  # pragma: no cover - best-effort load
                    logger.debug(
                        f"Failed to load epistemic graph from {resolved_path}: {e}"
                    )

    elif backend_type == "ladybug":
        from .contrib.ladybug_backend import LADYBUG_AVAILABLE, LadybugBackend

        if not LADYBUG_AVAILABLE:
            logger.warning(
                "LadybugDB requested but 'ladybug' package is not installed."
            )
            return None
        # Use centralized XDG-aware path resolver
        if db_path:
            resolved_path = db_path
        elif setting("GRAPH_DB_PATH"):
            resolved_path = setting("GRAPH_DB_PATH")
        else:
            from agent_utilities.core.paths import kg_db_path

            resolved = kg_db_path()
            # Ensure parent directory exists for XDG paths
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved_path = str(resolved)
        backend = LadybugBackend(resolved_path)

    elif backend_type == "falkordb":
        from .contrib.falkordb_backend import FalkorDBBackend

        resolved_host = host or setting("GRAPH_DB_HOST") or "localhost"
        resolved_port = port or setting("GRAPH_DB_PORT", 6379)
        resolved_name = db_name or setting("GRAPH_DB_NAME") or "agent_graph"
        backend = FalkorDBBackend(
            host=resolved_host, port=resolved_port, db_name=resolved_name
        )

    elif backend_type == "neo4j":
        from .contrib.neo4j_backend import Neo4jBackend

        resolved_uri = uri or setting("GRAPH_DB_URI") or "bolt://localhost:7687"
        resolved_user = user or setting("GRAPH_DB_USER") or "neo4j"
        resolved_password = password or setting("GRAPH_DB_PASSWORD") or "password"
        backend = Neo4jBackend(
            uri=resolved_uri, user=resolved_user, password=resolved_password
        )

    elif backend_type in ("postgresql", "age", "pggraph_age"):
        # GRAPH_PG_AGE=1 (or backend_type age/pggraph_age) selects the Apache AGE
        # backend — real openCypher-on-Postgres — over the regex-transpiler
        # PostgreSQLBackend. Both share the same DSN/pool config.
        _use_age = backend_type in ("age", "pggraph_age") or setting(
            "GRAPH_PG_AGE", ""
        ).lower() in ("1", "true", "yes")
        from .postgresql_backend import PostgreSQLBackend

        _PGBackend: type[PostgreSQLBackend]
        if _use_age:
            from .age_backend import AGEBackend

            _PGBackend = AGEBackend  # AGEBackend subclasses PostgreSQLBackend
        else:
            _PGBackend = PostgreSQLBackend

        resolved_uri = (
            uri
            or setting("GRAPH_DB_URI")
            or "postgresql://localhost:5432/agent_utilities"
        )
        resolved_name = db_name or setting("GRAPH_DB_NAME") or "agent_graph"
        pool_min = _PG_POOL_MIN
        pool_max = _PG_POOL_MAX
        pggraph_schema = setting("GRAPH_PGGRAPH_SCHEMA", "public")
        backend = _PGBackend(
            dsn=resolved_uri,
            graph_name=resolved_name,
            pool_min=pool_min,
            pool_max=pool_max,
            pggraph_schema=pggraph_schema,
        )

    elif backend_type == "jena_fuseki":
        from .sparql.jena_fuseki_backend import JenaFusekiBackend

        resolved_url = (
            kwargs.get("jena_fuseki_url")
            or setting("GRAPH_FUSEKI_URL")
            or "http://localhost:3030"
        )
        resolved_dataset = (
            kwargs.get("dataset") or setting("GRAPH_FUSEKI_DATASET") or "agent_kg"
        )
        resolved_jena_fuseki_user = kwargs.get("username") or setting(
            "GRAPH_FUSEKI_USER"
        )
        resolved_jena_fuseki_password = kwargs.get("password") or setting(
            "GRAPH_FUSEKI_PASSWORD"
        )
        backend = JenaFusekiBackend(
            jena_fuseki_url=resolved_url,
            dataset=resolved_dataset,
            username=resolved_jena_fuseki_user,
            password=resolved_jena_fuseki_password,
        )

    elif backend_type == "fanout":
        # Concurrent N-way mirroring (CONCEPT:KG-2.74): ONE authority store serves
        # reads + acks writes; every mutation is mirrored, losslessly, to the named
        # mirror connections via a durable outbox. Authority + mirrors are resolved
        # against kg_connections (CONCEPT:KG-2.63) so DSN/creds live in one place.
        from agent_utilities.core.config import config as _cfg

        from .fanout_backend import FanOutBackend

        conn_specs: dict[str, dict[str, Any]] = {}
        for spec in _cfg.kg_connections or []:
            d = dict(spec)
            nm = str(d.pop("name", "")).strip()
            # kg_connections (CONCEPT:KG-2.63) uses "backend"; create_backend's
            # parameter is "backend_type" — normalize so it isn't dropped into
            # **kwargs (which would silently recurse into the default backend).
            if "backend" in d and "backend_type" not in d:
                d["backend_type"] = d.pop("backend")
            if nm:
                conn_specs[nm] = d

        def _spec_for(name: str) -> dict[str, Any]:
            # A kg_connections name resolves to its spec; otherwise treat the value
            # as a bare backend type (e.g. "epistemic_graph", "age").
            return dict(conn_specs.get(name) or {"backend_type": name})

        authority_name = (
            setting("GRAPH_AUTHORITY") or _cfg.graph_authority or "epistemic_graph"
        )
        authority = _build_member(_spec_for(authority_name))
        if authority is None:
            logger.error(
                "fanout: authority connection '%s' could not be built; "
                "cannot serve graph.",
                authority_name,
            )
            return None

        target_names = (
            setting("GRAPH_MIRROR_TARGETS") or _cfg.graph_mirror_targets or []
        )
        if isinstance(target_names, str):
            target_names = [t.strip() for t in target_names.split(",") if t.strip()]
        mirrors: dict[str, GraphBackend] = {}
        for name in target_names:
            if name == authority_name:
                continue  # never mirror the authority onto itself
            member = _build_member(_spec_for(name))
            if member is None:
                logger.warning(
                    "fanout: mirror '%s' unavailable (missing driver / "
                    "unreachable); skipping.",
                    name,
                )
                continue
            mirrors[name] = member

        if not mirrors:
            logger.warning(
                "fanout: no mirrors configured/available; serving authority "
                "'%s' alone (set GRAPH_MIRROR_TARGETS to enable mirroring).",
                authority_name,
            )
            backend = authority
        else:
            from agent_utilities.core.paths import kg_db_path

            outbox_path = str(kg_db_path().parent / "graph_mirror_outbox.db")
            backend = FanOutBackend(authority, mirrors, outbox_path=outbox_path)

    elif backend_type == "tiered":
        # Two-tier write-through: L1 working store (epistemic-graph) in front of
        # an L2 durable tier. Sub-backends are built directly (not via recursive
        # create_backend) so they don't claim _ACTIVE_BACKEND.
        #
        # L2 (durable) selection — keep it zero-infra by default:
        #   * explicit GRAPH_BACKEND_L2 wins;
        #   * else if a Postgres DSN is configured (uri / GRAPH_DB_URI /
        #     PGGRAPH_DSN) → "postgresql" (preserves existing prod configs);
        #   * else → "ladybug" (embedded, no external server).
        from .epistemic_graph_backend import EpistemicGraphBackend
        from .tiered_backend import TieredGraphBackend

        l1_type = (setting("GRAPH_BACKEND_L1") or "epistemic_graph").lower().strip()
        if l1_type not in ("epistemic_graph", "memory", "file"):
            logger.warning(
                "tiered L1 '%s' unsupported; falling back to epistemic_graph",
                l1_type,
            )
        l1 = EpistemicGraphBackend()

        has_pg_dsn = bool(uri or setting("GRAPH_DB_URI") or setting("PGGRAPH_DSN"))
        l2_type = setting("GRAPH_BACKEND_L2", "").lower().strip() or (
            "postgresql" if has_pg_dsn else "ladybug"
        )

        # Durable-tier opener policy (CONCEPT:KG-2.8 / OS-5.9): the embedded
        # Ladybug/Kuzu DB is SINGLE-WRITER — if every process (host daemon + each
        # MCP server / CLI / script) opens it they contend on the file lock and a
        # host restart can't reacquire it (the "Graph DB locked / std::bad_alloc"
        # wedge). Gate it so only the singleton HOST (the flock holder) opens it;
        # other roles run L1-only (the shared epistemic-graph engine already holds
        # the full node+edge graph). Postgres/pggraph is multi-process (MVCC) and
        # is intentionally NOT gated — every role may open it concurrently.
        from ..core.host_lock import effective_daemon_role

        _role = effective_daemon_role()

        l3: GraphBackend | None = None
        if l2_type in ("postgres", "postgresql", "pggraph", "age", "pggraph_age"):
            # AGE durable tier when GRAPH_PG_AGE=1 or L2 explicitly names age.
            if l2_type in ("age", "pggraph_age") or setting(
                "GRAPH_PG_AGE", ""
            ).lower() in ("1", "true", "yes"):
                from .age_backend import AGEBackend as _PGBackend
            else:
                from .postgresql_backend import PostgreSQLBackend as _PGBackend

            resolved_uri = (
                uri
                or setting("GRAPH_DB_URI")
                or setting("PGGRAPH_DSN")
                or "postgresql://localhost:5432/agent_utilities"
            )
            resolved_name = db_name or setting("GRAPH_DB_NAME") or "agent_graph"
            pool_min = _PG_POOL_MIN
            pool_max = _PG_POOL_MAX
            pggraph_schema = setting("GRAPH_PGGRAPH_SCHEMA", "public")
            l3 = _PGBackend(
                dsn=resolved_uri,
                graph_name=resolved_name,
                pool_min=pool_min,
                pool_max=pool_max,
                pggraph_schema=pggraph_schema,
            )
        elif l2_type == "ladybug":
            from .contrib.ladybug_backend import LADYBUG_AVAILABLE, LadybugBackend

            if _role != "host":
                logger.info(
                    "tiered L2=ladybug: role=%s (not host) → L1-only. The singleton "
                    "host owns the single-writer durable tier; clients read the "
                    "shared engine. (CONCEPT:KG-2.8)",
                    _role,
                )
            elif not LADYBUG_AVAILABLE:
                logger.warning(
                    "tiered L2=ladybug requested but the 'ladybug' package is not "
                    "installed; running L1-only (no durable persistence)."
                )
            else:
                if db_path:
                    resolved_path = db_path
                elif setting("GRAPH_DB_PATH"):
                    resolved_path = setting("GRAPH_DB_PATH")
                else:
                    from agent_utilities.core.paths import kg_db_path

                    resolved = kg_db_path()
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    resolved_path = str(resolved)
                l3 = LadybugBackend(resolved_path)
        else:
            logger.warning(
                "tiered L2 '%s' unsupported; running L1-only (no durable "
                "persistence). Supported: ladybug, postgresql.",
                l2_type,
            )

        # CONCEPT:KG-2.74 — tee the durable L3 to mirror stores. When
        # GRAPH_MIRROR_TARGETS is set, every write that lands in the L3 authority
        # (e.g. pg-age) is also copied, losslessly, to the named mirrors (neo4j /
        # falkordb) via the durable outbox. The L3 authority is unchanged; reads
        # still come from L1 (epistemic). Backfill existing L3 data into fresh
        # mirrors with TieredGraphBackend.reconcile_to_durable().
        if l3 is not None:
            mirrors = _build_mirror_set()
            if mirrors:
                from agent_utilities.core.paths import kg_db_path

                from .fanout_backend import FanOutBackend

                outbox_path = str(kg_db_path().parent / "graph_mirror_outbox.db")
                l3 = FanOutBackend(l3, mirrors, outbox_path=outbox_path)
                logger.info(
                    "tiered L3 fan-out enabled: authority=durable tier, mirrors=[%s]",
                    ", ".join(mirrors),
                )

        # When no durable L2 could be built, degrade to the L1 working store
        # alone rather than crashing the whole engine.
        backend = TieredGraphBackend(l1=l1, l3=l3) if l3 is not None else l1

    elif backend_type == "stardog":
        # Stardog is primarily an OWLBackend; wrap it for GraphBackend compatibility
        logger.info(
            "Stardog backend requested — use OWL backend factory for full "
            "reasoning support: create_owl_backend('stardog')"
        )
        return None

    else:
        logger.error(
            f"Unknown graph backend type: '{backend_type}'. "
            f"Supported: memory, file, epistemic_graph, postgresql, tiered, "
            f"fanout, ladybug, falkordb, neo4j, jena_fuseki"
        )
        return None

    if backend:
        try:
            backend.create_schema()
            # Run schema migrations to add any missing columns/properties
            if (
                backend_type == "ladybug"
                and setting("AGENT_UTILITIES_TESTING") != "true"
            ):
                from ..migrations import migrate_graph

                migrate_graph(backend)
        except Exception as e:
            logger.debug(f"Failed to auto-initialize or migrate graph schema: {e}")

        # CONCEPT:KG-2.6 — wrap with the Company Brain write-path guard
        # (provenance + source-authority arbitration) only when enforcement is
        # on, so the default path stays byte-identical.
        try:
            from ..core.company_brain_runtime import brain_enforcement_enabled

            if brain_enforcement_enabled():
                from ..core.company_brain_runtime import get_company_brain
                from .brain_guarded_backend import BrainGuardedBackend

                backend = BrainGuardedBackend(  # type: ignore[assignment]
                    backend, get_company_brain()
                )
                logger.info("Company Brain write-path guard installed")
        except Exception as e:  # pragma: no cover - guard is best-effort
            logger.warning("Brain guard not installed: %s", e)

    if backend and _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = backend

    return backend
