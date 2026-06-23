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
    GRAPH_BACKEND: Backend type. Bare default: "epistemic_graph" — the engine IS
        the one database (compute + cache + semantic + durable persistence), a
        self-contained binary with no external dependencies. Set "fanout" to add
        MIRRORS (the engine stays the authority; writes fan out to the mirrors).
        Also: "memory", "file", "postgresql"/"age", "jena_fuseki", "stardog", plus
        opt-in contrib mirrors "ladybug", "falkordb", "neo4j".
    GRAPH_MIRROR_TARGETS: JSON/CSV list of mirror connection names for "fanout"
        (resolved via kg_connections). GRAPH_AUTHORITY names the authority
        (default "epistemic_graph").
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

# Single-writer, file-locked backends (Kuzu/LadybugDB hold an exclusive OS lock on
# their DB file). They can be a fan-out MIRROR, but only ONE process may own the
# file — so they are built only by the host write daemon (role="host"), never by
# the many client MCP processes that share the same config.json. Without this, every
# graph-os MCP child would try to open the same mirror file and contend on the lock.
_SINGLE_WRITER_BACKENDS = frozenset({"ladybug", "kuzu"})

# Supported backend-type tokens (a mirror target that is not a kg_connections name
# is treated as a bare backend type — anything outside this set is an operator
# misconfiguration, not a transient driver miss). Mirrors the dispatch in
# ``create_backend`` / the "Unknown graph backend type" error message.
_KNOWN_BACKEND_TYPES = frozenset(
    {
        "epistemic_graph",
        "fanout",
        "memory",
        "file",
        "postgresql",
        "age",
        "jena_fuseki",
        "stardog",
        "ladybug",
        "falkordb",
        "neo4j",
    }
)

_ACTIVE_BACKEND: Any = None


def _parse_mirror_targets(raw: Any) -> list[str]:
    """Parse ``GRAPH_MIRROR_TARGETS`` into a clean list of mirror names.

    CONCEPT:KG-2.203 — tolerant of every shape this value arrives in:

    * already a list (``config.json`` native, ``["a","b"]``) — used as-is;
    * a JSON-array *string* (``config.json`` injects list settings into the env
      as a JSON string: ``'["prod-neo4j","team-falkor"]'`` or ``"['a','b']"``);
    * a comma-separated string (``"prod-neo4j, team-falkor"``, trailing commas ok);
    * a single bare value (``"prod-neo4j"``).

    The previous fanout reader did a naive ``split(",")`` on the raw string, so a
    JSON array became fragments like ``'["prod-neo4j"'`` / ``'"team-falkor"]'`` —
    each then misread as a backend type ("Unknown graph backend type"). We try
    ``json.loads`` first (a list result wins); otherwise we comma-split and strip
    stray whitespace / quotes / brackets per item. Empties are filtered out.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    s = str(raw).strip()
    if not s:
        return []
    # JSON first: handles the env-injected '["a","b"]' shape losslessly.
    try:
        import json as _json

        parsed = _json.loads(s)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(t).strip() for t in parsed if str(t).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    # Fall back to comma-split, defensively stripping any leftover JSON
    # punctuation (brackets / quotes) from each fragment.
    return [
        item
        for raw_item in s.split(",")
        if (item := raw_item.strip().strip("[]").strip().strip("'\"").strip())
    ]


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
    "StardogSparqlBackend",
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
    if name == "StardogSparqlBackend":
        from .sparql.stardog_backend import StardogSparqlBackend

        return StardogSparqlBackend
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

    Used by the ``fanout`` backend: every write that lands in the engine authority
    is teed, losslessly, out to the named mirrors (e.g. pg-age / neo4j / falkordb).
    """
    from agent_utilities.core.config import config as _cfg

    targets = _parse_mirror_targets(
        setting("GRAPH_MIRROR_TARGETS") or _cfg.graph_mirror_targets or []
    )
    # CONCEPT:KG-2.89 — derive the mirror set from connections with role="mirror";
    # the explicit GRAPH_MIRROR_TARGETS above stays an optional override/addition.
    role_mirrors = [
        str(s.get("name") or "").strip()
        for s in (_cfg.kg_connections or [])
        if str(s.get("role") or "").strip().lower() == "mirror"
        and str(s.get("name") or "").strip()
    ]
    _seen: set[str] = set()
    _deduped: list[str] = []
    for t in targets + role_mirrors:
        if t and t not in _seen:
            _seen.add(t)
            _deduped.append(t)
    targets = _deduped
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
    _role: str | None = None
    for name in targets:
        if name in skip_names:
            continue
        spec = dict(conn_specs.get(name) or {"backend_type": name})
        backend_type = str(spec.get("backend_type") or name).strip().lower()
        # A single-writer file mirror is owned by exactly one process: the host
        # write daemon. Client processes (MCP children) skip it so they don't
        # contend on its exclusive file lock.
        if backend_type in _SINGLE_WRITER_BACKENDS:
            if _role is None:
                from ..core.host_lock import effective_daemon_role

                _role = effective_daemon_role()
            if _role != "host":
                logger.info(
                    "mirror '%s' (%s) is single-writer (file-locked); only the host "
                    "daemon owns it — skipping in role=%s.",
                    name,
                    backend_type,
                    _role,
                )
                continue
        member = _build_member(spec)
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
    environment variables, then to sensible defaults. The bare default is
    ``epistemic_graph`` — the engine is the one self-contained database (compute +
    cache + semantic + durable persistence), no external server required. Set
    ``fanout`` to add optional MIRRORS (Postgres/Neo4j/FalkorDB/Ladybug): the engine
    stays the authority serving every read and writes fan out losslessly to them.
    Contrib mirror backends (ladybug/falkordb/neo4j) are imported only when
    explicitly requested.

    Args:
        backend_type: One of "epistemic_graph" (default), "fanout" (engine +
            mirrors), "memory", "file", "postgresql"/"age", "jena_fuseki",
            "stardog", or the opt-in contrib mirrors "ladybug", "falkordb",
            "neo4j". Falls back to the ``GRAPH_BACKEND`` env var, then
            "epistemic_graph" (zero-infra self-contained engine).
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

    # Bare fallback is "epistemic_graph": the engine IS the one database — it does
    # compute, cache, semantic and durable persistence in a single binary with NO
    # external system dependencies (the self-contained, zero-infra default). To add
    # MIRRORS (Postgres/Neo4j/FalkorDB/Ladybug — optional interop/BI/DR fan-out
    # targets), set GRAPH_BACKEND=fanout + GRAPH_MIRROR_TARGETS: the engine stays the
    # authority serving every read, and writes fan out losslessly to the mirrors.
    # The unit suite pins GRAPH_BACKEND=memory (see tests/conftest.py) to stay
    # purely ephemeral.
    backend_type = (
        (backend_type or setting("GRAPH_BACKEND") or "epistemic_graph").lower().strip()
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

        # CONCEPT:KG-2.203 — tolerant parse: accepts a JSON-array string
        # ('["prod-neo4j","team-falkor"]', the shape config.json injects into the
        # env), a comma list, or a single value. The old naive comma-split turned
        # a JSON array into fragments ('["prod-neo4j"' / '"team-falkor"]') that were
        # each misread as a backend type, so every mirror was silently dropped.
        target_names = _parse_mirror_targets(
            setting("GRAPH_MIRROR_TARGETS") or _cfg.graph_mirror_targets or []
        )
        mirrors: dict[str, GraphBackend] = {}
        for name in target_names:
            if name == authority_name:
                continue  # never mirror the authority onto itself
            # Distinguish a genuine misconfiguration (a value that is neither a
            # known kg_connections name nor a supported backend type) from a
            # transient driver/reachability miss — the former is an operator error
            # worth a clear, specific warning.
            if name not in conn_specs and name not in _KNOWN_BACKEND_TYPES:
                logger.warning(
                    "fanout: mirror target '%s' is not a known kg_connections name "
                    "nor a supported backend type; skipping. Check "
                    "GRAPH_MIRROR_TARGETS.",
                    name,
                )
                continue
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

    elif backend_type == "stardog":
        # First-class SPARQL DATA backend (push/pull/query of instance data), usable
        # standalone, as a fan-out mirror, or an ad-hoc connection. The OWL
        # *reasoning* backend (TBox + inference) is separate:
        # ``create_owl_backend('stardog')``.
        from .sparql.stardog_backend import StardogSparqlBackend

        backend = StardogSparqlBackend(
            endpoint=kwargs.get("endpoint") or uri or setting("STARDOG_ENDPOINT"),
            database=db_name or kwargs.get("database") or setting("STARDOG_DATABASE"),
            username=user or kwargs.get("username") or setting("STARDOG_USER"),
            password=password or kwargs.get("password") or setting("STARDOG_PASSWORD"),
        )

    else:
        logger.error(
            f"Unknown graph backend type: '{backend_type}'. "
            f"Supported: epistemic_graph, fanout, memory, file, postgresql, age, "
            f"jena_fuseki, stardog, ladybug, falkordb, neo4j"
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
