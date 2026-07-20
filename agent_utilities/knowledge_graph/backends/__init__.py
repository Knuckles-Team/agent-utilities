#!/usr/bin/python
"""Graph Database Backends.

Provides the `GraphBackend` ABC and concrete primary implementations
(EpistemicGraph/file, PostgreSQL + pgvector). Demoted "contrib" backends
(LadybugDB, FalkorDB, Neo4j) live under ``backends.contrib`` and are opt-in:
they are never imported eagerly, so default backend selection works without
their optional driver packages. Use `create_backend()` to instantiate the
correct backend from typed configuration.

Architecture:
    EpistemicGraph is the single operational authority for graph storage,
    retrieval, semantics, and algorithms. ``FanOutBackend`` may add explicit
    mirrors without changing read authority. PostgreSQL/Apache AGE and contrib
    drivers remain opt-in interoperability endpoints rather than hidden cache or
    fallback layers.

Opt-in (contrib) access::

    from agent_utilities.knowledge_graph.backends.contrib.neo4j_backend import (
        Neo4jBackend,
    )
    from agent_utilities.knowledge_graph.backends import Neo4jBackend

Environment Variables:
    GRAPH_MIRROR_TARGETS: JSON/CSV list of projection connection names. A
        non-empty set automatically wraps the fixed epistemic-graph authority in
        ``FanOutBackend``; it never changes the read or write-ack authority.
    GRAPH_DB_CONNECTION_PROFILE_REF: Runtime secret reference resolving to one
        JSON connection profile. Endpoint, database, identity, credential, TLS,
        and local-path material are never read from separate environment fields.
"""

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


# Postgres connection-pool sizing (config discipline): sensible bounded defaults
# that work everywhere. CONCEPT:AU-KG.backend.authority-backend — the authority is the single shared
# write ceiling: under sustained ingest EVERY lane worker + the fan-out drainer
# threads + reads contend on this ONE pool, so a 10-conn cap starves and the
# acquire wait becomes the multi-second authority-write tail (profiled p50 ~3ms,
# MAX 11.7-16.4s). Raise the default ceiling so concurrent writers each get a
# connection, and let it be tuned per deploy (the box's max_connections is the
# real upper bound). Min is also env-tunable to keep warm connections ready.
def _pg_pool_size(env: str, default: int) -> int:
    try:
        return max(1, int(setting(env, str(default)) or default))
    except (TypeError, ValueError):
        return default


_PG_POOL_MIN = _pg_pool_size("GRAPH_DB_POOL_MIN", 4)
_PG_POOL_MAX = _pg_pool_size("GRAPH_DB_POOL_MAX", 32)

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

_CONNECTION_PROFILE_KEYS = frozenset(
    {
        "uri",
        "host",
        "port",
        "user",
        "password",
        "database",
        "db_name",
        "db_path",
        "graph_name",
        "tls_profile",
        "tls_profile_ref",
        "tls_profile_config",
        "connection_timeout",
        "max_connection_pool_size",
    }
)


def _resolve_connection_profile(reference: str | None) -> dict[str, Any]:
    """Resolve one transient graph connection profile from a secret reference."""

    ref = str(reference or "").strip()
    if not ref:
        return {}
    if not ref.startswith(("env://", "vault://", "secret://")):
        raise ValueError("graph connection profile must be a runtime secret reference")
    from agent_utilities.security.secrets_client import create_secrets_client

    resolved = create_secrets_client().resolve_ref(ref)
    if isinstance(resolved, Mapping):
        profile = dict(resolved)
    else:
        try:
            profile = json.loads(str(resolved))
        except (TypeError, ValueError):
            raise ValueError("graph connection profile is not valid JSON") from None
    if not isinstance(profile, dict):
        raise ValueError("graph connection profile must be a JSON object")
    if set(profile).difference(_CONNECTION_PROFILE_KEYS):
        raise ValueError("graph connection profile contains unsupported fields")
    if "port" in profile:
        profile["port"] = int(profile["port"])
    return profile


def _parse_mirror_targets(raw: Any) -> list[str]:
    """Parse ``GRAPH_MIRROR_TARGETS`` into a clean list of mirror names.

    CONCEPT:AU-KG.backend.tolerant-parse — tolerant of every shape this value arrives in:

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
    if isinstance(raw, list | tuple):
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

    Used by the ``fanout`` composite (CONCEPT:AU-KG.backend.mirror-health-repair): a member is a full
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
    """Build ``{name: backend}`` for ``GRAPH_MIRROR_TARGETS`` (CONCEPT:AU-KG.backend.mirror-health-repair),
    resolved against ``kg_connections``. Returns ``{}`` when none are configured.

    Used by the ``fanout`` backend: every write that lands in the engine authority
    is teed, losslessly, out to the named mirrors (e.g. pg-age / neo4j / falkordb).
    """
    from agent_utilities.core.config import config as _cfg

    targets = _parse_mirror_targets(_cfg.graph_mirror_targets or [])
    # CONCEPT:AU-KG.backend.derive-mirror-set — derive the mirror set from connections with role="mirror";
    # the explicit GRAPH_MIRROR_TARGETS above stays an optional override/addition.
    role_mirrors = [
        str(s.get("name") or "").strip()
        for s in (_cfg.kg_connections or [])
        if str(s.get("role") or "").strip().lower() == "mirror"
        and str(s.get("name") or "").strip()
    ]
    if getattr(_cfg, "continuous_stardog_mirror", False):
        role_mirrors.append("stardog")
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
        if backend_type in {"epistemic_graph", "fanout", "memory", "file"}:
            logger.warning(
                "mirror '%s' resolves to the operational authority; skipping "
                "because mirrors must be external projections.",
                name,
            )
            continue
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
    connection_profile_ref: str | None = None,
    **kwargs,
):
    """Factory function to create the appropriate graph backend.

    With no explicit backend type this is the current operational construction:
    the epistemic-graph engine is always the authority, and configured external
    projections are attached automatically. Explicit backend types are reserved
    for connection-registry source adapters, projection construction, and focused
    backend tests; they do not alter GraphOS authority.

    Args:
        backend_type: Explicit external source/projection adapter type. Omit for
            the operational epistemic-graph authority with automatic mirrors.
        connection_profile_ref: Runtime secret reference resolving to a JSON
            object containing the backend's transport fields.

    Returns:
        A configured ``GraphBackend`` instance, or ``None`` if the requested
        backend is not available (e.g., ladybug package not installed).
    """
    global _ACTIVE_BACKEND

    operational_authority = backend_type is None
    explicit_transport = any(
        value is not None
        for value in (db_path, host, port, uri, user, password, db_name)
    )
    external_transport = any(
        value is not None for value in (host, port, uri, user, password, db_name)
    )
    if operational_authority and (external_transport or connection_profile_ref):
        raise ValueError(
            "operational graph authority does not accept external transport fields"
        )

    requested_type = (
        "fanout" if operational_authority else str(backend_type).lower().strip()
    )
    if not operational_authority and requested_type == "fanout":
        raise ValueError(
            "fan-out is an automatic operational projection wrapper, not an "
            "explicit backend adapter"
        )
    if (
        not operational_authority
        and requested_type not in {"epistemic_graph", "fanout", "memory", "file"}
        and connection_profile_ref is None
        and not explicit_transport
    ):
        from agent_utilities.core.config import config as _cfg

        connection_profile_ref = _cfg.graph_db_connection_profile_ref
    profile = _resolve_connection_profile(connection_profile_ref)
    db_path = db_path if db_path is not None else profile.pop("db_path", None)
    host = host if host is not None else profile.pop("host", None)
    port = port if port is not None else profile.pop("port", None)
    uri = uri if uri is not None else profile.pop("uri", None)
    user = user if user is not None else profile.pop("user", None)
    password = password if password is not None else profile.pop("password", None)
    db_name = db_name if db_name is not None else profile.pop("db_name", None)
    if "database" in profile and "database" not in kwargs:
        kwargs["database"] = profile.pop("database")
    kwargs = {**profile, **kwargs}

    # Omission is the sole operational construction path. It always enters the
    # fixed-authority branch; that branch returns the bare engine when no external
    # projections are configured.
    backend_type = requested_type
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
            resolved_path = db_path or kwargs.get("json_path")
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
        else:
            from agent_utilities.core.paths import kg_db_path

            resolved = kg_db_path()
            # Ensure parent directory exists for XDG paths
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved_path = str(resolved)
        backend = LadybugBackend(resolved_path)

    elif backend_type == "falkordb":
        from .contrib.falkordb_backend import FalkorDBBackend

        if host is None or port is None or db_name is None:
            raise ValueError("FalkorDB requires a complete connection profile")
        resolved_host = host
        resolved_port = port
        resolved_name = db_name
        backend = FalkorDBBackend(
            host=resolved_host, port=resolved_port, db_name=resolved_name
        )

    elif backend_type == "neo4j":
        from .contrib.neo4j_backend import Neo4jBackend

        if uri is None or user is None or password is None:
            raise ValueError("Neo4j requires a complete connection profile")
        resolved_uri = uri
        resolved_user = user
        resolved_password = password
        backend = Neo4jBackend(
            uri=resolved_uri,
            user=resolved_user,
            password=resolved_password,
            database=(kwargs.get("database") or db_name or None),
            tls_profile=kwargs.get("tls_profile"),
            tls_profile_ref=kwargs.get("tls_profile_ref"),
            tls_profile_config=kwargs.get("tls_profile_config"),
            profile_resolver=kwargs.get("profile_resolver"),
            connection_timeout=kwargs.get("connection_timeout"),
            max_connection_pool_size=kwargs.get("max_connection_pool_size"),
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

        if uri is None or db_name is None:
            raise ValueError("PostgreSQL/AGE requires a complete connection profile")
        resolved_uri = uri
        resolved_name = db_name
        pool_min = _PG_POOL_MIN
        pool_max = _PG_POOL_MAX
        pggraph_schema = setting("GRAPH_PGGRAPH_SCHEMA", "public")
        backend = _PGBackend(
            dsn=resolved_uri,
            graph_name=resolved_name,
            pool_min=pool_min,
            pool_max=pool_max,
            pggraph_schema=pggraph_schema,
            tls_profile=kwargs.get("tls_profile"),
            tls_profile_ref=kwargs.get("tls_profile_ref"),
            tls_profile_config=kwargs.get("tls_profile_config"),
            profile_resolver=kwargs.get("profile_resolver"),
        )

    elif backend_type == "jena_fuseki":
        from agent_utilities.core.config import config as _cfg

        from .sparql.jena_fuseki_backend import JenaFusekiBackend

        resolved_url = kwargs.get("jena_fuseki_url") or _cfg.kg_fuseki_endpoint
        resolved_dataset = (
            kwargs.get("dataset") or setting("GRAPH_FUSEKI_DATASET") or "agent_kg"
        )
        resolved_jena_fuseki_user = kwargs.get("username") or setting(
            "GRAPH_FUSEKI_USER"
        )
        backend = JenaFusekiBackend(
            jena_fuseki_url=resolved_url,
            dataset=resolved_dataset,
            username=resolved_jena_fuseki_user,
            password_ref=(kwargs.get("password_ref") or _cfg.graph_fuseki_password_ref),
        )

    elif backend_type == "fanout":
        # Concurrent N-way projection (CONCEPT:AU-KG.backend.mirror-health-repair):
        # EpistemicGraphBackend is fixed as the read/write-ack authority. External
        # connections are write projections with durable replay and reconciliation.
        from .fanout_backend import FanOutBackend

        # Never mirror the configured authority connection onto itself — the
        # module-level ``_build_mirror_set`` already accepts a ``skip_names``
        # seam for exactly this (CONCEPT:AU-KG.backend.mirror-health-repair).
        from agent_utilities.core.config import config as _cfg

        authority_name = (
            setting("GRAPH_AUTHORITY") or _cfg.graph_authority or "epistemic_graph"
        )
        mirrors = _build_mirror_set(skip_names=(authority_name,))

        if not mirrors:
            from .epistemic_graph_backend import EpistemicGraphBackend

            backend = EpistemicGraphBackend()
        else:
            from agent_utilities.core.paths import kg_db_path

            outbox_path = str(kg_db_path().parent / "graph_mirror_outbox.db")
            backend = FanOutBackend(mirrors, outbox_path=outbox_path)

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

        # CONCEPT:AU-KG.backend.company-brain-write-guard — wrap with the Company Brain write-path guard
        # (provenance + source-authority arbitration) only when enforcement is
        # on, so the default path stays byte-identical.
        try:
            from ..core.company_brain_runtime import brain_enforcement_enabled

            if operational_authority and brain_enforcement_enabled():
                from ..core.company_brain_runtime import get_company_brain
                from .brain_guarded_backend import BrainGuardedBackend

                backend = BrainGuardedBackend(  # type: ignore[assignment]
                    backend, get_company_brain()
                )
                logger.info("Company Brain write-path guard installed")
        except Exception as e:  # pragma: no cover - guard is best-effort
            logger.warning("Brain guard not installed: %s", e)

    if backend and operational_authority and _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = backend

    return backend
