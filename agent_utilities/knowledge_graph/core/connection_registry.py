# CONCEPT:AU-KG.backend.multi-connection-registry - Named multi-connection graph registry: register N live graph backends (neo4j/falkordb/postgres-AGE/…) by name and run the same MCP/REST tools against any one or fan out to all, with the backend choice fully abstracted behind a `target` parameter.
# CONCEPT:AU-KG.backend.connection-registry - Role-aware multi-database registry plus live config mutation: every external graph DB is a read source or governed mirror; durable declarations contain only neutral metadata and runtime secret refs, with generic get/set/list of any config item live over both the MCP server and the API gateway, and a doctor health-check across all connections.
"""Named multi-connection graph registry.

CONCEPT:AU-KG.backend.multi-connection-registry — Multi-Connection Graph Registry. The engine has always been
vendor-agnostic (one ``GraphBackend`` interface, many implementations). The
operational authority is always epistemic-graph; this registry lets a deployment keep several live
connections side by side — e.g. ``prod-neo4j``, ``team-falkor``, ``pg-main`` —
and run the *same* graph tools against a named one (``target="pg-main"``) or fan
out to all of them (``target="all"``), with no per-backend special instructions.

Design (the sentences that matter):

* **The default is never duplicated.** The reserved name ``"default"`` always
  resolves to the existing process-wide ``IntelligenceGraphEngine.get_active()``
  singleton (built lazily via the injected ``default_engine_provider``). Building
  a *named* engine therefore happens only after the default exists, so the
  named engine never clobbers ``_ACTIVE_ENGINE`` (engine auto-registration only
  fires when no active engine exists).
* **Named connections are read adapters, not engines.** A named connection
  builds one backend and exposes only its server-enforced read contract. It
  never constructs another operational engine or Epistemic Graph transport.
* **One authority by default.** No ``target`` (or ``target="default"``) routes to
  the process-owned authoritative engine.

This mirrors the zero-infra-preserving shape of the shard topology
(:mod:`agent_utilities.knowledge_graph.core.shard_topology`, CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw):
config-list resolution, a default-preserving single-entry mode, and a
``status()`` health surface.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Reserved name for the process-wide authoritative engine.
DEFAULT_NAME = "default"

# Stable, non-sensitive fan-out result used when a connection cannot be built.
# Backend exceptions frequently include DSNs, filesystem locations, or driver
# payloads, so callers must never receive their text.
CONNECTION_UNAVAILABLE = "connection_unavailable"

#: Names that may not be used for a registered connection.
_RESERVED = {DEFAULT_NAME, "all", ""}

#: Named connections are read sources or governed outbox mirrors. Only the
#: reserved default operational engine accepts direct writes.
_ROLES = {"read", "mirror"}
DEFAULT_ROLE = "read"

_SECRETS_CLIENT: Any = None
_SECRET_REF_RE = re.compile(r"^(?:vault|env|secret)://[A-Za-z0-9_./#-]+$")
_PERSISTENCE_SENSITIVE_FIELDS = frozenset(
    {
        "uri",
        "endpoint",
        "host",
        "db_path",
        "user",
        "password",
        "auth_secret",
        "auth_profile_ref",
        "connection_profile_ref",
        "database",
        "db_name",
        "graph_name",
        "mapping_policy_ref",
        "mapping_profile_ref",
        "port",
        "tls_profile_ref",
        "variables_ref",
    }
)
_EXTERNAL_PROPERTY_GRAPH_FIELDS = frozenset(
    {
        "allow_empty_snapshot",
        "auth_profile_ref",
        "backend",
        "backend_type",
        "connection_profile_ref",
        "contextual",
        "discovery_max_depth",
        "discovery_max_types",
        "ingest_max_records",
        "ingest_max_pages",
        "ingest_max_row_bytes",
        "ingest_max_total_bytes",
        "ingest_max_nesting_depth",
        "ingest_max_collection_items",
        "ingest_operation",
        "ingest_page_size",
        "mapping_policy_ref",
        "name",
        "require_approval",
        "reconcile_deletions",
        "role",
        "schema_drift_policy",
        "semantic_mapping",
        "source_alias",
        "sync_mode",
        "tls_profile_ref",
        "variables_ref",
    }
)
_EXTERNAL_GRAPHQL_FIELDS = frozenset(
    {
        "allow_empty_snapshot",
        "allow_introspection",
        "auth_profile_ref",
        "backend",
        "backend_type",
        "connection_profile_ref",
        "contextual",
        "discovery_max_depth",
        "discovery_max_types",
        "ingest_max_records",
        "ingest_operation",
        "mapping_policy_ref",
        "name",
        "require_approval",
        "role",
        "schema_drift_policy",
        "semantic_mapping",
        "source_alias",
        "tls_profile_ref",
        "variables_ref",
    }
)
_EXTERNAL_GRAPH_ADAPTERS = frozenset(
    {"age", "epistemic_graph", "graphql", "ladybug", "neo4j", "opencypher"}
)
_PROPERTY_GRAPH_AUTH_BACKENDS = frozenset({"age", "neo4j", "opencypher"})
_PROPERTY_GRAPH_AUTH_FIELDS = frozenset({"password", "uri", "user"})
_MAX_RUNTIME_PROFILE_BYTES = 1024 * 1024


class ExternalGraphConnection:
    """Read-only view over one external backend, never an operational engine."""

    read_only = True

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        return list(self.backend.execute_read(query, params or {}) or [])

    def probe_connection(self) -> bool:
        """Prove the read transport without retaining source rows."""

        health_check = getattr(self.backend, "health_check", None)
        if callable(health_check):
            return bool(health_check())
        self.backend.execute_read("MATCH (n) RETURN n LIMIT 0", {})
        return True

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


def _resolve_secret(value: Any) -> Any:
    """Materialize a runtime secret reference for an internal backend constructor.

    Durable declarations accept ``vault://…`` / ``secret://…`` / ``env://VAR``
    references only. Already-materialized values may cross this private runtime
    boundary, but are never persisted or returned. An unresolved ref fails closed;
    treating the reference itself as a credential or endpoint both leaks metadata
    and creates confusing downstream transport errors.
    """
    if not isinstance(value, str) or not value.startswith(
        ("vault://", "env://", "secret://")
    ):
        return value
    global _SECRETS_CLIENT
    try:
        if _SECRETS_CLIENT is None:
            from agent_utilities.security.secrets_client import create_secrets_client

            _SECRETS_CLIENT = create_secrets_client()
        resolved = _SECRETS_CLIENT.resolve_ref(value)
        if resolved is None:
            raise ValueError("secret reference could not be resolved")
        return resolved
    except Exception as exc:
        logger.warning(
            "connection secret-reference resolution failed (%s)", type(exc).__name__
        )
        raise ValueError("connection secret reference could not be resolved") from None


def _resolve_runtime_profile(reference: str, label: str) -> dict[str, Any]:
    """Resolve one bounded JSON object for a transient connection boundary."""

    resolved = _resolve_secret(reference)
    if isinstance(resolved, Mapping):
        profile = dict(resolved)
    else:
        rendered = str(resolved)
        if len(rendered.encode("utf-8")) > _MAX_RUNTIME_PROFILE_BYTES:
            raise ValueError(f"{label} exceeds its size bound")
        try:
            profile = json.loads(rendered)
        except (TypeError, ValueError):
            raise ValueError(f"{label} is not valid JSON") from None
    if not isinstance(profile, dict):
        raise ValueError(f"{label} must be a JSON object")
    return profile


def _resolve_property_graph_auth_profile(
    reference: str,
    backend_kind: str,
) -> dict[str, str]:
    """Resolve credentials for one supported property-graph constructor."""

    if backend_kind not in _PROPERTY_GRAPH_AUTH_BACKENDS:
        raise ValueError("auth_profile_ref is unsupported for this graph backend")
    profile = _resolve_runtime_profile(reference, "auth profile")
    unsupported = set(profile).difference(_PROPERTY_GRAPH_AUTH_FIELDS)
    if unsupported:
        raise ValueError("auth profile contains unsupported or selector fields")
    auth: dict[str, str] = {}
    for key, value in profile.items():
        if not isinstance(value, str) or not value:
            raise ValueError("auth profile credential fields must be non-empty strings")
        auth[key] = value
    if not auth:
        raise ValueError("auth profile contains no credential fields")
    return auth


def validate_persistable_connection_spec(spec: dict[str, Any]) -> None:
    """Reject endpoint/path/identity material before a spec enters config.json.

    Programmatic transient registries may still use direct values.  The GraphOS
    durable registration action calls this gate before ``export_specs`` so the
    persisted shape contains only neutral aliases and secret references.
    """

    for field in _PERSISTENCE_SENSITIVE_FIELDS:
        value = spec.get(field)
        if value in (None, ""):
            continue
        if not isinstance(value, str) or not _SECRET_REF_RE.fullmatch(value):
            raise ValueError(
                f"persistent connection field {field!r} must be a secret reference"
            )
    backend_value = str(spec.get("backend") or "").strip()
    backend_type_value = str(spec.get("backend_type") or "").strip()
    if backend_value and backend_type_value and backend_value != backend_type_value:
        raise ValueError("persistent connection backend selectors disagree")
    backend = backend_type_value or backend_value
    if "source_alias" in spec and backend not in _EXTERNAL_GRAPH_ADAPTERS:
        raise ValueError("persistent external graph backend selector is unsupported")
    if backend in _EXTERNAL_GRAPH_ADAPTERS:
        graphql = backend == "graphql"
        allowed = (
            _EXTERNAL_GRAPHQL_FIELDS if graphql else _EXTERNAL_PROPERTY_GRAPH_FIELDS
        )
        if set(spec).difference(allowed):
            raise ValueError(
                "persistent external graph declarations contain unsupported inline material"
            )
        name = str(spec.get("name") or "").strip().lower()
        if name and not re.fullmatch(r"[a-z][a-z0-9-]{1,62}", name):
            raise ValueError(
                "persistent external graph declarations need a neutral name"
            )
        role = str(spec.get("role") or DEFAULT_ROLE).lower()
        if graphql and role != "read":
            raise ValueError("persistent GraphQL declarations must use role='read'")
        if role not in _ROLES:
            raise ValueError("persistent external graph role is invalid")
        source_alias = str(spec.get("source_alias") or "").strip().lower()
        if source_alias and not re.fullmatch(r"[a-z][a-z0-9-]{1,62}", source_alias):
            raise ValueError(
                "persistent external graph declarations need a neutral source_alias"
            )
        if graphql and not source_alias:
            raise ValueError(
                "persistent GraphQL declarations need a neutral source_alias"
            )
        required_refs = ["connection_profile_ref"]
        for required_ref in required_refs:
            if not _SECRET_REF_RE.fullmatch(str(spec.get(required_ref) or "")):
                raise ValueError(
                    f"persistent external graph declarations require {required_ref}"
                )
        if (
            graphql
            and not spec.get("mapping_policy_ref")
            and spec.get("allow_introspection") is not True
        ):
            raise ValueError(
                "persistent GraphQL declarations require mapping_policy_ref or "
                "explicit allow_introspection"
            )
        operation = str(spec.get("ingest_operation") or "")
        if operation and not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", operation):
            raise ValueError("persistent GraphQL ingest_operation is invalid")
        bounds = [
            ("discovery_max_types", 1, 500),
            ("ingest_max_records", 1, 10_000),
        ]
        if graphql:
            bounds.append(("discovery_max_depth", 1, 12))
        else:
            bounds.extend(
                (
                    ("ingest_page_size", 1, 1_000),
                    ("ingest_max_pages", 1, 1_000),
                    ("ingest_max_row_bytes", 256, 8_388_608),
                    ("ingest_max_total_bytes", 256, 67_108_864),
                    ("ingest_max_nesting_depth", 1, 64),
                    ("ingest_max_collection_items", 1, 100_000),
                )
            )
        for key, lower, upper in bounds:
            if spec.get(key) is None:
                continue
            if not graphql and isinstance(spec[key], bool):
                raise ValueError(f"persistent external graph {key} is invalid")
            try:
                bounded = int(spec[key])
            except (TypeError, ValueError):
                raise ValueError(
                    f"persistent external graph {key} is invalid"
                ) from None
            if not lower <= bounded <= upper:
                raise ValueError(f"persistent external graph {key} is out of range")
        if not graphql and int(spec.get("ingest_max_total_bytes") or 16_777_216) < int(
            spec.get("ingest_max_row_bytes") or 1_048_576
        ):
            raise ValueError(
                "persistent external graph total byte bound must cover one row"
            )
        if spec.get("require_approval") not in (None, True):
            raise ValueError("persistent external graph approval cannot be disabled")
        if spec.get("schema_drift_policy") not in (None, "fail_closed"):
            raise ValueError("persistent external graph schema drift must fail closed")
        if graphql and spec.get("semantic_mapping") not in (None, False):
            raise ValueError(
                "persistent GraphQL semantic_mapping is unsupported; "
                "use governed structural mapping proposals"
            )
        if not graphql and spec.get("sync_mode") not in (
            None,
            "auto",
            "cdc",
            "snapshot",
        ):
            raise ValueError("persistent external graph sync_mode is invalid")
        property_boolean_keys = ("reconcile_deletions",) if not graphql else ()
        for boolean_key in (
            "allow_empty_snapshot",
            "allow_introspection",
            "contextual",
            "semantic_mapping",
            *property_boolean_keys,
        ):
            if spec.get(boolean_key) is not None and not isinstance(
                spec.get(boolean_key), bool
            ):
                raise ValueError(
                    f"persistent external graph {boolean_key} must be boolean"
                )


class ConnectionRegistry:
    """Thread-safe registry of named live graph connections.

    Parameters
    ----------
    default_engine_provider:
        Zero-arg callable returning the process-wide active
        ``IntelligenceGraphEngine`` (creating it if needed). Injected by the MCP
        server (``kg_server._get_engine``) to avoid a circular import and to
        guarantee the active engine exists before any named engine is built.
    """

    def __init__(
        self, default_engine_provider: Callable[[], Any] | None = None
    ) -> None:
        self._lock = threading.RLock()
        self._specs: dict[str, dict[str, Any]] = {}
        self._engines: dict[str, Any] = {}
        self._default_provider = default_engine_provider

    # ── default engine resolution ──────────────────────────────────────────
    def _default_engine(self) -> Any:
        if self._default_provider is not None:
            return self._default_provider()
        # Fallback when no provider was injected (e.g. unit tests): the existing
        # active engine, or None.
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        return IntelligenceGraphEngine.get_active()

    # ── registration ───────────────────────────────────────────────────────
    def register(self, name: str, spec: dict[str, Any]) -> str:
        """Register (or replace) a named connection spec.

        ``spec`` is the kwargs for :func:`create_backend` (``backend``, ``uri``,
        ``host``, ``port``, ``user``, ``password``, ``db_name`` …). A
        ``backend_type``/``backend`` key selects the backend; for Postgres,
        ``backend="age"`` gets native openCypher (recommended for portability).
        Lazy: nothing connects until the connection is first targeted.
        """
        clean = (name or "").strip()
        if clean.lower() in _RESERVED:
            raise ValueError(
                f"Connection name {name!r} is reserved (cannot be one of {sorted(_RESERVED)})."
            )
        spec = dict(spec or {})
        embedded_name = spec.pop("name", None)
        if embedded_name not in (None, clean):
            raise ValueError("connection spec name does not match its registry alias")
        raw_selector = str(
            spec.get("backend_type") or spec.get("backend") or ""
        ).strip()
        if "source_alias" in spec and raw_selector not in _EXTERNAL_GRAPH_ADAPTERS:
            raise ValueError("external graph backend selector is unsupported")
        selector = raw_selector.lower().replace("-", "_")
        if selector == "graphql" and not re.fullmatch(r"[a-z][a-z0-9-]{1,62}", clean):
            raise ValueError(
                "GraphQL connection names must be neutral lowercase aliases"
            )
        # Accept both "backend" and "backend_type" as the selector key.
        if "backend" in spec and "backend_type" not in spec:
            spec["backend_type"] = spec.pop("backend")
        # CONCEPT:AU-KG.backend.connection-registry — normalize the connection role (default: read).
        role = str(spec.get("role") or DEFAULT_ROLE).strip().lower()
        if role not in _ROLES:
            raise ValueError(
                f"Invalid role {role!r} for connection {clean!r}; "
                f"must be one of {sorted(_ROLES)}"
            )
        spec["role"] = role
        with self._lock:
            self._specs[clean] = spec
            # Drop any stale cached engine so the next access rebuilds with the
            # new spec.
            old = self._engines.pop(clean, None)
        if old is not None:
            self._safe_close(old)
        return clean

    def remove(self, name: str) -> bool:
        """Remove a named connection and close its cached engine, if any."""
        clean = (name or "").strip()
        with self._lock:
            had = self._specs.pop(clean, None) is not None
            eng = self._engines.pop(clean, None)
        if eng is not None:
            self._safe_close(eng)
        return had

    def names(self) -> list[str]:
        """All addressable connection names, default first."""
        with self._lock:
            return [DEFAULT_NAME, *sorted(self._specs)]

    def role(self, name: str | None) -> str:
        """Return ``authority`` for default, else the named source role."""
        clean = (name or DEFAULT_NAME).strip() or DEFAULT_NAME
        if clean == DEFAULT_NAME:
            return "authority"
        with self._lock:
            spec = self._specs.get(clean)
        return str((spec or {}).get("role") or DEFAULT_ROLE)

    def backend_kind(self, name: str) -> str:
        """Normalized backend selector without endpoint/database material."""

        clean = (name or "").strip()
        with self._lock:
            spec = self._specs.get(clean)
        if spec is None:
            raise KeyError(f"Unknown connection '{clean}'")
        return str(spec.get("backend_type") or spec.get("backend") or "opencypher")

    def is_writable(self, name: str | None) -> bool:
        """Only the reserved operational authority accepts direct writes."""
        return (name or DEFAULT_NAME).strip() in {"", DEFAULT_NAME}

    def export_specs(self) -> list[dict[str, Any]]:
        """All registered connections as a config list (each entry = ``{name, **spec}``).

        For durable persistence to ``config.json`` (``kg_connections``). Every
        transport/identity/path field must already be a runtime secret reference;
        transient literal-backed registrations deliberately cannot cross this
        boundary. Resolution happens only at connect, on a copy
        (CONCEPT:AU-KG.backend.connection-registry)."""
        with self._lock:
            exported = [{"name": n, **s} for n, s in sorted(self._specs.items())]
        for entry in exported:
            validate_persistable_connection_spec(entry)
        return exported

    def default_name(self) -> str:
        return DEFAULT_NAME

    # ── engine resolution ──────────────────────────────────────────────────
    def get_engine(self, name: str | None) -> Any:
        """Return the live engine for ``name`` (building+caching on first use).

        ``None``/``""``/``"default"`` → the process-wide active engine. Raises
        ``KeyError`` for an unknown named connection and lets backend connection
        errors propagate (fail-loud) for a single explicit target.
        """
        clean = (name or DEFAULT_NAME).strip() or DEFAULT_NAME
        if clean == DEFAULT_NAME:
            return self._default_engine()

        with self._lock:
            eng = self._engines.get(clean)
            if eng is not None:
                return eng
            spec = self._specs.get(clean)
            if spec is None:
                raise KeyError(f"Unknown connection '{clean}'")

        # Build OUTSIDE the lock — connecting to a remote DB can be slow and we
        # must not serialise every other connection's first access behind it.
        # The registry alias is transient construction context. It is never
        # inserted into the persisted spec or handed to a database driver.
        engine = self._build_engine({**spec, "_registry_name": clean})

        with self._lock:
            # Double-checked: if a concurrent caller built it first, keep theirs.
            existing = self._engines.get(clean)
            if existing is not None:
                self._safe_close(engine)
                return existing
            self._engines[clean] = engine
            return engine

    def _build_engine(self, spec: dict[str, Any]) -> Any:
        from agent_utilities.knowledge_graph.backends import create_backend

        # CONCEPT:AU-KG.backend.connection-registry — ``role`` is registry metadata (not a backend kwarg), and
        # a credential may be a secret reference resolved at connect (never stored
        # raw in config.json).
        build_spec = {
            k: v for k, v in spec.items() if k not in {"role", "_registry_name"}
        }
        backend_kind = (
            str(build_spec.get("backend_type") or "").lower().replace("-", "_")
        )
        if backend_kind == "graphql":
            if str(spec.get("role") or DEFAULT_ROLE) != "read":
                raise ValueError("a GraphQL connector must have role='read'")
            from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
                GraphQLSourceAdapter,
            )

            return GraphQLSourceAdapter(
                connection=str(spec.get("_registry_name") or ""),
                source_alias=str(build_spec.get("source_alias") or ""),
                connection_profile_ref=str(
                    build_spec.get("connection_profile_ref") or ""
                ),
                mapping_policy_ref=str(build_spec.get("mapping_policy_ref") or ""),
                auth_profile_ref=str(build_spec.get("auth_profile_ref") or "") or None,
                tls_profile_ref=str(build_spec.get("tls_profile_ref") or "") or None,
                variables_ref=str(build_spec.get("variables_ref") or "") or None,
                ingest_operation=str(build_spec.get("ingest_operation") or "") or None,
                discovery_max_types=int(build_spec.get("discovery_max_types") or 200),
                discovery_max_depth=int(build_spec.get("discovery_max_depth") or 6),
                ingest_max_records=int(build_spec.get("ingest_max_records") or 1_000),
                contextual=bool(build_spec.get("contextual", True)),
                allow_introspection=bool(build_spec.get("allow_introspection", False)),
                allow_empty_snapshot=bool(
                    build_spec.get("allow_empty_snapshot", False)
                ),
            )
        for metadata_key in (
            "allow_empty_snapshot",
            "contextual",
            "discovery_max_depth",
            "discovery_max_types",
            "ingest_max_records",
            "ingest_max_pages",
            "ingest_max_row_bytes",
            "ingest_max_total_bytes",
            "ingest_max_nesting_depth",
            "ingest_max_collection_items",
            "ingest_operation",
            "ingest_page_size",
            "mapping_policy_ref",
            "require_approval",
            "reconcile_deletions",
            "schema_drift_policy",
            "semantic_mapping",
            "source_alias",
            "sync_mode",
            "variables_ref",
        ):
            build_spec.pop(metadata_key, None)
        connection_profile_ref = build_spec.pop("connection_profile_ref", None)
        if connection_profile_ref:
            runtime_profile = _resolve_runtime_profile(
                connection_profile_ref, "connection profile"
            )
            # The persisted selector/role is authoritative; runtime transport
            # material is resolved transiently from the encrypted profile.
            selector = build_spec.get("backend_type")
            build_spec = {**runtime_profile, **build_spec}
            if selector:
                build_spec["backend_type"] = selector
        auth_profile_ref = build_spec.pop("auth_profile_ref", None)
        if auth_profile_ref:
            selector = build_spec.get("backend_type")
            runtime_auth = _resolve_property_graph_auth_profile(
                str(auth_profile_ref), backend_kind
            )
            build_spec = {**build_spec, **runtime_auth}
            if selector:
                build_spec["backend_type"] = selector
        for key in (
            "auth_secret",
            "auth_secret_ref",
            "database",
            "db_name",
            "db_path",
            "endpoint",
            "endpoint_ref",
            "graph_name",
            "host",
            "password",
            "port",
            "uri",
            "user",
        ):
            if key in build_spec:
                build_spec[key] = _resolve_secret(build_spec[key])
        if "port" in build_spec:
            build_spec["port"] = int(build_spec["port"])

        backend_kind = (
            str(build_spec.get("backend_type") or "").lower().replace("-", "_")
        )
        if backend_kind == "epistemic_graph" and (
            build_spec.get("endpoint_ref") or build_spec.get("endpoint")
        ):
            if str(spec.get("role") or DEFAULT_ROLE) != "read":
                raise ValueError(
                    "a remote epistemic-graph connector must have role='read'"
                )
            from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
                RemoteEpistemicGraphReadAdapter,
            )

            endpoint = build_spec.get("endpoint_ref") or build_spec.get("endpoint")
            auth_secret = build_spec.get("auth_secret_ref") or build_spec.get(
                "auth_secret"
            )
            verified_context = build_spec.get("verified_context")
            if not auth_secret or not isinstance(verified_context, Mapping):
                raise ValueError(
                    "remote epistemic-graph profile requires current auth material"
                )
            return RemoteEpistemicGraphReadAdapter(
                endpoint=str(endpoint or ""),
                auth_secret=str(auth_secret),
                graph_name=str(build_spec.get("graph_name") or "default"),
                verified_context=dict(verified_context),
                tls_profile=(
                    str(build_spec.get("tls_profile"))
                    if build_spec.get("tls_profile")
                    else None
                ),
                tls_profile_ref=(
                    str(build_spec.get("tls_profile_ref"))
                    if build_spec.get("tls_profile_ref")
                    else None
                ),
                tls_server_name=(
                    str(build_spec.get("tls_server_name"))
                    if build_spec.get("tls_server_name")
                    else None
                ),
            )

        # The current generic openCypher source contract is Bolt read-only.
        # Reuse the hardened Neo4j-driver transport while retaining
        # ``opencypher`` as the discovery dialect reported by the registry.
        if backend_kind == "opencypher":
            build_spec["backend_type"] = "neo4j"
        backend = create_backend(**build_spec)
        if backend is None:
            raise RuntimeError(
                "Configured graph backend is unavailable "
                "(missing driver/package or unreachable)."
            )
        return ExternalGraphConnection(backend)

    # ── target resolution ──────────────────────────────────────────────────
    def resolve_names(self, target: Any) -> tuple[list[str], bool]:
        """Resolve a ``target`` into ``(names, fanout)``.

        * ``None`` / ``""`` / ``"default"`` → ``(["default"], False)`` — the
          authoritative engine.
        * a single name (``"pg-main"``) → ``(["pg-main"], False)`` — single shape.
        * ``"all"`` → ``(<all names>, True)``.
        * a comma list or an actual list/tuple → ``(<names>, True)``.

        ``fanout=True`` is the signal that callers (notably writes) require an
        *explicit* multi-target request before fanning out.
        """
        # Only an explicit str/list is a real target. Anything else (None, or an
        # unresolved pydantic ``FieldInfo`` default when a tool fn is called
        # directly rather than via ``_execute_tool``) routes to the default —
        # never a spurious fan-out.
        if target is None or not isinstance(target, str | list | tuple):
            return [DEFAULT_NAME], False
        if isinstance(target, list | tuple):
            names = [str(x).strip() for x in target if str(x).strip()]
            return (names or [DEFAULT_NAME]), len(names) > 1
        t = str(target).strip()
        if t == "" or t.lower() == DEFAULT_NAME:
            return [DEFAULT_NAME], False
        if t.lower() == "all":
            return self.names(), True
        if "," in t:
            names = [x.strip() for x in t.split(",") if x.strip()]
            return (names or [DEFAULT_NAME]), len(names) > 1
        return [t], False

    def safe_get_engine(self, name: str) -> tuple[Any, str | None]:
        """``get_engine`` variant for fan-out: returns ``(engine, error)`` instead
        of raising, so one bad/unreachable target never aborts the others."""
        try:
            return self.get_engine(name), None
        except Exception as exc:  # noqa: BLE001 — partial-success contract
            logger.warning(
                "Graph connection resolution failed (exception_type=%s)",
                type(exc).__name__,
            )
            return None, CONNECTION_UNAVAILABLE

    def probe(self, name: str) -> bool:
        """Validate and connect one named source through its native read seam.

        Runtime references, authentication, and TLS are resolved only inside the
        connector. No source row, endpoint, alias, reference, or exception text
        is returned to the caller; doctor reduces this boolean further to counts.
        """

        clean = (name or "").strip()
        if not clean or clean == DEFAULT_NAME:
            raise ValueError("only named external connections can be probed")
        target = self.get_engine(clean)
        probe = getattr(target, "probe_connection", None)
        if callable(probe):
            return bool(probe())
        validate = getattr(target, "validate_runtime_profiles", None)
        if callable(validate):
            validate()
        execute_read = getattr(target, "execute_read", None)
        if not callable(execute_read):
            raise RuntimeError("configured connection has no read probe")
        execute_read("MATCH (n) RETURN n LIMIT 0", {})
        return True

    # ── health / lifecycle ─────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """Per-connection health surface (CONCEPT:AU-KG.backend.multi-connection-registry / OS-5.28 style)."""
        conns: list[dict[str, Any]] = []
        # Default
        active = None
        try:
            active = self._default_engine()
        except Exception:  # noqa: BLE001
            active = None
        conns.append(
            {
                "name": DEFAULT_NAME,
                "role": "authority",
                "backend_type": _backend_type(active),
                "connected": active is not None,
                "is_default_target": True,
                "supports_sparql": _supports_sparql(active),
                "cypher_support": _cypher_support(active),
            }
        )
        with self._lock:
            specs = dict(self._specs)
            cached = dict(self._engines)
        for name, spec in sorted(specs.items()):
            eng = cached.get(name)
            entry: dict[str, Any] = {
                "name": name,
                "role": spec.get("role") or DEFAULT_ROLE,
                "backend_type": spec.get("backend_type") or spec.get("backend"),
                "connected": eng is not None,
                "is_default_target": False,
            }
            if eng is not None:
                entry["supports_sparql"] = _supports_sparql(eng)
                entry["cypher_support"] = _cypher_support(eng)
            conns.append(entry)
        return {"default_target": DEFAULT_NAME, "connections": conns}

    def spec_summary(self, name: str) -> dict[str, Any]:
        """Metadata-only summary; never returns endpoint, path, DB, or identity."""

        clean = (name or "").strip()
        with self._lock:
            spec = dict(self._specs.get(clean) or {})
        return {
            "backend": spec.get("backend_type") or spec.get("backend"),
            "endpoint_configured": any(
                spec.get(key) for key in ("uri", "host", "endpoint", "endpoint_ref")
            ),
            "database_configured": any(
                spec.get(key) for key in ("db_name", "database", "graph_name")
            ),
            "tls_profile_configured": bool(
                spec.get("tls_profile") or spec.get("tls_profile_ref")
            ),
            "connection_profile_configured": bool(spec.get("connection_profile_ref")),
        }

    def close_all(self) -> None:
        """Close every cached named engine's backend (default is left to the
        process lifecycle)."""
        with self._lock:
            engines = list(self._engines.values())
            self._engines.clear()
        for eng in engines:
            self._safe_close(eng)

    @staticmethod
    def _safe_close(engine: Any) -> None:
        backend = getattr(engine, "backend", None)
        target = backend if backend is not None else engine
        if hasattr(target, "close"):
            try:
                target.close()
            except Exception as exc:  # noqa: BLE001 — best-effort teardown
                logger.debug(
                    "Error closing backend during registry teardown (%s)",
                    type(exc).__name__,
                )


def _backend_type(engine: Any) -> str | None:
    backend = getattr(engine, "backend", None)
    if backend is not None:
        return type(backend).__name__
    value = getattr(engine, "backend_kind", None)
    return str(value) if value else None


def _supports_sparql(engine: Any) -> bool:
    backend = getattr(engine, "backend", None)
    return (
        bool(getattr(backend, "supports_sparql", False))
        if backend is not None
        else False
    )


def _cypher_support(engine: Any) -> str | None:
    backend = getattr(engine, "backend", None)
    if backend is not None:
        return str(getattr(backend, "cypher_support", "full"))
    value = getattr(engine, "cypher_support", None)
    return str(value) if value else None
