# CONCEPT:KG-2.63 - Named multi-connection graph registry: register N live graph backends (neo4j/falkordb/postgres-AGE/…) by name and run the same MCP/REST tools against any one or fan out to all, with the backend choice fully abstracted behind a `target` parameter.
# CONCEPT:KG-2.89 - Role-aware multi-database registry plus live config mutation: every external graph DB is labelled mirror, read (data source), or read_write, persisted durably in config.json, credentials given as literals or vault/env refs, with generic get/set/list of any config item live over both the MCP server and the API gateway, and a doctor health-check across all connections.
"""Named multi-connection graph registry.

CONCEPT:KG-2.63 — Multi-Connection Graph Registry. The engine has always been
vendor-agnostic (one ``GraphBackend`` interface, many implementations), but only
ONE backend was ever live at a time: a module-global default chosen at startup
from ``GRAPH_BACKEND``. This registry lets a deployment keep several live
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
* **Each named connection wraps its own engine.** A named connection builds a
  ``create_backend(**spec)`` backend and a dedicated ``IntelligenceGraphEngine``
  around it, so per-engine scoping / visibility / temporal filters still apply.
  Engines are built once and cached (lazy connect — an unreachable backend only
  fails when it is actually targeted, never at registration).
* **Backward compatible.** No ``target`` (or ``target="default"``) routes through
  the legacy single-engine path unchanged.

This mirrors the zero-infra-preserving shape of the shard topology
(:mod:`agent_utilities.knowledge_graph.core.shard_topology`, CONCEPT:KG-2.58):
config-list resolution, a default-preserving single-entry mode, and a
``status()`` health surface.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Reserved name for the process-wide active engine (the legacy default).
DEFAULT_NAME = "default"

#: Names that may not be used for a registered connection.
_RESERVED = {DEFAULT_NAME, "all", ""}

#: Connection roles (CONCEPT:KG-2.89). ``read`` = query-only data source;
#: ``read_write`` = query + write; ``mirror`` = receives fan-out replication of OUR
#: KG (never a direct ``target=`` write). The default/authority connection is always
#: read_write. New connections default to ``read`` (opt into writes).
_ROLES = {"read", "read_write", "mirror"}
DEFAULT_ROLE = "read"

_SECRETS_CLIENT: Any = None


def _resolve_secret(value: Any) -> Any:
    """Resolve a secret reference to its value; a plain literal passes through.

    Supports ``vault://…`` / ``env://VAR`` / ``sqlite://…`` (CONCEPT:KG-2.89 — a
    connection's password/user/uri may be a literal OR a secret reference, keeping
    secrets out of config.json). Fail-safe: an unresolvable ref returns the literal.
    """
    if not isinstance(value, str) or not value.startswith(
        ("vault://", "env://", "sqlite://")
    ):
        return value
    global _SECRETS_CLIENT
    try:
        if _SECRETS_CLIENT is None:
            from agent_utilities.security.secrets_client import create_secrets_client

            _SECRETS_CLIENT = create_secrets_client()
        resolved = _SECRETS_CLIENT.resolve_ref(value)
        return resolved if resolved is not None else value
    except Exception as e:  # noqa: BLE001 — fail-safe to the literal
        logger.warning("secret-ref resolution failed (%.24s): %s", value, e)
        return value


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
        self._default_target = DEFAULT_NAME
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
        # Accept both "backend" and "backend_type" as the selector key.
        if "backend" in spec and "backend_type" not in spec:
            spec["backend_type"] = spec.pop("backend")
        # CONCEPT:KG-2.89 — normalize the connection role (default: read).
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
            if self._default_target == clean:
                self._default_target = DEFAULT_NAME
        if eng is not None:
            self._safe_close(eng)
        return had

    def names(self) -> list[str]:
        """All addressable connection names, default first."""
        with self._lock:
            return [DEFAULT_NAME, *sorted(self._specs)]

    def role(self, name: str | None) -> str:
        """Role of a connection (CONCEPT:KG-2.89): ``read`` | ``read_write`` |
        ``mirror``. The default/authority connection is always ``read_write``."""
        clean = (name or DEFAULT_NAME).strip() or DEFAULT_NAME
        if clean == DEFAULT_NAME:
            return "read_write"
        with self._lock:
            spec = self._specs.get(clean)
        return str((spec or {}).get("role") or DEFAULT_ROLE)

    def is_writable(self, name: str | None) -> bool:
        """Whether ``target=name`` writes are allowed (only ``read_write``/default).

        ``read`` (data source) and ``mirror`` (fan-out replica — written only via the
        outbox, never directly) reject ``target=`` writes.
        """
        return self.role(name) == "read_write"

    def export_specs(self) -> list[dict[str, Any]]:
        """All registered connections as a config list (each entry = ``{name, **spec}``).

        For durable persistence to ``config.json`` (``kg_connections``). Secrets are
        kept exactly as registered (literal or ``vault://``/``env://`` ref); resolution
        happens only at connect, on a copy (CONCEPT:KG-2.89)."""
        with self._lock:
            return [{"name": n, **s} for n, s in sorted(self._specs.items())]

    def default_name(self) -> str:
        return self._default_target

    def set_default(self, name: str) -> str:
        """Repoint which connection an empty/`default` target resolves to."""
        clean = (name or "").strip() or DEFAULT_NAME
        if clean != DEFAULT_NAME:
            with self._lock:
                if clean not in self._specs:
                    raise KeyError(f"Unknown connection '{clean}'")
        self._default_target = clean
        return clean

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
        engine = self._build_engine(spec)

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
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        # Ensure the process-wide active engine exists FIRST, so this named
        # engine's construction never auto-registers itself as the global
        # default (auto-registration only fires when none exists).
        self._default_engine()

        # CONCEPT:KG-2.89 — ``role`` is registry metadata (not a backend kwarg), and
        # a credential may be a secret reference resolved at connect (never stored
        # raw in config.json).
        build_spec = {k: v for k, v in spec.items() if k != "role"}
        for key in ("password", "user", "uri"):
            if key in build_spec:
                build_spec[key] = _resolve_secret(build_spec[key])

        backend = create_backend(**build_spec)
        if backend is None:
            raise RuntimeError(
                f"Backend for connection spec {spec!r} is unavailable "
                "(missing driver/package or unreachable)."
            )
        return IntelligenceGraphEngine(backend=backend)

    # ── target resolution ──────────────────────────────────────────────────
    def resolve_names(self, target: Any) -> tuple[list[str], bool]:
        """Resolve a ``target`` into ``(names, fanout)``.

        * ``None`` / ``""`` / ``"default"`` → ``([default_target], False)`` — the
          legacy single-engine path.
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
            return [self._default_target], False
        if isinstance(target, list | tuple):
            names = [str(x).strip() for x in target if str(x).strip()]
            return (names or [self._default_target]), len(names) > 1
        t = str(target).strip()
        if t == "" or t.lower() == DEFAULT_NAME:
            return [self._default_target], False
        if t.lower() == "all":
            return self.names(), True
        if "," in t:
            names = [x.strip() for x in t.split(",") if x.strip()]
            return (names or [self._default_target]), len(names) > 1
        return [t], False

    def safe_get_engine(self, name: str) -> tuple[Any, str | None]:
        """``get_engine`` variant for fan-out: returns ``(engine, error)`` instead
        of raising, so one bad/unreachable target never aborts the others."""
        try:
            return self.get_engine(name), None
        except Exception as e:  # noqa: BLE001 — partial-success contract
            return None, str(e)

    # ── health / lifecycle ─────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """Per-connection health surface (CONCEPT:KG-2.63 / OS-5.28 style)."""
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
                "role": "read_write",
                "backend_type": _backend_type(active),
                "connected": active is not None,
                "is_default_target": self._default_target == DEFAULT_NAME,
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
                "is_default_target": self._default_target == name,
            }
            if eng is not None:
                entry["supports_sparql"] = _supports_sparql(eng)
                entry["cypher_support"] = _cypher_support(eng)
            conns.append(entry)
        return {"default_target": self._default_target, "connections": conns}

    def spec_summary(self, name: str) -> dict[str, Any]:
        """Redacted summary of a connection's spec — backend + endpoint, NO secrets.

        Used by the external-graph imprint so the catalog node never persists
        credentials. A ``user:pass@`` embedded in a URI is masked.
        """
        import re as _re

        clean = (name or "").strip()
        with self._lock:
            spec = dict(self._specs.get(clean) or {})
        endpoint = str(spec.get("uri") or spec.get("host") or "")
        endpoint = _re.sub(r"//[^/@]*@", "//***@", endpoint)
        return {
            "backend": spec.get("backend_type") or spec.get("backend"),
            "endpoint": endpoint,
            "db_name": spec.get("db_name") or spec.get("database"),
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
        if backend is not None and hasattr(backend, "close"):
            try:
                backend.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.debug(
                    "Error closing backend during registry teardown", exc_info=True
                )


def _backend_type(engine: Any) -> str | None:
    backend = getattr(engine, "backend", None)
    return type(backend).__name__ if backend is not None else None


def _supports_sparql(engine: Any) -> bool:
    backend = getattr(engine, "backend", None)
    return (
        bool(getattr(backend, "supports_sparql", False))
        if backend is not None
        else False
    )


def _cypher_support(engine: Any) -> str | None:
    backend = getattr(engine, "backend", None)
    return (
        str(getattr(backend, "cypher_support", "full")) if backend is not None else None
    )
