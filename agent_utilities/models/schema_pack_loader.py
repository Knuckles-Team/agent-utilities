#!/usr/bin/python
from __future__ import annotations

"""Schema-Pack active-profile resolution & lifecycle.

CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit — Schema-Pack Lifecycle and Audit

Resolves the *active* :class:`~agent_utilities.models.schema_pack.SchemaPack` for a
deployment and threads pack-change notifications to live consumers (the engine's
``HybridRetriever``, the entity extractor, the OWL bridge). Without this layer the
packs in ``models/schema_packs/`` are declared but never selected — the engine
historically constructed its retriever *pack-blind*.

Resolution precedence (highest first), mirroring gbrain's pack resolution chain:

1. Explicit ``name`` argument (per-call override).
2. ``GRAPH_SCHEMA_PACK`` environment variable.
3. ``graph.schema_pack`` (or ``graph_schema_pack``) key in the XDG ``config.json``.
4. The built-in ``core`` pack (today's behaviour — every Schema-Pack 2.0 signal is a
   no-op under ``core``, so this default is bit-for-bit backward compatible).

An unknown pack name never raises: it logs a warning and falls back to ``core`` so a
typo in config can never take the graph offline.
"""


import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path

from agent_utilities.core.config import setting

from .schema_pack import SchemaPack
from .schema_packs import get_schema_pack

logger = logging.getLogger(__name__)

DEFAULT_PACK_NAME = "core"

_lock = threading.RLock()
_active_pack: SchemaPack | None = None
_listeners: list[Callable[[SchemaPack], None]] = []


def _from_config_json() -> str | None:
    """Read the configured pack name from the XDG ``config.json`` if present.

    Uses the same APP_NAME/APP_AUTHOR/``AGENT_UTILITIES_CONFIG_DIR`` convention as
    ``agent_utilities.core.config._load_xdg_json_config`` so there is a single
    canonical config location.
    """
    try:
        override = setting("AGENT_UTILITIES_CONFIG_DIR")
        if override:
            cfg_dir = Path(override).expanduser()
        else:
            import platformdirs

            cfg_dir = Path(
                platformdirs.user_config_path("agent-utilities", "knuckles-team")
            )
        cfg_file = cfg_dir / "config.json"
        if not cfg_file.exists():
            return None
        data = json.loads(cfg_file.read_text())
    except Exception as e:  # pragma: no cover - defensive config read
        logger.debug("schema-pack config read failed: %s", e)
        return None

    # Accept either a nested {"graph": {"schema_pack": ...}} or a flat key.
    graph_section = data.get("graph")
    if isinstance(graph_section, dict) and graph_section.get("schema_pack"):
        return str(graph_section["schema_pack"])
    for flat in ("graph.schema_pack", "graph_schema_pack", "schema_pack"):
        if data.get(flat):
            return str(data[flat])
    return None


def resolve_pack_name(explicit: str | None = None) -> str:
    """Return the active pack *name* using the documented precedence."""
    if explicit:
        return explicit
    env = setting("GRAPH_SCHEMA_PACK")
    if env:
        return env
    cfg = _from_config_json()
    if cfg:
        return cfg
    return DEFAULT_PACK_NAME


def resolve_active_pack(explicit: str | None = None) -> SchemaPack:
    """Instantiate the active :class:`SchemaPack` (KG-2.35).

    Falls back to the ``core`` pack on any unknown name (warns, never raises).
    """
    name = resolve_pack_name(explicit)
    try:
        return get_schema_pack(name)
    except KeyError:
        logger.warning(
            "Unknown schema pack %r; falling back to %r", name, DEFAULT_PACK_NAME
        )
        return get_schema_pack(DEFAULT_PACK_NAME)


def get_active_pack() -> SchemaPack:
    """Return the process-wide active pack, resolving lazily on first use."""
    global _active_pack
    with _lock:
        if _active_pack is None:
            _active_pack = resolve_active_pack()
        return _active_pack


def set_active_pack(name: str | None) -> SchemaPack:
    """Switch the active pack and notify all registered live consumers (KG-2.35).

    Returns the newly-active pack. Listeners (e.g. the engine rebuilding its
    retriever and invalidating retrieval caches via ``pack.signature()``) are
    invoked synchronously; a failing listener is logged but does not abort the
    switch.
    """
    global _active_pack
    with _lock:
        pack = resolve_active_pack(name)
        _active_pack = pack
        listeners = list(_listeners)
    for cb in listeners:
        try:
            cb(pack)
        except Exception as e:  # pragma: no cover - listener robustness
            logger.warning("schema-pack listener failed: %s", e)
    logger.info("Active schema pack set to %r (sig=%s)", pack.name, pack.signature())
    return pack


def register_listener(callback: Callable[[SchemaPack], None]) -> None:
    """Register a callback invoked with the new pack whenever it changes (KG-2.35)."""
    with _lock:
        if callback not in _listeners:
            _listeners.append(callback)


def unregister_listener(callback: Callable[[SchemaPack], None]) -> None:
    """Remove a previously-registered pack-change listener."""
    with _lock:
        if callback in _listeners:
            _listeners.remove(callback)


__all__ = [
    "DEFAULT_PACK_NAME",
    "resolve_pack_name",
    "resolve_active_pack",
    "get_active_pack",
    "set_active_pack",
    "register_listener",
    "unregister_listener",
]
