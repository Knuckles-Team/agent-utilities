#!/usr/bin/python
from __future__ import annotations

"""Schema-Pack active-profile resolution & lifecycle.

CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit — Schema-Pack Lifecycle and Audit

Resolves the *active* :class:`~agent_utilities.models.schema_pack.SchemaPack` for a
deployment and threads pack-change notifications to live consumers (the engine's
``HybridRetriever``, the entity extractor, the OWL bridge). Without this layer the
packs in ``models/schema_packs/`` are declared but never selected. The active
pack is read exclusively from the validated :class:`AgentConfig` snapshot, and
an unknown name fails closed.
"""


import logging
import threading
from collections.abc import Callable

from agent_utilities.core.config import config

from .schema_pack import SchemaPack
from .schema_packs import get_schema_pack

logger = logging.getLogger(__name__)

DEFAULT_PACK_NAME = "core"

_lock = threading.RLock()
_active_pack: SchemaPack | None = None
_listeners: list[Callable[[SchemaPack], None]] = []


def resolve_pack_name() -> str:
    """Return the schema pack selected by the current AgentConfig snapshot."""
    return str(config.graph_schema_pack)


def resolve_active_pack() -> SchemaPack:
    """Instantiate the AgentConfig-selected :class:`SchemaPack` (KG-2.35)."""
    return get_schema_pack(resolve_pack_name())


def get_active_pack() -> SchemaPack:
    """Return the process-wide active pack, resolving lazily on first use."""
    global _active_pack
    with _lock:
        if _active_pack is None:
            _active_pack = resolve_active_pack()
        return _active_pack


def set_active_pack(name: str) -> SchemaPack:
    """Switch the active pack and notify all registered live consumers (KG-2.35).

    Returns the newly-active pack. Listeners (e.g. the engine rebuilding its
    retriever and invalidating retrieval caches via ``pack.signature()``) are
    invoked synchronously; a failing listener is logged but does not abort the
    switch.
    """
    global _active_pack
    with _lock:
        pack = get_schema_pack(name)
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
