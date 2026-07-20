from __future__ import annotations

"""Connector registry + factory — self-registering source discovery.

CONCEPT:AU-ECO.connector.factory-ingestion-adaptor — Connector Registry + Factory

A decorator-driven registry (mirroring the project's other ``register_*`` +
``pkgutil`` discovery patterns) that maps a ``source_type`` string to a
:class:`~agent_utilities.protocols.source_connectors.base.BaseSourceConnector`
subclass, plus a :func:`build_connector` factory that instantiates one from a
config dict.

Wire-First note (see AGENTS.md): ``check_wiring.py`` cannot see decorator
registration, so :func:`discover` (which imports the ``connectors`` subpackage to
run the ``@register_source`` decorators) is invoked on the **live ingestion
path** (the ``CONNECTOR`` adaptor) and asserted by a live-path test — not relied
on for import-graph reachability.
"""

import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import Any, TypeVar

from .base import BaseSourceConnector

logger = logging.getLogger(__name__)

__all__ = [
    "ConnectorGovernanceError",
    "register_source",
    "build_connector",
    "get_connector_class",
    "list_sources",
    "discover",
]

_SOURCE_REGISTRY: dict[str, type[BaseSourceConnector]] = {}
_DISCOVERED = False

C = TypeVar("C", bound=type[BaseSourceConnector])


class ConnectorGovernanceError(PermissionError):
    """A connector activation lacks a valid signed capability bundle."""


def _development_governance_bypass() -> bool:
    """Permit isolated unit fixtures only; production can never bypass the gate."""

    from agent_utilities.base_utilities import to_boolean
    from agent_utilities.core.config import setting
    from agent_utilities.core.profile_guard import is_production_profile

    return not is_production_profile() and to_boolean(
        setting("AGENT_UTILITIES_TESTING", False)
    )


def _governance_source(source_type: str, config: dict[str, Any]) -> str:
    """Resolve a transport connector to the package whose bundle owns it."""

    if source_type == "mcp_tool":
        from .connectors.mcp_tool import get_tool_preset, preset_provider

        preset = str(config.get("preset") or "").strip()
        provider = preset_provider(preset) if preset else None
        preset_config = get_tool_preset(preset) if preset else {}
        return str(
            provider
            or config.get("server")
            or preset_config.get("server")
            or source_type
        )
    if source_type == "mcp":
        return str(config.get("package") or config.get("server") or source_type)
    return source_type


def _require_certified_bundle(source_type: str, config: dict[str, Any]) -> None:
    if _development_governance_bypass():
        return
    from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (
        precheck_source,
    )

    governed_source = _governance_source(source_type, config)
    result = precheck_source(governed_source)
    if not result.get("checked") or not result.get("ok"):
        raise ConnectorGovernanceError(
            "connector activation refused: a signed, provider-verified capability "
            "bundle is required"
        )


def register_source(source_type: str) -> Callable[[C], C]:
    """Class decorator registering a connector under ``source_type``.

    CONCEPT:AU-ECO.connector.factory-ingestion-adaptor. Idempotent: re-registering the same key overwrites, so
    repeated :func:`discover` calls are safe.

    Example::

        @register_source("web")
        class WebCrawlerConnector(LoadConnector, PollConnector):
            ...
    """

    def _decorator(cls: C) -> C:
        if not isinstance(cls, type) or not issubclass(cls, BaseSourceConnector):
            raise TypeError(
                f"@register_source target must subclass BaseSourceConnector, got {cls!r}"
            )
        cls.source_type = source_type
        _SOURCE_REGISTRY[source_type] = cls
        logger.debug(
            "[ECO-4.27] registered source connector %r -> %s", source_type, cls.__name__
        )
        return cls

    return _decorator


def discover() -> dict[str, type[BaseSourceConnector]]:
    """Import the ``connectors`` subpackage so all decorators run; return the map.

    CONCEPT:AU-ECO.connector.factory-ingestion-adaptor — the discovery half of the self-registering pattern. Called
    on the live ingestion path (the ``CONNECTOR`` adaptor builds connectors via
    :func:`build_connector`, which calls ``discover`` first). Safe to call many
    times — module imports are cached and registration is idempotent.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return dict(_SOURCE_REGISTRY)
    try:
        pkg = importlib.import_module(f"{__package__}.connectors")
    except Exception as exc:  # noqa: BLE001 — discovery must not break ingestion
        logger.warning("[ECO-4.27] connector discovery failed: %s", exc)
        _DISCOVERED = True
        return dict(_SOURCE_REGISTRY)
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
        except Exception as exc:  # noqa: BLE001 — one bad connector must not block others
            logger.warning(
                "[ECO-4.27] failed to import connector %r: %s", mod.name, exc
            )
    _DISCOVERED = True
    return dict(_SOURCE_REGISTRY)


def get_connector_class(source_type: str) -> type[BaseSourceConnector] | None:
    """Return the connector class for ``source_type`` (after discovery)."""
    discover()
    return _SOURCE_REGISTRY.get(source_type)


def list_sources() -> list[str]:
    """Return all registered ``source_type`` keys (after discovery)."""
    discover()
    return sorted(_SOURCE_REGISTRY)


def build_connector(
    source_type: str, config: dict[str, Any] | None = None
) -> BaseSourceConnector:
    """Instantiate the connector registered under ``source_type``.

    CONCEPT:AU-ECO.connector.factory-ingestion-adaptor — the factory the ingestion adaptor and MCP tool call.

    Args:
        source_type: A registered key (see :func:`list_sources`).
        config: Connector-specific configuration forwarded to its constructor.

    Raises:
        KeyError: when ``source_type`` is not registered (message lists the
            available keys so the caller can correct the request).
    """
    cls = get_connector_class(source_type)
    if cls is None:
        raise KeyError(
            f"No source connector registered for {source_type!r}. "
            f"Available: {', '.join(list_sources()) or '(none)'}"
        )
    resolved_config = dict(config or {})
    _require_certified_bundle(source_type, resolved_config)
    return cls(**resolved_config)
