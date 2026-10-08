"""Serve installed ontology providers as MCP resources (spec: baseline-ingestion).

The fleet server factory already serves skills as ``skill://`` and prompts as
``prompt://``. This module adds each installed ``agent_utilities.ontology_providers``
package's ontologies and SHACL shapes through the connector SDK's own content
helper, so one URI contract covers SDK connectors and factory-built servers:

* ``<ontology root>/*.ttl`` -> ``ontology://<provider>/<file>.ttl``
* ``<ontology root>/shapes/*.ttl`` -> ``shapes://<provider>/<file>.ttl``

Skills and prompts stay on their existing providers, so nothing registers twice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_Register = Callable[[Any, str, Path], int]


def _ontology_sources() -> tuple[_Register, list[tuple[str, Path]]]:
    from agent_connector_sdk.mcp.content import register_ontology_resources

    from agent_utilities.core.providers import resolve_ontology_provider_dirs

    return register_ontology_resources, resolve_ontology_provider_dirs()


def _register_provider(register: _Register, mcp: Any, *, name: str, root: Path) -> int:
    try:
        return register(mcp, name, root)
    except Exception as exc:  # noqa: BLE001 - one provider must not sink the sweep
        logger.warning(
            "Could not register ontology provider %s: %s", name, type(exc).__name__
        )
        return 0


def register_ontology_providers(mcp: Any) -> int:
    """Register every ontology provider's resources on ``mcp``; return the count.

    Never raises: one unreadable provider logs a warning and is skipped, and
    any other failure degrades to one warning. Serving ontologies must never
    stop a server being built.
    """
    try:
        register, providers = _ontology_sources()
    except Exception as exc:  # noqa: BLE001 - optional surface; never block startup
        logger.warning("Could not register ontology providers: %s", type(exc).__name__)
        return 0
    registered = sum(
        _register_provider(register, mcp, name=name, root=root)
        for name, root in providers
    )
    logger.info("Registered %d ontology/shape resource(s)", registered)
    return registered


__all__ = ["register_ontology_providers"]
