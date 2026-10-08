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
from typing import Any

logger = logging.getLogger(__name__)


def register_ontology_providers(mcp: Any) -> int:
    """Register every ontology provider's resources on ``mcp``; return the count.

    Never raises: one unreadable provider logs a warning and is skipped, and
    any other failure degrades to one warning. Serving ontologies must never
    stop a server being built.
    """
    try:
        from agent_connector_sdk.mcp.content import register_ontology_resources

        from agent_utilities.core.providers import resolve_ontology_provider_dirs

        registered = 0
        for provider_name, root_dir in resolve_ontology_provider_dirs():
            try:
                registered += register_ontology_resources(mcp, provider_name, root_dir)
            except Exception as exc:  # noqa: BLE001 - one provider must not sink the sweep
                logger.warning(
                    "Could not register ontology provider %s: %s",
                    provider_name,
                    type(exc).__name__,
                )
        logger.info("Registered %d ontology/shape resource(s)", registered)
        return registered
    except Exception as exc:  # noqa: BLE001 - optional surface; never block startup
        logger.warning("Could not register ontology providers: %s", type(exc).__name__)
        return 0


__all__ = ["register_ontology_providers"]
