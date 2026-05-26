#!/usr/bin/python
"""OWL Reasoning Backends.

Provides the ``OWLBackend`` ABC and concrete implementations for Owlready2
(default, in-memory + SQLite quadstore) and Stardog (remote, via pystardog).

Use ``create_owl_backend()`` to instantiate the correct backend from
configuration or environment variables.

Environment Variables:
    OWL_BACKEND: Backend type ("owlready2", "stardog"). Default: "owlready2".
    OWL_DB_PATH: SQLite quadstore path for Owlready2. Default: "owl_store.db".
    STARDOG_ENDPOINT: Stardog server endpoint.
    STARDOG_DATABASE: Stardog database name.
    STARDOG_USER: Stardog username.
    STARDOG_PASSWORD: Stardog password.
"""

import logging
import os

from .base import OWLBackend

try:
    from .oxigraph_datalog_backend import OxigraphDatalogBackend
except ImportError:

    class OxigraphDatalogBackendFallback:
        pass

    OxigraphDatalogBackend = OxigraphDatalogBackendFallback  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "OWLBackend",
    "create_owl_backend",
    "OxigraphDatalogBackend",
]


def create_owl_backend(
    backend_type: str | None = None,
    **kwargs,
) -> OWLBackend:
    """Factory function to create the appropriate OWL reasoning backend.

    Args:
        backend_type: One of "owlready2", "stardog". Falls back to
            ``OWL_BACKEND`` env var, then "owlready2".
        **kwargs: Backend-specific configuration passed to the constructor.

    Returns:
        A configured ``OWLBackend`` instance.

    Raises:
        ValueError: If the requested backend type is unknown.
    """
    backend_type = (
        (backend_type or os.environ.get("OWL_BACKEND") or "oxigraph").lower().strip()
    )

    if backend_type == "oxigraph":
        try:
            from .oxigraph_datalog_backend import OxigraphDatalogBackend

            return OxigraphDatalogBackend(**kwargs)
        except ImportError as e:
            logger.warning(
                f"Failed to load high-performance OxigraphDatalogBackend: {e}. "
                "Falling back to legacy owlready2/JVM backend."
            )
            backend_type = "owlready2"

    if backend_type == "owlready2":
        from .owlready2_backend import Owlready2Backend

        return Owlready2Backend(**kwargs)

    elif backend_type == "stardog":
        from .stardog_backend import StardogBackend

        return StardogBackend(**kwargs)

    else:
        raise ValueError(
            f"Unknown OWL backend type: '{backend_type}'. Supported: oxigraph, owlready2, stardog"
        )
