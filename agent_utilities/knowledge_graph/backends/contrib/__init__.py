#!/usr/bin/python
"""Contributed (opt-in) graph backends.

These backends are demoted from the primary tier but remain fully supported
and importable on demand. They are NOT imported eagerly so that the default
backend selection (PostgreSQL + pgvector / EpistemicGraph / file) never
requires the optional ``neo4j``, ``falkordb``, or ``ladybug`` drivers to be
installed.

Opt-in usage::

    from agent_utilities.knowledge_graph.backends.contrib.neo4j_backend import (
        Neo4jBackend,
    )
    from agent_utilities.knowledge_graph.backends.contrib.falkordb_backend import (
        FalkorDBBackend,
    )
    from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
        LadybugBackend,
    )

Or via lazy attribute access on this package::

    from agent_utilities.knowledge_graph.backends import contrib

    contrib.Neo4jBackend  # resolved lazily

The lazy ``__getattr__`` below avoids importing optional driver packages at
package-import time; each class is only imported when its name is accessed.
"""

from typing import Any

__all__ = [
    "Neo4jBackend",
    "FalkorDBBackend",
    "LadybugBackend",
    "LADYBUG_AVAILABLE",
]


def __getattr__(name: str) -> Any:
    if name == "Neo4jBackend":
        from .neo4j_backend import Neo4jBackend

        return Neo4jBackend
    if name == "FalkorDBBackend":
        from .falkordb_backend import FalkorDBBackend

        return FalkorDBBackend
    if name in ("LadybugBackend", "LADYBUG_AVAILABLE"):
        from .ladybug_backend import LADYBUG_AVAILABLE, LadybugBackend

        if name == "LadybugBackend":
            return LadybugBackend
        return LADYBUG_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
