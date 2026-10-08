"""Shared ``nodes``/``edges`` store for lightweight in-memory graph-engine doubles.

Several independent test suites each hand-roll a small graph-engine stand-in
over a ``nodes`` dict and an ``edges`` list (``add_node``/``link_nodes`` style
doubles). :class:`NodeEdgeStore` holds just that common store, so each
double's own ``__init__`` states only what makes it distinct instead of
repeating the same two field declarations.
"""

from __future__ import annotations

from typing import Any

__all__ = ["NodeEdgeStore"]


class NodeEdgeStore:
    """The ``nodes``/``edges`` store shared by in-memory graph-engine doubles."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
