#!/usr/bin/python
"""OWL Backend Base Interface."""

from abc import ABC, abstractmethod
from typing import Any


class OWLBackend(ABC):
    """Abstract interface for OWL reasoning backends.

    Mirrors the ``GraphBackend`` ABC pattern used by LadybugDB/Neo4j/FalkorDB
    but provides OWL-specific operations: ontology loading, individual
    promotion, reasoning, inference retrieval, and RDF export.
    """

    @abstractmethod
    def load_ontology(self, ontology_path: str) -> None:
        """Load the base OWL ontology (TBox).

        Args:
            ontology_path: Filesystem path to an OWL/Turtle ontology file.
        """

    @abstractmethod
    def promote(self, stable_nodes: list[dict[str, Any]]) -> int:
        """Push stable LPG nodes into the OWL store as individuals (ABox).

        Args:
            stable_nodes: List of node dicts with at least ``id``, ``type``,
                and ``name`` keys. Additional properties are mapped to OWL
                datatype properties where a mapping exists.

        Returns:
            Number of individuals successfully created.
        """

    @abstractmethod
    def promote_edges(self, edges: list[dict[str, Any]]) -> int:
        """Push stable LPG edges into the OWL store as property assertions.

        Args:
            edges: List of edge dicts with ``source``, ``target``, and
                ``type`` keys.

        Returns:
            Number of property assertions created.
        """

    @abstractmethod
    def reason(self) -> list[dict[str, Any]]:
        """Run OWL reasoning and return newly inferred facts.

        Returns:
            List of dicts, each with ``subject``, ``predicate``, ``object``,
            and ``inference_type`` keys.
        """

    @abstractmethod
    def get_inferences(self) -> list[dict[str, Any]]:
        """Return all inferred facts currently available (cached or live).

        Returns:
            Same format as ``reason()``.
        """

    @abstractmethod
    def export_rdf(self, output_path: str, fmt: str = "turtle") -> None:
        """Serialize the full ontology + ABox to a file.

        Args:
            output_path: Filesystem path for the output.
            fmt: Serialization format (turtle, xml, ntriples, etc.).
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all individuals (ABox) while preserving the TBox."""

    @abstractmethod
    def close(self) -> None:
        """Release resources (file handles, connections, etc.)."""

    @abstractmethod
    def get_stats(self) -> dict[str, int]:
        """Return counts of individuals, classes, and properties.

        Returns:
            Dict with keys like ``individuals``, ``classes``, ``properties``.
        """
