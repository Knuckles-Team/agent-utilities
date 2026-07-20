"""
Unified ID Management System for Knowledge Graph.

This module provides unified ID generation and management across
document database, vector database, and knowledge graph storage layers.
"""

from .ontological_identifier import (
    OntologicalIdentifierManager,
    OntologicalIdentifierRegistry,
)

__all__ = ["OntologicalIdentifierManager", "OntologicalIdentifierRegistry"]
