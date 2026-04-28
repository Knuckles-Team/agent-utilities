"""
Unified ID Management System for Knowledge Graph.

This module provides unified ID generation and management across
document database, vector database, and knowledge graph storage layers.
"""

from .unified_id import UnifiedIDManager, UnifiedIDRegistry

__all__ = ["UnifiedIDManager", "UnifiedIDRegistry"]
