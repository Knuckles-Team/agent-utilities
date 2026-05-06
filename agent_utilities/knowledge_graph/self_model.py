"""Backward-compatible alias for the renamed memory_retriever module.

The ``SelfModel`` class was renamed to ``MemoryRetriever`` and its module
from ``self_model.py`` to ``memory_retriever.py`` during the CONCEPT:KG-2.0
migration.  This shim preserves the old import path so existing tests and
external integrations continue to work.
"""

from agent_utilities.knowledge_graph.memory_retriever import (
    MemoryRetriever as SelfModel,
)

__all__ = ["SelfModel"]
