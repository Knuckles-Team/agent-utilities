"""Embedding diagnostics — consolidated into knowledge_stability_engine.py.

CONCEPT:KG-2.6 — CKA and alignment diagnostics now live in
:mod:`agent_utilities.knowledge_graph.memory.knowledge_stability_engine`.
"""

from ..memory.knowledge_stability_engine import (  # noqa: F401
    CKAResult,
    EmbeddingHealthReport,
    FusionResult,
)
