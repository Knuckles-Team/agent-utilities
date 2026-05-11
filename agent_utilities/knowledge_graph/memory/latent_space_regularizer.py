"""Latent space regularizer — consolidated into knowledge_stability_engine.py.

CONCEPT:KG-2.6 — Anti-collapse now lives in
:mod:`agent_utilities.knowledge_graph.memory.knowledge_stability_engine`.
"""

from .knowledge_stability_engine import (  # noqa: F401
    CollapseReport,
    DiversityMetrics,
)
