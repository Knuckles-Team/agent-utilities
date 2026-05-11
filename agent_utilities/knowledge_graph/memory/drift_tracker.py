"""Drift tracker — consolidated into knowledge_stability_engine.py.

CONCEPT:AHE-3.4 — All drift detection now lives in
:mod:`agent_utilities.knowledge_graph.memory.knowledge_stability_engine`.
"""

from .knowledge_stability_engine import (  # noqa: F401
    DriftReport,
    calculate_cosine_distance,
    check_knowledge_drift,
)
