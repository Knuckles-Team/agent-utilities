"""EWC module — consolidated into knowledge_stability_engine.py.

CONCEPT:AHE-3.4 — All EWC functionality now lives in
:mod:`agent_utilities.knowledge_graph.memory.knowledge_stability_engine`.
"""

from .knowledge_stability_engine import (  # noqa: F401
    apply_ewc_consolidation,
    compute_fisher_diagonal_proxy,
)
