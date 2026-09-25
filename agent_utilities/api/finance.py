"""The finance agent port GraphOS calls (EH-419).

The flip explainer is an LLM agent role, so it lives in agent-utilities;
GraphOS reaches it through this public port only.
"""

from agent_utilities.domains.finance.flip_explainer import (
    Evidence,
    FlipExplanation,
    explain_flip,
)

__all__ = ["Evidence", "FlipExplanation", "explain_flip"]
