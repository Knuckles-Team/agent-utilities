"""Recursive Language Models (RLM) Module.

CONCEPT:ORCH-1.1 — RLM Execution
"""

from .client import RLM, RLMResponse  # CONCEPT:ORCH-1.54 — drop-in completion surface
from .config import RLMConfig

__all__ = ["RLMConfig", "RLM", "RLMResponse"]
