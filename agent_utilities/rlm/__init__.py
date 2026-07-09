"""Recursive Language Models (RLM) Module.

CONCEPT:AU-ORCH.execution.rlm-execution — RLM Execution
"""

from .client import (  # CONCEPT:AU-ORCH.execution.drop-rlm-completion-client — drop-in completion surface
    RLM,
    RLMResponse,
)
from .config import RLMConfig

__all__ = ["RLMConfig", "RLM", "RLMResponse"]
