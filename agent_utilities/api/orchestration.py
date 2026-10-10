"""Public orchestration entry point for AU application integrations.

Exposes the AU-owned ``Orchestrator`` to application consumers without making
them import the internal orchestration manager module.
"""

from agent_utilities.orchestration.manager import Orchestrator

__all__ = ["Orchestrator"]
