"""Orchestration subsystem for agent-utilities.

CONCEPT:OS-5.6 — Distributed Coordination & Recovery
CONCEPT:ORCH-1.0 — Agent Orchestration Layer

This package provides:
- Agent runner (ORCH-1.0) — Core agent execution lifecycle
- Orchestrator (ORCH-1.2) — Multi-agent pool management
- Durable execution (ORCH-1.22) — Persistent workflow execution
- Prediction linkage — Predictive routing for agent task allocation
- Distributed coordinator (OS-5.6) — NATS/local semantic task routing
- Recovery daemon (OS-5.6) — Homeostatic hung-process recovery
"""

from .agent_runner import run_agent
from .distributed_coordinator import DistributedCoordinator
from .durable_execution import DurableExecutionManager
from .manager import Orchestrator
from .prediction_linkage import PredictionLinkageLayer
from .recovery_daemon import RecoveryDaemon

__all__ = [
    # Core orchestration
    "run_agent",
    "Orchestrator",
    "DurableExecutionManager",
    "PredictionLinkageLayer",
    # Distributed coordination (OS-5.6)
    "DistributedCoordinator",
    "RecoveryDaemon",
]
