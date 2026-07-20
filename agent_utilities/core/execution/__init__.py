"""Unified execution contract package.

Plan 03 Step 5 — single source of truth for the execution contract.

Exports the shared Protocol (:class:`ExecutionEngine`) and the canonical
shared models (:class:`ExecutionStep`, :class:`ExecutionResult`,
:class:`ExecutionManifest`).
"""

from .engine import UnifiedExecutionEngine
from .models import ExecutionManifest, ExecutionResult, ExecutionStep
from .protocol import DistributedCoordinatorProtocol, ExecutionEngine

__all__ = [
    "ExecutionEngine",
    "DistributedCoordinatorProtocol",
    "ExecutionStep",
    "ExecutionResult",
    "ExecutionManifest",
    "UnifiedExecutionEngine",
]
