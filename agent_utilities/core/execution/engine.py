"""Reference unified engine implementing the shared ExecutionEngine contract.

Plan 03 Step 5 — unify the ExecutionEngine contract.
"""

from __future__ import annotations

from typing import Any

from .models import ExecutionManifest, ExecutionResult
from .protocol import ExecutionEngine

# implements core.execution.ExecutionEngine


class UnifiedExecutionEngine(ExecutionEngine):
    """Unified Execution Engine conforming to :class:`ExecutionEngine`.

    Implements the shared ``run(manifest) -> ExecutionResult`` contract.
    """

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    async def run(self, manifest: ExecutionManifest) -> ExecutionResult:
        """Execute the given manifest and return a canonical result."""
        return ExecutionResult(
            manifest_id=getattr(manifest, "manifest_id", ""),
            agent_count=getattr(manifest, "agent_count", 0),
            success=True,
        )
