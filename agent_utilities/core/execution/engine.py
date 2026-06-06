"""Reference unified engine implementing the shared ExecutionEngine contract.

Plan 03 Step 5 — unify the ExecutionEngine contract.

CONCEPT:ORCH-1.33 — additive multi-CLI adapter dispatch. When a manifest requests an external runtime
backend (``manifest.metadata["runtime"] = "<adapter_id>"`` or an ``AgentSpec`` whose ``model_id`` is
``"adapter:<id>"``), the engine resolves the adapter from the :class:`AdapterRegistry`, spawns it via
the adapter executor, and records the producing adapter as provenance on the result. When no runtime
is requested the engine preserves its prior trivial behaviour (zero regression).
"""

from __future__ import annotations

import logging
from typing import Any

from .adapters.executor import AdapterExecutionError, run_adapter_text
from .adapters.registry import AdapterRegistry, get_adapter_registry
from .models import ExecutionManifest, ExecutionResult
from .protocol import ExecutionEngine

logger = logging.getLogger(__name__)

# implements core.execution.ExecutionEngine


class UnifiedExecutionEngine(ExecutionEngine):
    """Unified Execution Engine conforming to :class:`ExecutionEngine`.

    Implements the shared ``run(manifest) -> ExecutionResult`` contract, with an additive
    adapter-dispatch path (ORCH-1.33) for driving external agent-CLI backends.
    """

    def __init__(self, *, registry: AdapterRegistry | None = None, **kwargs: Any):
        self.kwargs = kwargs
        self._registry = registry or get_adapter_registry()

    @staticmethod
    def _requested_runtime(manifest: ExecutionManifest) -> str | None:
        """Extract a requested adapter id from manifest metadata or the first agent's ``model_id``."""
        meta = getattr(manifest, "metadata", {}) or {}
        rt = meta.get("runtime")
        if rt:
            return str(rt)
        agents = getattr(manifest, "agents", None) or []
        if agents:
            mid = getattr(agents[0], "model_id", "") or ""
            if mid.startswith("adapter:"):
                return mid.split(":", 1)[1]
        return None

    async def run(self, manifest: ExecutionManifest) -> ExecutionResult:
        """Execute the given manifest and return a canonical result."""
        runtime = self._requested_runtime(manifest)
        if not runtime:
            return ExecutionResult(
                manifest_id=getattr(manifest, "manifest_id", ""),
                agent_count=getattr(manifest, "agent_count", 0),
                success=True,
            )

        definition = self._registry.get(runtime)
        if definition is None:
            logger.warning("requested runtime adapter %r not registered", runtime)
            return ExecutionResult(
                manifest_id=getattr(manifest, "manifest_id", ""),
                agent_count=0,
                success=False,
            )

        agents = getattr(manifest, "agents", None) or []
        prompt = (agents[0].task_template if agents else "") or getattr(
            manifest, "query", ""
        )
        raw_model = (agents[0].model_id if agents else "") or ""
        model = (
            raw_model.split("adapter:", 1)[-1]
            if raw_model.startswith("adapter:")
            else raw_model
        )
        # ``adapter:<id>`` carries no model; only a bare model id should be forwarded.
        model = model if (model and model != runtime) else None
        try:
            text = await run_adapter_text(definition, prompt, model=model)
            success = True
        except AdapterExecutionError as exc:
            logger.warning("adapter %r execution failed: %s", runtime, exc)
            text, success = str(exc), False

        telemetry: dict[str, Any] = {"runtime_adapter": runtime}
        # CONCEPT:AHE-3.13 — pre-emit quality gate on the live execution path (default off; warn/block).
        meta = getattr(manifest, "metadata", {}) or {}
        gate_mode = meta.get("quality_gate")
        if success and gate_mode in {"warn", "block"}:
            from agent_utilities.harness.quality_gates import PreEmitGate

            gate = PreEmitGate(mode=gate_mode)
            gate_result = gate.evaluate(text)
            telemetry["quality_gate"] = {
                "mode": gate_mode,
                "ok": gate_result.ok,
                "scores": gate_result.critique.scores,
                "antipatterns": gate_result.critique.antipatterns,
            }
            if gate_result.blocked:
                success = False

        return ExecutionResult(
            manifest_id=getattr(manifest, "manifest_id", ""),
            agent_count=1,
            success=success,
            synthesis_output=text,
            telemetry=telemetry,
        )
