"""Unified ExecutionEngine contract.

Plan 03 Step 5 — unify the ExecutionEngine contract.

The three historical engines

    - ``orchestration.engine.AgentOrchestrationEngine``
    - ``graph.executor`` (a step-function module, wrapped by
      ``graph.executor.GraphExecutorEngine``)
    - ``knowledge_graph.core.engine.IntelligenceGraphEngine``

each had their own bespoke execution entrypoint. The **most general common
shape** they share is::

    async def run(self, manifest) -> ExecutionResult

``AgentOrchestrationEngine`` already exposes this exact shape via its
``execute(manifest) -> ExecutionResult`` method, which is the canonical
Parallel Engine entrypoint (CONCEPT:ORCH-1.8). The other engines gain an
**additive** ``run`` adapter that conforms to this Protocol without changing
any existing behaviour.

The Protocol is ``runtime_checkable`` so conformance can be asserted
structurally (``isinstance(engine, ExecutionEngine)``) — it checks only that
a callable ``run`` attribute exists, per :mod:`typing` semantics.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import ExecutionResult


@runtime_checkable
class ExecutionEngine(Protocol):
    """Unified contract every execution engine conforms to.

    The single shared entrypoint is :meth:`run`, accepting an execution
    manifest and returning an :class:`ExecutionResult`. The ``manifest``
    parameter is intentionally typed as ``Any`` at the Protocol boundary so
    engines that accept either an ``ExecutionManifest`` or an engine-native
    spec (and normalise internally) still conform to the most general common
    shape.
    """

    async def run(self, manifest: Any) -> ExecutionResult:
        """Execute ``manifest`` and return an :class:`ExecutionResult`."""
        ...


@runtime_checkable
class DistributedCoordinatorProtocol(Protocol):
    """Protocol for distributed execution coordination."""

    async def coordinate(self, task: Any) -> Any: ...
