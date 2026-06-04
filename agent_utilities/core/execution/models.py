"""Canonical shared execution models — single source of truth.

Plan 03 Step 5 — unify the ExecutionEngine contract.

This module is the **single import path** for the shared execution-contract
models. To avoid creating a *duplicate* set of classes, it re-exports the
already-canonical definitions that live in :mod:`agent_utilities.models`:

    - ``ExecutionStep``      -> ``agent_utilities.models.graph.ExecutionStep``
                                (an alias of ``Task`` in ``models.sdd``)
    - ``ExecutionManifest``  -> ``agent_utilities.models.execution_manifest``
    - ``ExecutionResult``    -> ``agent_utilities.models.execution_manifest``

There is therefore exactly **one** class object per model; the engines and
``core.execution`` resolve to the *same* object (verified by ``is`` identity
in ``tests/unit/execution/test_execution_contract.py``). Importers may use
either the engine-local path or ``core.execution`` — they are the same class.
"""

from __future__ import annotations

from agent_utilities.models.execution_manifest import (
    ExecutionManifest,
    ExecutionResult,
)
from agent_utilities.models.graph import ExecutionStep

__all__ = [
    "ExecutionStep",
    "ExecutionManifest",
    "ExecutionResult",
]
