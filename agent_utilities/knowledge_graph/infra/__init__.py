#!/usr/bin/python
from __future__ import annotations

"""Infrastructure inventory + multi-objective placement (CONCEPT:KG-2.9 / KG-2.49).

Two pieces that turn the live hardware estate into an optimized workload plan:

  - :mod:`inventory_collector` — collects host/GPU/network profiles (via the
    tunnel-manager ``tm_system`` MCP) and persists them into the KG infra
    ontology, idempotent via the ingest manifest.
  - :mod:`placement_optimizer` — a multi-objective optimizer (efficiency /
    security / cost / resilience) over the infra subgraph that emits a placement
    plan + golden blueprint, propose-only.
"""

from .inventory_collector import (
    HostProfile,
    InfraInventoryCollector,
    collect_and_persist,
)
from .placement_optimizer import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    HostCapacity,
    PlacementPlan,
    ServicePlacement,
    ServiceSpec,
    optimize_from_graph,
    plan_placements,
)

__all__ = [
    "HostProfile",
    "InfraInventoryCollector",
    "collect_and_persist",
    "HostCapacity",
    "ServiceSpec",
    "ServicePlacement",
    "PlacementPlan",
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "plan_placements",
    "optimize_from_graph",
]
