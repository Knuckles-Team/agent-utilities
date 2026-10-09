"""Retirement inventory for knowledge_graph modules that duplicate the
epistemic-graph engine's native OWL/SPARQL/compute authority (AU-RETIRE-R001).

Each entry below records a single, reconciled disposition for one module:

- ``RETAINED``: no parity-verified epistemic-graph-native replacement has
  been identified yet; the module stays and keeps its real callers.
- ``DELETED``: a parity-verified epistemic-graph client replaced it; the
  path has already been removed from the tree.

This is the typed model for the AU-RETIRE-R001 census
(specs/au-engine-duplicate-retirement/tasks.md). The pinned set may only
shrink: a module moves from ``RETAINED`` to ``DELETED`` only once its file
is actually gone, and a ``DELETED`` path must never reappear. New duplicate
candidates are recorded here with a disposition as soon as they are found,
so the set never grows silently.

See ``tests/unit/knowledge_graph/test_retirement_inventory.py`` for the
regression test that enforces this.
"""

from __future__ import annotations

import dataclasses
import enum


class Disposition(str, enum.Enum):
    """The single owner disposition assigned to a duplicate-candidate module."""

    RETAINED = "retained"
    DELETED = "deleted"


@dataclasses.dataclass(frozen=True)
class RetirementEntry:
    """One reconciled census entry for a knowledge_graph duplicate candidate."""

    path: str
    disposition: Disposition
    reason: str


RETIREMENT_INVENTORY: tuple[RetirementEntry, ...] = (
    RetirementEntry(
        path="agent_utilities/knowledge_graph/core/graph_compute.py",
        disposition=Disposition.RETAINED,
        reason=(
            "GraphComputeEngine has 281 non-test callers across every domain "
            "(finance, security/RBAC, observability, MCP tools, deployment); no "
            "per-call-site EG-native parity map exists yet (AU-RETIRE-R001 census, "
            "2026-10-08)."
        ),
    ),
    RetirementEntry(
        path="agent_utilities/knowledge_graph/core/formal_reasoning_core.py",
        disposition=Disposition.RETAINED,
        reason=(
            "Every exported symbol has a live importer or a registered dynamic-load "
            "capability; no EG-native replacement has been identified for its formal "
            "graph-theory or causal/probabilistic reasoning surface (AU-RETIRE-R001 "
            "census, 2026-10-08)."
        ),
    ),
    RetirementEntry(
        path="agent_utilities/knowledge_graph/backends/sparql",
        disposition=Disposition.DELETED,
        reason=(
            "Legacy SPARQL backend and setup path removed (commit 3bde01e76); "
            "superseded by the epistemic-graph-native client."
        ),
    ),
    RetirementEntry(
        path="agent_utilities/knowledge_graph/core/owl_bridge.py",
        disposition=Disposition.DELETED,
        reason=(
            "OWL bridge module removed (commit 43197d7c6, 'move semantic authority "
            "to epistemic graph'); superseded by the epistemic-graph-native OWL/RDF "
            "authority."
        ),
    ),
)
