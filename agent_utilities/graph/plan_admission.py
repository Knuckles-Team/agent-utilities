"""The runtime admission of a committed topology plan.

EG decides the topology; this module turns the committed ``TopologyPlan`` into
the one runtime enforcement object AU already has,
:class:`~agent_utilities.graph.topology_engine.ElasticTopologyAdmission`, whose
caps ARE the plan's widths, depth and token share and whose digest binds the
decision record. A run can therefore never be wider, deeper or longer than the
plan it was admitted under: runtime only narrows -- widening is a new decision
and a new lease, never an override of this object (AU-CONTROL-R016, R019).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_utilities.graph.topology_engine import (
    ElasticTopologyAdmission,
    TopologyAdmissionError,
)


def _slots(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    slots = plan.get("slots") or []
    if not slots or not all(isinstance(slot, Mapping) for slot in slots):
        raise TopologyAdmissionError("a topology plan names at least one slot")
    return list(slots)


def _positive(slot: Mapping[str, Any], field: str) -> int:
    value = slot.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TopologyAdmissionError(f"plan slot {field} must be a positive integer")
    return value


def _token_ceiling(slots: Sequence[Mapping[str, Any]]) -> int | None:
    """The plan's declared tokens, when every slot declares them."""
    tokens = [slot.get("tokens") for slot in slots]
    if any(not isinstance(t, int) or isinstance(t, bool) for t in tokens):
        return None
    return sum(int(t) for t in tokens if isinstance(t, int)) or None


def admission_from_plan(
    plan: Mapping[str, Any],
    *,
    record_id: str,
    tenant: str,
    delegation_id: str,
    capabilities: Sequence[str] = ("topology.materialize",),
) -> ElasticTopologyAdmission:
    """The admission whose caps are exactly the committed ``plan``'s.

    * ``max_fan_out`` -- the widest slot;
    * ``max_parallelism`` and ``max_nodes`` -- every agent the plan runs at once;
    * ``max_depth`` -- the plan's sequential steps (one per slot per round);
    * ``max_tokens`` -- the plan's declared tokens when every slot declares
      them, else the admission default.
    """
    if not str(record_id).startswith("decision:"):
        raise TopologyAdmissionError("a plan admission names its decision record")
    slots = _slots(plan)
    widths = [_positive(slot, "width") for slot in slots]
    rounds = [_positive(slot, "rounds") for slot in slots]
    extra: dict[str, Any] = {}
    tokens = _token_ceiling(slots)
    if tokens is not None:
        extra["max_tokens"] = tokens
    return ElasticTopologyAdmission(
        tenant=tenant,
        delegation_id=delegation_id,
        capabilities=tuple(capabilities),
        max_nodes=sum(widths),
        max_depth=sum(rounds),
        max_fan_out=max(widths),
        max_parallelism=sum(widths),
        decision_record_id=str(record_id),
        **extra,
    )


__all__ = ["admission_from_plan"]
