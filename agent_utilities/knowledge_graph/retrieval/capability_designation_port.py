#!/usr/bin/python
from __future__ import annotations

"""Typed port through which the harness RECEIVES capability designations from EG.

Capability designation (which agent/tool/skill handles a task: capability,
tenant and policy filtering, ontology subsumption, cosine + reward + ontology
prior ranking, outcome-reward learning) is a *choice* and is owned by
epistemic-graph (``EG-DECISION-ENGINE-R172``: a native Rust index served through
``Decide`` with a ``capability_designation`` question and ``DecisionLog``
``capability_outcome``). agent-utilities keeps no index, no scoring and no
reward arithmetic for it: the run loop is *given* a
:class:`CapabilityDesignationPort` (bound to the EG client once the served
method ships) and only forwards requests and outcomes through it.

When no port is bound the harness has no designation — callers fall back to
their own non-scoring behaviour (keyword scan, declared order) exactly as they
did when designation was unavailable.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CapabilityDesignation",
    "CapabilityDesignationPort",
    "CapabilityDesignationRequest",
    "designation_port_of",
]


@dataclass(frozen=True)
class CapabilityDesignationRequest:
    """One designation question (mirrors EG ``CapabilityDesignationQuestionV1``)."""

    query_vector: Sequence[float]
    required_capabilities: tuple[str, ...] = ()
    required_policy_tags: tuple[str, ...] = ()
    tenant: str | None = None
    k: int = 5
    embedding_version: str | None = None


@dataclass(frozen=True)
class CapabilityDesignation:
    """One EG-designated entity (mirrors EG ``DesignatedCapabilityV1``)."""

    id: str
    score: float
    capabilities: frozenset[str] = frozenset()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class CapabilityDesignationPort(Protocol):
    """What the run loop is given: EG answers designation and learns outcomes."""

    def designate(
        self, request: CapabilityDesignationRequest
    ) -> list[CapabilityDesignation]: ...

    def record_outcome(
        self,
        entity_id: str,
        *,
        success: bool | None = None,
        reward: float | None = None,
    ) -> float | None: ...


def designation_port_of(holder: Any) -> CapabilityDesignationPort | None:
    """Return the port bound on ``holder`` (``capability_designator``), else ``None``."""
    if holder is None:
        return None
    if isinstance(holder, CapabilityDesignationPort):
        return holder
    port = getattr(holder, "capability_designator", None)
    return port if isinstance(port, CapabilityDesignationPort) else None
