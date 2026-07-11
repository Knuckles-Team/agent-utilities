#!/usr/bin/python
from __future__ import annotations

"""Versioned capability descriptor (CONCEPT:AU-P1-3 — X-4 ontology-driven tool/agent routing).

AU-P1-3 gave every callable resource (MCP tool / agent / skill) a flat capability
tag set plus tenant/policy scoping and a durable bandit reward. X-4 asks for a
richer, VERSIONED descriptor per capability: typed inputs/outputs, side effects,
required data/resource types, tenant/authz scopes, cost/latency/reliability/
locality, and a policy/approval class — the full contract a router needs to make
an accountable placement decision, not just "does it match".

Design, consistent with the rest of AU-P1-3's philosophy (``durable_outcome_store``'s
docstring: "Deliberately NOT a new node type / table"):

* :class:`CapabilityDescriptor` is a plain dataclass — the schema for a set of
  node properties on the SAME capability/tool node the engine already carries
  (``capabilities``, ``tenant``, ``policy_tags``, ``capability_reward``, ...).
  :meth:`to_node_properties` / :meth:`from_node_properties` are the exact
  projection to/from that property dict, so a descriptor rides the same
  bounded-cache / CDC-maintained pipeline every other capability field already
  uses — no new node type, no new store.
* ``reliability`` / ``success_count`` are NEVER written by
  :meth:`to_node_properties` — they are read-derived from the engine's
  ``capability_reward`` / ``capability_reward_count`` properties (the durable
  contextual-bandit outcome :mod:`.durable_outcome_store` already owns), so the
  descriptor's calibrated success history is always the live bandit state, never
  a second, driftable copy.
* :class:`CapabilityDescriptorRegistry` is the AU-side in-process index the X-4
  task explicitly asks for ("AU holds the descriptor index + explanations; engine/
  KG is the store") — a plain dict keyed by capability/tool id, hydratable from
  and persistable to the engine via :func:`persist_capability_descriptor` /
  :func:`load_capability_descriptor`.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CapabilityDescriptor",
    "CapabilityDescriptorRegistry",
    "persist_capability_descriptor",
    "load_capability_descriptor",
]

# The subset of node properties this module owns (written by to_node_properties /
# read by from_node_properties). Deliberately excludes capability_reward* — those
# belong to durable_outcome_store and are only ever READ here.
_DESCRIPTOR_PROPS: tuple[str, ...] = (
    "capability_version",
    "capability_type",
    "input_schema",
    "output_schema",
    "side_effects",
    "required_data_types",
    "required_resource_types",
    "tenant_scopes",
    "authz_scopes",
    "cost_estimate",
    "latency_ms_estimate",
    "locality",
    "policy_class",
    "approval_class",
    "descriptor_updated_at",
)

# Recognized side-effect tags — open vocabulary in practice (any string is
# accepted), but these are the ones the built-in approval-class defaulting
# understands.
SIDE_EFFECT_READ = "read"
SIDE_EFFECT_WRITE = "write"
SIDE_EFFECT_EXTERNAL_CALL = "external_call"
SIDE_EFFECT_DESTRUCTIVE = "destructive"

APPROVAL_AUTO = "auto"
APPROVAL_SUPERVISED = "supervised"
APPROVAL_HUMAN_REQUIRED = "human_approval_required"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class CapabilityDescriptor:
    """A versioned, typed contract for one MCP tool/agent capability instance.

    Attributes:
        id: The entity id this descriptor describes (same id the capability
            index / engine node use).
        capability_type: The ontology capability class this instance provides
            (e.g. ``"DNSCapability"``) — the subsumption anchor (X-4).
        version: Semantic version of THIS descriptor's contract. A tool that
            changes its input/output schema or side effects bumps this so a
            router can pin/require a minimum version.
        input_schema: Typed input contract — a JSON-Schema-shaped dict (or any
            structurally-typed mapping); opaque to this module, just persisted.
        output_schema: Typed output contract, same shape convention.
        side_effects: Declared side effects (``"read"``, ``"write"``,
            ``"external_call"``, ``"destructive"``, ...) — open vocabulary.
        required_data_types: Data types/classes this capability needs available
            to run (ontology class names, free-form otherwise).
        required_resource_types: Resource types it needs provisioned (e.g. GPU,
            a specific external service).
        tenant_scopes: Tenants this instance is authorized for. Empty = every
            tenant (mirrors :class:`~.capability_index.CapabilityIndex`'s
            tenant-agnostic default).
        authz_scopes: Auth scopes the CALLER must hold to invoke this capability
            (distinct from ``tenant_scopes``, which scope who owns the tool).
        cost_estimate: Estimated cost per call (currency- or token-denominated;
            unit is a deployment convention, not enforced here).
        latency_ms_estimate: Estimated latency per call, in milliseconds.
        locality: Where this capability executes (``"local"``, a region tag, a
            node id, ...) — feeds locality-aware routing / data-gravity policy.
        policy_class: Ontology-aligned policy classification (e.g.
            ``"standard"``, ``"restricted"``) — distinct from the flat
            ``policy_tags`` set-membership filter; this is the descriptor's own
            headline classification.
        approval_class: Required approval tier to invoke — one of
            :data:`APPROVAL_AUTO` / :data:`APPROVAL_SUPERVISED` /
            :data:`APPROVAL_HUMAN_REQUIRED`. Defaults to
            :data:`APPROVAL_HUMAN_REQUIRED` when ``side_effects`` includes
            :data:`SIDE_EFFECT_DESTRUCTIVE` and no explicit value was given —
            fail-closed, never silently auto-approved.
        reliability: Calibrated success rate — READ-ONLY here; it mirrors the
            durable contextual-bandit reward EMA
            (:func:`~.durable_outcome_store.read_capability_reward`), never an
            independently-tracked value. 0.5 = unproven/neutral.
        success_count: Number of outcomes the reliability EMA has absorbed
            (mirrors ``capability_reward_count``).
        updated_at: ISO-8601 timestamp of the last descriptor field write.
    """

    id: str
    capability_type: str
    version: str = "1.0.0"
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    side_effects: tuple[str, ...] = ()
    required_data_types: tuple[str, ...] = ()
    required_resource_types: tuple[str, ...] = ()
    tenant_scopes: tuple[str, ...] = ()
    authz_scopes: tuple[str, ...] = ()
    cost_estimate: float | None = None
    latency_ms_estimate: float | None = None
    locality: str | None = None
    policy_class: str = "standard"
    approval_class: str | None = None
    reliability: float = 0.5
    success_count: int = 0
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if self.approval_class is None:
            # Fail-closed default: a destructive capability is never auto-approved
            # unless the descriptor explicitly says so.
            self.approval_class = (
                APPROVAL_HUMAN_REQUIRED
                if SIDE_EFFECT_DESTRUCTIVE in self.side_effects
                else APPROVAL_AUTO
            )

    def to_node_properties(self) -> dict[str, Any]:
        """Project to the node-property dict this module owns (excludes reliability/success_count)."""
        return {
            "capability_version": self.version,
            "capability_type": self.capability_type,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "side_effects": list(self.side_effects),
            "required_data_types": list(self.required_data_types),
            "required_resource_types": list(self.required_resource_types),
            "tenant_scopes": list(self.tenant_scopes),
            "authz_scopes": list(self.authz_scopes),
            "cost_estimate": self.cost_estimate,
            "latency_ms_estimate": self.latency_ms_estimate,
            "locality": self.locality,
            "policy_class": self.policy_class,
            "approval_class": self.approval_class,
            "descriptor_updated_at": _now_iso(),
        }

    @classmethod
    def from_node_properties(
        cls, id: str, props: dict[str, Any]
    ) -> CapabilityDescriptor:
        """Reconstruct a descriptor from node properties (engine-hydrated or fake-engine test props).

        ``reliability``/``success_count`` are read from the durable bandit's OWN
        properties (``capability_reward``/``capability_reward_count``) — never a
        parallel descriptor-owned copy.
        """

        def _tuple(key: str) -> tuple[str, ...]:
            v = props.get(key) or ()
            if isinstance(v, str):
                return (v,)
            return tuple(str(x) for x in v)

        reward = props.get("capability_reward")
        return cls(
            id=id,
            capability_type=str(
                props.get("capability_type")
                or props.get("type")
                or props.get("node_type")
                or ""
            ),
            version=str(props.get("capability_version") or "1.0.0"),
            input_schema=dict(props.get("input_schema") or {}),
            output_schema=dict(props.get("output_schema") or {}),
            side_effects=_tuple("side_effects"),
            required_data_types=_tuple("required_data_types"),
            required_resource_types=_tuple("required_resource_types"),
            tenant_scopes=_tuple("tenant_scopes"),
            authz_scopes=_tuple("authz_scopes"),
            cost_estimate=(
                float(props["cost_estimate"])
                if props.get("cost_estimate") is not None
                else None
            ),
            latency_ms_estimate=(
                float(props["latency_ms_estimate"])
                if props.get("latency_ms_estimate") is not None
                else None
            ),
            locality=props.get("locality"),
            policy_class=str(props.get("policy_class") or "standard"),
            approval_class=props.get("approval_class"),
            reliability=float(reward) if reward is not None else 0.5,
            success_count=int(props.get("capability_reward_count") or 0),
            updated_at=props.get("descriptor_updated_at"),
        )


class CapabilityDescriptorRegistry:
    """AU-side in-process index of :class:`CapabilityDescriptor` (X-4).

    "AU holds the descriptor index + explanations; engine/KG is the store" — this
    is a plain dict keyed by entity id, hydrated from and persisted to the engine
    via the module-level :func:`persist_capability_descriptor` /
    :func:`load_capability_descriptor` functions. No CDC/ANN machinery of its own
    — it composes with :class:`~.capability_index.CapabilityIndex` /
    :mod:`.engine_capability_search` for candidate selection; this registry answers
    "what IS this capability's contract", not "which capability matches".
    """

    def __init__(self) -> None:
        self._by_id: dict[str, CapabilityDescriptor] = {}

    def register(self, descriptor: CapabilityDescriptor) -> None:
        self._by_id[descriptor.id] = descriptor

    def get(self, id: str) -> CapabilityDescriptor | None:
        return self._by_id.get(id)

    def __contains__(self, id: str) -> bool:
        return id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def all(self) -> list[CapabilityDescriptor]:
        return list(self._by_id.values())

    def remove(self, id: str) -> bool:
        return self._by_id.pop(id, None) is not None

    def hydrate_from_engine(self, engine: Any, ids: Any) -> int:
        """Load descriptors for ``ids`` from ``engine`` into this registry.

        Best-effort per id — a node with no descriptor properties still yields a
        minimal descriptor (defaults), so a candidate is never dropped from
        routing merely for lacking a full X-4 descriptor yet (incremental
        adoption). Returns the number of descriptors (re)hydrated.
        """
        graph = getattr(engine, "graph", None)
        getter = getattr(graph, "_get_node_properties", None) if graph else None
        count = 0
        for nid in ids:
            nid = str(nid)
            props: dict[str, Any] = {}
            if callable(getter):
                try:
                    props = getter(nid) or {}
                except Exception as e:  # noqa: BLE001 — best-effort hydration
                    logger.debug(
                        "CapabilityDescriptorRegistry: read failed for %r: %s", nid, e
                    )
            self.register(CapabilityDescriptor.from_node_properties(nid, props))
            count += 1
        return count


def persist_capability_descriptor(
    engine: Any, descriptor: CapabilityDescriptor
) -> bool:
    """Durably write ``descriptor``'s owned properties onto the engine node.

    Mirrors :func:`~.durable_outcome_store.persist_capability_reward`'s pattern
    exactly: a plain ``SET`` over the node's own properties, no new node type.
    Never writes ``capability_reward``/``capability_reward_count`` — those stay
    owned by the durable bandit. Returns ``False`` (no-op, never raises) when no
    engine backend is reachable.
    """
    backend = getattr(engine, "backend", None)
    if backend is None:
        return False
    props = descriptor.to_node_properties()
    set_clause = ", ".join(f"n.{key} = ${key}" for key in props)
    query = f"MATCH (n) WHERE n.id = $id SET {set_clause}"
    try:
        backend.execute(query, {"id": descriptor.id, **props})
        return True
    except Exception as e:  # noqa: BLE001 — durable persistence is best-effort
        logger.debug(
            "persist_capability_descriptor: write failed for %r: %s", descriptor.id, e
        )
        return False


def load_capability_descriptor(engine: Any, id: str) -> CapabilityDescriptor | None:
    """Read one descriptor back from the engine, or ``None`` if unreachable/unknown."""
    backend = getattr(engine, "backend", None)
    if backend is None:
        return None
    props_query = " , ".join(f"n.{key} AS {key}" for key in _DESCRIPTOR_PROPS)
    try:
        rows = backend.execute(
            f"MATCH (n) WHERE n.id = $id RETURN n.capability_reward AS capability_reward, "
            f"n.capability_reward_count AS capability_reward_count, {props_query}",
            {"id": id},
        )
    except Exception as e:  # noqa: BLE001 — durable read is best-effort
        logger.debug("load_capability_descriptor: read failed for %r: %s", id, e)
        return None
    for row in rows or ():
        if isinstance(row, dict):
            return CapabilityDescriptor.from_node_properties(id, row)
    return None
