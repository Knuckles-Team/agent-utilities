#!/usr/bin/python
from __future__ import annotations

"""Enterprise standards expressed as ontology interface types (CONCEPT:KG-2.49).

The north-star question — "do we flatten every org's standard into one superset,
or let orgs override?" — is answered by reusing the Foundry-parity **interface
type** layer (``ontology/interfaces.py``, CONCEPT:KG-2.38) verbatim:

  - An **enterprise standard IS an** :class:`Interface`. Its required
    :class:`InterfaceProperty` / :class:`InterfaceLinkConstraint` encode the
    *mandatory enterprise contract* — the agreed superset of must-have fields and
    links every governed asset must carry (owner, lifecycle_state,
    data_classification, the vendor-neutral ``capability`` tag, an
    organization link).
  - Each **org keeps its own concrete object type** that *implements* the
    interface and carries its own extra properties (``pci_scope``,
    ``pii_region``, …). Extensions are free: :meth:`Interface.gaps_for` ignores
    keys outside the shape, so specialization never breaks conformance.
  - :func:`drift_score` derives a per-asset 0..1 drift from
    :meth:`Interface.gaps_for` (the same call the structural interfaces use):
    ``0.0`` = fully conformant, ``1.0`` = nothing met.

This gives both at once — one enterprise contract *and* org-specific extension,
with measurable drift — instead of a bloated flattened superset or an
override-everything free-for-all.

Standards live in a **dedicated** :data:`ENTERPRISE_STANDARD_REGISTRY` (its own
:class:`InterfaceRegistry`, kept separate from the structural
``DEFAULT_INTERFACE_REGISTRY``) and are also materialized as ``EnterpriseStandard``
KG nodes (+ ``to_owl`` SHACL shapes) by :func:`materialize_standards` so they are
queryable and SHACL-enforceable. The registry follows the import-populated idiom
(real built-ins at import, never an empty shell).
"""

from typing import Any

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..ontology.interfaces import (
    Interface,
    InterfaceLinkConstraint,
    InterfaceProperty,
    InterfaceRegistry,
)

# Node type used for materialized standards in the KG.
ENTERPRISE_STANDARD_NODE = RegistryNodeType.ENTERPRISE_STANDARD.value


# ── Domain routing ──────────────────────────────────────────────────────────
# Which assets each standard governs. An asset is routed to a standard by its
# vendor-neutral ``capability`` tag first (the egeria harvest cross-vendor join
# key, CONCEPT:EG-009), then by its node ``type`` as a fallback. Lower-cased
# matching throughout.
STANDARD_DOMAINS: dict[str, dict[str, set[str]]] = {
    "ManagedApplication": {
        "capabilities": {
            "vcs",
            "itsm",
            "erp",
            "crm",
            "application",
            "enterprise-architecture",
            "observability",
            "identity",
            "secrets",
            "bpm",
        },
        "types": {
            "deployed_software_component",
            "enterprise_resource",
            "software_project",
            "company_software",
            "container_stack",
            "platform_service",
            "tool",
        },
    },
    "BusinessProcess": {
        "capabilities": {"process", "workflow", "automation"},
        "types": {
            "business_process",
            "process",
            "process_flow",
            "process_model",
            "value_stream",
        },
    },
    "DataAsset": {
        "capabilities": {"data", "dataset", "datastore", "analytics", "finance"},
        "types": {
            "dataset",
            "data_object",
            "data_connector",
            "datastore",
            "market_data_source",
        },
    },
}


def register_enterprise_standards(registry: InterfaceRegistry) -> None:
    """Register the built-in enterprise-standard interfaces into ``registry``.

    Three real, governing standards (one per capability domain). Each declares
    the mandatory enterprise contract as required interface-properties + link
    constraints; orgs implement them with their own concrete types + extensions.
    """
    registry.register(
        Interface(
            name="ManagedApplication",
            description=(
                "Enterprise standard for any managed application/service/system: "
                "it must declare an owner, a lifecycle state, a data "
                "classification, a cost center, the vendor-neutral capability it "
                "provides, and belong to an organization."
            ),
            properties=[
                InterfaceProperty(
                    name="owner",
                    type_ref="string",
                    description="Accountable owner (person/team identifier).",
                ),
                InterfaceProperty(
                    name="lifecycle_state",
                    type_ref="string",
                    description="Lifecycle state (e.g. active/deprecated/retired).",
                ),
                InterfaceProperty(
                    name="data_classification",
                    type_ref="string",
                    description="Data sensitivity class (public/internal/...).",
                ),
                InterfaceProperty(
                    name="cost_center",
                    type_ref="string",
                    description="Cost center the application is charged to.",
                ),
                InterfaceProperty(
                    name="capability",
                    type_ref="string",
                    description="Vendor-neutral capability the app provides.",
                ),
            ],
            link_constraints=[
                InterfaceLinkConstraint(
                    name="organization",
                    edge_type=RegistryEdgeType.BELONGS_TO_ORGANIZATION,
                    min_count=1,
                    description="Must belong to an owning organization/business unit.",
                ),
            ],
        )
    )
    registry.register(
        Interface(
            name="BusinessProcess",
            description=(
                "Enterprise standard for a business process: it must declare an "
                "owner, the capability/domain it serves, and a process tier, and "
                "apply to at least one system it operates on."
            ),
            properties=[
                InterfaceProperty(
                    name="owner",
                    type_ref="string",
                    description="Accountable process owner.",
                ),
                InterfaceProperty(
                    name="capability",
                    type_ref="string",
                    description="Capability/domain the process serves.",
                ),
                InterfaceProperty(
                    name="process_tier",
                    type_ref="string",
                    description="Criticality tier of the process.",
                ),
            ],
            # No mandatory link constraint: a process's contract is its owner /
            # capability / tier metadata; the system(s) it applies to are modelled
            # via APPLIES_TO links but not required for baseline conformance.
        )
    )
    registry.register(
        Interface(
            name="DataAsset",
            description=(
                "Enterprise standard for a data asset: it must declare an owner, "
                "a data classification, a retention policy, and a provenance link "
                "to the source it was derived from."
            ),
            properties=[
                InterfaceProperty(
                    name="owner",
                    type_ref="string",
                    description="Accountable data owner/steward.",
                ),
                InterfaceProperty(
                    name="data_classification",
                    type_ref="string",
                    description="Data sensitivity class.",
                ),
                InterfaceProperty(
                    name="retention_policy",
                    type_ref="string",
                    description="Retention/disposal policy identifier.",
                ),
            ],
            link_constraints=[
                InterfaceLinkConstraint(
                    name="provenance",
                    edge_type=RegistryEdgeType.WAS_DERIVED_FROM,
                    min_count=1,
                    description="Must declare a provenance link to its source.",
                ),
            ],
        )
    )


# CONCEPT:KG-2.49 — populated at import with real built-in standards, never empty.
ENTERPRISE_STANDARD_REGISTRY = InterfaceRegistry()
register_enterprise_standards(ENTERPRISE_STANDARD_REGISTRY)


def standard_names() -> list[str]:
    """Return the registered enterprise-standard names."""
    return [i.name for i in ENTERPRISE_STANDARD_REGISTRY.list_interfaces()]


def applicable_standard(asset: dict[str, Any]) -> str | None:
    """Route an asset dict to the enterprise standard that governs it.

    Routes by the vendor-neutral ``capability`` property first, then by the node
    ``type``. Returns the standard name, or ``None`` if no standard governs the
    asset (it is then excluded from drift/consolidation scoring).
    """
    cap = str(asset.get("capability", "") or "").lower()
    ntype = str(asset.get("type", "") or "").lower()
    for name, domain in STANDARD_DOMAINS.items():
        if cap and cap in domain["capabilities"]:
            return name
    for name, domain in STANDARD_DOMAINS.items():
        if ntype and ntype in domain["types"]:
            return name
    return None


def _required_slot_count(standard: Interface) -> int:
    """Number of checkable required slots (required props + link constraints).

    The denominator of the drift score. Each required property yields at most one
    gap (missing or type-invalid), and each link constraint yields at most one
    gap, so this bounds drift to ``[0, 1]``.
    """
    req_props = sum(
        1
        for p in standard.all_properties(ENTERPRISE_STANDARD_REGISTRY).values()
        if p.required
    )
    links = len(standard.all_link_constraints(ENTERPRISE_STANDARD_REGISTRY))
    return req_props + links


def drift_score(
    asset: dict[str, Any], standard_name: str
) -> tuple[float, list[str]]:
    """Return ``(drift, gaps)`` for an asset against a named enterprise standard.

    ``drift`` is ``len(gaps) / required_slots`` — ``0.0`` when the asset fully
    satisfies the standard's mandatory contract, ``1.0`` when it meets none of
    it. Delegates the conformance check to :meth:`Interface.gaps_for` (the same
    call the structural interface layer uses), so org-specific extra properties
    on the asset pass through without counting against it.

    Raises:
        ValueError: if ``standard_name`` is unknown.
    """
    standard = ENTERPRISE_STANDARD_REGISTRY.get(standard_name)
    if standard is None:
        raise ValueError(f"unknown enterprise standard: {standard_name!r}")
    gaps = standard.gaps_for(asset, registry=ENTERPRISE_STANDARD_REGISTRY)
    total = _required_slot_count(standard)
    drift = (len(gaps) / total) if total else 0.0
    return round(min(1.0, drift), 6), gaps


def materialize_standards(engine: Any) -> int:
    """Persist each enterprise standard as a queryable ``EnterpriseStandard`` node.

    Each node carries the interface name, version-stable ``content_hash`` (the
    interface :meth:`Interface.signature`), the serialized required properties /
    link constraints, the governed capability/type domain, and the ``to_owl``
    SHACL shape so a SHACL gate can enforce conformance natively. Idempotent:
    ``add_node`` MERGEs by id, and the ``content_hash`` changes only when the
    standard's shape is edited.

    Returns:
        The number of standards materialized.
    """
    add = getattr(engine, "add_node", None)
    if not callable(add):
        return 0
    count = 0
    for standard in ENTERPRISE_STANDARD_REGISTRY.list_interfaces():
        domain = STANDARD_DOMAINS.get(standard.name, {})
        props = {
            "name": standard.name,
            "description": standard.description,
            "content_hash": standard.signature(),
            "required_properties": sorted(
                p.name
                for p in standard.all_properties(
                    ENTERPRISE_STANDARD_REGISTRY
                ).values()
                if p.required
            ),
            "link_constraints": sorted(
                c.edge_type.value
                for c in standard.all_link_constraints(
                    ENTERPRISE_STANDARD_REGISTRY
                ).values()
            ),
            "governs_capabilities": sorted(domain.get("capabilities", set())),
            "governs_types": sorted(domain.get("types", set())),
            "shacl_shape": standard.to_owl(registry=ENTERPRISE_STANDARD_REGISTRY),
            "concept": "KG-2.49",
        }
        try:
            add(
                standard_node_id(standard.name),
                ENTERPRISE_STANDARD_NODE,
                properties=props,
            )
            count += 1
        except Exception:  # noqa: BLE001 - best-effort materialization
            continue
    return count


def standard_node_id(standard_name: str) -> str:
    """Stable KG node id for a materialized enterprise standard."""
    return f"enterprise_standard:{standard_name}"


__all__ = [
    "ENTERPRISE_STANDARD_REGISTRY",
    "ENTERPRISE_STANDARD_NODE",
    "STANDARD_DOMAINS",
    "register_enterprise_standards",
    "standard_names",
    "applicable_standard",
    "drift_score",
    "materialize_standards",
    "standard_node_id",
]
