"""CISO Assistant GRC source extractor (CONCEPT:AU-KG.enrichment.ciso-assistant-extraction / CISO-002).

Self-registering extractor that folds intuitem **CISO Assistant** GRC records
(policies, controls, risks, threats, assessments, frameworks, assets, incidents,
third-party entities) into the uniform ``ExtractionBatch`` shape (typed
``GraphNode`` + ``EnrichmentEdge``), so CISO Assistant's Governance/Risk/Compliance
system-of-record *federates* with the epistemic-graph KG through the one generic
writer — no edits to any shared hub file.

Emitted node types are the **canonical** governance concepts (the same classes the
Egeria extractor emits — see ``ontology_egeria.ttl`` / ``ontology_enterprise.ttl``),
so CISO Assistant data reconciles with the Egeria/Camunda crosswalk:

    policy                         -> ``Policy``               ciso_assistant_policy:{id}
    applied / reference control    -> ``Control``              ciso_assistant_control:{id}
    risk scenario                  -> ``Risk``                 ciso_assistant_risk:{id}
    threat                         -> ``Threat``               ciso_assistant_threat:{id}
    risk assessment                -> ``RiskAssessment``       ciso_assistant_risk_assessment:{id}
    compliance assessment / audit  -> ``ComplianceAssessment`` ciso_assistant_compliance_assessment:{id}
    framework                      -> ``Framework``            ciso_assistant_framework:{id}
    asset                          -> ``Asset``                ciso_assistant_asset:{id}
    incident                       -> ``Incident``             ciso_assistant_incident:{id}
    security exception / finding   -> ``SecurityException`` / ``Finding``
    third-party entity             -> ``Entity``               ciso_assistant_entity:{id}

**Crosswalk to Egeria & Camunda (bidirectional via the hub).** Every node carries
``domain="ciso_assistant"`` + ``externalToolId`` (the CISO uuid) +
``qualifiedName`` (the CISO ``urn``/``ref_id``) — the federation keys. When a CISO
record carries an explicit twin reference (an Egeria GUID or a Camunda/BPMN process
id, populated by an operator mapping), an ``ALIGNED_WITH`` equivalence edge is
emitted to ``egeria_policy:{guid}`` / ``bpmn_process:{id}`` exactly as the Camunda
extractor does — so a CISO control aligned to an Egeria policy, or a CISO
compliance process aligned to a Camunda BPMN process, resolves into one logical
concept under the OWL reasoner. Absent an explicit twin id, the shared
``qualifiedName``/``name`` lets the reasoner reconcile by ``sameAs``.

The CISO Assistant client is **injected** (duck-typed) via ``config["client"]`` —
the ``Api`` facade from the ``ciso_assistant_api`` package, which exposes the
generated list methods (``api_policies_list``, ``api_applied_controls_list``,
``api_risk_scenarios_list``, …) returning a ``Response`` whose ``.data`` is the
concatenated result list. Method presence is probed so the extractor tolerates
partial client surfaces; this module performs **no** network calls itself and
imports nothing CISO-specific (import-safe even when ``ciso_assistant_api`` is
absent — ``client is None`` yields an empty batch).
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "ciso_assistant"


def _get(record: Any, key: str, default: Any = None) -> Any:
    """Tolerant field access for dict records (or attr-style objects)."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _first(record: Any, *keys: str) -> Any:
    """Return the first present, non-empty value among ``keys``."""
    for key in keys:
        val = _get(record, key)
        if val is not None and val != "":
            return val
    return None


def _list(client: Any, name: str) -> list:
    """Call a generated list-method if present, returning its ``.data`` list.

    Tolerant of partial clients and of plain-list / ``Response`` return shapes.
    """
    method = getattr(client, name, None)
    if not callable(method):
        return []
    try:
        result = method()
    except Exception:  # noqa: BLE001 - a missing/forbidden endpoint yields nothing
        return []
    data = getattr(result, "data", result)
    if isinstance(data, dict):
        data = data.get("results") or data.get("items") or data.get("data") or []
    return list(data) if isinstance(data, list) else []


def _ref_id(rec: Any) -> Any:
    """A record's stable cross-system key: urn first, then ref_id, then name."""
    return _first(rec, "urn", "ref_id", "name")


def extract(config: Any) -> ExtractionBatch:
    """Extract CISO Assistant GRC artifacts into a uniform ``ExtractionBatch``.

    ``config`` is a dict (or attr-style object) carrying an injected ``client``
    (the ``ciso_assistant_api.Api`` facade). Returns an empty batch when no client
    is supplied, so the module is safe to register even when CISO Assistant is
    unavailable.
    """
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    seen: set[str] = set()

    def _base_props(rec: Any) -> dict[str, Any]:
        return {
            "domain": "ciso_assistant",
            "externalToolId": _first(rec, "id", "uuid"),
            "qualifiedName": _ref_id(rec),
            "name": _first(rec, "name", "ref_id", "str") or _first(rec, "urn"),
            "ref_id": _get(rec, "ref_id"),
            "description": _get(rec, "description"),
        }

    def _add(node_id: str, node_type: str, props: dict[str, Any]) -> bool:
        if node_id in seen:
            return False
        seen.add(node_id)
        nodes.append(GraphNode(id=node_id, type=node_type, props=props))
        return True

    def _crosswalk(rec: Any, node_id: str) -> None:
        """Emit ALIGNED_WITH equivalence edges to Egeria / Camunda twins.

        Fires only when the CISO record carries an explicit twin id (populated by
        an operator mapping); otherwise the shared ``qualifiedName`` lets the OWL
        reasoner reconcile by ``sameAs``.
        """
        egeria_guid = _first(rec, "egeria_guid", "egeria_id", "egeriaGuid")
        if egeria_guid:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"egeria_policy:{egeria_guid}",
                    rel_type="ALIGNED_WITH",
                )
            )
        bpmn_id = _first(rec, "bpmn_process_id", "camunda_process_id", "process_id")
        if bpmn_id:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"bpmn_process:{bpmn_id}",
                    rel_type="ALIGNED_WITH",
                )
            )

    # category → (list method, node-id prefix, canonical KG type)
    simple_kinds = [
        ("api_policies_list", "ciso_assistant_policy", "Policy"),
        ("api_applied_controls_list", "ciso_assistant_control", "Control"),
        ("api_reference_controls_list", "ciso_assistant_refcontrol", "Control"),
        ("api_threats_list", "ciso_assistant_threat", "Threat"),
        (
            "api_risk_assessments_list",
            "ciso_assistant_risk_assessment",
            "RiskAssessment",
        ),
        ("api_frameworks_list", "ciso_assistant_framework", "Framework"),
        ("api_assets_list", "ciso_assistant_asset", "Asset"),
        ("api_incidents_list", "ciso_assistant_incident", "Incident"),
        (
            "api_security_exceptions_list",
            "ciso_assistant_exception",
            "SecurityException",
        ),
        ("api_findings_list", "ciso_assistant_finding", "Finding"),
        ("api_entities_list", "ciso_assistant_entity", "Entity"),
        ("api_perimeters_list", "ciso_assistant_perimeter", "Perimeter"),
    ]
    for method, prefix, node_type in simple_kinds:
        for rec in _list(client, method):
            ext = _first(rec, "id", "uuid")
            if not ext:
                continue
            node_id = f"{prefix}:{ext}"
            if _add(node_id, node_type, _base_props(rec)):
                _crosswalk(rec, node_id)

    # --- Risk scenarios (Risk) + mitigation edges to applied controls --------
    for rec in _list(client, "api_risk_scenarios_list"):
        ext = _first(rec, "id", "uuid")
        if not ext:
            continue
        node_id = f"ciso_assistant_risk:{ext}"
        if not _add(node_id, "Risk", _base_props(rec)):
            continue
        _crosswalk(rec, node_id)
        # RiskScenario MITIGATED_BY each AppliedControl it references.
        for ctrl in _as_list(_first(rec, "applied_controls", "controls")):
            cid = _id_of(ctrl)
            if cid:
                edges.append(
                    EnrichmentEdge(
                        source=node_id,
                        target=f"ciso_assistant_control:{cid}",
                        rel_type="MITIGATED_BY",
                    )
                )
        ra = _id_of(_first(rec, "risk_assessment"))
        if ra:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"ciso_assistant_risk_assessment:{ra}",
                    rel_type="PART_OF",
                )
            )

    # --- Compliance assessments (CONFORMS_TO their framework) ----------------
    for rec in _list(client, "api_compliance_assessments_list"):
        ext = _first(rec, "id", "uuid")
        if not ext:
            continue
        node_id = f"ciso_assistant_compliance_assessment:{ext}"
        if not _add(node_id, "ComplianceAssessment", _base_props(rec)):
            continue
        _crosswalk(rec, node_id)
        fw = _id_of(_first(rec, "framework"))
        if fw:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"ciso_assistant_framework:{fw}",
                    rel_type="CONFORMS_TO",
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


def _id_of(value: Any) -> Any:
    """Resolve a related-object reference (uuid string, dict, or {'id': ...})."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value.get("id") or value.get("uuid") or value.get("str")
    return value


def _as_list(value: Any) -> list:
    """Coerce a scalar/None/list relation field into a clean list."""
    if value is None or value == "":
        return []
    if isinstance(value, list | tuple | set):
        return [v for v in value if v]
    return [value]


register_extractor(
    CATEGORY,
    extract,
    description="intuitem CISO Assistant GRC (policies / controls / risks / assessments) → KG",
)
