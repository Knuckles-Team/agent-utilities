"""ServiceNow source extractor — ITSM + CMDB + Technology Reference Model (KG-2.9).

Maps ServiceNow into the uniform ``ExtractionBatch``:

* ITSM — ``incident`` → :Incident, ``change_request`` → :Change (→ AFFECTS CI,
  ASSIGNED_TO person).
* CMDB inventory — ``cmdb_ci_*`` → :ConfigurationItem (an :AssetInstance).
* Technology Reference Model — ``cmdb_model`` → :TechnologyProduct; ``alm_hardware``/
  ``alm_asset`` → :AssetInstance (INSTANCE_OF its product), with lifecycle/risk
  attributes (lifecycleStage / endOfLifeDate / riskRating) and a :TechnologyRisk
  node (HAS_RISK) when a record carries risk/EOL signal.

Every node carries ``externalToolId`` (sys_id) + ``domain="servicenow"`` — the
federation key the write-back layer resolves against. The client is **injected**
(see :class:`source_adapters.ServiceNowSourceClient`); no network here.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "servicenow"
_DOMAIN = "servicenow"


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _ref(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value") or value.get("display_value")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first(record: Any, *keys: str) -> str | None:
    for k in keys:
        v = _ref(_get(record, k))
        if v:
            return v
    return None


def _key(record: Any) -> str | None:
    return _ref(_get(record, "sys_id")) or _ref(_get(record, "number"))


def _call(client: Any, name: str) -> list:
    method = getattr(client, name, None)
    if not callable(method):
        return []
    result = method()
    return list(result) if result else []


def _federation(raw_id: str, **extra: Any) -> dict[str, Any]:
    props: dict[str, Any] = {"externalToolId": raw_id, "domain": _DOMAIN}
    props.update({k: v for k, v in extra.items() if v is not None})
    return props


def _risk_props(rec: Any) -> dict[str, Any]:
    """Lifecycle/risk attributes (vendor-neutral TRM props), tolerant of field names."""
    out: dict[str, Any] = {}
    stage = _first(
        rec, "lifecycle_stage", "lifecycle", "install_status", "life_cycle_stage"
    )
    if stage:
        out["lifecycleStage"] = stage
    eol = _first(rec, "end_of_life", "eol_date", "end_of_support", "decommission_date")
    if eol:
        out["endOfLifeDate"] = eol
    rating = _first(rec, "risk_rating", "risk", "risk_score")
    if rating:
        out["riskRating"] = rating
    return out


def extract(config: Any) -> ExtractionBatch:
    """Extract ServiceNow ITSM + CMDB + TRM into a uniform ``ExtractionBatch``."""
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    def _emit_risk_node(owner_id: str, rec: Any, risk: dict[str, Any]) -> None:
        """A TechnologyRisk node + HAS_RISK edge when a record signals risk/EOL."""
        if not (risk.get("riskRating") or risk.get("endOfLifeDate")):
            return
        rid = f"snrisk:{owner_id}"
        nodes.append(
            GraphNode(
                id=rid,
                type="TechnologyRisk",
                props=_federation(
                    owner_id, name=f"Risk: {_get(rec, 'name') or owner_id}", **risk
                ),
            )
        )
        edges.append(EnrichmentEdge(source=owner_id, target=rid, rel_type="HAS_RISK"))

    # --- Incidents / Changes ----------------------------------------------
    for method, label, prefix in (
        ("incidents", "Incident", "incident"),
        ("changes", "Change", "change"),
    ):
        for rec in _call(client, method):
            key = _key(rec)
            if not key:
                continue
            node_id = f"{prefix}:{key}"
            nodes.append(
                GraphNode(
                    id=node_id,
                    type=label,
                    props=_federation(
                        key,
                        number=_ref(_get(rec, "number")),
                        short_description=_get(rec, "short_description"),
                        state=_ref(_get(rec, "state")),
                        priority=_ref(_get(rec, "priority")),
                    ),
                )
            )
            ci = _ref(_get(rec, "cmdb_ci"))
            if ci:
                edges.append(
                    EnrichmentEdge(
                        source=node_id, target=f"ci:{ci}", rel_type="AFFECTS"
                    )
                )
            assignee = _ref(_get(rec, "assigned_to"))
            if assignee:
                edges.append(
                    EnrichmentEdge(
                        source=node_id,
                        target=f"person:{assignee}",
                        rel_type="ASSIGNED_TO",
                    )
                )

    # --- CMDB Configuration Items (inventory) -----------------------------
    for rec in _call(client, "cmdb_cis"):
        key = _key(rec)
        if not key:
            continue
        node_id = f"ci:{key}"
        risk = _risk_props(rec)
        nodes.append(
            GraphNode(
                id=node_id,
                type="ConfigurationItem",
                props=_federation(
                    key,
                    name=_get(rec, "name"),
                    short_description=_get(rec, "short_description"),
                    ci_class=_ref(_get(rec, "ci_class")),
                    state=_ref(_get(rec, "state")),
                    **risk,
                ),
            )
        )
        _emit_risk_node(node_id, rec, risk)
        model = _first(rec, "model_id", "model")
        if model:
            edges.append(
                EnrichmentEdge(
                    source=node_id, target=f"snproduct:{model}", rel_type="INSTANCE_OF"
                )
            )

    # --- Technology Reference Model: products -----------------------------
    for rec in _call(client, "cmdb_models"):
        key = _key(rec)
        if not key:
            continue
        node_id = f"snproduct:{key}"
        risk = _risk_props(rec)
        nodes.append(
            GraphNode(
                id=node_id,
                type="TechnologyProduct",
                props=_federation(
                    key,
                    name=_first(rec, "display_name", "name"),
                    manufacturer=_first(rec, "manufacturer", "vendor"),
                    **risk,
                ),
            )
        )
        _emit_risk_node(node_id, rec, risk)

    # --- Technology Reference Model: asset instances ----------------------
    for rec in _call(client, "assets"):
        key = _key(rec)
        if not key:
            continue
        node_id = f"asset:{key}"
        risk = _risk_props(rec)
        nodes.append(
            GraphNode(
                id=node_id,
                type="AssetInstance",
                props=_federation(
                    key, name=_first(rec, "display_name", "name", "asset_tag"), **risk
                ),
            )
        )
        _emit_risk_node(node_id, rec, risk)
        model = _first(rec, "model", "model_id", "model_category")
        if model:
            edges.append(
                EnrichmentEdge(
                    source=node_id, target=f"snproduct:{model}", rel_type="INSTANCE_OF"
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY,
    extract,
    description="ServiceNow ITSM + CMDB + Technology Reference Model → KG",
)
