"""Apache Egeria open-metadata source extractor (CONCEPT:KG-2.9).

Self-registering extractor that folds Egeria's open-metadata / glossary /
governance / lineage records into the uniform ``ExtractionBatch`` shape (typed
``GraphNode`` + ``EnrichmentEdge``), so Egeria's metadata system-of-record
*federates* with the epistemic-graph KG through the one generic writer — no edits
to any shared hub file.

Emitted node types are the **canonical** ArchiMate concepts (see
``ontology_enterprise.ttl`` / ``ontology_egeria.ttl``), so Egeria data reconciles
with the ServiceNow/ERPNext/Camunda/infra crosswalk by GUID:

    asset (DataStore/Database/FileFolder) -> ``DataConnector``  egeria_asset:{guid}
    asset (DataSet/Table/DataFile)        -> ``DataObject``     egeria_asset:{guid}  (PART_OF store)
    glossary term                         -> ``Concept``        egeria_term:{guid}
    glossary category                     -> ``GlossaryCategory`` egeria_category:{guid}
    governance policy / rule              -> ``Policy``         egeria_policy:{guid}
    governance principle                  -> ``Principle``      egeria_policy:{guid}
    governance strategy / imperative      -> ``Goal``           egeria_policy:{guid}
    software server / host                -> ``Server``         egeria_server:{guid}
    connection                            -> ``DataConnector``  egeria_connection:{guid}
    lineage process                       -> ``ProcessModel``   egeria_process:{guid}

Lineage ``DataFlow`` records become ``flowsTo`` (source asset → process) and
``derivesFrom`` (process → target asset) edges — the unique payload Egeria
contributes that the KG does not model natively.

Every node carries ``externalToolId`` (the Egeria GUID) and ``domain="egeria"`` —
the federation key. Classifications (Confidentiality, Retention) are attached as
node **properties** plus a ``governedBy`` edge to the relevant ``Policy``, never as
nodes.

The Egeria client is **injected** (duck-typed) via ``config["client"]`` — the
``EgeriaApi`` facade from the ``egeria-mcp`` package, expected to expose
``list_assets()``, ``list_glossary_terms()``, ``list_glossary_categories()``,
``list_governance_definitions()``, ``list_software_servers()``,
``list_connections()`` and ``list_data_flows()``. Method presence is probed so the
extractor tolerates partial client surfaces, and this module performs **no**
network calls itself and imports nothing Egeria-specific (so it is import-safe even
when pyegeria/Egeria are absent — ``client is None`` yields an empty batch).
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_source

CATEGORY = "egeria"

# Egeria asset typeName fragment -> (KG node type, hosted/part-of role)
#   "store"  -> a :DataConnector that is HOSTED_ON a :Server
#   "object" -> a :DataObject that is PART_OF a store
_STORE_HINTS = ("datastore", "database", "filefolder", "filesystem", "datasource")
_OBJECT_HINTS = ("dataset", "datafile", "table", "relationaltable", "datafield")


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


def _call(client: Any, name: str) -> list:
    """Call a client list-method if present, returning a list (tolerant)."""
    method = getattr(client, name, None)
    if not callable(method):
        return []
    try:
        result = method()
    except TypeError:
        try:
            result = method({})
        except Exception:
            return []
    except Exception:
        return []
    if isinstance(result, dict):
        result = (
            result.get("items") or result.get("results") or result.get("elements") or []
        )
    return list(result) if result else []


def _classification_props(rec: Any) -> dict[str, Any]:
    """Lift Egeria classifications into flat node properties (props-first model)."""
    props: dict[str, Any] = {}
    conf = _first(rec, "confidentiality", "confidentialityLevel")
    if conf is not None:
        props["confidentialityLevel"] = conf
    ret = _first(rec, "retention", "retentionPeriod")
    if ret is not None:
        props["retentionPeriod"] = ret
    return props


def _asset_type(rec: Any) -> tuple[str, str]:
    """Resolve an Egeria asset record to (KG node type, role: store|object|other)."""
    type_name = str(_first(rec, "typeName", "type", "category") or "").lower()
    if any(h in type_name for h in _OBJECT_HINTS):
        return "DataObject", "object"
    if any(h in type_name for h in _STORE_HINTS):
        return "DataConnector", "store"
    return "EAFactSheet", "other"


def _gov_type(rec: Any) -> str:
    """Map an Egeria governance-definition typeName to a KG class."""
    type_name = str(_first(rec, "typeName", "type") or "").lower()
    if "principle" in type_name:
        return "Principle"
    if "strategy" in type_name or "imperative" in type_name:
        return "Goal"
    return "Policy"


def extract(config: Any) -> ExtractionBatch:
    """Extract Egeria open-metadata artifacts into a uniform ``ExtractionBatch``.

    ``config`` is a dict (or attr-style object) carrying an injected ``client``
    (the ``EgeriaApi`` facade). Returns an empty batch when no client is supplied,
    so the module is safe to register even when Egeria is unavailable.
    """
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    def _base_props(rec: Any) -> dict[str, Any]:
        return {
            "domain": "egeria",
            "externalToolId": _first(rec, "guid", "GUID", "id"),
            "qualifiedName": _first(rec, "qualifiedName", "qualified_name"),
        }

    seen: set[str] = set()

    def _add(node_id: str, node_type: str, props: dict[str, Any]) -> bool:
        if node_id in seen:
            return False
        seen.add(node_id)
        nodes.append(GraphNode(id=node_id, type=node_type, props=props))
        return True

    # --- Assets (data stores / data objects) --------------------------------
    for rec in _call(client, "list_assets"):
        guid = _first(rec, "guid", "GUID", "id")
        if not guid:
            continue
        node_id = f"egeria_asset:{guid}"
        node_type, role = _asset_type(rec)
        props = {
            **_base_props(rec),
            "name": _first(rec, "displayName", "name", "qualifiedName"),
            **_classification_props(rec),
        }
        _add(node_id, node_type, props)
        # store HOSTED_ON server; object PART_OF its store
        host = _first(rec, "hostGuid", "serverGuid", "hostedOnGuid")
        if role == "store" and host:
            edges.append(_edge(node_id, f"egeria_server:{host}", "HOSTED_ON"))
        parent = _first(rec, "storeGuid", "parentGuid", "assetGuid")
        if role == "object" and parent:
            edges.append(_edge(node_id, f"egeria_asset:{parent}", "PART_OF"))
        # classification → governedBy a Policy
        for pol in _as_list(_first(rec, "governedByGuids", "policyGuids")):
            edges.append(_edge(node_id, f"egeria_policy:{pol}", "governedBy"))

    # --- Glossary categories (parents before terms reference them) ----------
    for rec in _call(client, "list_glossary_categories"):
        guid = _first(rec, "guid", "GUID", "id")
        if not guid:
            continue
        node_id = f"egeria_category:{guid}"
        _add(
            node_id,
            "GlossaryCategory",
            {**_base_props(rec), "name": _first(rec, "displayName", "name")},
        )
        parent = _first(rec, "parentCategoryGuid", "parentGuid")
        if parent:
            edges.append(_edge(node_id, f"egeria_category:{parent}", "PART_OF"))

    # --- Glossary terms (federate as canonical :Concept) --------------------
    for rec in _call(client, "list_glossary_terms"):
        guid = _first(rec, "guid", "GUID", "id")
        if not guid:
            continue
        node_id = f"egeria_term:{guid}"
        _add(
            node_id,
            "Concept",
            {
                **_base_props(rec),
                "name": _first(rec, "displayName", "name"),
                "summary": _first(rec, "summary", "description"),
            },
        )
        cat = _first(rec, "categoryGuid", "category")
        if cat:
            edges.append(_edge(node_id, f"egeria_category:{cat}", "IN_CATEGORY"))

    # --- Governance definitions (Policy / Principle / Goal) -----------------
    for rec in _call(client, "list_governance_definitions"):
        guid = _first(rec, "guid", "GUID", "id")
        if not guid:
            continue
        node_id = f"egeria_policy:{guid}"
        _add(
            node_id,
            _gov_type(rec),
            {
                **_base_props(rec),
                "name": _first(rec, "title", "displayName", "name"),
                "governanceDomain": _first(rec, "domain", "domainIdentifier"),
            },
        )

    # --- Software servers / hosts -------------------------------------------
    for rec in _call(client, "list_software_servers"):
        guid = _first(rec, "guid", "GUID", "id")
        if not guid:
            continue
        _add(
            f"egeria_server:{guid}",
            "Server",
            {
                **_base_props(rec),
                "name": _first(rec, "displayName", "name", "hostName"),
            },
        )

    # --- Connections ---------------------------------------------------------
    for rec in _call(client, "list_connections"):
        guid = _first(rec, "guid", "GUID", "id")
        if not guid:
            continue
        node_id = f"egeria_connection:{guid}"
        _add(
            node_id,
            "DataConnector",
            {**_base_props(rec), "name": _first(rec, "displayName", "name")},
        )
        asset = _first(rec, "assetGuid", "connectsToGuid")
        if asset:
            edges.append(_edge(node_id, f"egeria_asset:{asset}", "CONNECTS_TO"))

    # --- Lineage data flows (the unique Egeria payload) ---------------------
    for rec in _call(client, "list_data_flows"):
        proc = _first(rec, "processGuid", "process", "transformationGuid")
        src = _first(rec, "sourceGuid", "source", "fromGuid")
        tgt = _first(rec, "targetGuid", "target", "toGuid")
        if proc:
            proc_id = f"egeria_process:{proc}"
            _add(
                proc_id,
                "ProcessModel",
                {
                    "domain": "egeria",
                    "externalToolId": proc,
                    "name": _first(rec, "processName", "name"),
                },
            )
            if src:
                edges.append(_edge(f"egeria_asset:{src}", proc_id, "flowsTo"))
            if tgt:
                edges.append(_edge(proc_id, f"egeria_asset:{tgt}", "derivesFrom"))
        elif src and tgt:
            # Cross-layer / process-less lineage edge (from reconciliation + harvest).
            # Ensure both endpoints exist as nodes — reconciled infrastructure assets
            # are not always returned by list_assets — then map the semantic label to
            # flowsTo (data movement) or dependsOn (structural/hosting).
            for g, nk, tk in (
                (src, "sourceName", "sourceType"),
                (tgt, "targetName", "targetType"),
            ):
                _add(
                    f"egeria_asset:{g}",
                    _kg_type(_first(rec, tk)),
                    {
                        "domain": "egeria",
                        "externalToolId": g,
                        "name": _first(rec, nk) or g,
                    },
                )
            edges.append(
                _edge(
                    f"egeria_asset:{src}",
                    f"egeria_asset:{tgt}",
                    _flow_rel(_first(rec, "label")),
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


def _edge(source: str, target: str, rel_type: str) -> EnrichmentEdge:
    """Construct a typed enrichment edge (keyword-only Pydantic model)."""
    return EnrichmentEdge(source=source, target=target, rel_type=rel_type)


# Cross-link labels that denote a structural dependency rather than data movement.
_STRUCTURAL = {
    "hosts",
    "realizes",
    "secures",
    "same-as",
    "means",
    "deploys",
    "monitors",
    "reads",
    "groups",
}


def _flow_rel(label: Any) -> str:
    """Map a reconciliation/lineage edge label to a KG relationship type."""
    return "dependsOn" if str(label or "").lower() in _STRUCTURAL else "flowsTo"


# Egeria asset typeName → KG node type (for lineage-edge endpoints).
_KG_TYPES = {
    "SoftwareServer": "Server",
    "DeployedSoftwareComponent": "Tool",
    "RelationalDatabase": "DataObject",
    "DeployedDatabaseSchema": "DataObject",
    "Process": "ProcessModel",
    "Collection": "Concept",
}


def _kg_type(type_name: Any) -> str:
    return _KG_TYPES.get(str(type_name or ""), "DataConnector")


def _as_list(value: Any) -> list:
    """Coerce a scalar/None/list relation field into a clean list."""
    if value is None or value == "":
        return []
    if isinstance(value, list | tuple | set):
        return [v for v in value if v]
    return [value]


register_source(
    CATEGORY,
    extract,
    description="Apache Egeria open metadata / glossary / governance / lineage → KG",
)
