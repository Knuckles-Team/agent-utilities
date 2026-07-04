"""Camunda BPMN source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor, step-level lift KG-2.53).

Self-registering extractor that maps Camunda process artifacts — process
definitions, tasks, and incidents — into the uniform ``ExtractionBatch`` shape
(typed ``GraphNode`` + ``EnrichmentEdge``) so they persist through the one
generic writer with no edits to any shared hub file.

Emitted node types are the **canonical** ArchiMate concepts, so Camunda data
folds into the same cross-vendor crosswalk as ServiceNow/ERPNext (see
``ontology_archimate.ttl``):

    process definition -> ``BusinessProcess``   id=bpmn_process:{id}
    task               -> ``BusinessTask``      id=bpmn_task:{id}   (PART_OF process)
    incident           -> ``Incident``          id=incident:{id}    (AFFECTS process)

**Step-level structure lift (CONCEPT:AU-KG.ontology.descriptive-process-world-gains).** When the injected client also
exposes ``get_process_definition_xml`` (the camunda-mcp ``camunda_process
action=xml`` surface), each definition's BPMN 2.0 XML is parsed (stdlib
ElementTree, namespace-tolerant) and the *static* process structure is lifted:

    flow element (task/userTask/serviceTask/.../gateway)
        -> ``BusinessTask``  id=bpmn_task:{processId}:{elementId}  (PART_OF process)
           typed via the ``task_type`` property (e.g. ``userTask``,
           ``exclusiveGateway``) — one node type, per the crosswalk
    sequenceFlow -> ``FLOWS_TO`` edges between lifted elements, collapsing
           through non-lifted pass-through elements (start/end/intermediate
           events) so ordering survives; a ``conditionExpression`` is preserved
           as the edge's ``condition`` property (gateway branching stays
           visible as multiple conditional FLOWS_TO edges)

Egeria reconciliation: a process record carrying an Egeria GUID
(``egeriaGuid``/``externalToolId``-style field) gets the ``externalToolId``
federation-key property (exactly how ``extractors/egeria.py`` stores external
ids) plus an ``ALIGNED_WITH`` equivalence edge to ``egeria_process:{guid}``
(the ontology's ``:alignedWith`` — the sameAs-style cross-source identity).

The Camunda client is **injected** (duck-typed) via ``config["client"]`` and is
expected to expose the camunda-mcp surface (``list_process_definitions()``,
``list_tasks()``, ``list_incidents()``). Method presence is probed so the
extractor tolerates the Camunda 7 vs 8 (Zeebe/Operate) client differences and
degrades to metadata-only extraction when the XML capability is absent. All
field access is tolerant of missing keys and this module performs **no** network
calls itself.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "camunda"

# BPMN 2.0 model namespace (the ``bpmn2`` prefix); parsing is tolerant of any
# namespace by matching local names, this is documentation of the canonical one.
BPMN2_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"

# Executable-step BPMN elements lifted to BusinessTask nodes (KG-2.53).
_TASK_TAGS = {
    "task",
    "userTask",
    "serviceTask",
    "scriptTask",
    "sendTask",
    "receiveTask",
    "businessRuleTask",
    "manualTask",
    "callActivity",
}
# Gateways are lifted too (typed via ``task_type``) so branching is queryable.
_GATEWAY_TAGS = {
    "exclusiveGateway",
    "parallelGateway",
    "inclusiveGateway",
    "eventBasedGateway",
    "complexGateway",
}
_LIFTED_TAGS = _TASK_TAGS | _GATEWAY_TAGS


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
    """Call a client method if present, returning a list (tolerant).

    camunda-mcp list methods accept an optional ``params`` dict; calling with no
    arguments returns the unfiltered collection.
    """
    method = getattr(client, name, None)
    if not callable(method):
        return []
    try:
        result = method()
    except TypeError:
        # Some client variants require an explicit (possibly empty) params arg.
        try:
            result = method({})
        except Exception:
            return []
    except Exception:
        return []
    if isinstance(result, dict):
        # Operate/Zeebe REST often wrap collections under "items".
        result = result.get("items") or result.get("results") or []
    return list(result) if result else []


def _fetch_bpmn_xml(client: Any, proc_id: Any) -> str | None:
    """Fetch a definition's BPMN 2.0 XML when the client supports it (KG-2.53).

    camunda-mcp exposes this as ``camunda_process action=xml`` →
    ``get_process_definition_xml(id=...)`` returning either the raw XML string
    or the Camunda 7 REST envelope ``{"id": ..., "bpmn20Xml": "<xml...>"}``.
    Absent capability or any failure degrades to metadata-only extraction.
    """
    method = getattr(client, "get_process_definition_xml", None)
    if not callable(method):
        return None
    try:
        result = method(proc_id)
    except TypeError:
        try:
            result = method(id=proc_id)
        except Exception:
            return None
    except Exception:
        return None
    if isinstance(result, dict):
        result = _first(result, "bpmn20Xml", "bpmnXml", "xml")
    if isinstance(result, str) and result.strip():
        return result
    return None


def _local(tag: Any) -> str:
    """Strip the XML namespace from a tag: ``{ns}userTask`` → ``userTask``."""
    text = str(tag)
    return text.rsplit("}", 1)[-1]


def _lift_process_structure(
    proc_node_id: str,
    proc_id: Any,
    xml_text: str,
    nodes: list[GraphNode],
    edges: list[EnrichmentEdge],
) -> None:
    """Lift one definition's BPMN XML into BusinessTask/FLOWS_TO structure.

    CONCEPT:AU-KG.ontology.descriptive-process-world-gains — the descriptive process world gains step-level shape:
    tasks/gateways become ``BusinessTask`` nodes (typed via ``task_type``,
    ``PART_OF`` the process) and sequence flows become ``FLOWS_TO`` edges.
    Non-lifted pass-through elements (start/end/intermediate events) are
    collapsed so ordering between lifted elements survives; a sequence flow's
    ``conditionExpression`` is preserved as the edge ``condition`` property.
    Parse failures are swallowed — structure lift is additive, never blocking.
    """
    import xml.etree.ElementTree as ET  # nosec B405 — BPMN XML comes from the operator-configured Camunda engine (trusted, injected client)

    try:
        root = ET.fromstring(xml_text)  # nosec B314 — same trusted operator-configured source, not untrusted input
    except ET.ParseError:
        return

    elements: dict[str, dict[str, Any]] = {}  # element_id -> {tag, name}
    flows: list[tuple[str, str, str | None]] = []  # (src, tgt, condition)
    for el in root.iter():
        tag = _local(el.tag)
        el_id = el.get("id")
        if not el_id:
            continue
        if tag == "sequenceFlow":
            src, tgt = el.get("sourceRef"), el.get("targetRef")
            if not src or not tgt:
                continue
            condition = None
            for child in el:
                if _local(child.tag) == "conditionExpression":
                    condition = (child.text or "").strip() or None
                    break
            flows.append((src, tgt, condition))
        else:
            elements[el_id] = {"tag": tag, "name": el.get("name")}

    lifted = {el_id for el_id, meta in elements.items() if meta["tag"] in _LIFTED_TAGS}
    if not lifted:
        return

    # Emit BusinessTask nodes for every lifted element.
    for el_id in sorted(lifted):
        meta = elements[el_id]
        node_id = f"bpmn_task:{proc_id}:{el_id}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="BusinessTask",
                props={
                    "name": meta["name"] or el_id,
                    "element_id": el_id,
                    "task_type": meta["tag"],
                    "is_gateway": meta["tag"] in _GATEWAY_TAGS,
                    "process_definition_id": str(proc_id),
                },
            )
        )
        edges.append(
            EnrichmentEdge(source=node_id, target=proc_node_id, rel_type="PART_OF")
        )

    # FLOWS_TO between lifted elements, collapsing through pass-through
    # elements (events). The first condition met along a collapsed path wins.
    outgoing: dict[str, list[tuple[str, str | None]]] = {}
    for src, tgt, condition in flows:
        outgoing.setdefault(src, []).append((tgt, condition))

    emitted: set[tuple[str, str]] = set()
    for src in sorted(lifted):
        # BFS forward through non-lifted elements until lifted targets.
        frontier: list[tuple[str, str | None]] = list(outgoing.get(src, []))
        visited: set[str] = {src}
        while frontier:
            tgt, condition = frontier.pop(0)
            if tgt in lifted:
                if (src, tgt) not in emitted:
                    emitted.add((src, tgt))
                    edges.append(
                        EnrichmentEdge(
                            source=f"bpmn_task:{proc_id}:{src}",
                            target=f"bpmn_task:{proc_id}:{tgt}",
                            rel_type="FLOWS_TO",
                            props={"condition": condition} if condition else {},
                        )
                    )
                continue
            if tgt in visited:
                continue  # cycle through pass-through elements — bounded walk
            visited.add(tgt)
            for nxt, nxt_condition in outgoing.get(tgt, []):
                frontier.append((nxt, condition or nxt_condition))


def extract(config: Any) -> ExtractionBatch:
    """Extract Camunda BPMN artifacts into a uniform ``ExtractionBatch``.

    ``config`` is a dict (or attr-style object) carrying an injected ``client``.
    Process definitions become ``BusinessProcess`` nodes; tasks become
    ``BusinessTask`` nodes linked ``PART_OF`` their process; incidents become
    canonical ``Incident`` nodes linked ``AFFECTS`` their process. When the
    client can serve BPMN XML, the step-level structure is lifted too
    (CONCEPT:AU-KG.ontology.descriptive-process-world-gains — see :func:`_lift_process_structure`).
    """
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    # --- Process definitions ------------------------------------------------
    for rec in _call(client, "list_process_definitions"):
        # Identity is the unique definition id so task/incident edges resolve;
        # the human-stable key is kept as a property.
        proc_id = _first(rec, "id", "key", "bpmnProcessId")
        if not proc_id:
            continue
        proc_node_id = f"bpmn_process:{proc_id}"
        props: dict[str, Any] = {
            "name": _first(rec, "name", "key", "bpmnProcessId"),
            "key": _first(rec, "key", "bpmnProcessId"),
            "version": _get(rec, "version"),
        }
        # Egeria reconciliation (KG-2.53): a record carrying an Egeria GUID is
        # federated exactly like the egeria extractor does — externalToolId
        # property + an alignedWith (sameAs-style) equivalence edge.
        egeria_guid = _first(rec, "egeriaGuid", "egeria_guid", "externalToolId")
        if egeria_guid:
            props["externalToolId"] = egeria_guid
            edges.append(
                EnrichmentEdge(
                    source=proc_node_id,
                    target=f"egeria_process:{egeria_guid}",
                    rel_type="ALIGNED_WITH",
                )
            )
        nodes.append(GraphNode(id=proc_node_id, type="BusinessProcess", props=props))

        # Step-level structure lift when the client can serve BPMN XML.
        xml_text = _fetch_bpmn_xml(client, proc_id)
        if xml_text:
            _lift_process_structure(proc_node_id, proc_id, xml_text, nodes, edges)

    # --- Tasks --------------------------------------------------------------
    for rec in _call(client, "list_tasks"):
        task_id = _first(rec, "id", "key")
        if not task_id:
            continue
        node_id = f"bpmn_task:{task_id}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="BusinessTask",
                props={
                    "name": _first(rec, "name", "taskDefinitionKey"),
                    "assignee": _get(rec, "assignee"),
                },
            )
        )
        proc_ref = _first(rec, "processDefinitionId", "processDefinitionKey")
        if proc_ref:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"bpmn_process:{proc_ref}",
                    rel_type="PART_OF",
                )
            )

    # --- Incidents ----------------------------------------------------------
    for rec in _call(client, "list_incidents"):
        inc_id = _first(rec, "id", "key")
        if not inc_id:
            continue
        node_id = f"incident:{inc_id}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Incident",
                props={
                    "short_description": _first(rec, "incidentMessage", "errorMessage"),
                    "incident_type": _get(rec, "incidentType"),
                },
            )
        )
        proc_ref = _first(rec, "processDefinitionId", "processDefinitionKey")
        if proc_ref:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"bpmn_process:{proc_ref}",
                    rel_type="AFFECTS",
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY,
    extract,
    description=(
        "Camunda BPMN (process definitions/tasks/incidents + step-level "
        "structure: BusinessTask/FLOWS_TO from BPMN XML) → KG"
    ),
)
