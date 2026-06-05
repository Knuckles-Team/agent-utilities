"""Camunda BPMN source extractor (CONCEPT:KG-2.9).

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

The Camunda client is **injected** (duck-typed) via ``config["client"]`` and is
expected to expose the camunda-mcp surface (``list_process_definitions()``,
``list_tasks()``, ``list_incidents()``). Method presence is probed so the
extractor tolerates the Camunda 7 vs 8 (Zeebe/Operate) client differences. All
field access is tolerant of missing keys and this module performs **no** network
calls itself.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_source

CATEGORY = "camunda"


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


def extract(config: Any) -> ExtractionBatch:
    """Extract Camunda BPMN artifacts into a uniform ``ExtractionBatch``.

    ``config`` is a dict (or attr-style object) carrying an injected ``client``.
    Process definitions become ``BusinessProcess`` nodes; tasks become
    ``BusinessTask`` nodes linked ``PART_OF`` their process; incidents become
    canonical ``Incident`` nodes linked ``AFFECTS`` their process.
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
        nodes.append(
            GraphNode(
                id=f"bpmn_process:{proc_id}",
                type="BusinessProcess",
                props={
                    "name": _first(rec, "name", "key", "bpmnProcessId"),
                    "key": _first(rec, "key", "bpmnProcessId"),
                    "version": _get(rec, "version"),
                },
            )
        )

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


register_source(
    CATEGORY,
    extract,
    description="Camunda BPMN (process definitions/tasks/incidents) → KG",
)
