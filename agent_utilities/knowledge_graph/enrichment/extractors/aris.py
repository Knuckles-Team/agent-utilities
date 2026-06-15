"""ARIS process / enterprise-architecture source extractor (CONCEPT:KG-2.9, step-level lift KG-2.53).

Self-registering extractor that maps Software AG **ARIS** models into the uniform
``ExtractionBatch`` shape (typed ``GraphNode`` + ``EnrichmentEdge``) so they
persist through the one generic writer with no edits to any shared hub file.

Emitted node types are the **canonical** ArchiMate concepts so ARIS folds into the
same cross-vendor crosswalk as Camunda (BPM) and LeanIX/ArchiMate (EA):

    process model       -> ``BusinessProcess``      id=aris_model:{id}
    architecture model  -> ``ApplicationComponent`` id=aris_model:{id}

**Step-level structure lift (CONCEPT:KG-2.53).** When the injected client also
exposes the EPC detail surface (``list_model_objects(model_id)`` +
``list_model_connections(model_id)`` — the ``aris-mcp`` ``aris_model
action=objects/connections`` tools), each *process* model's
Event-driven-Process-Chain is lifted to the same shape the Camunda BPMN extractor
produces, so an ARIS EPC and its Camunda implementation are queryable/reasoned
over identically:

    function (OT_FUNC)        -> ``BusinessTask``  id=aris_object:{model}:{obj}  (PART_OF model)
    rule/operator (OT_RULE)   -> ``BusinessTask``  (``is_gateway=True``, ``gateway_kind`` AND|OR|XOR)
                                 — the EPC analogue of a BPMN gateway, so the
                                 ProcessPlanCompiler collapses branching identically
    event (OT_EVT)            -> NOT lifted (collapsed through, like BPMN start/end
                                 events) so ``FLOWS_TO`` ordering between functions
                                 and rules survives
    connection                -> ``FLOWS_TO`` edges between lifted objects; a branch
                                 label/condition is preserved as the edge
                                 ``condition`` property (XOR branching stays visible)

Cross-vendor reconciliation: a model record carrying a Camunda key or Egeria GUID
(``camundaKey``/``egeriaGuid``/``externalToolId``-style field) gets the
``externalToolId`` federation-key property plus an ``ALIGNED_WITH`` equivalence
edge to ``bpmn_process:{key}`` / ``egeria_process:{guid}`` (the ontology's
``:alignedWith`` — the sameAs-style cross-source identity), so an ARIS process and
its Camunda/Egeria twin collapse to one identity under reasoning.

The ARIS client is **injected** (duck-typed) via ``config["client"]`` and is
expected to expose ``list_models()`` (and, for the step lift,
``list_model_objects``/``list_model_connections``). Method presence is probed so
the extractor degrades to model-level extraction when the EPC surface is absent.
All field access is tolerant and this module performs **no** network calls itself.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "aris"

# Model-type substrings marking a process (BPM) model vs an architecture model.
_BPM_HINTS = ("process", "epc", "bpmn", "value", "vad")


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


def _unwrap(result: Any) -> list:
    """Normalise a client return into a list (tolerant of envelope dicts)."""
    if isinstance(result, dict):
        result = (
            result.get("items")
            or result.get("objects")
            or result.get("connections")
            or result.get("models")
            or result.get("data")
            or []
        )
    return list(result) if result else []


def _call(client: Any, name: str) -> list:
    """Call a no-arg client method if present, returning a list (tolerant)."""
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
    return _unwrap(result)


def _call_arg(client: Any, name: str, arg: Any) -> list:
    """Call a single-arg client method (e.g. ``list_model_objects(model_id)``)."""
    method = getattr(client, name, None)
    if not callable(method):
        return []
    try:
        result = method(arg)
    except Exception:
        return []
    return _unwrap(result)


def _is_process(model_type: str) -> bool:
    t = (model_type or "").lower()
    return any(h in t for h in _BPM_HINTS)


def _object_kind(rec: Any) -> str:
    """Classify an ARIS object into ``function`` | ``rule`` | ``event`` | ``other``.

    Matches against the ARIS object-type code / symbol / type name tolerantly
    (``OT_FUNC``/``OT_RULE``/``OT_EVT`` and their symbol variants) so the lift
    survives the Connect-vs-Cloud vocabulary differences.
    """
    raw = str(
        _first(rec, "typeName", "type", "objectType", "symbolType", "symbol") or ""
    ).lower()
    if "func" in raw:
        return "function"
    if "rule" in raw or "operator" in raw or "opr" in raw:
        return "rule"
    if "evt" in raw or "event" in raw:
        return "event"
    return "other"


def _gateway_kind(rec: Any) -> str | None:
    """Resolve an ARIS rule operator to ``XOR`` | ``OR`` | ``AND`` when discernible."""
    raw = str(
        _first(rec, "symbol", "symbolType", "typeName", "type", "name") or ""
    ).lower()
    if "xor" in raw:
        return "XOR"
    if "and" in raw:
        return "AND"
    if "or" in raw:
        return "OR"
    return None


def _lift_epc_structure(
    model_node_id: str,
    model_id: Any,
    client: Any,
    nodes: list[GraphNode],
    edges: list[EnrichmentEdge],
) -> None:
    """Lift one process model's EPC into BusinessTask/FLOWS_TO structure (KG-2.53).

    Functions and rule operators become ``BusinessTask`` nodes (``PART_OF`` the
    model); events are collapsed pass-throughs so control-flow ordering between
    the lifted objects survives. Branch labels/conditions on connections are
    preserved as the ``FLOWS_TO`` edge ``condition`` property. A client without
    the EPC surface yields no detail (model-level node only).
    """
    raw_objects = _call_arg(client, "list_model_objects", model_id)
    if not raw_objects:
        return
    raw_connections = _call_arg(client, "list_model_connections", model_id)

    # object_id -> classification + display props
    kinds: dict[str, str] = {}
    metas: dict[str, dict[str, Any]] = {}
    for rec in raw_objects:
        obj_id = _first(rec, "id", "objectId", "guid", "ObjGuid")
        if not obj_id:
            continue
        kind = _object_kind(rec)
        kinds[str(obj_id)] = kind
        metas[str(obj_id)] = {
            "name": _first(rec, "name", "label") or str(obj_id),
            "kind": kind,
            "gateway_kind": _gateway_kind(rec) if kind == "rule" else None,
        }

    lifted = {oid for oid, kind in kinds.items() if kind in ("function", "rule")}
    if not lifted:
        return

    # Emit BusinessTask nodes for every lifted object.
    for obj_id in sorted(lifted):
        meta = metas[obj_id]
        node_id = f"aris_object:{model_id}:{obj_id}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="BusinessTask",
                props={
                    "name": meta["name"],
                    "object_id": obj_id,
                    "object_type": meta["kind"],
                    "is_gateway": meta["kind"] == "rule",
                    "gateway_kind": meta["gateway_kind"],
                    "aris_model_id": str(model_id),
                },
            )
        )
        edges.append(
            EnrichmentEdge(source=node_id, target=model_node_id, rel_type="PART_OF")
        )

    # Build the directed control-flow adjacency, then FLOWS_TO between lifted
    # objects, collapsing through non-lifted (event) objects — mirrors the
    # Camunda BPMN sequence-flow collapse so the two graphs are shaped alike.
    outgoing: dict[str, list[tuple[str, str | None]]] = {}
    for rec in raw_connections:
        src = _first(rec, "sourceObjectId", "source", "fromObjectId", "from", "sourceId")
        tgt = _first(rec, "targetObjectId", "target", "toObjectId", "to", "targetId")
        if not src or not tgt:
            continue
        condition = _first(rec, "condition", "label", "name")
        outgoing.setdefault(str(src), []).append(
            (str(tgt), str(condition) if condition else None)
        )

    emitted: set[tuple[str, str]] = set()
    for src in sorted(lifted):
        frontier: list[tuple[str, str | None]] = list(outgoing.get(src, []))
        visited: set[str] = {src}
        while frontier:
            tgt, condition = frontier.pop(0)
            if tgt in lifted:
                if (src, tgt) not in emitted:
                    emitted.add((src, tgt))
                    edges.append(
                        EnrichmentEdge(
                            source=f"aris_object:{model_id}:{src}",
                            target=f"aris_object:{model_id}:{tgt}",
                            rel_type="FLOWS_TO",
                            props={"condition": condition} if condition else {},
                        )
                    )
                continue
            if tgt in visited:
                continue  # bounded walk through pass-through events
            visited.add(tgt)
            for nxt, nxt_condition in outgoing.get(tgt, []):
                frontier.append((nxt, condition or nxt_condition))


def extract(config: Any) -> ExtractionBatch:
    """Extract ARIS models into a uniform ``ExtractionBatch``.

    ``config`` carries an injected ``client``. Process models become canonical
    ``BusinessProcess`` nodes; architecture models become ``ApplicationComponent``
    nodes — so they cross-link with the BPM and EA cohorts respectively. For
    process models, the EPC step structure is lifted too when the client can
    serve it (CONCEPT:KG-2.53 — see :func:`_lift_epc_structure`).
    """
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for rec in _call(client, "list_models"):
        mid = _first(rec, "id", "modelId", "name")
        if not mid:
            continue
        mtype = _first(rec, "type", "modelType") or "Model"
        is_proc = _is_process(mtype)
        label = "BusinessProcess" if is_proc else "ApplicationComponent"
        model_node_id = f"aris_model:{mid}"
        props: dict[str, Any] = {
            "name": _first(rec, "name", "id"),
            "model_type": mtype,
            "capability": "bpm" if is_proc else "enterprise-architecture",
        }
        # Cross-vendor reconciliation (KG-2.53): federate to a Camunda/Egeria twin.
        camunda_key = _first(rec, "camundaKey", "camunda_key", "bpmnProcessId")
        egeria_guid = _first(rec, "egeriaGuid", "egeria_guid", "externalToolId")
        if camunda_key:
            props["externalToolId"] = camunda_key
            edges.append(
                EnrichmentEdge(
                    source=model_node_id,
                    target=f"bpmn_process:{camunda_key}",
                    rel_type="ALIGNED_WITH",
                )
            )
        elif egeria_guid:
            props["externalToolId"] = egeria_guid
            edges.append(
                EnrichmentEdge(
                    source=model_node_id,
                    target=f"egeria_process:{egeria_guid}",
                    rel_type="ALIGNED_WITH",
                )
            )
        nodes.append(GraphNode(id=model_node_id, type=label, props=props))

        # Step-level EPC lift for process models when the client can serve it.
        if is_proc:
            _lift_epc_structure(model_node_id, mid, client, nodes, edges)

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY,
    extract,
    description=(
        "ARIS models (process/architecture + step-level EPC structure: "
        "BusinessTask/FLOWS_TO from functions/rules/events) → KG"
    ),
)
