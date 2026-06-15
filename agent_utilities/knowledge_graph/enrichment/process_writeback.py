"""Push KG-derived intelligence back INTO Camunda & ARIS (CONCEPT:KG-2.8).

The outbound twin of :mod:`capability_writeback`. Where that pushes minted
*capabilities* to EA tools, this pushes the **process intelligence** the KG
accumulates back onto the live process world, so the operator sees — on the
Camunda instance / ARIS model itself — what the graph knows about it:

1. **Capability / code lineage** — the Workflows that ``REALIZES`` the process
   and what they ``ORCHESTRATES`` (which agents/skills/code implement each step).
2. **Ontology inferences** — OWL-reasoned cross-source links: ``ALIGNED_WITH``
   twins (the Camunda⇆ARIS⇆Egeria same-process identity), ``governedBy`` policy
   attachments, and any inferred relationships touching the process.
3. **Operational signals** — open ``Incident`` nodes ``AFFECTS`` the process.
4. **Glossary / data lineage** — ``Concept`` (glossary) terms and ``DataObject``
   data the process's tasks touch.

The assembled ``kg_intelligence`` payload is written:

* **Camunda** — as a ``kg_intelligence`` **process-instance variable** (Json
  type) on every running instance of the definition, via the camunda-mcp
  ``Camunda7Api`` write surface (``list_process_instances`` →
  ``modify_process_instance_variables``). Camunda 7 has no runtime
  extension-property write, so instance variables are the correct vehicle;
  definitions with no running instances have nowhere to attach (reported as
  ``no_target``).
* **ARIS** — as a ``kg_intelligence`` **model attribute** via the aris-mcp
  ``set_model_attributes`` write surface (gated upstream on the tenant's API
  write tier).

Both clients are **duck-typed and optional** — pass whichever you have. Writes
are **idempotent**: the payload carries a content hash; before writing we read
the current value back and skip when the hash is unchanged. One failing target
never aborts the batch. No network or vendor import happens here; the caller
injects ready clients and a graph reader.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

#: the single namespaced variable/attribute key we own on the target side.
INTELLIGENCE_KEY = "kg_intelligence"


class ProcessWritebackResult:
    """Per-target counts of process-intelligence pushes (and skips)."""

    def __init__(self) -> None:
        self.camunda_pushed = 0
        self.aris_pushed = 0
        self.skipped_unchanged = 0
        self.no_target = 0
        self.errors = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "camunda_pushed": self.camunda_pushed,
            "aris_pushed": self.aris_pushed,
            "skipped_unchanged": self.skipped_unchanged,
            "no_target": self.no_target,
            "errors": self.errors,
        }


# ── graph reading ──────────────────────────────────────────────────────────
#
# Edge reads can't go through the epistemic backend's single-node ``execute``
# subset — they live in the compute layer — so we read via a tiny duck-typed
# ``reader`` (the engine's ``graph_compute``, adapted). This keeps the writeback
# logic a pure function over a reader + clients, so it unit-tests with a fake.


def _rel(props: Any) -> str:
    """Edge relation type from an edge-property dict (``rel_type`` key)."""
    if isinstance(props, dict):
        return str(props.get("rel_type") or props.get("relType") or "").upper()
    return ""


def _node_type(props: Any) -> str:
    if isinstance(props, dict):
        return str(props.get("type") or props.get("label") or "")
    return ""


def _name(props: Any, fallback: str = "") -> str:
    if isinstance(props, dict):
        return str(props.get("name") or props.get("short_description") or fallback)
    return fallback


def gather_intelligence(reader: Any, process_id: str) -> dict[str, Any]:
    """Assemble the ``kg_intelligence`` payload for one process node.

    ``reader`` is duck-typed: ``node_props(id) -> dict``,
    ``out_edges(id) -> [(tgt, props)]``, ``in_edges(id) -> [(src, props)]``.
    Every section is best-effort and tolerant of missing labels/edges.
    """
    capabilities: list[str] = []
    aligned_with: list[str] = []
    governance: list[str] = []
    incidents: list[str] = []
    glossary_terms: list[str] = []
    data_objects: list[str] = []

    out_edges = list(reader.out_edges(process_id) or [])
    in_edges = list(reader.in_edges(process_id) or [])

    # 1. Capability / code lineage — Workflows that REALIZES this process, and
    #    what they ORCHESTRATES (the implementing agents/skills/code).
    for src, props in in_edges:
        if _rel(props) == "REALIZES":
            wf_props = reader.node_props(src) or {}
            wf_name = _name(wf_props, src)
            orchestrated = [
                _name(reader.node_props(tgt) or {}, tgt)
                for tgt, ep in (reader.out_edges(src) or [])
                if _rel(ep) == "ORCHESTRATES"
            ]
            capabilities.append(
                f"{wf_name} ({', '.join(orchestrated)})" if orchestrated else wf_name
            )

    # 2. Ontology inferences — ALIGNED_WITH twins + governedBy policies (+ any
    #    edge reasoning marked inferred).
    for tgt, props in out_edges:
        rel = _rel(props)
        if rel == "ALIGNED_WITH":
            aligned_with.append(str(tgt))
        elif rel in ("GOVERNEDBY", "GOVERNED_BY"):
            governance.append(_name(reader.node_props(tgt) or {}, tgt))
    for src, props in in_edges:
        if _rel(props) == "ALIGNED_WITH":
            aligned_with.append(str(src))

    # 3. Operational signals — Incidents AFFECTS this process.
    for src, props in in_edges:
        if _rel(props) == "AFFECTS":
            inc = reader.node_props(src) or {}
            if "incident" in _node_type(inc).lower() or _rel(props) == "AFFECTS":
                incidents.append(_name(inc, str(src)))

    # 4. Glossary / data lineage — through the process's tasks (PART_OF) to the
    #    DataObjects they touch and the Concepts (glossary terms) they mention.
    task_ids = [
        src
        for src, props in in_edges
        if _rel(props) == "PART_OF"
        and "task" in _node_type(reader.node_props(src) or {}).lower()
    ]
    for task_id in task_ids:
        for tgt, ep in reader.out_edges(task_id) or []:
            tprops = reader.node_props(tgt) or {}
            ttype = _node_type(tprops).lower()
            rel = _rel(ep)
            if "dataobject" in ttype and rel in (
                "FLOWS_TO",
                "DERIVESFROM",
                "DERIVES_FROM",
            ):
                data_objects.append(_name(tprops, str(tgt)))
            elif "concept" in ttype and rel in ("MENTIONS", "RELATES_TO"):
                glossary_terms.append(_name(tprops, str(tgt)))

    payload: dict[str, Any] = {
        "capabilities": sorted(set(capabilities)),
        "aligned_with": sorted(set(aligned_with)),
        "governance": sorted(set(governance)),
        "incidents": sorted(set(incidents)),
        "glossary_terms": sorted(set(glossary_terms)),
        "data_objects": sorted(set(data_objects)),
    }
    return payload


def _is_empty(payload: dict[str, Any]) -> bool:
    """A payload with no signal in any of the four sections is skippable."""
    return not any(payload.get(k) for k in payload if not k.startswith("_"))


def _hashed(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload with a stable ``_hash`` of its content for idempotency."""
    body = {k: v for k, v in payload.items() if not k.startswith("_")}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return {**body, "_hash": digest}


def _existing_hash_camunda(client: Any, instance_id: str) -> str | None:
    """Read back the ``kg_intelligence`` instance variable's hash (or ``None``)."""
    getter = getattr(client, "get_process_instance_variables", None)
    if not callable(getter):
        return None
    try:
        variables = getter(instance_id) or {}
        var = variables.get(INTELLIGENCE_KEY) if isinstance(variables, dict) else None
        value = var.get("value") if isinstance(var, dict) else None
        parsed = json.loads(value) if isinstance(value, str) else value
        if isinstance(parsed, dict):
            return parsed.get("_hash")
    except Exception:  # noqa: BLE001 - read-back is best-effort
        return None
    return None


def _push_camunda(
    client: Any,
    process_key: str,
    payload: dict[str, Any],
    result: ProcessWritebackResult,
) -> None:
    """Write the payload as a ``kg_intelligence`` variable on running instances."""
    list_instances = getattr(client, "list_process_instances", None)
    modify = getattr(client, "modify_process_instance_variables", None)
    if not callable(list_instances) or not callable(modify):
        return
    try:
        instances = list_instances({"processDefinitionKey": process_key}) or []
    except Exception as exc:  # noqa: BLE001 - vendor transport
        logger.debug(
            "camunda list_process_instances failed for %s: %s", process_key, exc
        )
        result.errors += 1
        return
    if isinstance(instances, dict):
        instances = instances.get("items") or instances.get("results") or []
    if not instances:
        result.no_target += 1
        return

    hashed = _hashed(payload)
    value = json.dumps(hashed, default=str)
    for inst in instances:
        instance_id = (
            inst.get("id") if isinstance(inst, dict) else getattr(inst, "id", None)
        )
        if not instance_id:
            continue
        if _existing_hash_camunda(client, instance_id) == hashed["_hash"]:
            result.skipped_unchanged += 1
            continue
        try:
            modify(
                instance_id,
                {"modifications": {INTELLIGENCE_KEY: {"value": value, "type": "Json"}}},
            )
            result.camunda_pushed += 1
        except Exception as exc:  # noqa: BLE001 - vendor transport
            logger.debug("camunda modify vars failed for %s: %s", instance_id, exc)
            result.errors += 1


def _push_aris(
    client: Any,
    model_id: str,
    payload: dict[str, Any],
    result: ProcessWritebackResult,
) -> None:
    """Write the payload as a ``kg_intelligence`` attribute on the ARIS model."""
    setter = getattr(client, "set_model_attributes", None)
    if not callable(setter):
        return
    hashed = _hashed(payload)
    # Best-effort read-back for idempotency when the client can serve attributes.
    getter = getattr(client, "list_model_attributes", None) or getattr(
        client, "get_model", None
    )
    if callable(getter):
        try:
            attrs = getter(model_id) or {}
            current = attrs.get(INTELLIGENCE_KEY) if isinstance(attrs, dict) else None
            parsed = json.loads(current) if isinstance(current, str) else current
            if isinstance(parsed, dict) and parsed.get("_hash") == hashed["_hash"]:
                result.skipped_unchanged += 1
                return
        except Exception:  # noqa: BLE001 - read-back is best-effort
            pass
    try:
        setter(model_id, {INTELLIGENCE_KEY: json.dumps(hashed, default=str)})
        result.aris_pushed += 1
    except Exception as exc:  # noqa: BLE001 - vendor transport
        logger.debug("aris set_model_attributes failed for %s: %s", model_id, exc)
        result.errors += 1


def _process_targets(
    reader: Any, process_ids: Iterable[str] | None
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve the BusinessProcess nodes to enrich (explicit ids, or discovered)."""
    if process_ids:
        return [(pid, reader.node_props(pid) or {}) for pid in process_ids]
    discover = getattr(reader, "process_nodes", None)
    if callable(discover):
        return list(discover() or [])
    return []


def push_process_intelligence(
    reader: Any,
    *,
    camunda_client: Any | None = None,
    aris_client: Any | None = None,
    process_ids: Iterable[str] | None = None,
) -> ProcessWritebackResult:
    """Push KG process intelligence onto Camunda instances and/or ARIS models.

    ``reader`` is the graph reader (engine ``graph_compute`` adapter or a fake).
    A Camunda ``BusinessProcess`` (id ``bpmn_process:{key}``) routes to
    ``camunda_client``; an ARIS one (id ``aris_model:{id}``) routes to
    ``aris_client``. Processes with an empty payload are skipped entirely.
    """
    result = ProcessWritebackResult()
    if camunda_client is None and aris_client is None:
        return result

    for process_id, props in _process_targets(reader, process_ids):
        payload = gather_intelligence(reader, process_id)
        if _is_empty(payload):
            continue
        if camunda_client is not None and str(process_id).startswith("bpmn_process:"):
            key = str(props.get("key") or str(process_id).split(":", 1)[-1])
            _push_camunda(camunda_client, key, payload, result)
        elif aris_client is not None and str(process_id).startswith("aris_model:"):
            model_id = str(process_id).split(":", 1)[-1]
            _push_aris(aris_client, model_id, payload, result)

    return result


def resolve_process_writeback(
    engine: Any,
    *,
    camunda_client: Any = None,
    aris_client: Any = None,
    process_ids: Iterable[str] | None = None,
) -> ProcessWritebackResult | None:
    """Gated entry point: run process-intelligence writeback when enabled, else ``None``.

    Gated by the ``KG_PROCESS_WRITEBACK`` flag (default off → ``None`` → no-op).
    When on, it resolves the Camunda/ARIS clients (passed in, or best-effort from
    the connector packages via :func:`materialize.resolve_source_client`), adapts
    the engine's compute layer into a graph reader, and pushes.
    """
    if not setting("KG_PROCESS_WRITEBACK", False):
        return None
    if camunda_client is None and aris_client is None:
        from .materialize import resolve_source_client

        camunda_client = resolve_source_client("camunda")
        aris_client = resolve_source_client("aris")
    if camunda_client is None and aris_client is None:
        logger.debug("KG_PROCESS_WRITEBACK set but no Camunda/ARIS client available")
        return None
    reader = GraphComputeReader(engine)
    return push_process_intelligence(
        reader,
        camunda_client=camunda_client,
        aris_client=aris_client,
        process_ids=process_ids,
    )


class GraphComputeReader:
    """Adapt the engine's ``graph_compute`` into the duck-typed reader above.

    Exposes ``node_props``/``out_edges``/``in_edges``/``process_nodes`` over the
    compute layer's NX-style edge views — the only place process edges live
    (the L1 backend's ``execute`` is single-node-only).
    """

    def __init__(self, engine: Any) -> None:
        self._gc = getattr(engine, "graph_compute", None)

    def node_props(self, node_id: str) -> dict[str, Any]:
        if self._gc is None:
            return {}
        try:
            return dict(self._gc._get_node_properties(node_id) or {})
        except Exception:  # noqa: BLE001 - tolerant read
            return {}

    def out_edges(self, node_id: str) -> list[tuple[str, dict[str, Any]]]:
        if self._gc is None:
            return []
        try:
            return [(t, p) for _s, t, p in self._gc.out_edges(node_id, data=True)]
        except Exception:  # noqa: BLE001
            return []

    def in_edges(self, node_id: str) -> list[tuple[str, dict[str, Any]]]:
        if self._gc is None:
            return []
        try:
            return [(s, p) for s, _t, p in self._gc.in_edges(node_id, data=True)]
        except Exception:  # noqa: BLE001
            return []

    def process_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        """Every BusinessProcess node (by id prefix or type)."""
        if self._gc is None:
            return []
        out: list[tuple[str, dict[str, Any]]] = []
        try:
            for nid, props in self._gc.nodes(data=True):
                pid = str(nid)
                if (
                    pid.startswith(("bpmn_process:", "aris_model:"))
                    or _node_type(props) == "BusinessProcess"
                ):
                    out.append((pid, dict(props)))
        except Exception:  # noqa: BLE001
            return []
        return out
