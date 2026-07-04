"""Process write-back sink — Camunda & ARIS (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

Pushes the KG's per-process intelligence (capability/code lineage, OWL inferences,
incidents, glossary/data lineage) onto live Camunda instances and ARIS models as a
hash-idempotent ``kg_intelligence`` payload. Folds the former ``process_writeback``
module onto the unified sink contract (fail-closed ``KG_PROCESS_WRITEBACK``,
dry-run-first).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

INTELLIGENCE_KEY = "kg_intelligence"


def _rel(props: Any) -> str:
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
    """Assemble the ``kg_intelligence`` payload for one process node."""
    capabilities: list[str] = []
    aligned_with: list[str] = []
    governance: list[str] = []
    incidents: list[str] = []
    glossary_terms: list[str] = []
    data_objects: list[str] = []

    out_edges = list(reader.out_edges(process_id) or [])
    in_edges = list(reader.in_edges(process_id) or [])

    for src, props in in_edges:
        if _rel(props) == "REALIZES":
            wf_name = _name(reader.node_props(src) or {}, src)
            orchestrated = [
                _name(reader.node_props(tgt) or {}, tgt)
                for tgt, ep in (reader.out_edges(src) or [])
                if _rel(ep) == "ORCHESTRATES"
            ]
            capabilities.append(
                f"{wf_name} ({', '.join(orchestrated)})" if orchestrated else wf_name
            )

    for tgt, props in out_edges:
        rel = _rel(props)
        if rel == "ALIGNED_WITH":
            aligned_with.append(str(tgt))
        elif rel in ("GOVERNEDBY", "GOVERNED_BY"):
            governance.append(_name(reader.node_props(tgt) or {}, tgt))
    for src, props in in_edges:
        if _rel(props) == "ALIGNED_WITH":
            aligned_with.append(str(src))

    for src, props in in_edges:
        if _rel(props) == "AFFECTS":
            incidents.append(_name(reader.node_props(src) or {}, str(src)))

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

    return {
        "capabilities": sorted(set(capabilities)),
        "aligned_with": sorted(set(aligned_with)),
        "governance": sorted(set(governance)),
        "incidents": sorted(set(incidents)),
        "glossary_terms": sorted(set(glossary_terms)),
        "data_objects": sorted(set(data_objects)),
    }


def _is_empty(payload: dict[str, Any]) -> bool:
    return not any(payload.get(k) for k in payload if not k.startswith("_"))


def _hashed(payload: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in payload.items() if not k.startswith("_")}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return {**body, "_hash": digest}


def _existing_hash_camunda(client: Any, instance_id: str) -> str | None:
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
    except Exception:  # noqa: BLE001
        return None
    return None


def _push_camunda(
    client: Any, process_key: str, payload: dict[str, Any], result: WritebackResult
) -> None:
    list_instances = getattr(client, "list_process_instances", None)
    modify = getattr(client, "modify_process_instance_variables", None)
    if not callable(list_instances) or not callable(modify):
        return
    try:
        instances = list_instances({"processDefinitionKey": process_key}) or []
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "camunda list_process_instances failed for %s: %s", process_key, exc
        )
        result.errors += 1
        return
    if isinstance(instances, dict):
        instances = instances.get("items") or instances.get("results") or []
    if not instances:
        result.skipped += 1
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
            result.skipped += 1
            continue
        try:
            modify(
                instance_id,
                {"modifications": {INTELLIGENCE_KEY: {"value": value, "type": "Json"}}},
            )
            result.enriched += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("camunda modify vars failed for %s: %s", instance_id, exc)
            result.errors += 1


def _push_aris(
    client: Any, model_id: str, payload: dict[str, Any], result: WritebackResult
) -> None:
    setter = getattr(client, "set_model_attributes", None)
    if not callable(setter):
        return
    hashed = _hashed(payload)
    getter = getattr(client, "list_model_attributes", None) or getattr(
        client, "get_model", None
    )
    if callable(getter):
        try:
            attrs = getter(model_id) or {}
            current = attrs.get(INTELLIGENCE_KEY) if isinstance(attrs, dict) else None
            parsed = json.loads(current) if isinstance(current, str) else current
            if isinstance(parsed, dict) and parsed.get("_hash") == hashed["_hash"]:
                result.skipped += 1
                return
        except Exception:  # noqa: BLE001
            pass
    try:
        setter(model_id, {INTELLIGENCE_KEY: json.dumps(hashed, default=str)})
        result.enriched += 1
    except Exception as exc:  # noqa: BLE001
        logger.debug("aris set_model_attributes failed for %s: %s", model_id, exc)
        result.errors += 1


def _process_targets(
    reader: Any, process_ids: Iterable[str] | None
) -> list[tuple[str, dict[str, Any]]]:
    if process_ids:
        return [(pid, reader.node_props(pid) or {}) for pid in process_ids]
    discover = getattr(reader, "process_nodes", None)
    return list(discover() or []) if callable(discover) else []


def push_process_intelligence(
    reader: Any,
    *,
    camunda_client: Any | None = None,
    aris_client: Any | None = None,
    process_ids: Iterable[str] | None = None,
    dry_run: bool = False,
    result: WritebackResult | None = None,
) -> WritebackResult:
    """Push KG process intelligence onto Camunda instances and/or ARIS models."""
    result = result or WritebackResult(target="process")
    if camunda_client is None and aris_client is None:
        return result
    for process_id, props in _process_targets(reader, process_ids):
        payload = gather_intelligence(reader, process_id)
        if _is_empty(payload):
            continue
        is_camunda = str(process_id).startswith("bpmn_process:")
        is_aris = str(process_id).startswith("aris_model:")
        if dry_run:
            if (is_camunda and camunda_client is not None) or (
                is_aris and aris_client is not None
            ):
                result.proposals.append(
                    {"op": "write_intelligence", "process": str(process_id)}
                )
            continue
        if camunda_client is not None and is_camunda:
            key = str(props.get("key") or str(process_id).split(":", 1)[-1])
            _push_camunda(camunda_client, key, payload, result)
        elif aris_client is not None and is_aris:
            _push_aris(aris_client, str(process_id).split(":", 1)[-1], payload, result)
    return result


class GraphComputeReader:
    """Adapt the engine's ``graph_compute`` into the duck-typed process reader."""

    def __init__(self, engine: Any) -> None:
        self._gc = getattr(engine, "graph_compute", None)

    def node_props(self, node_id: str) -> dict[str, Any]:
        if self._gc is None:
            return {}
        try:
            return dict(self._gc._get_node_properties(node_id) or {})
        except Exception:  # noqa: BLE001
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


class ProcessSink:
    """Write-back sink for Camunda & ARIS process intelligence."""

    domain = "process"
    enable_flag = "KG_PROCESS_WRITEBACK"

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        target = str(ops.get("target") or "both").lower()
        camunda_client = ops.get("camunda_client")
        aris_client = ops.get("aris_client")
        if camunda_client is None and aris_client is None:
            from ...materialize import resolve_source_client

            if target in ("camunda", "both"):
                camunda_client = resolve_source_client("camunda")
            if target in ("aris", "both"):
                aris_client = resolve_source_client("aris")
        if camunda_client is None and aris_client is None:
            result.skipped += 1
            return result
        reader = GraphComputeReader(ctx.engine)
        return push_process_intelligence(
            reader,
            camunda_client=camunda_client,
            aris_client=aris_client,
            process_ids=ops.get("process_ids"),
            dry_run=dry_run,
            result=result,
        )


register_sink(ProcessSink())
