"""ServiceNow CMDB write-back sink (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Backfeeds the KG into ServiceNow — creates inventory CIs/products, enriches
existing records, writes inferred CI relations, and retires decommissioned CIs —
fail-closed (``SERVICENOW_ENABLE_WRITE``), dry-run-first. Resolves KG nodes to
sys_ids via the ``externalToolId``/``domain="servicenow"`` federation key. Uses
the ``servicenow-api`` CMDB write surface (``create/patch_cmdb_instance``,
``create_cmdb_relation``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..core import (
    PROVENANCE_TAG,
    WritebackClientMixin,
    WritebackInvocation,
    WritebackResult,
    register_sink,
)

logger = logging.getLogger(__name__)

_SOURCE = "agent-utilities"
_ATTEMPT_FAILED = object()

# KG node type → ServiceNow CMDB class for created inventory. Covers the fleet's
# emitted infra/container/host types (see INVENTORY_TYPES), not just the generic
# EA/TRM ones; unmapped types fall back to the generic ``cmdb_ci``.
_CLASS_MAP: dict[str, str] = {
    # generic EA / TRM / CMDB
    "server": "cmdb_ci_server",
    "hardwarenode": "cmdb_ci_server",
    "service": "cmdb_ci_service",
    "container": "cmdb_ci_docker_container",
    "application": "cmdb_ci_appl",
    "itcomponent": "cmdb_ci_appl",
    "technologyproduct": "cmdb_model",
    "assetinstance": "alm_hardware",
    "configurationitem": "cmdb_ci",
    # hosts / nodes
    "host": "cmdb_ci_server",
    "node": "cmdb_ci_server",
    "swarmnode": "cmdb_ci_server",
    "hostgroup": "cmdb_ci_cluster",
    # container / orchestration workloads
    "containerimage": "cmdb_model",
    "pod": "cmdb_ci_docker_container",
    "deployment": "cmdb_ci_appl",
    "workload": "cmdb_ci_appl",
    "k8sservice": "cmdb_ci_service",
    "swarmservice": "cmdb_ci_service",
    "namespace": "cmdb_ci_cloud_namespace",
    "stack": "cmdb_ci_appl",
    "repository": "cmdb_model",
    # network / storage components
    "networkinterface": "cmdb_ci_network_adapter",
    "diskvolume": "cmdb_ci_storage_volume",
    "tunnel": "cmdb_ci_ip_network",
}


def _class_for(node_type: str, default: str = "cmdb_ci") -> str:
    return _CLASS_MAP.get((node_type or "").lower(), default)


def _extract_sys_id(res: Any) -> str | None:
    """Best-effort pull of the created CI's sys_id from the client return shape."""
    if res is None:
        return None
    if isinstance(res, str):
        return res or None
    for obj in (res, getattr(res, "data", None), getattr(res, "result", None)):
        if isinstance(obj, dict):
            inner = obj.get("result")
            if isinstance(inner, dict) and inner.get("sys_id"):
                return str(inner["sys_id"])
            if obj.get("sys_id"):
                return str(obj["sys_id"])
    sid = getattr(res, "sys_id", None)
    return str(sid) if sid else None


class ServiceNowSink(WritebackClientMixin):
    """Write-back sink for ServiceNow CMDB."""

    domain = "servicenow"
    enable_flag = "SERVICENOW_ENABLE_WRITE"
    client_module = "servicenow_api"
    client_label = "servicenow"

    def _attempt(
        self,
        result: WritebackResult,
        action: str,
        call: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run one ServiceNow client call, sharing the try/except/log/count
        shape every op kind below needs (CX-DUP-ENFORCE: dupehound flagged
        this repeated try/except-log-count block across ``_run``'s five op
        loops as a structural duplicate; this is the extracted helper)."""
        try:
            return call(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("servicenow %s failed", action, exc_info=True)
            result.errors += 1
            return _ATTEMPT_FAILED

    def _run(self, invocation: WritebackInvocation) -> WritebackResult:
        ctx, ops, client, result, dry_run = invocation.unpack()

        resolve = ctx.resolver("servicenow")

        # creations — new inventory CIs/products.
        for c in ops.get("creations") or []:
            name = c.get("name")
            if not name:
                continue
            class_name = c.get("className") or _class_for(c.get("type", ""))
            attrs = {"name": name, **(c.get("attributes") or {})}
            if dry_run:
                result.proposals.append(
                    {"op": "create_cmdb_instance", "class": class_name, "name": name}
                )
                continue

            response = self._attempt(
                result,
                "create_cmdb_instance",
                client.create_cmdb_instance,  # type: ignore[union-attr]  # client None-checked above
                className=class_name,
                attributes=attrs,
                source=_SOURCE,
            )
            if response is not _ATTEMPT_FAILED:
                result.created += 1
                # Round-trip the sys_id back onto the source node → idempotent re-runs.
                ctx.stamp_external_id(
                    c.get("node"),
                    self.domain,
                    _extract_sys_id(response),
                    node_type=c.get("type", ""),
                )

        # enrichments — patch attributes onto existing CIs.
        for item in ops.get("enrichments") or []:
            sys_id = resolve(item.get("node"))
            attrs = item.get("attributes") or {}
            if not (sys_id and attrs):
                result.skipped += 1
                continue
            class_name = item.get("className") or "cmdb_ci"
            if dry_run:
                result.proposals.append(
                    {"op": "patch_cmdb_instance", "class": class_name, "sys_id": sys_id}
                )
                continue
            response = self._attempt(
                result,
                "patch_cmdb_instance",
                client.patch_cmdb_instance,  # type: ignore[union-attr]  # client None-checked above
                className=class_name,
                sys_id=sys_id,
                attributes=attrs,
                source=_SOURCE,
            )
            if response is not _ATTEMPT_FAILED:
                result.enriched += 1

        # inferred relations — between existing CIs.
        for edge in ops.get("inferences") or []:
            src = resolve(edge.get("source"))
            tgt = resolve(edge.get("target"))
            rel = edge.get("rel_type") or edge.get("type")
            if not (src and tgt and rel):
                result.skipped += 1
                continue
            class_name = edge.get("className") or "cmdb_ci"
            if dry_run:
                result.proposals.append(
                    {
                        "op": "create_cmdb_relation",
                        "sys_id": src,
                        "type": rel,
                        "target": tgt,
                        "provenance": PROVENANCE_TAG,
                    }
                )
                continue
            response = self._attempt(
                result,
                "create_cmdb_relation",
                client.create_cmdb_relation,  # type: ignore[union-attr]  # client None-checked above
                className=class_name,
                sys_id=src,
                outbound_relations=[{"type": rel, "target": tgt}],
                source=_SOURCE,
            )
            if response is not _ATTEMPT_FAILED:
                result.relations_written += 1

        # work_notes — append a review note to an existing ticket/demand record
        # (e.g. a TRM u_trm_request, an incident) WITHOUT touching CMDB CI
        # fields — a ticket is a different table than a CI, so this reuses the
        # generic table-patch surface (matching incident_router.ServiceNowAdapter's
        # update_ticket) rather than the CMDB-only create/patch_cmdb_instance
        # calls above. Portfolio-intelligence's TRM recommendation writeback is
        # the first consumer (CONCEPT:AU-KG.enrichment.portfolio-intelligence).
        for item in ops.get("work_notes") or []:
            sys_id = item.get("sys_id") or resolve(item.get("node"))
            note = item.get("note")
            table = item.get("table") or "task"
            if not (sys_id and note):
                result.skipped += 1
                continue
            # X5 (CONCEPT:AU-KG.temporal.bi-temporal-memory-layers): work_notes IS the
            # ServiceNow-native audit-trail field, so the KG-state instant this
            # recommendation derived from is appended directly into the note text
            # sent to ServiceNow — not just our own returned proposal/manifest.
            if ctx.as_of:
                note = f"{note}\n\n(KG state as of {ctx.as_of})"
            if dry_run:
                result.proposals.append(
                    {"op": "work_notes", "table": table, "sys_id": sys_id, "note": note}
                )
                continue
            response = self._attempt(
                result,
                "work_notes patch",
                client.patch_table_record,  # type: ignore[union-attr]  # client None-checked above
                table=table,
                table_record_sys_id=sys_id,
                data={"work_notes": note},
            )
            if response is not _ATTEMPT_FAILED:
                result.enriched += 1

        # retirements — mark CIs retired (install_status=7).
        for item in ops.get("retirements") or []:
            sys_id = resolve(item.get("node"))
            if not sys_id:
                result.skipped += 1
                continue
            class_name = item.get("className") or "cmdb_ci"
            if dry_run:
                result.proposals.append(
                    {"op": "retire", "class": class_name, "sys_id": sys_id}
                )
                continue
            response = self._attempt(
                result,
                "retire",
                client.patch_cmdb_instance,  # type: ignore[union-attr]  # client None-checked above
                className=class_name,
                sys_id=sys_id,
                attributes={"install_status": "7"},
                source=_SOURCE,
            )
            if response is not _ATTEMPT_FAILED:
                result.retired += 1

        return result


register_sink(ServiceNowSink())
