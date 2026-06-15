"""ServiceNow CMDB write-back sink (CONCEPT:KG-2.9).

Backfeeds the KG into ServiceNow — creates inventory CIs/products, enriches
existing records, writes inferred CI relations, and retires decommissioned CIs —
fail-closed (``SERVICENOW_ENABLE_WRITE``), dry-run-first. Resolves KG nodes to
sys_ids via the ``externalToolId``/``domain="servicenow"`` federation key. Uses
the ``servicenow-api`` CMDB write surface (``create/patch_cmdb_instance``,
``create_cmdb_relation``).
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import PROVENANCE_TAG, WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

_SOURCE = "agent-utilities"

# KG node type → ServiceNow CMDB class for created inventory.
_CLASS_MAP: dict[str, str] = {
    "server": "cmdb_ci_server",
    "hardwarenode": "cmdb_ci_server",
    "service": "cmdb_ci_service",
    "container": "cmdb_ci_docker_container",
    "application": "cmdb_ci_appl",
    "itcomponent": "cmdb_ci_appl",
    "technologyproduct": "cmdb_model",
    "assetinstance": "alm_hardware",
    "configurationitem": "cmdb_ci",
}


def _class_for(node_type: str, default: str = "cmdb_ci") -> str:
    return _CLASS_MAP.get((node_type or "").lower(), default)


class ServiceNowSink:
    """Write-back sink for ServiceNow CMDB."""

    domain = "servicenow"
    enable_flag = "SERVICENOW_ENABLE_WRITE"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from servicenow_api.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001 - connector absent / unconfigured
            logger.debug("servicenow write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result
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
            try:
                client.create_cmdb_instance(  # type: ignore[union-attr]  # client None-checked above
                    className=class_name, attributes=attrs, source=_SOURCE
                )
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("servicenow create_cmdb_instance failed", exc_info=True)
                result.errors += 1

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
            try:
                client.patch_cmdb_instance(  # type: ignore[union-attr]  # client None-checked above
                    className=class_name,
                    sys_id=sys_id,
                    attributes=attrs,
                    source=_SOURCE,
                )
                result.enriched += 1
            except Exception:  # noqa: BLE001
                logger.debug("servicenow patch_cmdb_instance failed", exc_info=True)
                result.errors += 1

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
            try:
                client.create_cmdb_relation(  # type: ignore[union-attr]  # client None-checked above
                    className=class_name,
                    sys_id=src,
                    outbound_relations=[{"type": rel, "target": tgt}],
                    source=_SOURCE,
                )
                result.relations_written += 1
            except Exception:  # noqa: BLE001
                logger.debug("servicenow create_cmdb_relation failed", exc_info=True)
                result.errors += 1

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
            try:
                client.patch_cmdb_instance(  # type: ignore[union-attr]  # client None-checked above
                    className=class_name,
                    sys_id=sys_id,
                    attributes={"install_status": "7"},
                    source=_SOURCE,
                )
                result.retired += 1
            except Exception:  # noqa: BLE001
                logger.debug("servicenow retire failed", exc_info=True)
                result.errors += 1

        return result


register_sink(ServiceNowSink())
