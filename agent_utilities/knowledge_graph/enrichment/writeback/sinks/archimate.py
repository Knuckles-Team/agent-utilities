"""ArchiMate model write-back sink (CONCEPT:KG-2.9).

Writes KG-inferred elements/relationships back into the ArchiMate model so the
model stays a live mirror of the cross-tool reality (ServiceNow/LeanIX/Camunda all
map to ArchiMate classes). Standard tier (a local model file), fail-closed
(``ARCHIMATE_ENABLE_WRITE``), dry-run-first. Resolves KG nodes → ArchiMate element
ids via ``externalToolId``/``domain="archimate"``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

# UPPER_SNAKE KG rel type → ArchiMate relationship type.
_REL_TYPE = {
    "REALIZES": "Realization",
    "REALIZED_BY": "Realization",
    "SERVES": "Serving",
    "ASSIGNED_TO": "Assignment",
    "DEPENDS_ON": "Serving",
    "PART_OF": "Composition",
    "FLOWS_TO": "Flow",
    "TRIGGERS": "Triggering",
}


class ArchimateSink:
    domain = "archimate"
    enable_flag = "ARCHIMATE_ENABLE_WRITE"
    risk_tier = "standard"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from archimate_mcp.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("archimate write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result
        resolve = ctx.resolver("archimate")

        for c in ops.get("creations") or []:
            name = c.get("name")
            etype = c.get("type") or "Grouping"
            if not name:
                continue
            if dry_run:
                result.proposals.append(
                    {"op": "add_element", "type": etype, "name": name}
                )
                continue
            try:
                client.add_element(
                    type=etype, name=name, documentation=c.get("documentation", "")
                )
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("archimate add_element failed", exc_info=True)
                result.errors += 1

        for edge in ops.get("inferences") or []:
            src = resolve(edge.get("source"))
            tgt = resolve(edge.get("target"))
            rel = _REL_TYPE.get(edge.get("rel_type") or edge.get("type"), "Association")
            if not (src and tgt):
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append(
                    {
                        "op": "add_relationship",
                        "source": src,
                        "target": tgt,
                        "type": rel,
                    }
                )
                continue
            try:
                client.add_relationship(src, tgt, rel)
                result.relations_written += 1
            except Exception:  # noqa: BLE001
                logger.debug("archimate add_relationship failed", exc_info=True)
                result.errors += 1

        return result


register_sink(ArchimateSink())
