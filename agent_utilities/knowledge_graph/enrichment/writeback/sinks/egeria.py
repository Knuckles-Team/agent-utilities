"""Egeria write-back sink (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Catalogs KG-derived entities into the Egeria open-metadata estate as governed
assets — the first-class write-back twin of the Egeria harvest, now on the unified
write-back surface. Standard tier, fail-closed (``EGERIA_ENABLE_WRITE``),
dry-run-first.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

# KG node type → Egeria open-metadata asset type.
_ASSET_TYPE = {
    "application": "DeployedSoftwareComponent",
    "itcomponent": "DeployedSoftwareComponent",
    "service": "SoftwareServer",
    "server": "SoftwareServer",
    "dataobject": "DeployedDatabaseSchema",
    "topic": "KafkaTopic",
    "configurationitem": "DeployedSoftwareComponent",
    "assetinstance": "DeployedSoftwareComponent",
    "technologyproduct": "DeployedSoftwareComponent",
}


class EgeriaSink:
    domain = "egeria"
    enable_flag = "EGERIA_ENABLE_WRITE"
    risk_tier = "standard"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from egeria_mcp.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("egeria write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result

        for c in ops.get("creations") or []:
            name = c.get("name")
            if not name:
                continue
            asset_type = c.get("asset_type") or _ASSET_TYPE.get(
                (c.get("type") or "").lower(), "DeployedSoftwareComponent"
            )
            qn = c.get("qualified_name") or f"KG::{c.get('type', 'Asset')}::{name}"
            if dry_run:
                result.proposals.append(
                    {"op": "create_asset", "type": asset_type, "name": name}
                )
                continue
            try:
                res = client.create_asset(  # type: ignore[union-attr]  # client None-checked above
                    asset_type,
                    qn,
                    name,
                    description=c.get("description", "KG-derived asset."),
                    additional_properties={"source": "agent-utilities-kg"},
                )
                if isinstance(res, dict) and res.get("guid"):
                    result.created += 1
                else:
                    result.errors += 1
            except Exception:  # noqa: BLE001
                logger.debug("egeria create_asset failed", exc_info=True)
                result.errors += 1

        return result


register_sink(EgeriaSink())
