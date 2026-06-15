"""Ansible Tower/AWX source extractor — managed-host inventory (CONCEPT:KG-2.9).

Maps Tower hosts into the uniform ExtractionBatch as :Server (joining the same
infra/CMDB plane), stamped ``externalToolId`` + ``domain="ansible"``. The action
side (launching remediation jobs) lives in the write-back sink. Client
(``ansible_tower_mcp.auth.get_client()``) is injected; tolerant.
"""

from __future__ import annotations

from typing import Any

from ..models import ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "ansible"
_DOMAIN = "ansible"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def extract(config: Any) -> ExtractionBatch:
    """Extract Ansible Tower managed hosts into a uniform ExtractionBatch."""
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])

    lister = getattr(client, "list_hosts", None)
    if not callable(lister):
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])
    try:
        hosts = lister() or []
    except Exception:  # noqa: BLE001
        hosts = []

    for h in hosts:
        if not isinstance(h, dict):
            continue
        hid = h.get("id") or h.get("name")
        name = h.get("name")
        if not (hid and name):
            continue
        nodes.append(
            GraphNode(
                id=f"ansible_host:{hid}",
                type="Server",
                props={
                    "name": name,
                    "enabled": h.get("enabled"),
                    "externalToolId": str(hid),
                    "domain": _DOMAIN,
                },
            )
        )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])


register_extractor(
    CATEGORY, extract, description="Ansible Tower hosts (inventory) → KG"
)
