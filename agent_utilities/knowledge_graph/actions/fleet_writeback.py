"""Governed fleet write-back — the write side of the KG-2.59 source connector.

CONCEPT:KG-2.42 — vendor write-back via governed ontology Actions.

Ingestion (KG-2.59) *reads* records out of the ~58-server MCP fleet into the
Knowledge Graph. This is the symmetric *write* path: a single governed
``fleet.write_record`` action that pushes a mutation back to the source of record
through the very same fleet tools (``call_tool_once``, the write-side twin of the
source connector). Because it runs through the action executor, every external
write is authorization-gated, approval-gateable, and audited as an
``ActionInvocation`` — turning the KG from a read model into a *system of action*
over the enterprise without a single bespoke per-vendor write path.

The caller supplies the exact ``server`` / ``tool`` / ``action`` / ``params`` (the
fleet ``action`` + ``params_json`` convention), exactly as a source preset supplies
the read shape — e.g. patch a ServiceNow incident::

    execute_action("fleet.write_record", {
        "server": "servicenow-mcp", "tool": "servicenow_table_api",
        "action": "patch_table_record",
        "params": {"table": "incident", "sys_id": "...", "state": "6"},
    })
"""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
    _run_async,
)
from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
    call_tool_once,
)

from .models import ActionEffect, ActionParameter, OntologyAction
from .registry import ActionRegistry

logger = logging.getLogger(__name__)


FLEET_WRITE_RECORD = OntologyAction(
    name="fleet.write_record",
    verb="write",
    description=(
        "Push a governed mutation back to a fleet MCP system — the write side of "
        "the KG-2.59 source connector. Authorization-gated and audited by the action "
        "executor, so a Knowledge-Graph decision can update the source of record "
        "(patch a ServiceNow incident, update an ERPNext document, …) through the "
        "same fleet tools ingestion reads from."
    ),
    parameters=[
        ActionParameter(
            name="server",
            type="string",
            required=True,
            description="Fleet MCP server name (resolved via mcp_config), e.g. 'servicenow-mcp'.",
        ),
        ActionParameter(
            name="tool",
            type="string",
            required=True,
            description="The mutating tool on that server, e.g. 'servicenow_table_api'.",
        ),
        ActionParameter(
            name="action",
            type="string",
            required=False,
            description="Tool routing action, e.g. 'patch_table_record' (fleet action+params_json convention).",
        ),
        ActionParameter(
            name="params",
            type="object",
            required=False,
            description="Arguments for the tool action (JSON-encoded into params_json).",
        ),
    ],
    acts_on=["concept", "document", "fact", "node"],
    required_capability="fleet_write",
    produces_effect=ActionEffect.EXTERNAL,
    idempotent=False,
)


def _handle_fleet_write_record(params: dict[str, Any]) -> dict[str, Any]:
    """Invoke one fleet MCP mutating tool and return its decoded result."""
    server = params["server"]
    tool = params["tool"]
    action = params.get("action") or ""
    tool_params = params.get("params") or {}
    result = _run_async(
        call_tool_once(server=server, tool=tool, action=action, params=tool_params)
    )
    return {"server": server, "tool": tool, "action": action, "result": result}


def register_fleet_writeback(registry: ActionRegistry) -> None:
    """Register the governed fleet write-back action(s) into ``registry``."""
    registry.register(FLEET_WRITE_RECORD, _handle_fleet_write_record)
