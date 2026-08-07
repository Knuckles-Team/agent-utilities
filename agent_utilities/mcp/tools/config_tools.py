"""``graph_config`` MCP tool — governed AgentConfig read/describe/diff/reload/set.

CONCEPT:AU-OS.config.two-surfaces-by-default

Thin action router over :mod:`agent_utilities.core.config_admin` (the one core).
No behaviour lives in this module: the tool body validates nothing, redacts
nothing and writes nothing itself, so the MCP surface and any future REST/CLI
twin cannot drift.

Why the tool exists at all: ``AgentConfig`` is the platform's entire behavioural
contract — including the MCP always-load set (``MCP_ALWAYS_LOAD`` /
``MCP_ALWAYS_LOAD_TOOLS``, CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog) — and until now an
agent could neither discover what was settable nor change it without a human
editing ``config.json`` and restarting.

Read posture (``get``/``describe``/``diff``/``reload``) is ungated, matching the
other graph surfaces, and every value it returns passes through the shared
redactor. ``set`` is governed: model-validated, ActionPolicy-gated
(``config.set``, approval_required), written through the existing
``save_config_item`` precedence, and recorded as a ``:ConfigChange`` provenance
event.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field

from agent_utilities.mcp import kg_server


def register_config_tools(mcp):
    """Register the ``graph_config`` tool onto the MCP server."""

    @mcp.tool(
        name="graph_config",
        description=(
            "Inspect and govern this deployment's AgentConfig "
            "(CONCEPT:AU-OS.config.two-surfaces-by-default). Actions: "
            "'describe' (key optional — a field's docstring, type, default, "
            "current value and env alias, all derived from the pydantic model, "
            "so you can discover what is settable without reading source; omit "
            "key for the whole inventory, narrow with contains), "
            "'get' (key -> effective value), "
            "'diff' (effective config vs shipped defaults — the fastest answer "
            "to 'why is this deployment behaving differently'), "
            "'reload' (re-read config without a restart; reports which fields "
            "are wired at startup and therefore still need one), "
            "'set' (key+value -> GOVERNED write: validated against the model, "
            "gated by ActionPolicy config.set, persisted through the standard "
            "config precedence, recorded as provenance). "
            "Secrets are redacted by reference, never by value: a vault://, "
            "env:// or ${VAR} reference is shown verbatim, an inline secret is "
            "never echoed and can never be written. Use this to change the MCP "
            "always-load set (MCP_ALWAYS_LOAD / MCP_ALWAYS_LOAD_TOOLS)."
        ),
        tags=["graph-os", "configure", "config"],
    )
    def graph_config(
        action: Literal["describe", "get", "diff", "reload", "set"] = Field(
            default="describe",
            description="describe | get | diff | reload | set",
        ),
        key: str = Field(
            default="",
            description=(
                "AgentConfig env alias (MCP_ALWAYS_LOAD) or field name "
                "(mcp_always_load). Required for get/set; optional for describe."
            ),
        ),
        value: str = Field(
            default="",
            description=(
                "New value for 'set'. JSON is parsed (arrays/objects/booleans); "
                "anything else is taken as a string and validated against the "
                "field's declared type."
            ),
        ),
        reason: str = Field(
            default="",
            description="Why this change is happening (audit trail, 'set').",
        ),
        contains: str = Field(
            default="",
            description="Substring filter for a keyless 'describe'.",
        ),
    ) -> str:
        from agent_utilities.core.config_admin import ConfigAdminError, dispatch

        try:
            return json.dumps(
                dispatch(
                    action,
                    key=key,
                    value=value,
                    reason=reason,
                    contains=contains,
                ),
                default=str,
            )
        except ConfigAdminError as exc:
            return json.dumps({"error": exc.code, "detail": exc.message})

    kg_server.REGISTERED_TOOLS["graph_config"] = graph_config
