"""The single MCP tool contract of the ecosystem (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse).

Every MCP server built on agent-utilities — graph-os and every fleet connector —
serves the same contract: a handful of **intent tools** (``ask``, ``find``,
``act``, …), each with a condensed input schema:

``action``
    The operation to run, as an operation id from the server's internal
    operation table (``"<tool>.<op>"``, or ``"<tool>"`` for a single-operation
    tool), or ``"describe"``. An enum where the set is small and fixed; a
    string validated against the table where it is not.
``params``
    The operation's arguments as one JSON object.
``intent``
    Optional natural language. With no ``action``, the server routes it.

``action="describe"`` returns the operations an intent tool can run, and
``params={"action": "<operation id>"}`` returns one operation's full argument
schema — so per-operation schemas are served on demand instead of being
embedded in every ``tools/list``.

This module holds the parts both surfaces share: the contract's field
definitions, argument shaping onto a backing tool, schema projection, and the
token-budget measure the surface gates use.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from pydantic import Field

#: Maximum number of tools in a server's default ``tools/list``.
DEFAULT_SURFACE_TOOL_LIMIT = 10
#: Token budget for a server's whole default ``tools/list``.
DEFAULT_SURFACE_TOKEN_BUDGET = 4000
#: Token budget for any single intent tool's listed definition.
INTENT_TOOL_TOKEN_BUDGET = 400

#: The pseudo-action every intent tool answers with its operation catalog.
DESCRIBE_ACTION = "describe"

#: Leading action-name tokens that read without mutating.
READ_ACTION_PREFIXES = frozenset(
    {
        "check",
        "count",
        "describe",
        "discover",
        "doctor",
        "explain",
        "export",
        "fetch",
        "find",
        "get",
        "has",
        "history",
        "inspect",
        "list",
        "lookup",
        "metrics",
        "preflight",
        "profile",
        "query",
        "read",
        "recall",
        "report",
        "search",
        "show",
        "status",
        "validate",
        "view",
    }
)

ACTION_FIELD_DESCRIPTION = (
    "Operation id ('<tool>.<op>' or '<tool>') or 'describe'; empty routes `intent`."
)
PARAMS_FIELD_DESCRIPTION = (
    "Operation arguments. With action='describe', {'action': '<id>'} returns "
    "that operation's argument schema."
)
INTENT_FIELD_DESCRIPTION = "Natural-language request, routed when action is empty."


def estimate_tokens(payload: Any) -> int:
    """Token estimate of a JSON-serializable tool definition.

    Four characters per token over the JSON text a client receives — within a
    few percent of the cl100k BPE count on these tool definitions — so the
    budget gates need no tokenizer dependency.
    """
    text = json.dumps(payload, default=str)
    return math.ceil(len(text) / 4)


def tool_definition(tool: Any) -> dict[str, Any]:
    """The ``tools/list`` entry a client receives for a FastMCP tool."""
    return tool.to_mcp_tool().model_dump(mode="json", by_alias=True, exclude_none=True)


def operation_id(tool: str, action: str | None) -> str:
    """Operation id of ``tool``'s ``action`` (``tool`` for a single operation)."""
    return tool if action is None else f"{tool}.{action}"


def split_operation_id(op_id: str) -> tuple[str, str | None]:
    """Inverse of :func:`operation_id`."""
    tool, _, action = op_id.partition(".")
    return tool, (action or None)


def is_read_name(name: str) -> bool:
    """Whether an action or tool-name token sequence reads without mutating."""
    tokens = [token for token in name.casefold().split("_") if token]
    return bool(tokens) and tokens[0] in READ_ACTION_PREFIXES


def schema_without_action(schema: Mapping[str, Any]) -> dict[str, Any]:
    """A backing tool's input schema minus its ``action`` selector."""
    properties = dict(schema.get("properties") or {})
    properties.pop("action", None)
    required = [name for name in schema.get("required") or () if name != "action"]
    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    if "$defs" in schema:
        out["$defs"] = schema["$defs"]
    return out


def shape_arguments(
    accepted: set[str] | frozenset[str],
    params: Mapping[str, Any],
    action: str | None,
    *,
    accepts_var_keyword: bool = False,
) -> dict[str, Any]:
    """Map a contract ``params`` object onto a backing tool's arguments.

    Arguments the backing tool declares pass through by name. A backing tool
    that takes a ``params_json`` envelope (the action-routed dispatchers)
    receives every remaining argument inside it, so callers never hand-encode
    JSON strings.
    """
    if "action" in params:
        raise ValueError(
            "pass the operation as the intent tool's 'action', not inside params"
        )
    kwargs = {key: value for key, value in params.items() if value is not None}
    if "params_json" in accepted and "params_json" not in kwargs:
        extra = {key: kwargs.pop(key) for key in list(kwargs) if key not in accepted}
        if extra or not accepts_var_keyword:
            kwargs["params_json"] = json.dumps(extra, default=str)
    if action is not None:
        kwargs["action"] = action
    return kwargs


def make_intent_tool(
    name: str,
    handler: Any,
    *,
    actions: tuple[str, ...] | None = None,
    execute_default: bool = True,
) -> Any:
    """Build an intent tool function with the condensed contract signature.

    ``handler(action, params, intent, execute)`` does the work. ``actions``
    publishes ``action`` as an enum (used when the operation set is small and
    fixed); the handler still validates every value it receives.
    """
    action_field: dict[str, Any] = {
        "default": "",
        "description": (
            "One of the listed actions; 'describe' explains each."
            if actions
            else ACTION_FIELD_DESCRIPTION
        ),
    }
    if actions:
        action_field["json_schema_extra"] = {"enum": ["", *actions]}

    async def _tool(
        action: str = Field(**action_field),
        params: dict[str, Any] = Field(
            default_factory=dict, description=PARAMS_FIELD_DESCRIPTION
        ),
        intent: str = Field(default="", description=INTENT_FIELD_DESCRIPTION),
        execute: bool = Field(
            default=execute_default,
            description="Run now; when false, return the resolved plan.",
        ),
    ) -> Any:
        return await handler(action, dict(params or {}), intent, execute)

    _tool.__name__ = name
    return _tool


__all__ = [
    "ACTION_FIELD_DESCRIPTION",
    "DEFAULT_SURFACE_TOKEN_BUDGET",
    "DEFAULT_SURFACE_TOOL_LIMIT",
    "DESCRIBE_ACTION",
    "INTENT_FIELD_DESCRIPTION",
    "INTENT_TOOL_TOKEN_BUDGET",
    "PARAMS_FIELD_DESCRIPTION",
    "READ_ACTION_PREFIXES",
    "estimate_tokens",
    "is_read_name",
    "make_intent_tool",
    "operation_id",
    "schema_without_action",
    "shape_arguments",
    "split_operation_id",
    "tool_definition",
]
