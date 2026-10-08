"""The intent contract for agent-utilities-built MCP servers (fleet connectors).

CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse — the same single contract
graph-os serves (:mod:`agent_utilities.mcp.intent_contract`), applied by the
shared builder (:func:`agent_utilities.mcp.verbose_tools.register_tool_surface`)
to every connector:

``find``
    Discover operations: rank them against a natural-language ``intent``, list
    them (``action="describe"``), or return one operation's argument schema.
``ask``
    Run a read-only operation.
``act``
    Run any operation, including mutations (each backing tool keeps its own
    destructive-operation confirmation).

Operation ids are ``"<tool>.<action>"`` for an action-routed backing tool and
``"<tool>"`` for a single-operation tool — exactly the names the connector's
tools always had, so a call ``servicenow_cmdb(action="get_instance", …)`` is
``act(action="servicenow_cmdb.get_instance", params={…})``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_utilities.mcp.action_dispatch import is_destructive_action
from agent_utilities.mcp.intent_contract import (
    DESCRIBE_ACTION,
    is_read_name,
    make_intent_tool,
    operation_id,
    schema_without_action,
    shape_arguments,
)

#: The intent tools a connector serves, in listing order.
CONNECTOR_INTENT_TOOLS = ("find", "ask", "act")
#: ``find``'s fixed action set.
FIND_ACTIONS = ("search", DESCRIBE_ACTION)
_FIND_TOP_K = 8

_DESCRIPTIONS = {
    "find": "Discover this server's operations: search by intent, or describe "
    "one operation's arguments.",
    "ask": "Run a read-only operation (action='<tool>.<op>'); action='describe' "
    "lists them.",
    "act": "Run any operation, including changes (action='<tool>.<op>'); "
    "action='describe' lists them.",
}


@dataclass(frozen=True)
class Operation:
    """One addressable operation of a connector's backing tools."""

    id: str
    tool: str
    action: str | None
    description: str
    reads: bool


def _first_line(text: str | None) -> str:
    return (text or "").strip().splitlines()[0] if (text or "").strip() else ""


def _reads(tool: Any, action: str | None) -> bool:
    annotations = getattr(tool, "annotations", None)
    if action is None:
        hint = getattr(annotations, "readOnlyHint", None)
        if hint is not None:
            return bool(hint)
    name = action if action is not None else str(getattr(tool, "name", ""))
    if is_destructive_action(name):
        return False
    if action is not None:
        return is_read_name(action)
    tokens = name.casefold().split("_")
    return any(is_read_name(token) for token in tokens[1:])


def connector_operations(backing: Any) -> dict[str, Operation]:
    """``{operation id: Operation}`` over every backing tool, cached on ``backing``."""
    cached = getattr(backing, "_intent_operations", None)
    if cached is not None:
        return cached
    from agent_utilities.mcp.verbose_tools import (
        _ACTION_PROVIDERS_ATTR,
        _provider_tools,
        _tool_action_names,
    )

    providers: dict[str, Any] = getattr(backing, _ACTION_PROVIDERS_ATTR, {})
    table: dict[str, Operation] = {}
    for name, tool in sorted(_provider_tools(backing).items()):
        description = _first_line(getattr(tool, "description", None))
        actions: list[str | None] = list(_tool_action_names(tool, providers))
        for action in actions or [None]:
            op = Operation(
                id=operation_id(name, action),
                tool=name,
                action=action,
                description=description,
                reads=_reads(tool, action),
            )
            table[op.id] = op
    backing._intent_operations = table
    return table


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(c if c.isalnum() else " " for c in text.casefold()).split() if len(t) > 2}


def rank_operations(
    operations: Mapping[str, Operation], intent: str, *, top_k: int = _FIND_TOP_K
) -> list[dict[str, Any]]:
    """Rank operations by token overlap between ``intent`` and their id/description."""
    wanted = _tokens(intent)
    scored = []
    for op in operations.values():
        overlap = len(wanted & _tokens(f"{op.id} {op.description}"))
        if overlap:
            scored.append((overlap, op.id, op))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "action": op.id,
            "description": op.description,
            "tool": "ask" if op.reads else "act",
            "score": score,
        }
        for score, _, op in scored[:top_k]
    ]


def describe_operations(
    backing: Any, operations: Mapping[str, Operation], target: str | None
) -> dict[str, Any]:
    """All operation ids (grouped by tool), or one operation's argument schema."""
    from agent_utilities.mcp.verbose_tools import _provider_tools

    if not target:
        grouped: dict[str, list[str]] = {}
        for op in operations.values():
            grouped.setdefault(op.tool, []).append(op.id)
        return {"operations": grouped}
    op = operations.get(target)
    if op is None:
        return {"error": f"Unknown operation {target!r}; describe lists them."}
    tool = _provider_tools(backing).get(op.tool)
    return {
        "action": op.id,
        "reads": op.reads,
        "description": getattr(tool, "description", None) or "",
        "params_schema": schema_without_action(getattr(tool, "parameters", None) or {}),
    }


async def run_operation(
    backing: Any, op: Operation, params: Mapping[str, Any], *, execute: bool
) -> Any:
    """Shape ``params`` onto the backing tool and call it through FastMCP.

    Calling through the backing server keeps every backing tool's ``Depends``
    client binding, ``Context`` injection, and result coercion.
    """
    from agent_utilities.mcp.verbose_tools import _provider_tools

    tool = _provider_tools(backing)[op.tool]
    schema = getattr(tool, "parameters", None) or {}
    accepted = frozenset((schema.get("properties") or {}).keys())
    arguments = shape_arguments(
        accepted,
        params,
        op.action,
        accepts_var_keyword=bool(schema.get("additionalProperties")),
    )
    if not execute:
        return {"executed": False, "action": op.id, "arguments": sorted(arguments)}
    return await backing.call_tool(op.tool, arguments)


def _intent_handler(verb: str, backing: Any):
    async def _handle(
        action: str, params: dict[str, Any], intent: str, execute: bool
    ) -> Any:
        operations = connector_operations(backing)
        if verb == "find" or action == DESCRIBE_ACTION:
            if action == DESCRIBE_ACTION or not intent:
                return describe_operations(backing, operations, params.get("action"))
            return {"results": rank_operations(operations, intent)}
        if not action:
            return {
                "executed": False,
                "results": rank_operations(operations, intent),
                "hint": f"Resubmit {verb} with one of these operation ids as action.",
            }
        op = operations.get(action)
        if op is None:
            return {"error": f"Unknown operation {action!r}; call find to list them."}
        if verb == "ask" and not op.reads:
            return {"error": f"{op.id!r} changes state; run it with act."}
        return await run_operation(backing, op, params, execute=execute)

    return _handle


def register_connector_intent_tools(mcp: Any, backing: Any, *, service: str) -> list[str]:
    """Serve the intent tools over ``backing``'s operations on ``mcp``."""
    mcp._intent_backing = backing
    for verb in CONNECTOR_INTENT_TOOLS:
        fn = make_intent_tool(
            verb,
            _intent_handler(verb, backing),
            actions=FIND_ACTIONS if verb == "find" else None,
        )
        mcp.tool(
            name=verb,
            description=f"{service}: {_DESCRIPTIONS[verb]}",
            tags={"intent"},
        )(fn)
    return list(CONNECTOR_INTENT_TOOLS)


__all__ = [
    "CONNECTOR_INTENT_TOOLS",
    "FIND_ACTIONS",
    "Operation",
    "connector_operations",
    "describe_operations",
    "rank_operations",
    "register_connector_intent_tools",
    "run_operation",
]
