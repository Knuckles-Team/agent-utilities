#!/usr/bin/env python
"""Generate the graph-os verbose action manifest (CONCEPT:ECO-4.82).

graph-os is a thin action-routed MCP wrapper over the API gateway's action core
(``_execute_tool``). Unlike a connector it has no per-method client to introspect
and its tools use bespoke action dispatch, so there's no manifest for the shared
``register_verbose_tools`` to read. This script harvests one — statically — so the
verbose 1:1 surface (one tool per CRUD action, e.g. ``graph_write_add_node``) can
be generated like any other agent.

For each action-routed tool in ``ACTION_TOOL_ROUTES`` it collects the action string
literals from the tool's source: ``action == "x"``, ``action in {...}``,
``resolve_action(action, [...])`` — resolving module-level frozenset/list constants
referenced by name. A tool with no discoverable actions is a single operation
(``graph_query`` takes a query, not an action) and is emitted as one verbose op
with ``action=None``.

Output: ``agent_utilities/mcp/_graphos_action_manifest.py`` (``GRAPHOS_ACTIONS``).
Regenerate after changing the graph-os tool surface; ``ruff format`` afterwards.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

# Resolve ``agent_utilities`` from THIS repo (the script's own tree), not the
# editable-installed copy that ``sys.path[0]`` would otherwise prefer when the
# script is run from a worktree — otherwise the regenerated manifest reflects the
# wrong checkout and silently misses newly-added actions.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DISCOVERY = {"list_actions", "help", "actions"}


def _resolve_const(name: str, namespace: dict) -> set[str]:
    val = namespace.get(name)
    if isinstance(val, (frozenset, set, list, tuple)):
        return {v for v in val if isinstance(v, str)}
    return set()


def harvest_actions(func) -> set[str]:
    """Action string literals dispatched inside a tool function's source."""
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return set()
    import textwrap

    tree = ast.parse(textwrap.dedent(src))
    ns = getattr(func, "__globals__", {})
    actions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.comparators) == 1:
            left, right = node.left, node.comparators[0]
            op = node.ops[0]
            # action == "x"  /  "x" == action
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if (
                    isinstance(left, ast.Name)
                    and left.id == "action"
                    and isinstance(right, ast.Constant)
                    and isinstance(right.value, str)
                ):
                    actions.add(right.value)
                if (
                    isinstance(right, ast.Name)
                    and right.id == "action"
                    and isinstance(left, ast.Constant)
                    and isinstance(left.value, str)
                ):
                    actions.add(left.value)
            # action in {...}/[...]/(...)  or  action in CONST
            if (
                isinstance(op, ast.In)
                and isinstance(left, ast.Name)
                and left.id == "action"
            ):
                for c in node.comparators:
                    if isinstance(c, (ast.Tuple, ast.List, ast.Set)):
                        actions |= {
                            e.value
                            for e in c.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)
                        }
                    elif isinstance(c, ast.Name):
                        actions |= _resolve_const(c.id, ns)
        # resolve_action(action, <list|set|Name>, ...)
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "resolve_action"
            and len(node.args) >= 2
        ):
            arg = node.args[1]
            if isinstance(arg, (ast.Tuple, ast.List, ast.Set)):
                actions |= {
                    e.value
                    for e in arg.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
            elif isinstance(arg, ast.Name):
                actions |= _resolve_const(arg.id, ns)
    return actions - _DISCOVERY


def build_manifest() -> list[dict]:
    import sys

    sys.argv = ["graph-os"]
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()
    from agent_utilities.mcp.kg_server import ACTION_TOOL_ROUTES, REGISTERED_TOOLS

    # The low-level engine_<domain> tools (CONCEPT:ECO-4.99) are generic
    # client-introspection dispatchers — their actions are NOT string literals in
    # source (harvest_actions can't see them), so enumerate them from the engine
    # surface manifest instead (CONCEPT:KG-2.277).
    from agent_utilities.mcp.tools.engine_tools import ENGINE_DOMAINS

    engine_actions: dict[str, set[str]] = {
        f"engine_{domain}": set(methods) for domain, methods in ENGINE_DOMAINS.items()
    }

    ops: list[dict] = []
    for tool in sorted(ACTION_TOOL_ROUTES):
        func = REGISTERED_TOOLS.get(tool)
        if tool in engine_actions:
            actions = engine_actions[tool]
        else:
            actions = harvest_actions(func) if func else set()
        if actions:
            for action in sorted(actions):
                ops.append({"tool": tool, "action": action, "name": f"{tool}_{action}"})
        else:
            # Single-operation tool (no action switch) — itself is the verbose op.
            ops.append({"tool": tool, "action": None, "name": tool})
    return ops


def main() -> None:
    ops = build_manifest()
    out = (
        Path(__file__).resolve().parent.parent
        / "agent_utilities"
        / "mcp"
        / "_graphos_action_manifest.py"
    )
    header = (
        '"""Auto-generated by scripts/gen_graphos_manifest.py — do not edit by hand.\n\n'
        "The graph-os verbose 1:1 tool surface (CONCEPT:ECO-4.82): one entry per CRUD\n"
        "action over the API gateway action core. Each becomes an MCP tool ``name`` that\n"
        "dispatches ``_execute_tool(tool, action=action, **params)``. ``action=None`` is a\n"
        'single-operation tool. Regenerate after changing the graph-os tool surface."""\n'
    )
    # Emit a precise TypedDict so consumers index ``op["tool"]``/``["name"]`` as
    # ``str`` (not ``object``) — the loose inference over ~300 dict literals
    # otherwise widens the list to ``list[object]`` and breaks type-checking.
    typedef = (
        "from typing import TypedDict\n\n\n"
        "class GraphosAction(TypedDict):\n"
        "    tool: str\n"
        "    action: str | None\n"
        "    name: str\n"
    )
    body = "GRAPHOS_ACTIONS: list[GraphosAction] = " + repr(ops) + "\n"
    out.write_text(header + "\n" + typedef + "\n\n" + body)
    print(
        f"Wrote {len(ops)} verbose ops across "
        f"{len({o['tool'] for o in ops})} tools -> {out}"
    )


if __name__ == "__main__":
    main()
