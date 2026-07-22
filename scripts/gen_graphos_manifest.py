#!/usr/bin/env python
"""Generate the graph-os verbose action manifest (CONCEPT:AU-ECO.mcp.tool-mode-standardization).

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


def _literal_strings(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return set()
    return {
        item.value
        for item in node.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def _roots_at_action(node: ast.AST, aliases: set[str]) -> bool:
    """True if ``node`` is ``action``/a known alias, possibly wrapped in the
    common normalization idiom: ``(action or "default").strip().lower()``,
    ``str(action or "")``, attribute/method chains, or boolean-or fallbacks.
    Handles the ``action_key = ...`` / ``action_norm = ...`` pattern several
    action-routed tools use instead of comparing ``action`` directly — without
    this, ``harvest_actions`` silently sees zero actions for those tools and
    the generated manifest collapses them to a single ``action=None`` op.
    """
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ast.Name):
            if current.id == "action" or current.id in aliases:
                return True
        elif isinstance(current, ast.Attribute):
            stack.append(current.value)
        elif isinstance(current, ast.Call):
            stack.append(current.func)
            stack.extend(current.args)
        elif isinstance(current, ast.BoolOp):
            stack.extend(current.values)
        elif isinstance(current, (ast.IfExp,)):
            stack.append(current.body)
            stack.append(current.orelse)
    return False


def harvest_actions(func) -> set[str]:
    """Action string literals dispatched inside a tool function's source."""
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return set()
    import textwrap

    tree = ast.parse(textwrap.dedent(src))
    ns = getattr(func, "__globals__", {})
    local_constants: dict[str, set[str]] = {}
    action_aliases: set[str] = set()
    # Two passes: first collect local string-set constants (for `action in X`
    # where X is a module/local constant) AND local aliases of the `action`
    # parameter (`action_key = (action or "default").strip().lower()`), since
    # an alias assignment may appear anywhere in source order relative to its
    # use.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            values = _literal_strings(node.value)
            if values:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        local_constants[target.id] = values
            elif _roots_at_action(node.value, action_aliases):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        action_aliases.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            values = _literal_strings(node.value) if node.value is not None else set()
            if values:
                local_constants[node.target.id] = values
            elif node.value is not None and _roots_at_action(
                node.value, action_aliases
            ):
                action_aliases.add(node.target.id)

    def _is_action_name(n: ast.AST) -> bool:
        return isinstance(n, ast.Name) and (n.id == "action" or n.id in action_aliases)

    actions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.comparators) == 1:
            left, right = node.left, node.comparators[0]
            op = node.ops[0]
            # action == "x"  /  "x" == action  (or a normalized alias thereof)
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if (
                    _is_action_name(left)
                    and isinstance(right, ast.Constant)
                    and isinstance(right.value, str)
                ):
                    actions.add(right.value)
                if (
                    _is_action_name(right)
                    and isinstance(left, ast.Constant)
                    and isinstance(left.value, str)
                ):
                    actions.add(left.value)
            # action in/not in {...}/[...]/(...) or a local/module constant.
            if isinstance(op, (ast.In, ast.NotIn)) and _is_action_name(left):
                for c in node.comparators:
                    if isinstance(c, (ast.Tuple, ast.List, ast.Set)):
                        actions |= _literal_strings(c)
                    elif isinstance(c, ast.Name):
                        actions |= local_constants.get(c.id, set())
                        actions |= _resolve_const(c.id, ns)
        # resolve_action(action, <list|set|Name>, ...)
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "resolve_action"
            and len(node.args) >= 2
        ):
            arg = node.args[1]
            if isinstance(arg, (ast.Tuple, ast.List, ast.Set)):
                actions |= _literal_strings(arg)
            elif isinstance(arg, ast.Name):
                actions |= local_constants.get(arg.id, set())
                actions |= _resolve_const(arg.id, ns)
    return actions - _DISCOVERY


def build_manifest() -> list[dict]:
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp.optional_tool_features import OPTIONAL_TOOL_ACTIONS

    # The low-level engine_<domain> tools (CONCEPT:AU-ECO.mcp.full-api-mcp-surface) are generic
    # client-introspection dispatchers — their actions are NOT string literals in
    # source (harvest_actions can't see them), so enumerate them from the engine
    # surface manifest instead (CONCEPT:AU-KG.compute.engine-surface-manifest).
    from agent_utilities.mcp.tools.engine_tools import ENGINE_DOMAINS

    engine_actions: dict[str, set[str]] = {
        f"engine_{domain}": set(methods) for domain, methods in ENGINE_DOMAINS.items()
    }

    registered_before = dict(kg_server.REGISTERED_TOOLS)
    routes_before = dict(kg_server.ACTION_TOOL_ROUTES)
    try:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.ACTION_TOOL_ROUTES.clear()
        kg_server.ACTION_TOOL_ROUTES.update(kg_server.BASE_ACTION_TOOL_ROUTES)
        kg_server._build_server(
            bootstrap=False,
            tool_profile="intent",
            canonical_surface=True,
        )

        ops: list[dict] = []
        for tool in sorted(kg_server.ACTION_TOOL_ROUTES):
            func = kg_server.REGISTERED_TOOLS.get(tool)
            if tool in engine_actions:
                actions = engine_actions[tool]
            else:
                actions = harvest_actions(func) if func else set()
            if actions:
                for action in sorted(actions):
                    ops.append(
                        {
                            "tool": tool,
                            "action": action,
                            "name": f"{tool}_{action}",
                        }
                    )
            else:
                # Single-operation tool (no action switch) — itself is the verbose op.
                ops.append({"tool": tool, "action": None, "name": tool})
        actions_by_tool: dict[str, set[str]] = {}
        for entry in ops:
            if entry["action"] is not None:
                actions_by_tool.setdefault(entry["tool"], set()).add(entry["action"])
        for tool, optional_actions in OPTIONAL_TOOL_ACTIONS.items():
            actual = actions_by_tool.get(tool)
            expected = set(optional_actions)
            if actual is not None and actual != expected:
                raise RuntimeError(
                    f"optional tool {tool!r} action declaration drift: "
                    f"runtime={sorted(actual)}, declared={sorted(expected)}"
                )
            if actual is None:
                ops.extend(
                    {
                        "tool": tool,
                        "action": action,
                        "name": f"{tool}_{action}",
                    }
                    for action in optional_actions
                )
        return sorted(ops, key=lambda entry: entry["name"])
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(registered_before)
        kg_server.ACTION_TOOL_ROUTES.clear()
        kg_server.ACTION_TOOL_ROUTES.update(routes_before)


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
        "The graph-os verbose 1:1 tool surface (CONCEPT:AU-ECO.mcp.tool-mode-standardization): one entry per CRUD\n"
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
