#!/usr/bin/env python
"""Generate the graph-os verbose action manifest (CONCEPT:AU-ECO.mcp.tool-mode-standardization).

graph-os is a thin action-routed MCP wrapper over the API gateway's action core
(``_execute_tool``). Unlike a connector it has no per-method client to introspect
and its tools use bespoke action dispatch, so there's no manifest for the shared
``register_verbose_tools`` to read. This script harvests one — statically — so the
verbose 1:1 surface (one tool per CRUD action, e.g. ``graph_write_add_node``) can
be generated like any other agent.

For each action-routed tool in ``ACTION_TOOL_ROUTES`` it collects a static
``Literal[...]`` action annotation plus action string literals from the tool's
source: comparisons, closed callable dispatch maps, ``resolve_action`` calls, and
focused helpers that receive the public action value. Module-level constants and
closure-owned dispatch maps are resolved from the registered callable itself;
descriptions are never action authority. A tool with no discoverable actions is a
single operation (``graph_query`` takes a query, not an action) and is emitted as
one verbose op with ``action=None``.

Output: ``agent_utilities/mcp/_graphos_action_manifest.py`` (``GRAPHOS_ACTIONS``).
Regenerate after changing the graph-os tool surface. The emitted file is
formatted in place with ``ruff format``, so a no-change regeneration is a
byte-level no-op rather than a whole-file reformat diff (D-KCI-6).
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

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
    if isinstance(val, (frozenset, set, list, tuple, dict)):
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


def _literal_dispatch_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    return {
        key.value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and not isinstance(value, ast.Constant)
    }


def _dispatch_keys(value: object) -> set[str]:
    if not isinstance(value, dict) or not all(
        callable(item) for item in value.values()
    ):
        return set()
    return {key for key in value if isinstance(key, str)}


def _action_root_step(node: ast.AST, aliases: set[str]) -> tuple[bool, list[ast.AST]]:
    if isinstance(node, ast.Name):
        return node.id == "action" or node.id in aliases, []
    if isinstance(node, ast.Attribute):
        direct = (
            node.attr == "action"
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        )
        return direct, [node.value]
    if isinstance(node, ast.Call):
        return False, [node.func, *node.args]
    if isinstance(node, ast.BoolOp):
        return False, list(node.values)
    if isinstance(node, ast.IfExp):
        return False, [node.body, node.orelse]
    return False, []


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
        matched, children = _action_root_step(current, aliases)
        if matched:
            return True
        stack.extend(children)
    return False


def _callable_namespace(func: object) -> dict[str, object]:
    namespace = dict(getattr(func, "__globals__", {}))
    closure = getattr(func, "__closure__", None)
    code = getattr(func, "__code__", None)
    if closure and code:
        for name, cell in zip(code.co_freevars, closure, strict=True):
            try:
                namespace[name] = cell.cell_contents
            except ValueError:
                continue
    return namespace


def _resolve_callable(node: ast.AST, namespace: dict[str, object]) -> object | None:
    if isinstance(node, ast.Name):
        value = namespace.get(node.id)
        return value if callable(value) else None
    return None


def _source_tree(func: object) -> ast.Module | None:
    try:
        import textwrap

        return ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        return None


def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> list[ast.AST]:
    return node.targets if isinstance(node, ast.Assign) else [node.target]


def _local_dispatch_mappings(tree: ast.Module) -> dict[str, set[str]]:
    local_mappings: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        keys = _literal_dispatch_keys(node.value) if node.value is not None else set()
        if not keys:
            continue
        local_mappings.update(
            (target.id, keys)
            for target in _assignment_targets(node)
            if isinstance(target, ast.Name)
        )
    return local_mappings


def _returned_mapping_keys(func: object) -> set[str]:
    """Return literal keys from a helper that owns an action dispatch mapping."""

    tree = _source_tree(func)
    if tree is None:
        return set()
    local_mappings = _local_dispatch_mappings(tree)
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        result |= _literal_dispatch_keys(node.value)
        if isinstance(node.value, ast.Name):
            result |= local_mappings.get(node.value.id, set())
    return result


def _parameter_names(func: object) -> list[str]:
    try:
        return list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return []


def _root_parameters_for_call(
    node: ast.Call,
    callee: object,
    roots: set[str],
    *,
    positional_offset: int = 0,
) -> set[str]:
    parameters = _parameter_names(callee)
    rooted: set[str] = set()
    for index, argument in enumerate(node.args[positional_offset:]):
        parameter_index = index
        if parameter_index < len(parameters) and _roots_at_action(argument, roots):
            rooted.add(parameters[parameter_index])
    for keyword in node.keywords:
        if keyword.arg in parameters and _roots_at_action(keyword.value, roots):
            rooted.add(keyword.arg)
    return rooted


def _hint_actions(func: object, roots: set[str]) -> tuple[set[str], bool]:
    try:
        hints = get_type_hints(func)
    except (NameError, TypeError):
        hints = {}
    actions: set[str] = set()
    for parameter in roots:
        action_type = hints.get(parameter)
        if get_origin(action_type) is Literal:
            actions.update(
                value for value in get_args(action_type) if isinstance(value, str)
            )
    return actions, bool(actions)


def _named_targets(node: ast.Assign | ast.AnnAssign) -> set[str]:
    return {
        target.id
        for target in _assignment_targets(node)
        if isinstance(target, ast.Name)
    }


def _record_call_assignment_facts(
    value: ast.Call,
    targets: set[str],
    namespace: dict[str, object],
    aliases: set[str],
    factories: dict[str, object],
    carriers: set[str],
) -> None:
    factory = _resolve_callable(value.func, namespace)
    if factory is not None:
        factories.update((target, factory) for target in targets)
    carries_action = any(
        keyword.arg == "action" and _roots_at_action(keyword.value, aliases)
        for keyword in value.keywords
    )
    if carries_action:
        carriers.update(targets)


def _record_assignment_facts(
    node: ast.Assign | ast.AnnAssign,
    namespace: dict[str, object],
    constants: dict[str, set[str]],
    factories: dict[str, object],
    aliases: set[str],
    carriers: set[str],
) -> None:
    value = node.value
    if value is None:
        return
    targets = _named_targets(node)
    values = _literal_strings(value) | _literal_dispatch_keys(value)
    if values:
        constants.update((target, values) for target in targets)
        return
    if _roots_at_action(value, aliases):
        aliases.update(targets)
        return
    if not isinstance(value, ast.Call):
        return
    _record_call_assignment_facts(
        value, targets, namespace, aliases, factories, carriers
    )


def _assignment_facts(
    tree: ast.Module,
    namespace: dict[str, object],
    roots: set[str],
) -> tuple[dict[str, set[str]], dict[str, object], set[str], set[str]]:
    constants: dict[str, set[str]] = {}
    factories: dict[str, object] = {}
    aliases = set(roots)
    carriers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            _record_assignment_facts(
                node, namespace, constants, factories, aliases, carriers
            )
    return constants, factories, aliases, carriers


def _is_action_reference(node: ast.AST, aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "action"
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    )


def _equality_actions(node: ast.Compare, aliases: set[str]) -> set[str]:
    if len(node.comparators) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
        return set()
    left, right = node.left, node.comparators[0]
    result: set[str] = set()
    if _is_action_reference(left, aliases) and isinstance(right, ast.Constant):
        if isinstance(right.value, str):
            result.add(right.value)
    if _is_action_reference(right, aliases) and isinstance(left, ast.Constant):
        if isinstance(left.value, str):
            result.add(left.value)
    return result


def _membership_actions(
    node: ast.Compare,
    aliases: set[str],
    constants: dict[str, set[str]],
    namespace: dict[str, object],
) -> tuple[set[str], bool]:
    if len(node.comparators) != 1 or not isinstance(node.ops[0], (ast.In, ast.NotIn)):
        return set(), False
    if not _is_action_reference(node.left, aliases):
        return set(), False
    candidate = node.comparators[0]
    if isinstance(candidate, (ast.Tuple, ast.List, ast.Set)):
        actions = _literal_strings(candidate)
    elif isinstance(candidate, ast.Name):
        actions = constants.get(candidate.id, set()) | _resolve_const(
            candidate.id, namespace
        )
    else:
        actions = set()
    return actions, isinstance(node.ops[0], ast.NotIn)


def _mapping_actions(
    node: ast.AST,
    aliases: set[str],
    constants: dict[str, set[str]],
    factories: dict[str, object],
    namespace: dict[str, object],
) -> tuple[set[str], bool]:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) == 1
        and _roots_at_action(node.args[0], aliases)
    ):
        return set(), False
    owner = node.func.value
    if not isinstance(owner, ast.Name):
        return _literal_dispatch_keys(owner), True
    actions = constants.get(owner.id, set()) | _dispatch_keys(namespace.get(owner.id))
    factory = factories.get(owner.id)
    if factory is not None:
        actions |= _returned_mapping_keys(factory)
    return actions, True


def _resolver_actions(
    node: ast.AST,
    constants: dict[str, set[str]],
    namespace: dict[str, object],
) -> tuple[set[str], bool]:
    if not (
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "resolve_action"
        and len(node.args) >= 2
    ):
        return set(), False
    candidate = node.args[1]
    if isinstance(candidate, (ast.Tuple, ast.List, ast.Set)):
        return _literal_strings(candidate), True
    if isinstance(candidate, ast.Name):
        return (
            constants.get(candidate.id, set())
            | _resolve_const(candidate.id, namespace),
            True,
        )
    return set(), True


def _return_call_ids(tree: ast.Module) -> set[int]:
    return {
        id(call)
        for result in ast.walk(tree)
        if isinstance(result, ast.Return) and result.value is not None
        for call in ast.walk(result.value)
        if isinstance(call, ast.Call)
    }


def _loop_helpers(loop: ast.For, namespace: dict[str, object]) -> list[object]:
    if not isinstance(loop.iter, (ast.Tuple, ast.List)):
        return []
    return [
        helper
        for item in loop.iter.elts
        if (helper := _resolve_callable(item, namespace)) is not None
    ]


def _loop_action_calls(loop: ast.For) -> list[ast.Call]:
    if not isinstance(loop.target, ast.Name):
        return []
    return [
        node
        for statement in loop.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == loop.target.id
    ]


def _harvest_helper_calls(
    helpers: list[object],
    calls: list[ast.Call],
    aliases: set[str],
    seen: set[tuple[int, tuple[str, ...]]],
) -> set[str]:
    actions: set[str] = set()
    for helper in helpers:
        for call in calls:
            roots = _root_parameters_for_call(call, helper, aliases)
            if roots:
                actions |= _harvest_actions(helper, roots, seen)
    return actions


def _iterated_helper_actions(
    tree: ast.Module,
    namespace: dict[str, object],
    aliases: set[str],
    seen: set[tuple[int, tuple[str, ...]]],
) -> set[str]:
    actions: set[str] = set()
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        actions |= _harvest_helper_calls(
            _loop_helpers(loop, namespace), _loop_action_calls(loop), aliases, seen
        )
    return actions


def _follows_action_helper(
    call: ast.Call, callee: object, return_calls: set[int]
) -> bool:
    name = getattr(callee, "__name__", "")
    return (
        id(call) in return_calls
        or name.startswith("_resolve_")
        or name.endswith("_action_or_error")
    )


def _recursive_call_actions(
    node: ast.AST,
    func: object,
    namespace: dict[str, object],
    aliases: set[str],
    carriers: set[str],
    return_calls: set[int],
    seen: set[tuple[int, tuple[str, ...]]],
) -> set[str]:
    if not isinstance(node, ast.Call):
        return set()
    actions: set[str] = set()
    callee = _resolve_callable(node.func, namespace)
    if (
        callee is not None
        and callee is not func
        and _follows_action_helper(node, callee, return_calls)
    ):
        roots = _root_parameters_for_call(node, callee, aliases)
        if roots:
            actions |= _harvest_actions(callee, roots, seen)
    if not node.args:
        return actions
    delegated = _resolve_callable(node.args[0], namespace)
    if delegated is None:
        return actions
    roots = _root_parameters_for_call(
        node, delegated, aliases | carriers, positional_offset=1
    )
    if roots:
        actions |= _harvest_actions(delegated, roots, seen)
    return actions


def _harvest_actions(
    func: object,
    root_parameters: set[str],
    seen: set[tuple[int, tuple[str, ...]]],
) -> set[str]:
    """Harvest action authorities reachable from one callable.

    Current GraphOS tools delegate through exact dispatch mappings and focused
    helper functions.  Follow only data flow rooted at the public action
    parameter; prose descriptions and unrelated string dictionaries are never
    treated as authority.
    """

    identity = (id(func), tuple(sorted(root_parameters)))
    if identity in seen:
        return set()
    seen.add(identity)
    parameters = set(_parameter_names(func))
    if not root_parameters & parameters:
        return set()
    actions, has_closed_authority = _hint_actions(func, root_parameters)
    tree = _source_tree(func)
    if tree is None:
        return actions
    ns = _callable_namespace(func)
    local_constants, mapping_factories, action_aliases, action_carriers = (
        _assignment_facts(tree, ns, root_parameters)
    )
    return_calls = _return_call_ids(tree)
    actions |= _iterated_helper_actions(tree, ns, action_aliases, seen)

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            actions |= _equality_actions(node, action_aliases)
            found, closed = _membership_actions(
                node, action_aliases, local_constants, ns
            )
            actions |= found
            has_closed_authority |= closed
        found, closed = _mapping_actions(
            node, action_aliases, local_constants, mapping_factories, ns
        )
        actions |= found
        has_closed_authority |= closed
        found, closed = _resolver_actions(node, local_constants, ns)
        actions |= found
        has_closed_authority |= closed
        if not has_closed_authority:
            actions |= _recursive_call_actions(
                node,
                func,
                ns,
                action_aliases,
                action_carriers,
                return_calls,
                seen,
            )
    return actions - _DISCOVERY


def harvest_actions(func) -> set[str]:
    """Action literals reachable from one tool's canonical action parameter."""

    if func is None:
        return set()
    return _harvest_actions(func, {"action"}, set())


def _validate_operations(ops: list[dict]) -> None:
    """Reject duplicate identities or verbose names before generation."""

    identities: set[tuple[str, str | None]] = set()
    names: dict[str, tuple[str, str | None]] = {}
    for entry in ops:
        identity = (entry["tool"], entry["action"])
        if identity in identities:
            raise RuntimeError(f"duplicate GraphOS operation identity: {identity!r}")
        identities.add(identity)
        previous = names.setdefault(entry["name"], identity)
        if previous != identity:
            raise RuntimeError(
                f"conflicting GraphOS verbose name {entry['name']!r}: "
                f"{previous!r} vs {identity!r}"
            )


def build_manifest() -> list[dict]:
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp.optional_tool_features import OPTIONAL_TOOL_ACTIONS

    # The low-level engine_<domain> tools (CONCEPT:AU-ECO.mcp.full-api-mcp-surface) are generic
    # client-introspection dispatchers — their actions are NOT string literals in
    # source (harvest_actions can't see them), so enumerate them from the engine
    # surface manifest instead (CONCEPT:AU-KG.compute.engine-surface-manifest).
    from agent_utilities.mcp.tools.engine_tools import ENGINE_DOMAINS

    if not ENGINE_DOMAINS:
        # ENGINE_DOMAINS is populated by introspecting the installed
        # `epistemic_graph` client (see engine_tools._discover_domains); an
        # empty dict here means that import failed and EVERY engine_<domain>
        # tool — and every engine_<domain>_<method> verbose op derived from it
        # — will be silently absent from this run's manifest. This is the
        # dominant known cause of a badly shrunken regen (a box missing the
        # epistemic-graph wheel/numeric-kernel extra); `main()` treats a net
        # entry-count regression as fatal, but this loud print fires even when
        # some other addition happens to offset the loss in the final count.
        raise RuntimeError(
            "ENGINE_DOMAINS is empty: the epistemic-graph client surface is "
            "unavailable, so authoritative manifest generation cannot proceed"
        )

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
        _validate_operations(ops)
        return sorted(ops, key=lambda entry: entry["name"])
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(registered_before)
        kg_server.ACTION_TOOL_ROUTES.clear()
        kg_server.ACTION_TOOL_ROUTES.update(routes_before)


def _format_in_place(path: Path) -> None:
    """Run the repo formatter over the emitted file.

    The generator writes one dict per line; `ruff format` then applies the repo's
    own line-length wrapping (`[tool.ruff] line-length = 88`), which explodes the
    longer entries across several lines. Deferring to the formatter — rather than
    reimplementing its wrapping — is what makes a no-change regeneration a
    byte-level no-op against the checked-in, formatter-owned file (D-KCI-6).

    A missing formatter is loud and fatal: emitting an unformatted manifest would
    silently reintroduce the very diff-noise this exists to remove, and the next
    `ruff-format` hook would rewrite the file anyway.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, repo-local tool
            ["ruff", "format", "--quiet", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"cannot format {path}: 'ruff' is not runnable ({exc}). The manifest "
            "must be emitted in the repo's formatted style or every regeneration "
            "shows as a spurious whole-file diff."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"'ruff format' failed on {path} (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def main() -> None:
    out = (
        Path(__file__).resolve().parent.parent
        / "agent_utilities"
        / "mcp"
        / "_graphos_action_manifest.py"
    )
    ops = build_manifest()

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

    # Emit ONE dict per line, matching the formatter's own output, so a no-change
    # regeneration is a byte-level no-op. `repr(ops)` writes the whole list on a
    # single ~80KB line; the checked-in file is formatter-pretty-printed across
    # 2400+ lines. That mismatch meant every faithful regen showed as a ~-2367
    # line diff, so the file's own "regenerate after changing the tool surface"
    # instruction looked destructive and was rationally never followed — the
    # manifest then drifted from its generator and had to be hand-edited
    # (D-KCI-6). Formatting is the contract here, not decoration.
    def _lit(value: str | None) -> str:
        # json.dumps gives the double-quoted, correctly-escaped form the formatter
        # emits. A blanket `repr(...).replace("'", '"')` would corrupt any value
        # containing an apostrophe, so quote per-value instead of per-line.
        return "None" if value is None else json.dumps(value)

    rows = "".join(
        f'    {{"tool": {_lit(op["tool"])}, "action": {_lit(op["action"])}, '
        f'"name": {_lit(op["name"])}}},\n'
        for op in ops
    )
    body = f"GRAPHOS_ACTIONS: list[GraphosAction] = [\n{rows}]\n"
    out.write_text(header + "\n" + typedef + "\n\n" + body)
    _format_in_place(out)
    print(
        f"Wrote {len(ops)} verbose ops across "
        f"{len({o['tool'] for o in ops})} tools -> {out}"
    )


if __name__ == "__main__":
    main()
