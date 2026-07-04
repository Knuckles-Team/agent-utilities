"""CONCEPT:AU-ECO.toolkit.tool-ref-integrity-gate — Capability-bound tool/skill resolution (Layers 2 & 3).

The problem (see ``scripts/check_tool_refs.py``, Layer 1): agents referenced tools/skills by
mutable display name, so renames and mcp-multiplexer prefixes silently broke them. The fix is to
separate the *reference* from the *binding*:

* **Layer 3 — capability binding.** An agent declares bounded INTENT tags (``code-intelligence``,
  ``test-execution``, ...), never tool names. :func:`resolve_capabilities` maps each tag to the
  concrete tool functions at construction time — deterministically via :data:`CAPABILITY_TOOLS`,
  and (optionally) via the KG capability index for the long tail. Renames change the function,
  not the contract.
* **Layer 2 — prefix/alias boundary.** The mcp-multiplexer prefixes tools (``go__graph_query``).
  :func:`build_alias_map` / :func:`resolve_mcp_name` turn a canonical name into the deployment's
  actual prefixed name by reading the live catalog — so a reference is prefix-proof.

This is the durable answer to the name-drift question: don't hardcode names; declare intents;
resolve through the registry/KG; let pure-intent escalation (ECO-4.36 ``find_tools``) cover the
rest at runtime.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_PREFIX = re.compile(r"^[a-z0-9]+__")


# --------------------------------------------------------------------------- #
# Layer 3 — capability -> tool binding
# --------------------------------------------------------------------------- #

# Intent tag -> the names of tool functions that provide it. Names index the live tool
# universe (resolved to function objects lazily), so a rename is a one-line edit here, never a
# fleet-wide break. Extend this as new capabilities/tools land.
CAPABILITY_TOOLS: dict[str, list[str]] = {
    "code-intelligence": [
        "find_definition",
        "who_calls",
        "impacted_tests",
        "call_graph",
        "dependencies",
    ],
    "code-navigation": ["find_definition", "who_calls", "call_graph", "dependencies"],
    "file-editing": ["read_file", "write_file", "edit_file"],
    "shell-execution": ["run_command"],
    "test-execution": ["run_tests"],
    "tdd": ["run_tests", "read_file", "edit_file"],
    "debugging": ["run_command", "run_tests", "read_file"],
    "web-browsing": ["browse"],
    "software-engineering": [
        "find_definition",
        "who_calls",
        "impacted_tests",
        "call_graph",
        "dependencies",
        "run_command",
        "read_file",
        "write_file",
        "edit_file",
        "run_tests",
    ],
}


def known_capabilities() -> set[str]:
    """The capability intent tags the static registry can resolve (used by the CI gate)."""
    return set(CAPABILITY_TOOLS)


def _tool_universe() -> dict[str, Callable[..., Any]]:
    """Map tool-function name -> function object for every resolvable tool (lazy, cycle-safe)."""
    from agent_utilities.tools.code_intelligence_tools import CODE_INTELLIGENCE_TOOLS
    from agent_utilities.tools.swe_workspace_tools import SWE_WORKSPACE_TOOLS

    universe: dict[str, Callable[..., Any]] = {}
    for fn in (*CODE_INTELLIGENCE_TOOLS, *SWE_WORKSPACE_TOOLS):
        universe[getattr(fn, "__name__", "")] = fn
    return universe


def resolve_capabilities(
    capabilities: list[str],
    *,
    kg: Any = None,
    embed: Callable[[str], Any] | None = None,
) -> list[Callable[..., Any]]:
    """Resolve intent tags to concrete tool functions (deduped, order-stable).

    Static :data:`CAPABILITY_TOOLS` is the primary path. Unknown tags fall back to the KG
    capability index (``kg.retrieval.designate`` over an embedding of the tag) when ``kg`` and
    ``embed`` are supplied — the dynamic, rename-proof long tail. Truly unresolved tags are
    logged, never raised (a missing capability must not crash agent construction).
    """
    universe = _tool_universe()
    seen: set[str] = set()
    resolved: list[Callable[..., Any]] = []

    def _add(name: str) -> None:
        fn = universe.get(name)
        if fn is not None and name not in seen:
            seen.add(name)
            resolved.append(fn)

    for tag in capabilities:
        names = CAPABILITY_TOOLS.get(tag)
        if names:
            for n in names:
                _add(n)
            continue
        # Long-tail: ask the KG which tools provide this capability.
        for n in _kg_resolve(tag, kg=kg, embed=embed):
            _add(n)
        if tag not in CAPABILITY_TOOLS and not _kg_resolve(tag, kg=kg, embed=embed):
            logger.debug("capability %r did not resolve to any known tool", tag)
    return resolved


def _kg_resolve(tag: str, *, kg: Any, embed: Callable[[str], Any] | None) -> list[str]:
    """Best-effort KG semantic resolution of a capability tag -> tool names."""
    if kg is None or embed is None:
        return []
    try:
        retrieval = getattr(kg, "retrieval", None)
        if retrieval is None:
            return []
        designations = retrieval.designate(embed(tag), k=5)
        out: list[str] = []
        for d in designations or []:
            name = getattr(d, "entity_id", None) or getattr(d, "name", None)
            if name:
                out.append(str(name))
        return out
    except Exception as exc:  # noqa: BLE001 - semantic path is optional
        logger.debug("KG capability resolve failed for %r: %s", tag, exc)
        return []


def register_capability_tools(
    agent: Any, capabilities: list[str], *, kg: Any = None
) -> int:
    """Resolve ``capabilities`` to tools and register them on ``agent``. Returns the count."""
    tools = resolve_capabilities(capabilities, kg=kg)
    count = 0
    for fn in tools:
        try:
            agent.tool(fn)
            count += 1
        except Exception as exc:  # noqa: BLE001 - duplicate registration tolerated
            logger.debug(
                "capability tool %s not registered: %s",
                getattr(fn, "__name__", fn),
                exc,
            )
    return count


# --------------------------------------------------------------------------- #
# Layer 2 — mcp-multiplexer prefix/alias boundary
# --------------------------------------------------------------------------- #


def strip_mcp_prefix(name: str) -> str:
    """Strip a single ``<server>__`` mcp-multiplexer prefix, if present."""
    return _PREFIX.sub("", name)


def build_alias_map(catalog: dict[str, list[str]]) -> dict[str, str]:
    """Build a canonical-name -> prefixed-runtime-name map from a multiplexer catalog.

    ``catalog`` maps server prefix -> list of base tool names (the shape ``list_catalog`` /
    ``multiplexer_status`` expose). The canonical (base) name resolves to ``<prefix>__<base>``.
    Last server wins on a base-name collision (deterministic, logged by the caller if needed).
    """
    alias: dict[str, str] = {}
    for prefix, tools in (catalog or {}).items():
        for base in tools or []:
            alias[base] = f"{prefix}__{base}"
    return alias


def resolve_mcp_name(canonical: str, alias_map: dict[str, str]) -> str:
    """Resolve a canonical tool name to its deployment's actual (prefixed) runtime name."""
    if canonical in alias_map:
        return alias_map[canonical]
    base = strip_mcp_prefix(canonical)
    return alias_map.get(base, canonical)
