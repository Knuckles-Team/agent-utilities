"""Every entrypoint resolves to the SAME execution plane / engine authority.

CONCEPT:AU-ECO.ui.one-engine-authority

``AGENTS.md`` → *Universal capability — ONE core, thin entrypoints* says every
user/system-facing surface (messaging, the A2A protocol layer every
``agents/*/agent_server.py`` shares, ``agent-webui``, ``agent-terminal-ui``,
``geniusbot``) is a thin transport over the ONE orchestrator, backed by the ONE
process-wide :class:`IntelligenceGraphEngine` authority
(``get_active()``/``get_or_create()``). D-WD-7 found this violated: a second,
hand-rolled construction path inside ``agent_webui.api_extensions.get_engine()``
could win the process-wide singleton race with a disconnected local backend,
silently becoming the authority the REST route (and every other route sharing
the process) then read from.

Two things are pinned here, per the Wire-First taxonomy:

* **Wiring** — the real messaging entrypoint (:meth:`MessagingService._resolve_engine`)
  and the real MCP/gateway entrypoint (:func:`kg_server._get_engine`) both reach
  the literal same seam, :meth:`IntelligenceGraphEngine.get_active`, using
  ``observe()`` (a real pass-through wrapper — the production classmethod still
  runs) rather than ``patch()``, so this proves the edge was actually reached,
  not that a mock stood in for it.
* **Contract** — across the five entrypoint trees on disk, the only
  ``IntelligenceGraphEngine`` attributes ever referenced are ``get_active`` and
  ``get_or_create`` — an exact-set surface (``assert_surface``), so a future
  entrypoint reaching for ``_ACTIVE_ENGINE`` directly, calling ``set_active()``
  from a request path, or reintroducing direct construction fails this test
  instead of shipping a second flavor of the graph.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from tests.wiring import assert_surface, observe, past_the_seam

# ``AGENTS.md`` → *Universal capability*: the entrypoint surface this contract
# governs. Paths are relative to the ``agent-packages/`` workspace root (this
# repo's parent directory) so the contract also reaches the sibling frontend
# repos, not just agent-utilities' own messaging/protocols trees.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent

ENTRYPOINT_TREES = {
    "messaging": _REPO_ROOT / "agent_utilities" / "messaging",
    "a2a-protocol": _REPO_ROOT / "agent_utilities" / "protocols",
    "agent-webui": _WORKSPACE_ROOT / "agent-webui" / "agent" / "agent_webui",
    "agent-terminal-ui": _WORKSPACE_ROOT / "agent-terminal-ui" / "agent_terminal_ui",
    "geniusbot": _WORKSPACE_ROOT / "geniusbot" / "geniusbot",
}

#: The ONLY sanctioned way for a thin entrypoint to reach the engine authority.
SANCTIONED_SEAMS = frozenset({"get_active", "get_or_create"})

_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}
_TEST_MARKERS = {"tests", "test", "__tests__"}


def _engine_attribute_accesses(tree: ast.Module) -> set[str]:
    """Every ``IntelligenceGraphEngine.<attr>`` access in one module's AST."""
    imported_as: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "IntelligenceGraphEngine":
                    imported_as.add(alias.asname or alias.name)

    accesses: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in imported_as
        ):
            accesses.add(node.attr)
    return accesses


def _entrypoint_engine_surface() -> set[str]:
    """Union of every ``IntelligenceGraphEngine.<attr>`` access across every
    entrypoint tree that exists on this checkout (non-test files only)."""
    surface: set[str] = set()
    for path in ENTRYPOINT_TREES.values():
        if not path.exists():
            continue
        for py in path.rglob("*.py"):
            rel_parts = py.relative_to(path).parts
            if any(part in _SKIP_DIRS for part in rel_parts):
                continue
            if any(
                part in _TEST_MARKERS or part.startswith("test_") for part in rel_parts
            ):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            surface |= _engine_attribute_accesses(tree)
    return surface


def test_entrypoint_engine_access_surface_is_exactly_get_active_and_get_or_create():
    """Contract: no entrypoint reaches the engine any way other than the two
    sanctioned classmethods -- not ``_ACTIVE_ENGINE`` directly, not
    ``set_active()`` from a request path, not a reintroduced direct-construction
    workaround.
    """
    scanned = [name for name, path in ENTRYPOINT_TREES.items() if path.exists()]
    assert scanned, "no entrypoint tree found on this checkout -- contract vacuous"
    assert_surface(
        _entrypoint_engine_surface(),
        SANCTIONED_SEAMS,
        surface="entrypoint IntelligenceGraphEngine access methods",
        invariant=SANCTIONED_SEAMS,
        parameterisation=f"trees={sorted(scanned)}",
    )


def test_messaging_entrypoint_reaches_the_same_engine_seam_as_the_mcp_entrypoint():
    """Wiring: the messaging entrypoint and the MCP/gateway entrypoint's engine
    accessors both call the literal same classmethod -- proof they can never
    disagree about which engine is "active", because there is only one seam
    for either of them to disagree through.
    """
    from agent_utilities.mcp import kg_server
    from agent_utilities.messaging.service import MessagingService

    MessagingService._instance = None  # isolate from any prior test's singleton
    try:
        with observe(IntelligenceGraphEngine, "get_active") as seen:
            messaging_service = MessagingService.instance(None)
            messaging_service._resolve_engine()
            with past_the_seam():
                kg_server._get_engine()
    finally:
        MessagingService._instance = None

    # Not an exact count: kg_server._get_engine() double-checks get_active()
    # under its own lock (calls it twice when no engine is active yet) -- an
    # internal locking detail, not part of this contract. What matters is that
    # BOTH entrypoints' calls land in the one observation, i.e. through the
    # literal same classmethod object.
    seen.assert_called(
        why=(
            "MessagingService._resolve_engine (the messaging entrypoint) and "
            "kg_server._get_engine (the MCP/gateway entrypoint) must both "
            "consult IntelligenceGraphEngine.get_active() -- not a second, "
            "independently-constructed engine reference."
        ),
    )
    assert seen.count >= 2, (
        f"expected at least one get_active() call from each of the two "
        f"entrypoints exercised above, saw {seen.count}"
    )


@pytest.mark.parametrize("name,path", sorted(ENTRYPOINT_TREES.items()))
def test_no_entrypoint_file_bypasses_get_or_create_with_direct_construction(
    name: str, path: Path
):
    """Belt-and-suspenders: the same invariant the standalone gate script
    (``scripts/check_entrypoint_engine_construction.py``) enforces in CI/pre-commit,
    reproduced here as a pytest-collected regression so it runs under the normal
    unit-test gate too, parameterised per entrypoint tree.
    """
    if not path.exists():
        pytest.skip(f"{name} tree not present on this checkout")

    from scripts.check_entrypoint_engine_construction import scan_tree

    violations = scan_tree(path)
    assert not violations, f"{name}: {violations}"
