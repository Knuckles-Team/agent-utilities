"""CONTRACT — the graph-os **dispatch** registry, pinned per ``MCP_TOOL_MODE``.

CONCEPT:AU-AHE.evaluation.surface-contract-test

Taxonomy (``AGENTS.md`` → *Wire-First*): a **contract/surface** test, plus one
**wiring** assertion that the registry is populated from the live entrypoint.

Registered is not served, and served is not registered
------------------------------------------------------
graph-os has two different surfaces and they are pinned by two different tests:

* **served** — what a client sees over ``tools/list``. Mode-dependent *by design*
  (``intent`` shows 6 verbs + 5 meta-tools; the rest stay hidden but loadable).
  Pinned by ``tests/unit/mcp/test_served_tool_surface.py``, which also encodes the
  operator invariant that the five fleet meta-tools appear in EVERY mode. This
  file does not duplicate it.
* **dispatch** — ``kg_server.REGISTERED_TOOLS``, the table ``_execute_tool`` looks
  every call up in. It backs the MCP tools, the intent verbs, *and* the whole REST
  route table (``_mount_rest_routes`` → "Handlers dispatch through
  ``REGISTERED_TOOLS``"). It is supposed to be mode-**in**dependent: the mode
  chooses what is *visible*, never what is *callable*. That is what this file
  pins, and nothing pinned it before.

Conflating the two is how the 118-vs-11 regression stayed invisible, and it is why
"the tool list looks right" is not evidence the tool works.

Cost note: ``_build_server`` registers ~750 tools, so each mode is built exactly
once per module and the global ``REGISTERED_TOOLS`` is snapshotted and restored —
a test that leaves that dict mutated silently changes every later test's world.
"""

from __future__ import annotations

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.verbose_tools import VALID_TOOL_MODES
from tests.wiring import assert_surface, observe

#: Modes in which graph-os serves its own action core. ``verbose`` JOINED this
#: list when D-WS-1 was fixed: ``register_tool_surface`` now runs the condensed
#: registrars in ``verbose`` too and merely gates their *visibility*
#: (``gate_condensed_visibility = mode == "intent" or (mode == "verbose" and
#: has_verbose)``), so the dispatch core is populated and the REST route table is
#: live. The defect-pin test that asserted the old empty-registry bug was deleted
#: per its own instruction ("Move 'verbose' into DISPATCHING_MODES and delete
#: this test") when reconciliation gate 2 merged the fix.
DISPATCHING_MODES = ("intent", "condensed", "both", "verbose")

#: The collapsed intent surface — additive on dispatch, subtractive on view.
INTENT_VERBS = frozenset({"ask", "find", "act", "why", "write", "manage"})

#: Representative tools the REST table and the intent verbs both dispatch into.
#: Not the full 112 — an exhaustive list would be a snapshot nobody maintains;
#: these are the ones whose absence breaks a documented public surface.
CORE_DISPATCH_TOOLS = frozenset(
    {
        "graph_query",
        "graph_search",
        "graph_ingest",
        "graph_orchestrate",
        "graph_write",
        "graph_workflows",
    }
)


def _build(mode: str) -> set[str]:
    """Build graph-os for ``mode`` and return the dispatch registry it produced."""
    saved = dict(kg_server.REGISTERED_TOOLS)
    kg_server.REGISTERED_TOOLS.clear()
    try:
        kg_server._build_server(bootstrap=False, tool_profile=mode)
        return set(kg_server.REGISTERED_TOOLS)
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(saved)


@pytest.fixture(scope="module")
def registries() -> dict[str, set[str]]:
    """``{mode: dispatch tool names}`` for every valid mode, built once."""
    return {mode: _build(mode) for mode in sorted(VALID_TOOL_MODES)}


class TestDispatchRegistryContract:
    @pytest.mark.parametrize("mode", DISPATCHING_MODES)
    def test_the_dispatch_registry_is_identical_in_every_dispatching_mode(
        self, registries, mode
    ) -> None:
        """Visibility is mode-dependent; callability is not.

        An exact-set assertion against the ``condensed`` baseline. A superset check
        would pass while a mode quietly dropped a domain's registrar — which is the
        shape of every regression this standard exists to catch.
        """
        baseline = registries["condensed"]
        # `intent` additionally registers the six verbs — which dispatch INTO the
        # same registry, so they are an addition to the contract, not a swap.
        expected = baseline | (INTENT_VERBS if mode == "intent" else set())
        assert_surface(
            registries[mode],
            expected,
            surface="kg_server.REGISTERED_TOOLS",
            invariant=CORE_DISPATCH_TOOLS,
            parameterisation=f"MCP_TOOL_MODE={mode}",
        )

    def test_intent_adds_the_verbs_without_removing_any_granular_tool(
        self, registries
    ) -> None:
        """``intent`` is additive on dispatch even though it is subtractive on view.

        The six verbs dispatch INTO the same registry rather than replacing it —
        so the granular tools must all still be callable. This is the assertion
        that distinguishes "collapsed the view" from "deleted the surface".
        """
        assert registries["condensed"] <= registries["intent"]
        assert INTENT_VERBS <= registries["intent"]

    def test_every_registrar_actually_contributed_to_the_registry(self) -> None:
        """WIRING — the live entrypoint reaches ``register_tool_surface`` for real.

        ``tests/test_verbose_tools.py`` exercises ``register_tool_surface`` against
        a synthetic ``MockMCP``, proving the *mechanism*; it does not prove
        graph-os is wired to it (and, being at ``tests/`` root, it is outside
        ``pytest.ini``'s ``testpaths`` and does not run in the default suite at
        all). This asserts the edge ``_build_server`` → ``register_tool_surface``
        with the real registrar list, and that the list is not silently empty.
        """
        from agent_utilities.mcp import verbose_tools

        saved = dict(kg_server.REGISTERED_TOOLS)
        kg_server.REGISTERED_TOOLS.clear()
        try:
            with observe(verbose_tools, "register_tool_surface") as surfaced:
                kg_server._build_server(bootstrap=False, tool_profile="condensed")
            call = surfaced.assert_called(
                why="graph-os registers its tools through the one fleet-wide surface builder"
            )
            registrars = call.arg("registrars")
            assert registrars, "graph-os passed an empty registrar list"
            assert call.arg("service") == "graph-os"
            # Each registrar is a real function, not a placeholder that no-ops.
            assert all(callable(fn) for fn in registrars)
            assert len(kg_server.REGISTERED_TOOLS) >= len(registrars), (
                "more registrars than registered tools — at least one registrar "
                "ran without contributing anything to the dispatch core"
            )
        finally:
            kg_server.REGISTERED_TOOLS.clear()
            kg_server.REGISTERED_TOOLS.update(saved)

