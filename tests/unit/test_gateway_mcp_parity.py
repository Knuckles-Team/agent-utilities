"""Contract test: the MCP tool surface and the REST gateway must stay in lockstep.

Both surfaces dispatch through ``_execute_tool`` against the same in-process
engine, so ``_execute_tool`` is the single source of truth. This test enforces
that every action-routed MCP tool in ``REGISTERED_TOOLS`` has exactly one
collapsed REST twin (via ``ACTION_TOOL_ROUTES``) and that the twin is actually
mounted by ``_mount_rest_routes`` — and vice versa. If a new MCP tool is added
without a REST route (or a route is added for a non-existent tool), this fails,
preventing the two surfaces from drifting.
"""

from __future__ import annotations

from agent_utilities.mcp import kg_server


class _RecordingApp:
    """Minimal Starlette-compatible app that records mounted routes."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, list[str]]] = []

    def add_route(self, path, handler, methods=None) -> None:  # noqa: ANN001
        self.routes.append((path, list(methods or [])))


def _mounted_paths(prefix: str = "") -> set[str]:
    kg_server.ensure_tools_registered()
    app = _RecordingApp()
    kg_server._mount_rest_routes(app, prefix=prefix)
    return {path for path, _methods in app.routes}


def test_every_mcp_tool_has_a_rest_route():
    kg_server.ensure_tools_registered()
    tools = set(kg_server.REGISTERED_TOOLS)
    mapped = set(kg_server.ACTION_TOOL_ROUTES)

    missing = tools - mapped
    assert not missing, (
        f"MCP tools with no REST twin in ACTION_TOOL_ROUTES: {sorted(missing)}. "
        "Add a collapsed REST route so the gateway reaches every MCP capability."
    )


def test_no_phantom_routes_for_missing_tools():
    kg_server.ensure_tools_registered()
    tools = set(kg_server.REGISTERED_TOOLS)
    mapped = set(kg_server.ACTION_TOOL_ROUTES)

    phantom = mapped - tools
    assert not phantom, (
        f"ACTION_TOOL_ROUTES references tools not in REGISTERED_TOOLS: {sorted(phantom)}."
    )


def test_mapped_routes_are_actually_mounted():
    paths = _mounted_paths()
    for tool, path in kg_server.ACTION_TOOL_ROUTES.items():
        assert path in paths, (
            f"Tool '{tool}' maps to '{path}' but that route is not mounted by "
            "_mount_rest_routes."
        )


def test_prefix_is_applied_to_mounted_routes():
    # The API gateway mounts these under /api — confirm the prefix is honoured.
    paths = _mounted_paths(prefix="/api")
    for path in kg_server.ACTION_TOOL_ROUTES.values():
        assert ("/api" + path) in paths


# ── Third leg: MCP verb ⇄ kg-* skill coverage (CONCEPT:AU-ECO.mcp.kg-skill-verb-coverage) ──────────────
# Beyond REST⇄MCP parity, every graph-os verb must also be wrapped by a
# discoverable ``kg-*`` skill (and no skill may reference a dead verb), so the
# operator-facing skill suite can never silently drift from the tool surface. The
# contract itself lives in ``agent_utilities.mcp.skill_coverage`` (shared by the
# ``kg-coverage-doctor`` skill CLI); this test is its CI/pre-commit enforcement.


def test_every_verb_has_a_wrapping_kg_skill():
    from agent_utilities.mcp import skill_coverage

    report = skill_coverage.compute_coverage()
    assert not report.uncovered, (
        "graph-os verbs with no wrapping kg-* skill: "
        f"{report.uncovered}. Author a kg-<slug> skill (slug = verb minus "
        "'graph_', '_'→'-'), fold it into an existing skill's `wraps:` list, or "
        "add it to skill_coverage.INTENTIONALLY_UNSKILLED with a reason."
    )


def test_no_orphan_kg_skills():
    from agent_utilities.mcp import skill_coverage

    report = skill_coverage.compute_coverage()
    assert not report.orphans, (
        "kg-* skills whose slug/`wraps:` points at a non-existent verb: "
        f"{report.orphans}. Fix the slug/wraps, or tag the skill "
        "`tier: meta|surface` if it is not a verb wrapper."
    )
    assert not report.bad_tiers, (
        f"kg-* skills with an invalid `tier:` value: {report.bad_tiers}. "
        "Use one of core|modality|meta|surface."
    )
