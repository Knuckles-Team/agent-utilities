"""Contract test: the MCP tool surface and the REST gateway must stay in lockstep.

Both surfaces dispatch through ``_execute_tool`` against the same in-process
engine, while the immutable ToolSpec universe is the capability source of
truth. This test enforces that every canonical profile tool has exactly one
collapsed REST twin (via ``ACTION_TOOL_ROUTES``) and that the twin is actually
mounted by ``_mount_rest_routes`` — and vice versa. If a new MCP tool is added
without a REST route (or a route is added for a non-existent tool), this fails,
preventing the two surfaces from drifting.
"""

from __future__ import annotations

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tool_specs import (
    INTENT_VERBS,
    TOOL_SPECS_BY_NAME,
    canonical_tool_names,
)


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


def _canonical_runtime_names() -> frozenset[str]:
    tools = set(kg_server.REGISTERED_TOOLS)
    unknown = tools - set(TOOL_SPECS_BY_NAME)
    assert not unknown, (
        f"runtime registered unknown ToolSpec families: {sorted(unknown)}"
    )
    features = frozenset(
        spec.feature
        for name, spec in TOOL_SPECS_BY_NAME.items()
        if name in tools and spec.feature is not None
    )
    include_intent = set(INTENT_VERBS) <= tools
    return canonical_tool_names(features=features, include_intent=include_intent)


def test_every_mcp_tool_has_a_rest_route():
    kg_server.ensure_tools_registered()
    tools = set(kg_server.REGISTERED_TOOLS)
    mapped = set(kg_server.ACTION_TOOL_ROUTES)

    assert tools == set(_canonical_runtime_names())

    missing = tools - mapped
    assert not missing, (
        f"MCP tools with no REST twin in ACTION_TOOL_ROUTES: {sorted(missing)}. "
        "Add a collapsed REST route so the gateway reaches every MCP capability."
    )


def test_no_phantom_routes_for_missing_tools():
    kg_server.ensure_tools_registered()
    tools = set(kg_server.REGISTERED_TOOLS)
    mapped = set(kg_server.ACTION_TOOL_ROUTES)

    assert tools == set(_canonical_runtime_names())

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


# ── Third leg: MCP verb ⇄ domain-skill coverage (CONCEPT:AU-ECO.mcp.kg-skill-verb-coverage) ──
# Beyond REST⇄MCP parity, every Graph-OS verb must be claimed explicitly by a
# retained workflow skill's ``agents/graph-os.yaml`` sidecar. The contract lives
# in ``agent_utilities.mcp.skill_coverage`` and has no slug inference or waivers.


def test_every_verb_has_explicit_domain_skill_coverage():
    from agent_utilities.mcp import skill_coverage

    report = skill_coverage.compute_coverage()
    assert not report.uncovered, (
        "Graph-OS verbs missing from every domain skill sidecar: "
        f"{report.uncovered}. Add each verb to the owning workflow skill's "
        "agents/graph-os.yaml file."
    )


def test_domain_skill_sidecars_are_valid_and_have_no_orphans():
    from agent_utilities.mcp import skill_coverage

    report = skill_coverage.compute_coverage()
    assert not report.orphans, (
        "Domain skills claiming verbs outside the live Graph-OS surface: "
        f"{report.orphans}. Fix the explicit sidecar claim."
    )
    assert not report.duplicates, (
        "Graph-OS verbs with ambiguous domain ownership: "
        f"{report.duplicates}. Keep each verb in exactly one domain sidecar."
    )
    assert not report.invalid_sidecars, (
        "Invalid agents/graph-os.yaml sidecars: "
        f"{report.invalid_sidecars}. Follow the closed version-2 schema."
    )
