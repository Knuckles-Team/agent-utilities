"""README MCP-tools-table auto-generation (ECO-4.82)."""

from __future__ import annotations

from fastmcp import FastMCP

from agent_utilities.mcp.readme_tools import (
    END,
    START,
    _toggle_env,
    render_tools_table,
    sync_readme,
)


def _server() -> FastMCP:
    mcp = FastMCP("t")

    @mcp.tool(name="svc_cmdb", tags={"cmdb"})
    def _c():
        "Manage CMDB operations."

    @mcp.tool(name="svc_incidents", tags={"incidents"})
    def _i():
        "Manage incidents."

    @mcp.tool(name="svc_get_cmdb_instance", tags={"verbose", "cmdb"})
    def _v():
        "verbose 1:1 op."

    return mcp


def test_toggle_env_ignores_structural_only_tags():
    assert _toggle_env(set()) == "—"
    assert _toggle_env({"verbose"}) == "—"
    assert _toggle_env({"granular", "verbose"}) == "—"


def test_toggle_env_selects_domain_among_structural_tags():
    assert _toggle_env({"series", "granular"}) == "`SERIESTOOL`"
    assert _toggle_env({"podcasts", "granular", "verbose"}) == "`PODCASTSTOOL`"


def _surface() -> FastMCP:
    """A connector built through the shared builder (the intent contract)."""
    import types

    from agent_utilities.mcp.verbose_tools import register_tool_surface

    mod = types.ModuleType("fake_pkg_mcp")

    def register_cmdb_tools(mcp):
        @mcp.tool(name="svc_cmdb", tags={"cmdb"})
        def _c():
            "Manage CMDB operations."

    mod.register_cmdb_tools = register_cmdb_tools
    mcp = FastMCP("t")
    register_tool_surface(mcp, service="svc-api", tools_module=mod)
    return mcp


def test_render_lists_the_intent_tools_and_their_operations():
    table = render_tools_table(_surface())
    assert START in table and END in table
    assert "#### Intent tools" in table
    for verb in ("find", "ask", "act"):
        assert f"| `{verb}` | intent |" in table
    assert "#### Operations (`action` values)" in table
    assert "| `svc_cmdb` | act · `CMDBTOOL` |" in table
    assert "3 intent tool(s) · 1 operation(s)" in table


def test_render_omits_the_operations_section_without_a_backing_server():
    table = render_tools_table(_server())
    assert "#### Operations" not in table
    assert "0 operation(s)" in table


def test_sync_inserts_under_heading_and_is_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# X\n\n## Available MCP Tools\n\nplaceholder\n")
    assert sync_readme(_server(), readme) is True
    body = readme.read_text()
    assert START in body and END in body
    # second run is a no-op (table already current)
    assert sync_readme(_server(), readme) is False


def test_sync_replaces_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# X\n\n{START}\nstale table\n{END}\n\n## After\n")
    sync_readme(_server(), readme)
    body = readme.read_text()
    assert "stale table" not in body
    assert "## After" in body  # content after the markers preserved
    assert body.count(START) == 1 and body.count(END) == 1


def test_check_mode_does_not_write(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# X\n")
    changed = sync_readme(_server(), readme, check=True)
    assert changed is True  # would change
    assert START not in readme.read_text()  # but did not write
