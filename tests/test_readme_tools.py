"""README MCP-tools-table auto-generation (ECO-4.82)."""

from __future__ import annotations

from fastmcp import FastMCP

from agent_utilities.mcp.readme_tools import (
    END,
    START,
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


def test_render_includes_condensed_and_verbose_sections():
    table = render_tools_table(_server())
    assert START in table and END in table
    # condensed action-routed tools render in the default section
    assert "Condensed action-routed tools" in table
    assert "| `svc_cmdb` | `CMDBTOOL` |" in table
    assert "| `svc_incidents` | `INCIDENTSTOOL` |" in table
    # verbose 1:1 tools are now INCLUDED in their own (collapsible) section
    assert "Verbose 1:1 API-mapped tools" in table
    assert "<details>" in table
    assert "| `svc_get_cmdb_instance` |" in table
    # summary distinguishes the two surfaces
    assert "2 action-routed tool(s) (default) · 1 verbose 1:1 tool(s)" in table


def test_render_omits_verbose_section_when_none():
    mcp = FastMCP("t")

    @mcp.tool(name="svc_cmdb", tags={"cmdb"})
    def _c():
        "Manage CMDB operations."

    table = render_tools_table(mcp)
    assert "Condensed action-routed tools" in table
    assert "Verbose 1:1 API-mapped tools" not in table  # no verbose -> no section
    assert "1 action-routed tool(s) (default) · 0 verbose 1:1 tool(s)" in table


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
