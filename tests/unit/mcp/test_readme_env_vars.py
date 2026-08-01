"""README environment-variables table auto-generation."""

from __future__ import annotations

from agent_utilities.mcp.readme_env_vars import (
    END,
    START,
    parse_env_example,
    render_env_table,
    sync_readme,
)

SAMPLE = """\
# ==============================================================================
# Demo Agent Environment Configuration
# ==============================================================================

# --- Connection ---
DEMO_URL=https://demo.example.com # Base service URL
DEMO_TOKEN=your_token_here

# --- MCP Server Settings ---
TRANSPORT=stdio
# A description that wraps
# across multiple comment lines.
# DEMO_OPTIONAL=off
"""


def test_parse_extracts_name_example_description():
    rows = {name: (ex, desc) for name, ex, desc in parse_env_example(SAMPLE)}
    assert rows["DEMO_URL"] == ("https://demo.example.com", "Base service URL")
    # standalone-comment description carries to the next var
    assert rows["DEMO_TOKEN"][0] == "your_token_here"
    # commented-out assignment kept as an optional var
    assert "DEMO_OPTIONAL" in rows
    assert rows["DEMO_OPTIONAL"][1] == (
        "A description that wraps across multiple comment lines."
    )
    # section banners are not parsed as variables
    assert not any(n.startswith("-") for n in rows)


def test_render_has_package_and_inherited_sections():
    table = render_env_table(SAMPLE)
    assert START in table and END in table
    assert "Package environment variables" in table
    assert "| `DEMO_URL` | `https://demo.example.com` | Base service URL |" in table
    assert "Inherited agent-utilities variables" in table
    # TRANSPORT is declared by the package, so it must NOT be duplicated in inherited
    assert table.count("`TRANSPORT`") == 1
    # an inherited-only var shows up
    assert "`EUNOMIA_TYPE`" in table
    # Direct credential examples never get copied into generated documentation.
    assert "| `DEMO_TOKEN` | secret-injected |" in table
    assert "your_token_here" not in table
    # Runtime secret references remain visible as the supported configuration seam.
    assert "| `OIDC_CLIENT_SECRET_REF` |" in table
    assert "`secret://identity/oidc-client-secret`" in table
    assert "| `OIDC_CLIENT_SECRET` |" not in table
    assert "| `MCP_BASIC_AUTH_PASSWORD_REF` |" in table
    assert "| `MCP_BASIC_AUTH_PASSWORD` |" not in table
    assert r"`oidc-client-credentials` \| `basic` \| `none`" in table


def test_sync_inserts_and_is_idempotent(tmp_path):
    env = tmp_path / ".env.example"
    env.write_text(SAMPLE)
    readme = tmp_path / "README.md"
    readme.write_text("# X\n\n## Environment Variables\n\nplaceholder\n")
    assert sync_readme(env, readme) is True
    assert sync_readme(env, readme) is False  # second run is a no-op
    body = readme.read_text()
    assert START in body and body.count(START) == 1
