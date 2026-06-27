"""Tests for the env-var drift detector (CONCEPT:OS-5.72)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.mcp import check_env_var_drift as drift


def _make_pkg(tmp_path: Path, *, env_example: str, mcp_config: dict, code: str) -> Path:
    root = tmp_path / "demo-agent"
    (root / "demo_agent").mkdir(parents=True)
    (root / ".env.example").write_text(env_example, encoding="utf-8")
    (root / "mcp_config.json").write_text(json.dumps(mcp_config), encoding="utf-8")
    (root / "demo_agent" / "auth.py").write_text(code, encoding="utf-8")
    return root


def _types(report: dict, kind: str) -> set[str]:
    return {f["var"] for f in report["findings"] if f["type"] == kind}


def test_dead_var_flagged(tmp_path: Path) -> None:
    """A var in mcp_config that no code reads is DEAD."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {
                        "DEMO_BASE_URL": "x",
                        "DEMO_TOKEN": "x",  # read by nothing -> DEAD
                        "MCP_TOOL_MODE": "condensed",
                    }
                }
            }
        },
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    report = drift.analyze(root)
    assert "DEMO_TOKEN" in _types(report, "DEAD")
    assert "DEMO_BASE_URL" not in _types(report, "DEAD")


def test_runtime_allowlist_not_dead(tmp_path: Path) -> None:
    """Generic process vars (TERM, NO_COLOR) are not flagged dead."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {"TERM": "xterm", "NO_COLOR": "1", "MCP_TOOL_MODE": "both"}
                }
            }
        },
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    report = drift.analyze(root)
    assert _types(report, "DEAD") == set()


def test_missing_tool_mode_flagged(tmp_path: Path) -> None:
    """An mcp_config env block without MCP_TOOL_MODE is flagged."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"DEMO_BASE_URL": "x"}}}},
        code='from agent_utilities.core.config import setting\nsetting("DEMO_BASE_URL", "")\n',
    )
    report = drift.analyze(root)
    assert "MCP_TOOL_MODE" in _types(report, "MISSING_TOOL_MODE")


def test_host_alias_not_undocumented(tmp_path: Path) -> None:
    """A legacy host alias (DEMO_HOST) is not UNDOCUMENTED when DEMO_BASE_URL is documented."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "from agent_utilities.core.config import setting\n"
            'setting("DEMO_BASE_URL", "") or setting("DEMO_HOST", "d")\n'
        ),
    )
    report = drift.analyze(root)
    assert "DEMO_HOST" not in _types(report, "UNDOCUMENTED")


def test_os_getenv_read_not_dead(tmp_path: Path) -> None:
    """A var read via bare os.getenv / os.environ is a real read, not DEAD."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={
            "mcpServers": {
                "demo": {
                    "env": {
                        "DEMO_BASE_URL": "x",
                        "HARVEST_HOST": "h",  # read via os.getenv below
                        "HARVEST_PORT": "1",  # read via os.environ[] below
                        "MCP_TOOL_MODE": "condensed",
                    }
                }
            }
        },
        code='import os\nos.getenv("HARVEST_HOST")\nos.environ["HARVEST_PORT"]\n',
    )
    report = drift.analyze(root)
    assert "HARVEST_HOST" not in _types(report, "DEAD")
    assert "HARVEST_PORT" not in _types(report, "DEAD")


def test_env_write_not_a_read(tmp_path: Path) -> None:
    """os.environ["X"] = ... is a write (cross-process signal), not a documentable read."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "import os\n"
            'setting("DEMO_BASE_URL", "")\n'
            'os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"\n'
            'os.environ["DEMO_SIGNAL"] = "1"\n'
        ),
    )
    flagged = {f["var"] for f in drift.analyze(root)["findings"]}
    assert "DEMO_SIGNAL" not in flagged
    assert "FASTMCP_LOG_LEVEL" not in flagged


def test_read_inside_string_literal_ignored(tmp_path: Path) -> None:
    """An os.environ.get("X") nested in a codegen template string is not a real read."""
    root = _make_pkg(
        tmp_path,
        env_example="DEMO_BASE_URL=http://x\n",
        mcp_config={"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}},
        code=(
            "import os\n"
            'setting("DEMO_BASE_URL", "")\n'
            "lines = []\n"
            "lines.append('    token = os.environ.get(\"GENERATED_TOKEN\")')\n"
        ),
    )
    assert "GENERATED_TOKEN" not in {f["var"] for f in drift.analyze(root)["findings"]}


def test_derived_toggle_undocumented(tmp_path: Path) -> None:
    """A register_<tag>_tools registrar implies <TAG>TOOL; undocumented if absent from .env.example."""
    root = tmp_path / "demo-agent"
    (root / "demo_agent").mkdir(parents=True)
    (root / ".env.example").write_text("DEMO_BASE_URL=http://x\n", encoding="utf-8")
    (root / "mcp_config.json").write_text(
        json.dumps({"mcpServers": {"demo": {"env": {"MCP_TOOL_MODE": "condensed"}}}}),
        encoding="utf-8",
    )
    (root / "demo_agent" / "tools.py").write_text(
        "def register_demo_reports_tools(mcp):\n    pass\n", encoding="utf-8"
    )
    report = drift.analyze(root)
    assert "DEMO_REPORTSTOOL" in _types(report, "UNDOCUMENTED")
