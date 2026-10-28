"""Exercise connector classification without a network or fleet checkout."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_connector_import_census import census, inspect_file


def _source(root: Path, package: str, contents: str) -> None:
    path = root / package / "package/source.py"
    path.parent.mkdir(parents=True)
    path.write_text(contents)


@pytest.mark.spec("AU-BOUNDARY-R006.1")
def test_fj_census_classifies_clean_and_banned_imports(tmp_path: Path) -> None:
    _source(tmp_path, "github-agent", "import json\n")
    _source(
        tmp_path,
        "fan-manager",
        "from agent_utilities.core.config import config, setting\n",
    )
    result = census(tmp_path, "fj")
    assert result["github-agent"].hits == ()
    assert result["github-agent"].unresolved == ()
    assert "agent_utilities.core.config" in result["fan-manager"].hits


@pytest.mark.spec("AU-BOUNDARY-R009.1")
def test_tz_census_classifies_static_imports_and_unknown_dynamic(
    tmp_path: Path,
) -> None:
    _source(tmp_path, "technitium-dns-mcp", "import json\n")
    _source(tmp_path, "twenty-mcp", "from agent_utilities.mcp import context_helpers\n")
    _source(tmp_path, "vault", "import importlib\nimportlib.import_module(name)\n")
    result = census(tmp_path, "tz")
    assert not result["technitium-dns-mcp"].hits
    assert "agent_utilities.mcp.context_helpers" in result["twenty-mcp"].hits
    assert result["vault"].unresolved


def test_aliases_and_dynamic_imports_are_detected(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "import agent_utilities as au\n"
        "from importlib import import_module as load\n"
        "import builtins as builtins_alias\n"
        "au.core.config.setting\n"
        "load(name='agent_utilities.base_utilities')\n"
        "builtins_alias.__import__('agent_utilities.core.exceptions')\n"
        "getattr(au, 'mcp')\n"
    )
    result = inspect_file(source)
    assert "agent_utilities.core.config" in result.hits
    assert "agent_utilities.base_utilities" in result.hits
    assert "agent_utilities.core.exceptions" in result.hits
    assert "agent_utilities.mcp" in result.hits


def test_relative_and_star_imports_remain_indeterminate(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "from agent_utilities import *\n"
        "import importlib\n"
        "importlib.import_module('.mcp', 'agent_utilities')\n"
        "__import__('agent_utilities', fromlist=['mcp'])\n"
    )
    assert len(inspect_file(source).unresolved) == 3


def test_missing_empty_or_unreadable_sources_never_report_clean(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no connector checkouts"):
        census(tmp_path, "fj")
    (tmp_path / "fan-manager").mkdir()
    with pytest.raises(ValueError, match="no Python source"):
        census(tmp_path, "fj")
    _source(tmp_path, "fan-manager", "def invalid(\n")
    with pytest.raises(SyntaxError):
        census(tmp_path, "fj")


def test_cli_reports_violations_and_does_not_treat_unknown_as_clean(
    tmp_path: Path,
) -> None:
    _source(
        tmp_path, "github-agent", "import importlib\nimportlib.import_module(name)\n"
    )
    _source(tmp_path, "fan-manager", "import agent_utilities.core.config\n")
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[2]
            / "scripts/check_connector_import_census.py"
        ),
        "--fleet-root",
        str(tmp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "fan-manager: agent_utilities.core.config" in result.stdout
    assert "github-agent: indeterminate" in result.stdout
    assert "unresolved dynamic import:" in result.stdout
