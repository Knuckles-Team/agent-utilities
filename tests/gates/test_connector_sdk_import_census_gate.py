"""Gate: the connector-SDK import census correctly classifies packages.

Binds AU-BOUNDARY-R006.1 and AU-BOUNDARY-R009.1 -- AU's per-package import
census instrument for the F-J and T-Z connector-package migration rows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "security"))

from check_connector_sdk_import_census import census  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def fleet(tmp_path: Path) -> Path:
    """A self-contained fixture fleet: compliant, violating and skipped code."""
    # F-J range: one compliant package, one violating, one violating only in tests.
    _write(tmp_path / "github-agent" / "github_agent" / "a.py", "import json\n")
    _write(
        tmp_path / "fan-manager" / "fan_manager" / "agent_server.py",
        "from agent_utilities.mcp import server\nimport agent_utilities.core.config\n",
    )
    _write(
        tmp_path / "jira-mcp" / "tests" / "test_x.py",
        "from agent_utilities.base_utilities import x\n",
    )
    # T-Z range.
    _write(tmp_path / "technitium-dns-mcp" / "pkg" / "a.py", "import agent_utilities\n")
    _write(
        tmp_path / "twenty-mcp" / "twenty_mcp" / "auth.py",
        "from agent_utilities.core.exceptions import Boom\n",
    )
    # Outside both ranges: never visited.
    _write(
        tmp_path / "mealie-mcp" / "m.py",
        "from agent_utilities.mcp import server\n",
    )
    return tmp_path


@pytest.mark.spec("AU-BOUNDARY-R006.1")
def test_fj_census_classifies_known_packages(fleet: Path) -> None:
    result = census(fleet, "f", "j")
    assert "github-agent" not in result
    assert result["fan-manager"] == [
        "agent_utilities.core.config",
        "agent_utilities.mcp",
    ]
    # Imports under tests/ are not part of the shipped surface.
    assert "jira-mcp" not in result
    # The T-Z and out-of-range packages are not in the F-J census.
    assert "twenty-mcp" not in result
    assert "mealie-mcp" not in result


@pytest.mark.spec("AU-BOUNDARY-R009.1")
def test_tz_census_classifies_known_packages(fleet: Path) -> None:
    result = census(fleet, "t", "z")
    assert "technitium-dns-mcp" not in result
    assert result["twenty-mcp"] == ["agent_utilities.core.exceptions"]
    assert "fan-manager" not in result


@pytest.mark.spec("AU-BOUNDARY-R006.1")
def test_no_arg_invocation_checks_both_ranges_instead_of_erroring(fleet: Path) -> None:
    """D-ML-1: the contract-checks forwarder invokes every
    scripts/security/check_*.py with NO arguments at all
    (scripts/security/run_contract_checks.py has no per-script args table).
    Before this fix, --range was required, so that bare invocation failed
    argparse validation (exit 2) on every push. The no-arg path must now
    succeed by censusing BOTH the F-J and T-Z ranges -- strictly more
    coverage than either range alone, not a weaker/exempted check.
    """
    script = (
        _REPO_ROOT / "scripts" / "security" / "check_connector_sdk_import_census.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--fleet-root", str(fleet)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    # Fixtures from both ranges must both be visible in one no-arg run.
    assert "fan-manager:" in completed.stdout
    assert "twenty-mcp:" in completed.stdout
    assert "mealie-mcp" not in completed.stdout
