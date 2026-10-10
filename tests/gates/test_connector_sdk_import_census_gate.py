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

import os as _os

FLEET_ROOT = Path(
    _os.environ.get(
        "AGENT_FLEET_ROOT",
        "/home/apps/workspace/agent-packages/agents",
    )
)

pytestmark = pytest.mark.skipif(
    not FLEET_ROOT.is_dir(),
    reason="connector-package fleet checkout is not present in this environment",
)


@pytest.mark.spec("AU-BOUNDARY-R006.1")
def test_fj_census_classifies_known_packages() -> None:
    result = census(FLEET_ROOT, "f", "j")
    # github-agent has already dropped the banned AU import surface.
    assert "github-agent" not in result
    # fan-manager still imports it (agent_server.py / kg_control.py / kg_ingest.py).
    assert "fan-manager" in result
    assert result["fan-manager"]


@pytest.mark.spec("AU-BOUNDARY-R009.1")
def test_tz_census_classifies_known_packages() -> None:
    result = census(FLEET_ROOT, "t", "z")
    # technitium-dns-mcp has already dropped the banned AU import surface.
    assert "technitium-dns-mcp" not in result
    # twenty-mcp still imports it (auth.py).
    assert "twenty-mcp" in result
    assert result["twenty-mcp"]


@pytest.mark.spec("AU-BOUNDARY-R006.1")
def test_no_arg_invocation_checks_both_ranges_instead_of_erroring() -> None:
    """D-ML-1: the contract-checks forwarder invokes every
    scripts/security/check_*.py with NO arguments at all
    (scripts/security/run_contract_checks.py has no per-script args table).
    Before this fix, --range was required, so that bare invocation failed
    argparse validation (exit 2) on every push. The no-arg path must now
    succeed by censusing BOTH the F-J and T-Z ranges -- strictly more
    coverage than either range alone, not a weaker/exempted check.
    """
    script = _REPO_ROOT / "scripts" / "security" / "check_connector_sdk_import_census.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--fleet-root", str(FLEET_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    # Known fixtures from both ranges must both be visible in one no-arg run.
    assert "fan-manager:" in completed.stdout
    assert "twenty-mcp:" in completed.stdout
