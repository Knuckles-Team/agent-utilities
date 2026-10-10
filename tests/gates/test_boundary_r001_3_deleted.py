"""AU-BOUNDARY-R001.3 — `agent_utilities/gateway/artifacts_api.py` deletion gate.

Asserts the module is gone from the checkout and no production code still imports it
(directly, via ``from ... import``, or by name) now that AU's duplicated gateway module
is retired in favor of graph-os serving the Live Artifacts HTTP surface.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DELETED_MODULE_PATH = REPO_ROOT / "agent_utilities" / "gateway" / "artifacts_api.py"

# Matches a dotted reference to the deleted module, e.g.
#   from agent_utilities.gateway.artifacts_api import artifacts_router
#   from agent_utilities.gateway import artifacts_api
#   agent_utilities.gateway.artifacts_api.register_artifact_source(...)
_IMPORT_PATTERN = re.compile(
    r"gateway\.artifacts_api|gateway\s+import\s+artifacts_api|install_kg_artifact_source"
)


@pytest.mark.spec("AU-BOUNDARY-R001.3")
def test_gateway_artifacts_api_module_is_deleted():
    assert not DELETED_MODULE_PATH.exists(), (
        f"{DELETED_MODULE_PATH} must be deleted per AU-BOUNDARY-R001.3"
    )


@pytest.mark.spec("AU-BOUNDARY-R001.3")
def test_no_production_importers_of_gateway_artifacts_api():
    result = subprocess.run(
        [
            "git",
            "grep",
            "-nE",
            _IMPORT_PATTERN.pattern,
            "--",
            "*.py",
            # Exclude this gate file itself: it names the retired symbols in the
            # search pattern/docstring, which is not an import of them.
            ":!tests/gates/test_boundary_r001_3_deleted.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # `git grep` exits 1 when there are no matches — that's the success case here.
    assert result.returncode in (0, 1), result.stderr
    matches = [line for line in result.stdout.splitlines() if line.strip()]
    assert matches == [], (
        "found remaining importers of the deleted "
        f"agent_utilities/gateway/artifacts_api.py:\n" + "\n".join(matches)
    )
