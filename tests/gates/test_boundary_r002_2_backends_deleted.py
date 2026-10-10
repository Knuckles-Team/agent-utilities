"""Deletion census for AU-BOUNDARY-R002.2 (and future R001/R010-family
siblings sharing this module).

AU-BOUNDARY-R002 retires AU's deployment and skill-certification tooling in
favor of graph-os. R002.1 found that the PR which moved `R002` to LANDED
never actually deleted `agent_utilities/deployment/backends.py` (or
`certification_oidc.py`); both files were still present at `origin/main`.
R002.2 is the bite-sized row that deletes `backends.py` for real and
repoints its one production importer (the CLI's `deploy-plan` console
command, which collided with graph-os's own deployment-planner entry point
and is removed outright per R002's "AU removes the console scripts that
... target the moved modules" text) and its dedicated unit test.

This gate asserts the file is gone and that no importer -- production or
test -- references it anymore, so the row cannot silently regress to the
R002.1 state (text says deleted, file still on disk).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gates._deletion_support import files_importing

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "agent_utilities"
TESTS_ROOT = REPO_ROOT / "tests"

DELETED_MODULE_PATH = PACKAGE_ROOT / "deployment" / "backends.py"


@pytest.mark.spec("AU-BOUNDARY-R002.2")
def test_deployment_backends_deleted() -> None:
    """`agent_utilities/deployment/backends.py` is gone and no importer --
    production or test -- still references `agent_utilities.deployment.backends`."""
    assert not DELETED_MODULE_PATH.exists(), (
        f"{DELETED_MODULE_PATH} still present; AU-BOUNDARY-R002.2 requires it deleted"
    )

    importers = files_importing(
        (PACKAGE_ROOT, TESTS_ROOT),
        ("agent_utilities.deployment.backends",),
        REPO_ROOT,
    )
    assert importers == frozenset(), (
        "no importer of agent_utilities.deployment.backends may remain, found: "
        f"{sorted(importers)}"
    )
