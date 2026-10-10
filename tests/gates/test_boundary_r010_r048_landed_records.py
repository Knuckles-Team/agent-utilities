"""Landed-state records for AU-BOUNDARY-R010.1 and AU-BOUNDARY-R048.1."""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "agent_utilities"


@pytest.mark.spec("AU-BOUNDARY-R010.1")
def test_connector_toolkit_base_utilities_not_yet_deleted() -> None:
    """The R010 deletion has not landed: ``base_utilities.py`` is still present."""
    assert (PACKAGE_ROOT / "base_utilities.py").is_file()


@pytest.mark.spec("AU-BOUNDARY-R048.1")
def test_images_directory_already_relocated() -> None:
    """``agent_utilities/images/`` no longer exists."""
    assert not (PACKAGE_ROOT / "images").exists()
