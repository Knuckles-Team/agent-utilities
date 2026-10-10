"""AU-INTEGRATION-R004: AU's dependency lock does not pull in nltk."""

import tomllib
from pathlib import Path

import pytest

LOCK = Path(__file__).resolve().parents[3] / "uv.lock"


@pytest.mark.spec("AU-INTEGRATION-R004")
def test_uv_lock_has_no_nltk_package() -> None:
    data = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    names = {pkg["name"].lower() for pkg in data["package"]}
    assert "nltk" not in names
    assert len(names) > 50
