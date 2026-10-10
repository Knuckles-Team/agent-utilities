"""AU-BOUNDARY-R015.2: ``protocols/a2a_epistemic.py`` and the ACP script are gone."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gates._deletion_support import files_importing

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.spec("AU-BOUNDARY-R015.2")
def test_a2a_epistemic_module_and_script_absent() -> None:
    assert not (ROOT / "agent_utilities/protocols/a2a_epistemic.py").exists()
    assert not files_importing(
        [ROOT / "agent_utilities", ROOT / "scripts"],
        ("protocols.a2a_epistemic",),
        ROOT,
    )
    assert "agent-utilities-acp" not in (ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )
