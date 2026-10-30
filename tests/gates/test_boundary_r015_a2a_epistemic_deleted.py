"""The second A2A task store and the ACP script are gone from Meristem."""

import pytest
from _census import assert_absent, console_scripts, references


@pytest.mark.spec("MERISTEM-BOUNDARY-R015.2")
def test_a2a_epistemic_module_and_script_absent() -> None:
    assert_absent("protocols/a2a_epistemic.py")
    assert "agent-utilities-acp" not in console_scripts()
    assert references(("a2a_epistemic",), include_tests=True) == []
