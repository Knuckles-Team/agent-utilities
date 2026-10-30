"""The source-connector lifecycle package is deleted from Meristem."""

import pytest
from _census import assert_absent, references


@pytest.mark.spec("MERISTEM-BOUNDARY-R011.3")
def test_source_connectors_package_absent() -> None:
    assert_absent("protocols/source_connectors", "protocols")
    assert references(("source_connectors",), include_tests=True) == []
