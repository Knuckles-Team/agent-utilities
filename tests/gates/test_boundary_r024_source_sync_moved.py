"""Source synchronization is no longer in Meristem."""

import pytest
from _census import assert_absent, references


@pytest.mark.spec("MERISTEM-BOUNDARY-R024.2")
def test_source_sync_module_absent_from_au() -> None:
    assert_absent("knowledge_graph/core/source_sync.py")
    assert references(("core.source_sync",), include_tests=True) == []
