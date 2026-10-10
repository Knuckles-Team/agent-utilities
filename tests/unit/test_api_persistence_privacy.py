"""Public API re-exports the persistence reference helper."""

from __future__ import annotations

import pytest

from agent_utilities.api import persistence_privacy
from agent_utilities.security import persistence_privacy as internal


@pytest.mark.spec("AU-BOUNDARY-R013.1")
def test_public_persistence_reference_is_the_internal_definition() -> None:
    assert persistence_privacy.persistence_reference is internal.persistence_reference
    assert persistence_privacy.__all__ == ["persistence_reference"]
