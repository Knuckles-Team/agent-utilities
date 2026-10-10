"""Public API exports approval-lease helpers (AU-BOUNDARY-R013.6)."""

from __future__ import annotations

import pytest

from agent_utilities.api import action_policy
from agent_utilities.orchestration import action_policy as internal


@pytest.mark.spec("AU-BOUNDARY-R013.6")
def test_approval_exports_are_the_internal_definitions() -> None:
    assert action_policy.ACTION_APPROVAL_KIND is internal.ACTION_APPROVAL_KIND
    assert action_policy.approval_lease_to_props is internal.approval_lease_to_props
