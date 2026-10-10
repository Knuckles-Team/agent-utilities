"""Public API exports the AU elevation contract."""

from __future__ import annotations

import pytest

from agent_utilities.api import elevation
from agent_utilities.security import elevation as core_elevation


@pytest.mark.spec("AU-BOUNDARY-R013.7")
def test_public_elevation_models_are_the_canonical_definitions() -> None:
    for name in ("ElevationRequest", "ElevationApproval", "ElevationRevocation"):
        assert getattr(elevation, name) is getattr(core_elevation, name)
        assert name in elevation.__all__


@pytest.mark.spec("AU-BOUNDARY-R013.8")
def test_public_elevation_service_and_surface_are_canonical() -> None:
    for name in ("ElevationService", "ElevationSurface"):
        assert getattr(elevation, name) is getattr(core_elevation, name)
        assert name in elevation.__all__
