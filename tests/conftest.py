"""Shared fixtures for the harness suite."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent_utilities.runtime import DevWorkspace, LocalWorkspace


@pytest.fixture
async def workspace(tmp_path) -> AsyncIterator[DevWorkspace]:
    """A started local developer workspace rooted in ``tmp_path``."""
    async with DevWorkspace(LocalWorkspace(root=tmp_path), run_id="testrun") as ws:
        yield ws
