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


@pytest.fixture(autouse=True)
def _clear_host_ports():
    """Every test starts and ends with no host provider registered."""
    from agent_utilities import host_ports

    registrars = (
        host_ports.register_run_decision_provider,
        host_ports.register_run_history_provider,
        host_ports.register_run_checkpoint_provider,
        host_ports.register_scaffold_publication_factory,
        host_ports.register_model_provider,
        host_ports.register_tool_caller,
        host_ports.register_run_event_sink,
        host_ports.register_harness_cycle_provider,
        host_ports.register_native_toolset_provider,
        host_ports.register_remote_toolset_provider,
    )
    for clear in registrars:
        clear(None)
    yield
    for clear in registrars:
        clear(None)


@pytest.fixture
def events() -> list[dict]:
    """Run events captured through a fake host run event sink."""
    from agent_utilities import host_ports

    captured: list[dict] = []
    host_ports.register_run_event_sink(lambda event: captured.append(dict(event)))
    return captured
