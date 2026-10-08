"""One installed fake EG runner per test (the ``eg`` fixture every consumer test uses)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from agent_utilities import decide
from tests.unit.decide.fakes import FakeTransport, runner


@pytest.fixture
def eg() -> Iterator[FakeTransport]:
    transport = FakeTransport()
    token = decide.use_runner(runner(transport))
    yield transport
    decide.reset_runner(token)
