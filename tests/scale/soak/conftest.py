"""Shared fixtures for the installed SCALE-P2-1 soak/chaos harness."""

from __future__ import annotations

from types import ModuleType

import pytest

from scripts.scale import loadgen as installed_loadgen


@pytest.fixture(scope="session")
def loadgen() -> ModuleType:
    return installed_loadgen


@pytest.fixture()
def contract(loadgen: ModuleType):
    return loadgen.load_workload_contract()


@pytest.fixture()
def engine(loadgen: ModuleType):
    """A fresh, fast (near-zero-latency) mock engine for deterministic chaos tests.

    Chaos scenarios assert exact CAS win/lose outcomes and timing-sensitive lease
    expiry — near-zero synthetic latency keeps them fast and avoids flaking on
    shared-CI-runner jitter; :func:`test_steady_burst` (the throughput/SLO scenario)
    uses the loadgen's default (measured-anchor-calibrated) latency instead.
    """
    return loadgen.FakeScaleEngine(
        latency=loadgen.LatencyModel(
            write_mean_s=0.0,
            write_jitter_s=0.0,
            query_mean_s=0.0,
            query_jitter_s=0.0,
        )
    )
