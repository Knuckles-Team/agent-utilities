"""Shared fixtures for the SCALE-P2-1 soak/chaos harness.

Dynamically loads ``scripts/scale/loadgen.py`` (and, transitively, ``fake_engine.py``
+ ``docs/scaling/workload_contract.py``) the same way ``tests/scale/test_capacity_model.py``
loads ``capacity_model.py`` — neither ``scripts/`` nor ``docs/scaling/`` is a packaged,
installed module, so importlib-by-path is this repo's established convention rather than
a relative/absolute package import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOADGEN_PATH = _REPO_ROOT / "scripts" / "scale" / "loadgen.py"
_CONTRACT_PATH = _REPO_ROOT / "docs" / "scaling" / "workload_contract.yml"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def loadgen() -> ModuleType:
    return _load_module("agent_utilities_scale_loadgen", _LOADGEN_PATH)


@pytest.fixture()
def contract(loadgen: ModuleType):
    return loadgen.load_workload_contract(str(_CONTRACT_PATH))


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
