"""D-CDX-19: the tracked delegation probe fails closed only in required mode."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _probe() -> ModuleType:
    source = Path(__file__).parents[3] / "scripts" / "delegation_probe.py"
    spec = importlib.util.spec_from_file_location("delegation_probe", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("samples", ([10.1], [float("inf")]))
def test_required_grounding_rejects_latency_and_timeout(samples: list[float]) -> None:
    reason = _probe()._grounding_gate_failure("required", samples, 10.0, False)

    assert reason is not None
    assert "latency budget exceeded" in reason


def test_required_grounding_rejects_retrieval_quality_failure() -> None:
    reason = _probe()._grounding_gate_failure(
        "required", [0.1, 0.2, 0.3], 10.0, True
    )

    assert reason == "retrieval_quality_gate_failed"


@pytest.mark.parametrize("grounding", ["best_effort", "none"])
def test_nonrequired_grounding_continues_after_measurement_failure(
    grounding: str,
) -> None:
    probe = _probe()
    reason = probe._grounding_gate_failure(
        grounding, [float("inf")], 10.0, True
    )

    assert reason is None


def test_run_returns_stage_four_for_required_grounding_failure(monkeypatch) -> None:
    probe = _probe()

    async def _pass(*_args, **_kwargs):
        return "ok"

    async def _grounding(*_args, **_kwargs):
        reason = probe._required_grounding_failure([float("inf")], 10.0, False)
        assert reason is not None
        raise RuntimeError("grounding='required' fails closed: " + reason)

    monkeypatch.setattr(probe, "_stage_config", _pass)
    monkeypatch.setattr(probe, "_stage_identity", _pass)
    monkeypatch.setattr(probe, "_stage_engine", _pass)
    monkeypatch.setattr(probe, "_stage_grounding", _grounding)
    args = argparse.Namespace(
        skill="",
        server="",
        tool="",
        mode="auto",
        identity_mode="process",
        transport="streamable-http",
        stop_after=None,
        model_class="standard",
        grounding_budget=90.0,
        grounding="required",
    )

    assert asyncio.run(probe.run(args)) == 4
