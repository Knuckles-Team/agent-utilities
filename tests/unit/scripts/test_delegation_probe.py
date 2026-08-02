"""D-CDX-19: the tracked delegation probe fails closed only in required mode."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def test_quality_gate_failure_aggregates_every_completed_sample() -> None:
    probe = _probe()

    assert probe._any_retrieval_quality_gate_failed([True, False, False]) is True
    assert probe._any_retrieval_quality_gate_failed([False, False, False]) is False


def test_stage_aggregates_quality_failure_across_real_sample_bundles(
    monkeypatch,
) -> None:
    probe = _probe()
    from agent_utilities.core import contextual_model, model_factory

    bundles = iter(
        [
            SimpleNamespace(retrieval_quality_gate_failed=True),
            SimpleNamespace(retrieval_quality_gate_failed=False),
        ]
    )
    calls = 0

    def _compile(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return object(), next(bundles)

    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 60.0)
    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _compile
    )
    monkeypatch.setattr(
        model_factory,
        "create_model",
        lambda **_kwargs: SimpleNamespace(model_name="test-model"),
    )

    with pytest.raises(RuntimeError, match="retrieval_quality_gate_failed"):
        asyncio.run(
            probe._stage_grounding(
                "standard", 1.0, "required", 2, "benchmark"
            )
        )

    assert calls == 2


@pytest.mark.parametrize(
    ("grounding", "expected"),
    [
        ("required", (1, "preflight")),
        ("best_effort", (0, "functional")),
        ("none", (0, "functional")),
    ],
)
def test_grounding_sample_plan_has_policy_aware_defaults(
    grounding: str, expected: tuple[int, str]
) -> None:
    assert _probe()._grounding_sample_plan(grounding, None) == expected


@pytest.mark.parametrize("grounding", ["required", "best_effort", "none"])
def test_positive_grounding_samples_enable_benchmarking(grounding: str) -> None:
    assert _probe()._grounding_sample_plan(grounding, 4) == (4, "benchmark")


def test_required_grounding_rejects_zero_samples() -> None:
    with pytest.raises(ValueError, match="required cannot use"):
        _probe()._grounding_sample_plan("required", 0)


def test_degraded_grounding_allows_explicit_zero_samples() -> None:
    assert _probe()._grounding_sample_plan("best_effort", 0) == (0, "functional")


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
        grounding_samples=1,
        grounding_sample_mode="preflight",
        traceback=False,
    )

    assert asyncio.run(probe.run(args)) == 4


def test_programmatic_required_zero_samples_returns_stage_four(monkeypatch) -> None:
    probe = _probe()

    async def _pass(*_args, **_kwargs):
        return "ok"

    monkeypatch.setattr(probe, "_stage_config", _pass)
    monkeypatch.setattr(probe, "_stage_identity", _pass)
    monkeypatch.setattr(probe, "_stage_engine", _pass)
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
        grounding_samples=0,
        grounding_sample_mode="functional",
        traceback=False,
    )

    assert asyncio.run(probe.run(args)) == 4


@pytest.mark.parametrize("grounding", ["best_effort", "none"])
def test_degraded_run_reaches_and_passes_grounding_stage(
    monkeypatch, grounding: str
) -> None:
    probe = _probe()
    reached: list[str] = []

    def _stage(name: str):
        async def _pass(*_args, **_kwargs):
            reached.append(name)
            return "ok"

        return _pass

    async def _grounding(*_args, **_kwargs):
        assert _args[2:] == (grounding, 0, "functional")
        reached.append("grounding")
        return "synthetic compile skipped; proceeding to real delegation"

    monkeypatch.setattr(probe, "_stage_config", _stage("config"))
    monkeypatch.setattr(probe, "_stage_identity", _stage("identity"))
    monkeypatch.setattr(probe, "_stage_engine", _stage("engine"))
    monkeypatch.setattr(probe, "_stage_grounding", _grounding)
    monkeypatch.setattr(probe, "_stage_model", _stage("model"))
    monkeypatch.setattr(probe, "_stage_skill", _stage("skill"))
    monkeypatch.setattr(probe, "_stage_toolset", _stage("toolset"))
    monkeypatch.setattr(probe, "_stage_delegate", _stage("delegate"))
    monkeypatch.setattr(probe, "_stage_provenance", _stage("provenance"))
    sample_count, sample_mode = probe._grounding_sample_plan(grounding, None)
    args = argparse.Namespace(
        skill="skill",
        server="server",
        tool="tool",
        mode="auto",
        identity_mode="process",
        transport="streamable-http",
        stop_after=None,
        model_class="standard",
        grounding_budget=90.0,
        grounding=grounding,
        grounding_samples=sample_count,
        grounding_sample_mode=sample_mode,
        live_model=False,
        traceback=False,
    )

    assert asyncio.run(probe.run(args)) == 0
    assert reached == probe.STAGES
