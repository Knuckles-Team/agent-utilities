"""Regression tests for the ContextCompiler static architecture gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _gate() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "check_context_compiler_boundary.py"
    )
    spec = importlib.util.spec_from_file_location("context_boundary_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_rejects_every_direct_pydantic_agent_import_and_constructor() -> None:
    gate = _gate()
    cases = (
        "from pydantic_ai import Agent\nagent = Agent(model)",
        (
            "from pydantic_ai import Agent as PydanticAgent\n"
            "agent = PydanticAgent(model)"
        ),
        "import pydantic_ai\nagent = pydantic_ai.Agent(model)",
        "import pydantic_ai\nAgent = pydantic_ai.Agent",
    )
    for source in cases:
        failures = gate._source_violations("agent_utilities/example.py", source)
        assert any("direct Agent" in failure for failure in failures)


def test_gate_does_not_confuse_domain_agent_names_with_pydantic_agent() -> None:
    gate = _gate()
    failures = gate._source_violations(
        "agent_utilities/example.py",
        'worker = SwarmAgent("id", role)\ntrader = TradingAgent(strategy)',
    )
    assert failures == []


def test_repository_has_one_governed_agent_constructor_boundary() -> None:
    assert _gate().violations() == []
