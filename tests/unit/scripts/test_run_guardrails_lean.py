from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "scripts" / "run_guardrails_lean.py"
SPEC = importlib.util.spec_from_file_location("run_guardrails_lean", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_runner_derives_frozen_sync_from_the_workflow() -> None:
    command = RUNNER._parse_sync_command(RUNNER._steps(RUNNER._load_workflow()))

    assert command[:3] == ["uv", "sync", "--frozen"]
    assert command.count("--no-install-package") == 2
    assert "epistemic-graph" in command
    assert "langfuse-agent" in command


def test_runner_discovers_nested_and_non_check_gate_scripts() -> None:
    gates = RUNNER._parse_gates(
        [
            {"name": "nested", "run": "python3 scripts/release/check_one.py"},
            {"name": "docs", "run": "python3 scripts/docs_contract.py --check"},
            {"name": "tests", "run": "python3 -m pytest tests/gates -q"},
        ]
    )

    assert [gate.name for gate in gates] == ["nested", "docs", "tests"]


def test_source_contract_environment_does_not_synthesize_kernel_absence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)

    environment = RUNNER._gate_env(tmp_path)

    assert "PYTHONPATH" not in environment
    assert not (tmp_path / "_kernel_block").exists()
    assert environment["PATH"].startswith(f"{tmp_path / 'bin'}{os.pathsep}")
