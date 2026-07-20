from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_gate():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts/check_external_graph_contract.py"
    spec = importlib.util.spec_from_file_location("external_graph_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_graph_gate_detects_environment_specific_literals() -> None:
    gate = _load_gate()
    fixture = gate.ROOT / "fixture.py"

    failures = gate.environment_literal_violations(
        fixture,
        'endpoint = "https' + '://private.example.test/graphql"\n'
        'owner = "person' + '@example.test"\n'
        'local_path = "/ho' + 'me/example/private"\n',
    )

    assert len(failures) == 3
    assert all("environment-specific literal" in failure for failure in failures)


def test_external_graph_gate_accepts_reference_only_examples() -> None:
    gate = _load_gate()
    fixture = gate.ROOT / "fixture.md"

    assert gate.environment_literal_violations(
        fixture,
        '"connection_profile_ref": "secret://source/connection"\n'
        '"mapping_policy_ref": "vault://source/mapping"\n',
    ) == []
