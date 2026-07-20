"""Contracts for the exact-installed local certification source gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "check_exact_local_gates_harness",
    _ROOT / "scripts" / "check_exact_local_gates_harness.py",
)
gate = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gate)


def _sources() -> tuple[str, str, str, str, str]:
    return (
        gate.HARNESS.read_text(encoding="utf-8"),
        gate.DOCUMENTATION.read_text(encoding="utf-8"),
        gate.NAVIGATION.read_text(encoding="utf-8"),
        gate.WORKFLOW.read_text(encoding="utf-8"),
        gate.A2A_TEST.read_text(encoding="utf-8"),
    )


def test_current_exact_local_certification_contract_is_complete() -> None:
    assert gate.check_contract(*_sources()) == ()


def test_gate_rejects_intent_matrix_and_exact_artifact_drift() -> None:
    harness, documentation, navigation, workflow, a2a = _sources()
    harness = harness.replace('    "why",\n', "", 1).replace(
        '    parser.add_argument("--engine-sha256", required=True)\n', "", 1
    )

    violations = gate.check_contract(
        harness, documentation, navigation, workflow, a2a
    )

    assert "intent_surface_drift" in violations
    assert "exact_artifact_arguments_drift" in violations


def test_gate_rejects_runtime_skips_and_incomplete_release_documentation() -> None:
    harness, documentation, navigation, workflow, a2a = _sources()
    harness += "\n# pytest.skip must never close an exact-installed gate\n"
    documentation = documentation.replace(
        "There is no artifact discovery or fallback path.", ""
    )

    violations = gate.check_contract(
        harness, documentation, navigation, workflow, a2a
    )

    assert "runtime_skip_path_present" in violations
    assert "documentation_contract_incomplete" in violations


def test_gate_rejects_environment_specific_literals_and_missing_navigation() -> None:
    harness, documentation, navigation, workflow, a2a = _sources()
    harness += '\n_LOCAL_MACHINE = "/home/example/private"\n'
    navigation = navigation.replace(gate.NAV_ENTRY, "")
    workflow = workflow.replace(gate.CI_ENTRY, "")

    violations = gate.check_contract(
        harness, documentation, navigation, workflow, a2a
    )

    assert "environment_specific_literal_present" in violations
    assert "documentation_navigation_invalid" in violations
    assert "continuous_gate_wiring_invalid" in violations


def test_gate_rejects_missing_optimizer_and_separate_process_controls() -> None:
    harness, documentation, navigation, workflow, a2a = _sources()
    harness = harness.replace("_expect_resource_limit(client, graph, invalid)", "pass")
    a2a = a2a.replace("first_process.kill()", "first_process.join()")

    violations = gate.check_contract(
        harness, documentation, navigation, workflow, a2a
    )

    assert "optimizer_behavior_control_missing" in violations
    assert "a2a_separate_process_crash_control_missing" in violations


def test_gate_rejects_avatar_artifact_and_plan_evidence_drift() -> None:
    harness, documentation, navigation, workflow, a2a = _sources()
    harness = harness.replace(
        '    "avatar": ("tool_policy",),\n',
        '    "avatar": ("instruction_proposal",),\n',
        1,
    )

    violations = gate.check_contract(
        harness, documentation, navigation, workflow, a2a
    )

    assert "optimizer_avatar_evidence_drift" in violations


def test_gate_rejects_work_item_bus_and_permission_case_drift() -> None:
    harness, documentation, navigation, workflow, a2a = _sources()
    harness = harness.replace('    "fairness_scoped_claim",\n', "", 1).replace(
        '    "test_injected_context_must_verify",\n', "", 1
    )

    violations = gate.check_contract(
        harness, documentation, navigation, workflow, a2a
    )

    assert "g08_runtime_case_matrix_drift" in violations
    assert "g35_permission_cases_incomplete" in violations


def test_gate_rejects_missing_runtime_worker_and_documentation_controls() -> None:
    harness, documentation, navigation, workflow, a2a = _sources()
    harness = harness.replace(
        "def _work_item_bus_worker(\n",
        "def _missing_work_item_bus_worker(\n",
        1,
    )
    documentation = documentation.replace(
        "eight-case permission-governance campaign", "permission campaign"
    )

    violations = gate.check_contract(
        harness, documentation, navigation, workflow, a2a
    )

    assert "work_item_bus_behavior_control_missing" in violations
    assert "documentation_contract_incomplete" in violations


def test_gate_rejects_work_item_read_before_graph_bootstrap() -> None:
    harness, documentation, navigation, workflow, a2a = _sources()
    harness = harness.replace(
        "        client.tenants.create(graph)\n"
        "        adapter = _ExactWorkItemAdapter(client, graph)\n",
        "        adapter = _ExactWorkItemAdapter(client, graph)\n"
        "        client.tenants.create(graph)\n",
        1,
    )

    violations = gate.check_contract(
        harness, documentation, navigation, workflow, a2a
    )

    assert "work_item_graph_bootstrap_order_invalid" in violations
