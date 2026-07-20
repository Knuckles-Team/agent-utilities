#!/usr/bin/env python3
"""Fail closed when the exact-installed local certification contract drifts."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "scripts" / "certification" / "exact_local_gates.py"
DOCUMENTATION = ROOT / "docs" / "release" / "exact-local-gates.md"
NAVIGATION = ROOT / "mkdocs.yml"
WORKFLOW = ROOT / ".github" / "workflows" / "guardrails.yml"
A2A_TEST = ROOT / "tests" / "integration" / "protocols" / "test_a2a_epistemic_live.py"

EXPECTED_INTENT_TOOLS = (
    "act",
    "ask",
    "find",
    "find_tools",
    "list_catalog",
    "load_tools",
    "manage",
    "multiplexer_status",
    "unload_tools",
    "why",
    "write",
)
EXPECTED_MODALITIES = (
    "text",
    "document",
    "image",
    "audio",
    "video",
    "graph",
    "table",
    "time_series",
    "vector",
    "spatial",
    "tensor",
    "code",
    "trace",
    "binary",
)
EXPECTED_OPTIMIZER_FAMILIES = (
    ("labeled_few_shot", ("labeled_few_shot",), "native_kernel", 0, 0),
    ("bootstrap_few_shot", ("bootstrap_few_shot",), "native_kernel", 0, 0),
    (
        "bootstrap_few_shot_with_random_search",
        ("bootstrap_few_shot_with_random_search",),
        "native_kernel",
        0,
        0,
    ),
    ("avatar", ("avatar",), "model_transport_plan", 8, 0),
    ("copro", ("copro",), "model_transport_plan", 8, 0),
    ("mipro_v2", ("mipro_v2",), "model_transport_plan", 8, 0),
    ("simba", ("simba",), "model_transport_plan", 8, 0),
    ("gepa", ("gepa",), "model_transport_plan", 8, 0),
    ("better_together", ("better_together",), "composite_plan", 8, 32),
    ("bootstrap_finetune", ("bootstrap_finetune",), "trainer_plan", 0, 32),
    ("knn_few_shot", ("knn_few_shot",), "graph_kernel_plan", 0, 0),
    ("ensemble", ("ensemble",), "native_kernel", 0, 0),
    ("infer_rules", ("infer_rules",), "model_transport_plan", 8, 0),
)
EXPECTED_A2A_SCENARIOS = (
    "test_native_create_reconcile_and_terminal_commit_are_atomic",
    "test_native_transaction_conflict_never_partially_commits",
    "test_native_delivery_lease_renews_and_stale_generation_is_fenced",
    "test_native_poison_bounds_and_record_tamper_fail_closed",
    "test_native_cancellation_wins_and_late_completion_is_rejected",
    "test_native_crash_restart_recovers_and_fences_precrash_completion",
)
EXPECTED_G08_RUNTIME_CASES = (
    "fairness_scoped_claim",
    "renewable_lease",
    "checkpoint_fencing",
    "retry_schedule",
    "dependency_release",
    "dead_letter",
    "stale_worker_rejection",
    "idempotent_terminal_commit",
)
EXPECTED_G09_RUNTIME_CASES = (
    "atomic_inbox_workitem_commit",
    "crash_window_replay_idempotent",
)
EXPECTED_G35_RUNTIME_CASES = (
    "referenced_signing_bootstrap",
    "closed_world_function_denial",
    "mcp_identity_denial",
    "ontology_permission_denial",
    "action_capability_denial",
    "governed_constructor_denial",
    "delegation_context_preserved",
    "invalid_authority_and_policy_rejected",
)

REQUIRED_G26_CASES = (
    "test_intent_mode_registers_verbs_and_keeps_the_granular_surface",
    "test_intent_verbs_and_gating_are_active_by_default",
    "test_intent_surface_selection_accuracy_meets_measured_floor",
    "test_ask_routes_to_the_right_tool_and_dispatches_via_execute_tool",
    "test_load_tools_reveals_a_gated_local_tool",
    "test_auto_unload_retracts_the_tool_after_its_next_call",
    "test_manage_verb_lifecycle_action_loads_and_unloads",
    "test_non_read_requires_bound_preview_and_surfaces_safety_plan",
    "test_destructive_plan_requires_exact_tool_approval",
    "test_candidate_build_preserves_exact_authority_order",
    "test_candidate_build_fails_closed_on_any_packaged_verb_drift",
    "test_pinned_execution_and_caller_feedback_cannot_poison_learning",
    "test_outcome_partition_covers_verified_tenant_policy_audience_and_scopes",
    "test_prompt_injection_is_denied_before_resolution_or_execution",
    "test_ambiguous_non_read_intent_never_executes",
    "test_human_approval_class_routes_to_exact_tool_even_when_not_destructive",
)
REQUIRED_G32_CASES = (
    "test_marker_v2_is_closed_bounded_path_free_and_atomic",
    "test_unmarked_operator_destination_is_never_replaced",
    "test_generation_activation_replaces_view_without_partial_or_deleted_files",
    "test_zero_assets_deactivates_old_ontology_instead_of_restamping_it",
    "test_source_marker_symlink_and_special_file_are_rejected",
    "test_failed_stage_leaves_previous_complete_generation_active",
    "test_serialized_concurrent_activation_leaves_one_complete_generation",
    "test_registration_is_distribution_owned_and_never_imports_provider",
    "test_duplicate_and_casefold_provider_owners_fail_closed",
    "test_empty_or_tampered_generation_falls_back_to_current_prompt_source",
    "test_install_api_has_no_force_and_result_is_path_free",
)
REQUIRED_G35_CASES = (
    "test_permissions_kernel_requires_explicit_strong_authority",
    "test_non_empty_capabilities_are_additional_closed_world_grants",
    "test_missing_configured_policy_fails_closed",
    "test_malformed_policy_clears_existing_policy_set",
    "test_resolves_reference_and_returns_verified_pair",
    "test_missing_reference_fails_without_process_authority",
    "test_injected_context_must_verify",
    "test_flag_mcp_tool_definitions_requires_identity_policy",
    "test_flag_mcp_tool_definitions_hard_deny_is_not_approvable",
    "test_flag_mcp_tool_definitions_authorization_error_fails_closed",
    "test_flag_mcp_tool_definitions_unknown_decision_fails_closed",
    "test_missing_marking_store_fails_closed",
    "test_missing_acl_is_denied",
    "test_missing_governed_id_raises_instead_of_leaking_projection",
    "test_unverified_or_tenantless_actor_is_rejected",
    "test_action_executor_requires_injected_kernel",
    "test_permission_deny_blocks_and_audits",
    "test_broad_role_allow_cannot_replace_required_capability",
    "test_create_context_agent_rejects_raw_mcp_without_permission_context",
    "test_graph_builder_injects_one_verified_permission_context",
    "test_graph_builder_rejects_mismatched_permission_context",
)

REQUIRED_ARGUMENTS = frozenset(
    {
        "--release-python",
        "--release-sha256",
        "--release-id",
        "--release-manifest",
        "--release-manifest-sha256",
        "--graphos",
        "--graphos-sha256",
        "--engine-binary",
        "--engine-sha256",
        "--signer-id",
        "--signing-key-env",
        "--output",
    }
)

REQUIRED_SOURCE_TOKENS = (
    "release_distribution_is_editable",
    "release_distribution_has_direct_url",
    "release_distribution_record_mismatch",
    "release_distribution_unlisted_file",
    "release_distribution_uses_source_checkout",
    "graphos_release_interpreter_mismatch",
    "graphos_engine_binary_mismatch",
    "g26_initial_surface_mismatch",
    "g26_dynamic_load_mismatch",
    "g26_dynamic_unload_mismatch",
    "g26_dynamic_approval_not_preserved",
    "g26_plan_binding_tamper_accepted",
    "g26_exact_tool_approval_route_failed",
    "g26_injection_not_denied",
    "g26_poisoned_feedback_not_denied",
    "program_optimization_result",
    "optimizer_budget_projection_mismatch",
    "optimizer_valid_promotion_missing",
    "optimizer_modality_regression_promoted",
    "optimizer_duplicate_runtime_loaded",
    "optimizer_governed_plan_invalid",
    "optimizer_executor_output_missing",
    "optimizer_candidate_cardinality_invalid",
    "optimizer_invalid_budget_wrong_failure",
    "optimizer_avatar_contract_invalid",
    "provider_concurrent_activation_failed",
    "provider_concurrent_doctor_failed",
    "provider_final_doctor_failed",
    "provider_tamper_not_detected",
    "provider_link_source_accepted",
    "provider_special_file_accepted",
    "provider_doctor_contains_local_reference",
    "work_item_fairness_claim_failed",
    "work_item_renewable_lease_failed",
    "work_item_checkpoint_fence_failed",
    "work_item_retry_schedule_failed",
    "work_item_dependency_release_failed",
    "work_item_dead_letter_failed",
    "work_item_stale_worker_not_rejected",
    "work_item_terminal_replay_failed",
    "bus_inbox_transaction_incomplete",
    "bus_crash_window_replay_failed",
    "permission_reference_bootstrap_failed",
    "permission_closed_world_function_failed",
    "permission_mcp_denial_failed",
    "permission_ontology_denial_failed",
    "permission_action_denial_failed",
    "permission_constructor_denial_failed",
    "permission_delegation_context_lost",
    "permission_invalid_authority_accepted",
    "permission_invalid_policy_accepted",
    "integration/protocols/test_a2a_epistemic_live.py",
    "exact_installed_pytest_gate_failed",
    "_assert_evidence_safe",
    "evidence_destination_must_be_new_absolute_path",
    "src_dir_fd=parent_descriptor",
    "distribution_closure_sha256",
    "ed25519",
    "RLIMIT_CORE",
    "killpg",
    "TemporaryDirectory",
    '"-I"',
)
REQUIRED_DOCUMENTATION_TOKENS = (
    "# Exact installed local certification",
    "exactly six intent verbs and five control tools",
    "exactly 13 semantic optimizer families",
    "one current spelling each",
    "`tool_policy` artifacts",
    "`compare_tool_use` plans",
    "all 14 modalities",
    "exact six-scenario suite",
    "eight-case native WorkItem lifecycle",
    "two-case transactional AgentBus campaign",
    "eight-case permission-governance campaign",
    "referenced signing material",
    "closed-world denial",
    "path-independent installed-tree digest",
    "release manifest",
    "Ed25519",
    "RECORD-verified",
    "Linux or WSL",
    "There is no artifact discovery or fallback path.",
    "path-free doctor output",
    "remain open until the exact installed campaign",
    "--release-python",
    "--release-sha256",
    "--release-manifest-sha256",
    "--graphos-sha256",
    "--engine-sha256",
)
NAV_ENTRY = "Exact Installed Local Certification: release/exact-local-gates.md"
CI_ENTRY = "run: python3 scripts/check_exact_local_gates_harness.py"
_LOCAL_LITERAL = re.compile(
    r"(?i)(?:[a-z]:[\\/]users[\\/][a-z0-9_.-]+|/mnt/[a-z]/|/home/[a-z0-9_.-]+)"
)
_BUILD_PROGRAMS = frozenset(
    {"cargo", "git", "maturin", "pip", "pip3", "rustc", "uv", "uvx"}
)


def _literal_assignments(tree: ast.AST) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def _subprocess_programs(tree: ast.AST) -> set[str]:
    programs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else ""
        if name not in {"Popen", "run"}:
            continue
        command = node.args[0]
        if not isinstance(command, ast.List) or not command.elts:
            continue
        first = command.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            programs.add(first.value.casefold())
    return programs


def _function(
    tree: ast.AST, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == name
        ),
        None,
    )


def _called_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    names: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Name):
            names.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
    return names


def _attribute_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {
        candidate.attr
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Attribute)
    }


def _required_arguments(tree: ast.AST) -> set[str]:
    parser = _function(tree, "_parser")
    if parser is None:
        return set()
    arguments: set[str] = set()
    for call in ast.walk(parser):
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "add_argument"
            or not call.args
            or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, str)
        ):
            continue
        required = next(
            (
                keyword.value.value
                for keyword in call.keywords
                if keyword.arg == "required"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, bool)
            ),
            False,
        )
        if required:
            arguments.add(call.args[0].value)
    return arguments


def _has_broad_suppression(node: ast.AST | None) -> bool:
    if node is None:
        return True
    for handler in (
        candidate for candidate in ast.walk(node) if isinstance(candidate, ast.ExceptHandler)
    ):
        if handler.type is None and any(isinstance(item, ast.Pass) for item in handler.body):
            return True
        if (
            isinstance(handler.type, ast.Name)
            and handler.type.id in {"BaseException", "Exception"}
            and any(isinstance(item, ast.Pass) for item in handler.body)
        ):
            return True
    return False


def _hardcoded_authority(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if any(name.casefold() in {"auth_secret", "signer_key"} for name in names):
            return True
    return False


def check_contract(
    harness_text: str,
    documentation_text: str,
    navigation_text: str,
    workflow_text: str,
    a2a_text: str,
) -> tuple[str, ...]:
    """Return stable, non-sensitive contract violation codes."""

    violations: list[str] = []
    try:
        tree = ast.parse(harness_text, filename="exact_local_gates.py")
    except SyntaxError:
        return ("harness_syntax_invalid",)
    assignments = _literal_assignments(tree)
    if assignments.get("INTENT_TOOLS") != EXPECTED_INTENT_TOOLS:
        violations.append("intent_surface_drift")
    if assignments.get("MODALITIES") != EXPECTED_MODALITIES:
        violations.append("modality_matrix_drift")
    optimizer_families = assignments.get("OPTIMIZER_FAMILIES")
    one_spelling_per_family = (
        isinstance(optimizer_families, tuple)
        and len(optimizer_families) == 13
        and all(
            isinstance(row, tuple)
            and len(row) == 5
            and isinstance(row[1], tuple)
            and row[1] == (row[0],)
            for row in optimizer_families
        )
    )
    if (
        optimizer_families != EXPECTED_OPTIMIZER_FAMILIES
        or not one_spelling_per_family
    ):
        violations.append("optimizer_matrix_drift")
    artifact_kinds = assignments.get("OPTIMIZER_ARTIFACT_KINDS")
    plan_step_kinds = assignments.get("ARTIFACT_PLAN_STEP_KINDS")
    plan_executors = assignments.get("EXPECTED_PLAN_EXECUTORS")
    if (
        not isinstance(artifact_kinds, dict)
        or artifact_kinds.get("avatar") != ("tool_policy",)
        or not isinstance(plan_step_kinds, dict)
        or plan_step_kinds.get("tool_policy") != {"compare_tool_use"}
        or not isinstance(plan_executors, dict)
        or plan_executors.get("avatar") != {"model_transport"}
    ):
        violations.append("optimizer_avatar_evidence_drift")
    if assignments.get("G34_SCENARIOS") != EXPECTED_A2A_SCENARIOS:
        violations.append("a2a_scenario_matrix_drift")
    if assignments.get("G08_RUNTIME_CASES") != EXPECTED_G08_RUNTIME_CASES:
        violations.append("g08_runtime_case_matrix_drift")
    if assignments.get("G09_RUNTIME_CASES") != EXPECTED_G09_RUNTIME_CASES:
        violations.append("g09_runtime_case_matrix_drift")
    if assignments.get("G35_RUNTIME_CASES") != EXPECTED_G35_RUNTIME_CASES:
        violations.append("g35_runtime_case_matrix_drift")

    g26_cases = assignments.get("G26_REQUIRED_TESTS", ())
    if g26_cases != REQUIRED_G26_CASES:
        violations.append("g26_adversarial_cases_incomplete")
    g32_cases = assignments.get("G32_REQUIRED_TESTS", ())
    if g32_cases != REQUIRED_G32_CASES:
        violations.append("g32_provider_cases_incomplete")
    g35_cases = assignments.get("G35_REQUIRED_TESTS", ())
    if g35_cases != REQUIRED_G35_CASES:
        violations.append("g35_permission_cases_incomplete")
    if _required_arguments(tree) != REQUIRED_ARGUMENTS:
        violations.append("exact_artifact_arguments_drift")

    campaign_calls = _called_names(_function(tree, "_campaign"))
    if not {
        "_read_release_manifest",
        "_sign_evidence",
        "_assert_evidence_safe",
        "_invoke_identity",
        "_test_catalog_sha256",
    } <= campaign_calls:
        violations.append("release_provenance_control_missing")
    optimizer = _function(tree, "_optimizer_worker")
    if not {
        "_expect_resource_limit",
        "_materialize_optimizer_artifacts",
        "program_optimization_result",
        "_forbidden_runtime_modules_loaded",
    } <= _called_names(optimizer) or _has_broad_suppression(optimizer):
        violations.append("optimizer_behavior_control_missing")
    provider_calls = _called_names(_function(tree, "_provider_worker"))
    if not {"_validate_provider_install_result", "_check_unified_install"} <= provider_calls:
        violations.append("provider_behavior_control_missing")
    work_item_calls = _called_names(_function(tree, "_work_item_bus_worker"))
    if not {
        "submit_work_item",
        "claim_specific",
        "claim_next",
        "heartbeat",
        "checkpoint_work_item",
        "commit_result",
        "commit_message_to_work_item",
    } <= work_item_calls:
        violations.append("work_item_bus_behavior_control_missing")
    work_item_worker = _function(tree, "_work_item_bus_worker")
    tenant_create_lines = [
        call.lineno
        for call in ast.walk(work_item_worker)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "create"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "tenants"
    ] if work_item_worker is not None else []
    adapter_lines = [
        call.lineno
        for call in ast.walk(work_item_worker)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_ExactWorkItemAdapter"
    ] if work_item_worker is not None else []
    if (
        len(tenant_create_lines) != 1
        or len(adapter_lines) != 1
        or tenant_create_lines[0] >= adapter_lines[0]
    ):
        violations.append("work_item_graph_bootstrap_order_invalid")
    permission_calls = _called_names(
        _function(tree, "_permission_governance_worker")
    )
    if not {
        "resolve_permission_context",
        "authorize_tool",
        "flag_mcp_tool_definitions",
        "markings_for",
        "ActionExecutor",
        "create_context_agent",
        "_build_graph_config",
        "PermissionsKernel",
    } <= permission_calls:
        violations.append("permission_governance_behavior_control_missing")
    worker_entry_calls = _called_names(_function(tree, "_worker_entry"))
    if not {
        "_work_item_bus_worker",
        "_permission_governance_worker",
    } <= worker_entry_calls:
        violations.append("exact_runtime_worker_entry_missing")
    intent_calls = _called_names(_function(tree, "_certify_intent_stdio"))
    if not {"_start_graphos", "_call_tool", "_tool_names"} <= intent_calls:
        violations.append("intent_behavior_control_missing")
    writer_calls = _called_names(_function(tree, "_write_evidence"))
    if not {"_open_output_parent", "link", "fsync"} <= writer_calls:
        violations.append("evidence_publication_control_missing")

    for token in REQUIRED_SOURCE_TOKENS:
        if token not in harness_text:
            violations.append("required_source_control_missing")
            break
    if "pytest.skip" in harness_text or "pytest.xfail" in harness_text:
        violations.append("runtime_skip_path_present")
    if "sys.path.insert(0, str(REPOSITORY_ROOT" in harness_text:
        violations.append("source_checkout_import_path_present")
    if _BUILD_PROGRAMS & _subprocess_programs(tree):
        violations.append("artifact_build_or_discovery_present")
    if _hardcoded_authority(tree):
        violations.append("hardcoded_runtime_authority_present")

    try:
        a2a_tree = ast.parse(a2a_text, filename="test_a2a_epistemic_live.py")
    except SyntaxError:
        violations.append("a2a_harness_syntax_invalid")
    else:
        crash_case = _function(
            a2a_tree,
            "test_native_crash_restart_recovers_and_fences_precrash_completion",
        )
        crash_controls = _called_names(crash_case) | _attribute_names(crash_case)
        if not {"Pipe", "Process", "kill", "crash", "restart"} <= crash_controls:
            violations.append("a2a_separate_process_crash_control_missing")

    for value in (
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ):
        if _LOCAL_LITERAL.search(value):
            violations.append("environment_specific_literal_present")
            break

    normalized_documentation = " ".join(documentation_text.split())
    for token in REQUIRED_DOCUMENTATION_TOKENS:
        if " ".join(token.split()) not in normalized_documentation:
            violations.append("documentation_contract_incomplete")
            break
    if navigation_text.count(NAV_ENTRY) != 1:
        violations.append("documentation_navigation_invalid")
    if workflow_text.count(CI_ENTRY) != 1:
        violations.append("continuous_gate_wiring_invalid")
    return tuple(dict.fromkeys(violations))


def main() -> int:
    try:
        violations = check_contract(
            HARNESS.read_text(encoding="utf-8"),
            DOCUMENTATION.read_text(encoding="utf-8"),
            NAVIGATION.read_text(encoding="utf-8"),
            WORKFLOW.read_text(encoding="utf-8"),
            A2A_TEST.read_text(encoding="utf-8"),
        )
    except OSError:
        violations = ("certification_source_unavailable",)
    if violations:
        print(
            "exact-local certification source gate failed: "
            f"{len(violations)} violation(s)"
        )
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("exact-local certification source gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
