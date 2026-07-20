#!/usr/bin/env python3
"""Check the current-only exact-artifact closure source contract."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "release" / "exact_local_gates_manifest.py"
BINDER = ROOT / "scripts" / "release" / "exact_artifact_closure.py"
COMPATIBILITY = ROOT / "scripts" / "release" / "check_compatibility.py"
CAMPAIGN = ROOT / "scripts" / "certification" / "exact_local_gates.py"
ORCHESTRATOR = ROOT / "scripts" / "certification" / "run_exact_engine_campaigns.py"
MANIFEST_SCHEMA = ROOT / "deploy" / "release" / "exact-local-gates-manifest.schema.json"
CLOSURE_SCHEMA = (
    ROOT / "deploy" / "release" / "exact-artifact-closure-evidence.schema.json"
)
RELEASE_SCHEMA = ROOT / "deploy" / "release" / "release-manifest.schema.json"
SOURCE_FREEZE = ROOT / "deploy" / "release" / "source-freeze-gates.json"
DOCUMENTATION = ROOT / "docs" / "release" / "exact-artifact-closure.md"
NAVIGATION = ROOT / "mkdocs.yml"
PROJECT = ROOT / "pyproject.toml"

EXPECTED_GATES = {
    "G-01",
    "G-02",
    "G-04",
    "G-05",
    "G-08",
    "G-09",
    "G-14",
    "G-15",
    "G-17",
    "G-26",
    "G-30",
    "G-32",
    "G-34",
    "G-35",
    "G-37",
}
EXPECTED_RELEASE_GATES = {
    "G-01",
    "G-02",
    "G-03",
    "G-04",
    "G-05",
    "G-06",
    "G-07",
    "G-08",
    "G-09",
    "G-11",
    "G-12",
    "G-13",
    "G-14",
    "G-15",
    "G-17",
    "G-18",
    "G-22",
    "G-25",
    "G-26",
    "G-27",
    "G-29",
    "G-30",
    "G-31",
    "G-32",
    "G-33",
    "G-34",
    "G-35",
    "G-36",
    "G-37",
    "G-38",
}
EXPECTED_CAMPAIGNS = {
    "faultRestart",
    "protocolAuthorization",
    "workItemAgentBus",
    "performance",
    "multimodal",
    "knowledgeBatch",
    "reasoningRepair",
    "exactLocal",
    "permissionGovernance",
}
EXPECTED_ENGINE_CAMPAIGN_ORDER = (
    "performance",
    "fault-restart",
    "protocol-authorization",
    "multimodal",
    "knowledge-batch",
    "reasoning-repair",
)
EXPECTED_ENGINE_PRODUCERS = {
    "performance": "certify_exact_performance.py",
    "fault-restart": "certify_exact_fault_restart.py",
    "protocol-authorization": "certify_exact_protocol_authorization.py",
    "multimodal": "certify_exact_multimodal.py",
    "knowledge-batch": "certify_exact_knowledge_batch.py",
    "reasoning-repair": "certify_exact_reasoning_repair.py",
}
EXPECTED_ENGINE_OUTPUTS = {
    "performance": "performance.json",
    "fault-restart": "fault-restart.json",
    "protocol-authorization": "protocol-authorization.json",
    "multimodal": "multimodal.json",
    "knowledge-batch": "knowledge-batch.json",
    "reasoning-repair": "reasoning-repair.json",
}
EXPECTED_ENGINE_TIMEOUTS = {
    "performance": 14_400,
    "fault-restart": 7_200,
    "protocol-authorization": 3_600,
    "multimodal": 7_200,
    "knowledge-batch": 3_600,
    "reasoning-repair": 3_600,
}
EXPECTED_MANIFEST_KEYS = {
    "agent_utilities_sha256",
    "distribution_closure_sha256",
    "engine_sha256",
    "evidence_schema_version",
    "graphos_sha256",
    "harness_sha256",
    "promotion_evidence_sha256",
    "release_id",
    "release_python_sha256",
    "release_spec_sha256",
    "schema_version",
    "test_catalog_sha256",
}


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            try:
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "frozenset"
                    and len(value.args) == 1
                    and not value.keywords
                ):
                    return frozenset(ast.literal_eval(value.args[0]))
                return ast.literal_eval(value)
            except (ValueError, TypeError):
                return None
    return None


def _hardened_orchestrator_process_boundary(tree: ast.Module) -> bool:
    popen_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "Popen"
    ]
    if len(popen_calls) != 1:
        return False
    keywords = {
        keyword.arg: keyword.value
        for keyword in popen_calls[0].keywords
        if keyword.arg is not None
    }

    def constant(name: str, expected: object) -> bool:
        value = keywords.get(name)
        return isinstance(value, ast.Constant) and value.value is expected

    environment = keywords.get("env")
    stdin = keywords.get("stdin")
    if (
        not constant("shell", False)
        or not constant("start_new_session", True)
        or not constant("close_fds", True)
        or not isinstance(environment, ast.Name)
        or environment.id != "environment"
        or not isinstance(stdin, ast.Attribute)
        or not isinstance(stdin.value, ast.Name)
        or stdin.value.id != "subprocess"
        or stdin.attr != "DEVNULL"
    ):
        return False
    if any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr == "environ"
        for node in ast.walk(tree)
    ):
        return False
    return any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "prefix"
            for target in node.targets
        )
        and isinstance(node.value, ast.List)
        and [item.value for item in node.value.elts if isinstance(item, ast.Constant)]
        == ["-E", "-s", "-B"]
        for node in ast.walk(tree)
    )


def check_contract(root: Path = ROOT) -> tuple[str, ...]:
    """Return stable findings for any release-closure source drift."""

    del root
    findings: list[str] = []
    required = (
        GENERATOR,
        BINDER,
        COMPATIBILITY,
        CAMPAIGN,
        ORCHESTRATOR,
        MANIFEST_SCHEMA,
        CLOSURE_SCHEMA,
        RELEASE_SCHEMA,
        SOURCE_FREEZE,
        DOCUMENTATION,
        NAVIGATION,
        PROJECT,
    )
    if any(not path.is_file() or path.is_symlink() for path in required):
        return ("closure-file-inventory",)
    try:
        generator_tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
        binder_tree = ast.parse(BINDER.read_text(encoding="utf-8"))
        compatibility_tree = ast.parse(COMPATIBILITY.read_text(encoding="utf-8"))
        campaign_tree = ast.parse(CAMPAIGN.read_text(encoding="utf-8"))
        orchestrator_tree = ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"))
        manifest_schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
        closure_schema = json.loads(CLOSURE_SCHEMA.read_text(encoding="utf-8"))
        release_schema = json.loads(RELEASE_SCHEMA.read_text(encoding="utf-8"))
        source_freeze = json.loads(SOURCE_FREEZE.read_text(encoding="utf-8"))
        project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        SyntaxError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ):
        return ("closure-source-unreadable",)

    generator_keys = set(_literal_assignment(generator_tree, "MANIFEST_KEYS") or ())
    campaign_keys = set(
        _literal_assignment(campaign_tree, "RELEASE_MANIFEST_KEYS") or ()
    )
    if (
        generator_keys != EXPECTED_MANIFEST_KEYS
        or campaign_keys != EXPECTED_MANIFEST_KEYS
    ):
        findings.append("closure-manifest-key-drift")
    generator_tests = tuple(
        _literal_assignment(generator_tree, "EXACT_LOCAL_TEST_FILES") or ()
    )
    campaign_tests = tuple(
        _literal_assignment(campaign_tree, "CERTIFICATION_TESTS") or ()
    )
    if not generator_tests or generator_tests != campaign_tests:
        findings.append("closure-test-catalog-drift")
    binder_gates = set(_literal_assignment(binder_tree, "GATES") or ())
    if binder_gates != EXPECTED_GATES:
        findings.append("closure-gate-inventory")
    compatibility_gates = set(
        _literal_assignment(compatibility_tree, "_EXACT_ARTIFACT_GATES") or ()
    )
    compatibility_authorities = _literal_assignment(
        compatibility_tree, "_EXACT_GATE_AUTHORITIES"
    )
    source_exact_gates = {
        str(gate.get("id") or "")
        for gate in (
            source_freeze.get("gates", []) if isinstance(source_freeze, dict) else ()
        )
        if isinstance(gate, dict)
        and "exact-artifact" in (gate.get("evidence_classes") or ())
    }
    release_gate_schema = release_schema.get("properties", {}).get(
        "exactGateEvidence", {}
    )
    release_gate_properties = set(release_gate_schema.get("properties") or {})
    release_gate_required = set(release_gate_schema.get("required") or ())
    if (
        compatibility_gates != EXPECTED_GATES
        or not isinstance(compatibility_authorities, dict)
        or set(compatibility_authorities) != EXPECTED_RELEASE_GATES
        or source_exact_gates != EXPECTED_RELEASE_GATES
        or release_gate_properties != EXPECTED_RELEASE_GATES
        or release_gate_required != EXPECTED_RELEASE_GATES
        or any(
            "certification:exactArtifactClosureEvidence"
            not in compatibility_authorities.get(gate, ())
            for gate in EXPECTED_GATES
        )
    ):
        findings.append("closure-release-gate-inventory")
    if (
        tuple(_literal_assignment(orchestrator_tree, "CAMPAIGN_ORDER") or ())
        != EXPECTED_ENGINE_CAMPAIGN_ORDER
        or _literal_assignment(orchestrator_tree, "PRODUCER_SCRIPTS")
        != EXPECTED_ENGINE_PRODUCERS
        or _literal_assignment(orchestrator_tree, "OUTPUT_FILES")
        != EXPECTED_ENGINE_OUTPUTS
        or _literal_assignment(orchestrator_tree, "CAMPAIGN_TIMEOUT_SECONDS")
        != EXPECTED_ENGINE_TIMEOUTS
        or _literal_assignment(orchestrator_tree, "_MAX_CHILD_OUTPUT_BYTES")
        != 1024 * 1024
        or not _hardened_orchestrator_process_boundary(orchestrator_tree)
    ):
        findings.append("closure-engine-campaign-orchestrator")

    if (
        manifest_schema.get("additionalProperties") is not False
        or set(manifest_schema.get("required") or ()) != EXPECTED_MANIFEST_KEYS
        or manifest_schema.get("properties", {}).get("schema_version", {}).get("const")
        != 1
    ):
        findings.append("closure-manifest-schema")
    closure_properties = closure_schema.get("properties") or {}
    closure_gates = closure_properties.get("gates", {}).get("properties", {})
    closure_campaigns = closure_properties.get("campaigns", {}).get("properties", {})
    if (
        closure_schema.get("additionalProperties") is not False
        or closure_properties.get("kind", {}).get("const")
        != "ExactArtifactClosureEvidence"
        or set(closure_properties.get("gates", {}).get("required") or ())
        != EXPECTED_GATES
        or set(closure_gates) != EXPECTED_GATES
        or set(closure_campaigns) != EXPECTED_CAMPAIGNS
        or any(item.get("const") != "passed" for item in closure_gates.values())
    ):
        findings.append("closure-evidence-schema")

    scripts = project.get("project", {}).get("scripts", {})
    if scripts.get("generate-exact-local-gates-manifest") != (
        "scripts.release.exact_local_gates_manifest:main"
    ):
        findings.append("closure-generator-entry-point")
    if scripts.get("bind-exact-local-release-evidence") != (
        "scripts.release.exact_artifact_closure:main"
    ):
        findings.append("closure-binder-entry-point")

    documentation = DOCUMENTATION.read_text(encoding="utf-8")
    navigation = NAVIGATION.read_text(encoding="utf-8")
    required_terms = (
        "generate-exact-local-gates-manifest",
        "bind-exact-local-release-evidence",
        "run_exact_engine_campaigns.py",
        "EXACT_ARTIFACT_CLOSURE_SIGNER_COMMAND",
        "EXACT_ARTIFACT_CLOSURE_VERIFIER_COMMAND",
        "G-01",
        "G-02",
        "G-37",
        "performance.json",
    )
    if any(term not in documentation for term in required_terms):
        findings.append("closure-documentation")
    if "release/exact-artifact-closure.md" not in navigation:
        findings.append("closure-navigation")
    return tuple(sorted(set(findings)))


def main() -> int:
    findings = check_contract()
    if findings:
        print(f"exact artifact closure contract: FAIL ({len(findings)} findings)")
        return 1
    print("exact artifact closure contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
