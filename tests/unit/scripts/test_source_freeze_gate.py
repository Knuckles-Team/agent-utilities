"""Focused contract tests for the deterministic source-freeze runner."""

from __future__ import annotations

import builtins
import copy
import json
import os
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from scripts import source_freeze_gate as gate


def _roots(tmp_path: Path) -> dict[str, Path]:
    fleet = tmp_path / "provider-fleet"
    roots = {
        "agent-utilities": fleet / "agent-utilities",
        "epistemic-graph": fleet / "epistemic-graph",
        "langfuse-agent": fleet / "agents" / "langfuse-agent",
        "provider-fleet": fleet,
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    (fleet / "skills").mkdir()
    manager = fleet / "agents" / "repository-manager" / "repository_manager"
    manager.mkdir(parents=True)
    (manager / "workspace.yml").write_text("repositories: []\n", encoding="utf-8")
    for identifier, package in (
        ("agent-utilities", "agent_utilities"),
        ("langfuse-agent", "langfuse_agent"),
    ):
        (roots[identifier] / "pyproject.toml").write_text(
            "[project]\nname='fixture'\n", encoding="utf-8"
        )
        (roots[identifier] / package).mkdir()
        (roots[identifier] / "scripts").mkdir()
    (roots["epistemic-graph"] / "Cargo.toml").write_text(
        "[package]\nname='fixture'\n", encoding="utf-8"
    )
    (roots["epistemic-graph"] / "crates").mkdir()
    (roots["epistemic-graph"] / "scripts").mkdir()
    roots = {identifier: root.resolve() for identifier, root in roots.items()}
    return roots


def _manifest_value(
    script: str = "check_fixture.py", *, command_count: int = 1
) -> dict[str, Any]:
    command_ids = [f"fixture-command-{index}" for index in range(command_count)]
    commands = [
        {
            "id": identifier,
            "repository": "agent-utilities",
            "argv": ["{python}", f"scripts/{script}"],
            "timeout_seconds": 15,
            "covers": ["G-02"],
        }
        for identifier in command_ids
    ]
    gates: list[dict[str, object]] = []
    for identifier in gate.EXPECTED_GATES:
        evidence_classes = (
            ["local-source", "exact-artifact"] if identifier == "G-02" else ["external"]
        )
        if identifier == "G-39":
            evidence_classes = ["terminal"]
        gates.append(
            {
                "id": identifier,
                "evidence_classes": evidence_classes,
                "scope": ["agent-utilities"],
                "command_ids": command_ids if identifier == "G-02" else [],
                "rationale": "Synthetic source-gate contract.",
            }
        )
    return {
        "schema": gate.MANIFEST_SCHEMA,
        "repositories": [
            {"id": "agent-utilities", "kind": "repository"},
            {"id": "epistemic-graph", "kind": "repository"},
            {"id": "langfuse-agent", "kind": "repository"},
            {"id": "provider-fleet", "kind": "fleet"},
        ],
        "commands": commands,
        "gates": gates,
    }


def _load(tmp_path: Path, value: dict[str, Any]) -> gate.Manifest:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return gate.load_manifest(path)


def _script(root: Path, source: str) -> None:
    scripts = root / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "check_fixture.py").write_text(source, encoding="utf-8")


def test_checked_in_manifest_is_complete_and_current() -> None:
    manifest = gate.load_manifest(
        Path(__file__).resolve().parents[3]
        / "deploy"
        / "release"
        / "source-freeze-gates.json"
    )

    assert tuple(item.identifier for item in manifest.gates) == gate.EXPECTED_GATES
    assert manifest.digest == gate.CANONICAL_MANIFEST_SHA256
    assert len(manifest.commands) == 49
    assert all(item.evidence_classes for item in manifest.gates)
    assert any("exact-artifact" in item.evidence_classes for item in manifest.gates)
    assert any(
        {"local-source", "external"} <= set(item.evidence_classes)
        for item in manifest.gates
    )
    assert manifest.gates[-1].evidence_classes == ("terminal",)
    assert {item.repository for item in manifest.commands} >= {
        "agent-utilities",
        "epistemic-graph",
        "langfuse-agent",
    }
    commands = {item.identifier: item for item in manifest.commands}
    gates = {item.identifier: item for item in manifest.gates}
    assert gates["G-01"].evidence_classes == (
        "local-source",
        "exact-artifact",
        "external",
    )
    assert set(gates["G-01"].command_ids) == {
        "au-tenant-identity-contract",
        "eg-read-policy",
        "eg-current-only",
        "eg-release-harness-contract",
    }
    assert commands["au-release-source-contract"].argv == (
        "{python}",
        "scripts/release/check_release_catalogs.py",
    )
    assert "{repo:provider-fleet}/agents" in commands["au-connector-manifests"].argv
    assert commands["au-ontology-library"].argv[-2:] == (
        "--provider-root",
        "{repo:provider-fleet}/agents",
    )
    assert commands["au-fleet-supply-chain"].argv == (
        "{python}",
        "scripts/check_fleet_supply_chain.py",
        "--source-snapshot-root",
        "{repo:provider-fleet}/agents",
        "--snapshot-workspace",
        "{repo:provider-fleet}/agents/repository-manager/repository_manager/workspace.yml",
    )
    assert "{repo:provider-fleet}" in commands["au-skill-collisions"].argv
    assert "--strict" in commands["au-skill-collisions"].argv


@pytest.mark.parametrize("missing", ["rdflib", "pyshacl", "owlrl"])
def test_ontology_gate_requires_every_validator(
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import check_ontology

    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] == missing:
            raise ImportError("simulated unavailable validator")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert check_ontology.check() == 1
    assert capsys.readouterr().out == (
        "check_ontology: required validation dependencies unavailable; "
        "failing closed.\n"
    )


def test_canonical_manifest_pin_rejects_semantically_valid_byte_drift(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "deploy"
        / "release"
        / "source-freeze-gates.json"
    )
    changed = tmp_path / "source_freeze_gates.json"
    changed.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(gate.GateError, match="manifest-pin"):
        gate.load_canonical_manifest(changed)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value["gates"].pop(), "manifest-gate-set"),
        (
            lambda value: value["gates"].append(copy.deepcopy(value["gates"][-1])),
            "manifest-gate-set",
        ),
        (
            lambda value: value["commands"].append(
                {
                    "id": "orphan-command",
                    "repository": "agent-utilities",
                    "argv": ["{python}", "scripts/check_fixture.py"],
                    "timeout_seconds": 15,
                    "covers": ["G-03"],
                }
            ),
            "manifest-command-orphan",
        ),
        (
            lambda value: value["gates"][1]["command_ids"].append("not-listed"),
            "manifest-command-unlisted",
        ),
        (
            lambda value: value["commands"][0]["argv"].append("--live"),
            "manifest-command-forbidden",
        ),
    ],
)
def test_manifest_fails_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    code: str,
) -> None:
    value = _manifest_value()
    mutation(value)

    with pytest.raises(gate.GateError, match=code):
        _load(tmp_path, value)


def test_repository_roots_are_explicit_and_non_symlinked(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    arguments = [
        f"{identifier}={roots[identifier]}" for identifier in gate.REPOSITORY_IDS
    ]

    assert gate.parse_repository_roots(arguments) == roots
    with pytest.raises(gate.GateError, match="repository-set"):
        gate.parse_repository_roots(arguments[:-1])
    link = tmp_path / "linked-root"
    link.symlink_to(roots["agent-utilities"], target_is_directory=True)
    linked = [
        f"{identifier}={link if identifier == 'agent-utilities' else roots[identifier]}"
        for identifier in gate.REPOSITORY_IDS
    ]
    with pytest.raises(gate.GateError, match="repository-type"):
        gate.parse_repository_roots(linked)


def test_success_evidence_contains_no_paths_or_command_output(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _script(
        roots["agent-utilities"],
        "import os\nprint(os.environ['SOURCE_FREEZE_PROTECTED_ROOTS'])\n",
    )
    manifest = _load(tmp_path, _manifest_value())
    evidence_path = tmp_path / "evidence.json"

    evidence = gate.execute_manifest(manifest, roots, evidence_path)

    retained = evidence_path.read_text(encoding="utf-8")
    assert evidence["status"] == "passed"
    assert evidence["source_digest_before"] == evidence["source_digest_after"]
    assert [item["id"] for item in evidence["tools"]] == ["git", "rg"]
    assert len(evidence["gates"]) == 39
    assert str(tmp_path) not in retained
    assert "SOURCE_FREEZE_PROTECTED_ROOTS" not in retained
    assert "argv" not in retained
    assert "duration_ms" not in retained
    assert "stdout_sha256" not in retained
    assert (
        evidence["commands"][0]["source_digest_before"]
        == evidence["commands"][0]["source_digest_after"]
    )
    assert evidence["gates"][1]["source_status"] == "passed"
    assert evidence["gates"][1]["remaining_evidence"] == ["exact-artifact"]
    assert evidence["gates"][-1]["source_status"] == "not-applicable"
    assert stat_mode(evidence_path) == 0o600


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_existing_evidence_fails_before_command_execution(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _script(roots["agent-utilities"], "raise SystemExit(77)\n")
    manifest = _load(tmp_path, _manifest_value())
    evidence = tmp_path / "evidence.json"
    evidence.write_text("owned", encoding="utf-8")

    with pytest.raises(gate.GateError, match="evidence-output-exists"):
        gate.execute_manifest(manifest, roots, evidence)
    assert evidence.read_text(encoding="utf-8") == "owned"


def test_evidence_parent_must_be_owned_private_and_symlink_free(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    _script(roots["agent-utilities"], "raise AssertionError('not executed')\n")
    manifest = _load(tmp_path, _manifest_value())
    public = tmp_path / "public"
    public.mkdir(mode=0o755)

    with pytest.raises(gate.GateError, match="evidence-parent-security"):
        gate.execute_manifest(manifest, roots, public / "evidence.json")

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    linked = tmp_path / "linked-evidence"
    linked.symlink_to(private, target_is_directory=True)
    with pytest.raises(gate.GateError, match="evidence-parent"):
        gate.execute_manifest(manifest, roots, linked / "evidence.json")


def test_evidence_is_complete_and_private_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "private-evidence"
    parent.mkdir(mode=0o700)
    evidence = parent / "evidence.json"
    target = gate._validate_evidence_target(evidence, {})
    real_link = gate.os.link
    observed = False

    def inspect_link(source: str, destination: str, **kwargs: Any) -> None:
        nonlocal observed
        temporary = parent / source
        assert destination == evidence.name
        assert not evidence.exists()
        assert stat_mode(temporary) == 0o600
        assert json.loads(temporary.read_text(encoding="utf-8")) == {"status": "passed"}
        observed = True
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(gate.os, "link", inspect_link)
    try:
        gate._write_exclusive(target, {"status": "passed"})
    finally:
        target.close()

    assert observed is True
    assert json.loads(evidence.read_text(encoding="utf-8")) == {"status": "passed"}
    assert stat_mode(evidence) == 0o600
    assert not tuple(parent.glob(".source-freeze-*.tmp"))


@pytest.mark.parametrize(
    "source",
    [
        "from pathlib import Path\nPath(__file__).with_name('edit').write_text('x')\n",
        "import socket\nsocket.socket()\n",
        "import subprocess\nsubprocess.run(['cargo', 'build'], check=True)\n",
        "import ctypes\nctypes.CDLL('libc.so.6')\n",
    ],
)
def test_process_guard_rejects_edits_network_and_builds(
    tmp_path: Path, source: str
) -> None:
    roots = _roots(tmp_path)
    _script(roots["agent-utilities"], source)
    manifest = _load(tmp_path, _manifest_value())

    with pytest.raises(gate.GateError, match="source-command-failed"):
        gate.execute_manifest(manifest, roots, tmp_path / "evidence.json")


def test_isolated_bootstrap_never_executes_unbound_site_startup_code() -> None:
    command = gate._bootstrap_command(
        ("/reviewed/python", "scripts/check_fixture.py"), Path("/private/guard")
    )
    source = gate._PROCESS_GUARD + gate._BOOTSTRAP

    assert command[:4] == (
        "/reviewed/python",
        "-S",
        "-B",
        "/private/guard/bootstrap.py",
    )
    assert source.index("sys.addaudithook(_audit)") < source.index("sys.path.append")
    assert "site.addsitedir" not in source
    assert "import site" not in gate._BOOTSTRAP


def test_process_guard_allows_standard_library_ctypes_handle(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _script(roots["agent-utilities"], "import ctypes\nassert ctypes.pythonapi\n")
    manifest = _load(tmp_path, _manifest_value())

    evidence = gate.execute_manifest(manifest, roots, tmp_path / "evidence.json")

    assert evidence["status"] == "passed"


def test_site_paths_include_active_venv_without_executing_pth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "venv"
    executable = prefix / (
        "Scripts/python.exe" if gate.os.name == "nt" else "bin/python"
    )
    executable.parent.mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text("home = isolated\n", encoding="utf-8")
    scheme = "nt_venv" if gate.os.name == "nt" else "posix_venv"
    if scheme not in sysconfig.get_scheme_names():
        scheme = "venv"
    value = str(prefix)
    variables = {
        "base": value,
        "platbase": value,
        "installed_base": value,
        "installed_platbase": value,
        "prefix": value,
        "exec_prefix": value,
    }
    expected = Path(sysconfig.get_paths(scheme=scheme, vars=variables)["purelib"])
    expected.mkdir(parents=True)
    monkeypatch.setattr(gate.sys, "executable", str(executable))

    assert str(expected.resolve()) in gate._site_package_paths()


def test_command_environment_disables_unreviewed_pydantic_plugins(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    guard = tmp_path / "guard"
    guard.mkdir()
    environment = gate._command_environment(
        guard,
        roots["agent-utilities"],
        roots,
        {"git": guard / "git", "rg": guard / "rg"},
    )

    assert environment["PYDANTIC_DISABLE_PLUGINS"] == "__all__"


def test_process_guard_allows_only_reviewed_rg_read(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _script(
        roots["agent-utilities"],
        "import subprocess\nsubprocess.run(['rg', '--files', 'scripts'], check=True)\n",
    )
    manifest = _load(tmp_path, _manifest_value())

    evidence = gate.execute_manifest(manifest, roots, tmp_path / "evidence.json")

    assert evidence["status"] == "passed"
    assert evidence["commands"][0]["termination"] == "exited"


def test_output_is_bounded_without_retaining_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _script(roots["agent-utilities"], "print('x' * 4096)\n")
    manifest = _load(tmp_path, _manifest_value())
    monkeypatch.setattr(gate, "_MAX_OUTPUT_BYTES", 1024)
    evidence_path = tmp_path / "evidence.json"

    with pytest.raises(gate.GateError, match="source-command-failed"):
        gate.execute_manifest(manifest, roots, evidence_path)

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["commands"][0]["termination"] == "output-limit"
    assert "xxxx" not in evidence_path.read_text(encoding="utf-8")


def test_release_cli_rejects_manifest_override() -> None:
    with pytest.raises(SystemExit):
        gate.main(["--manifest", "unreviewed.json"])


def test_release_cli_requires_isolated_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "_runner_isolated", lambda: False)

    assert gate.main(["--evidence", str(tmp_path / "evidence.json")]) == 1


def test_post_run_digest_detects_an_external_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _script(roots["agent-utilities"], "raise AssertionError('not executed')\n")
    manifest = _load(tmp_path, _manifest_value())

    def edit_during_run(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        (roots["epistemic-graph"] / "changed.py").write_text(
            "changed", encoding="utf-8"
        )
        return {"exit_code": 0, "termination": "exited"}

    monkeypatch.setattr(gate, "_run_bounded", edit_during_run)

    with pytest.raises(gate.GateError, match="source-tree-edited"):
        gate.execute_manifest(manifest, roots, tmp_path / "evidence.json")


def test_commands_are_serial_and_stop_at_first_failure(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _script(roots["agent-utilities"], "raise SystemExit(9)\n")
    manifest = _load(tmp_path, _manifest_value(command_count=2))
    evidence_path = tmp_path / "evidence.json"

    with pytest.raises(gate.GateError, match="source-command-failed"):
        gate.execute_manifest(manifest, roots, evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in evidence["commands"]] == ["fixture-command-0"]
    assert evidence["status"] == "failed"


def test_source_digest_is_root_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "source.py").write_text("value = 1\n", encoding="utf-8")
    (second / "source.py").write_text("value = 1\n", encoding="utf-8")

    assert gate.source_tree_digest(first) == gate.source_tree_digest(second)
    (second / "source.py").write_text("value = 2\n", encoding="utf-8")
    assert gate.source_tree_digest(first) != gate.source_tree_digest(second)


def test_source_digest_includes_empty_and_artifact_named_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    initial = gate.source_tree_digest(root)
    (root / "target").mkdir()
    with_empty = gate.source_tree_digest(root)
    (root / "target" / "tracked.rs").write_text("pub fn value() {}\n", encoding="utf-8")

    assert initial != with_empty
    assert with_empty != gate.source_tree_digest(root)


def test_source_digest_excludes_runtime_only_directories(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    initial = gate.source_tree_digest(root)

    for name in (
        ".acp-sessions",
        ".benchmarks",
        ".pytest_tmp",
        ".worktrees",
        "workspace",
    ):
        runtime = root / name
        runtime.mkdir()
        (runtime / "private-state.json").write_text("runtime\n", encoding="utf-8")

    assert gate.source_tree_digest(root) == initial


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX special files")
def test_pytest_tmp_is_not_traversed_for_symlink_or_special_fixtures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    initial = gate.source_tree_digest(root)

    pytest_tmp = root / ".pytest_tmp"
    pytest_tmp.mkdir()
    (pytest_tmp / "external-fixture").symlink_to(outside, target_is_directory=True)
    os.mkfifo(pytest_tmp / "special-fixture")

    assert gate.source_tree_digest(root) == initial


def test_json_schemas_are_strict_and_match_current_formats() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest_schema = json.loads(
        (root / "deploy" / "release" / "source-freeze-gates.schema.json").read_text()
    )
    evidence_schema = json.loads(
        (root / "deploy" / "release" / "source-freeze-evidence.schema.json").read_text()
    )

    assert manifest_schema["additionalProperties"] is False
    assert manifest_schema["properties"]["schema"]["const"] == gate.MANIFEST_SCHEMA
    assert manifest_schema["properties"]["gates"]["minItems"] == 39
    gate_schema = manifest_schema["$defs"]["gate"]
    assert "evidence_classes" in gate_schema["required"]
    assert "classification" not in gate_schema["properties"]
    assert evidence_schema["additionalProperties"] is False
    assert evidence_schema["properties"]["schema"]["const"] == gate.EVIDENCE_SCHEMA
    evidence_gates = evidence_schema["properties"]["gates"]
    assert evidence_gates["minItems"] == len(gate.EXPECTED_GATES)
    assert evidence_gates["maxItems"] == len(gate.EXPECTED_GATES)
    assert [
        item["properties"]["id"]["const"]
        for item in evidence_gates["prefixItems"]
    ] == list(gate.EXPECTED_GATES)
    evidence_gate = evidence_gates["items"]
    assert "source_status" in evidence_gate["required"]
    assert "remaining_evidence" in evidence_gate["required"]


def test_evidence_schema_accepts_only_the_executor_gate_sequence() -> None:
    root = Path(__file__).resolve().parents[3]
    schema = json.loads(
        (root / "deploy" / "release" / "source-freeze-evidence.schema.json").read_text()
    )
    validator = Draft202012Validator(schema)
    digest = "1" * 64
    evidence = {
        "schema": gate.EVIDENCE_SCHEMA,
        "status": "passed",
        "manifest_sha256": digest,
        "source_digest_before": digest,
        "source_digest_after": digest,
        "tools": [
            {"id": "git", "sha256": digest},
            {"id": "rg", "sha256": digest},
        ],
        "repositories": [
            {
                "id": identifier,
                "sha256_before": digest,
                "sha256_after": digest,
            }
            for identifier in gate.REPOSITORY_IDS
        ],
        "commands": [],
        "gates": [
            {
                "id": identifier,
                "required_evidence": ["terminal"]
                if identifier == "G-39"
                else ["external"],
                "source_status": "not-applicable",
                "remaining_evidence": ["terminal"]
                if identifier == "G-39"
                else ["external"],
            }
            for identifier in gate.EXPECTED_GATES
        ],
    }
    validator.validate(evidence)

    missing_g01 = copy.deepcopy(evidence)
    missing_g01["gates"] = missing_g01["gates"][1:]
    with pytest.raises(ValidationError):
        validator.validate(missing_g01)

    reordered = copy.deepcopy(evidence)
    reordered["gates"][0], reordered["gates"][1] = (
        reordered["gates"][1],
        reordered["gates"][0],
    )
    with pytest.raises(ValidationError):
        validator.validate(reordered)
