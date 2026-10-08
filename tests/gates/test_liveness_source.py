"""Opt-in source comparisons must not turn unresolved expiry into a clean gate."""

import importlib.util
import subprocess
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "source_liveness",
    Path(__file__).resolve().parents[2] / "scripts/check_liveness_source.py",
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
ALLOWED = "tests/unit/scripts/test_security_contract_runner.py"
EXPIRED = "dead_definitions\tagent_utilities/deferred.py\t# owner=@owner review-by=2026-10-01\n"


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    path = tmp_path / ALLOWED
    path.parent.mkdir(parents=True)
    path.write_text("# before\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/liveness_deferred.tsv").write_text(EXPIRED)
    git(tmp_path, "add", ALLOWED, "scripts/liveness_deferred.tsv")
    git(tmp_path, "commit", "-qm", "fixture")
    base = git(tmp_path, "rev-parse", "HEAD")
    path.write_text("# after\n")
    git(tmp_path, "add", ALLOWED)
    return tmp_path, base, git(tmp_path, "write-tree")


def report(**findings):
    details = {
        category: findings.get(category, [])
        for category in gate._strict_gate().CATEGORIES
    }
    return {
        "counts": {key: len(value) for key, value in details.items()},
        "details": details,
        "coverage": False,
        "semantic": {},
    }


def test_scope_and_materialization_use_frozen_index_not_working_tree(repository):
    repo, base, candidate = repository
    (repo / ALLOWED).write_text("# unstaged diversion\n")
    gate._scope(repo, base, candidate)
    with gate._snapshot(repo, base, candidate) as snapshot:
        assert (snapshot / ALLOWED).read_text() == "# after\n"
        assert git(snapshot, "write-tree") == candidate
    assert not snapshot.exists()
    assert (repo / ALLOWED).read_text() == "# unstaged diversion\n"


@pytest.mark.parametrize(
    "path",
    [
        "agent_utilities/deferred.py",
        "agent_utilities/core/registry/service_adapter.py",
        "scripts/liveness_reconciler.py",
        "scripts/liveness_deferred.tsv",
        "scripts/check_liveness.py",
        ".github/workflows/release.yml",
        "scripts/release/verify.py",
    ],
)
def test_protected_paths_require_strict_gate(repository, path):
    repo, base, _ = repository
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("protected\n")
    git(repo, "add", path)
    git(repo, "commit", "-qm", "protected fixture")
    base = git(repo, "rev-parse", "HEAD")
    target.write_text("modified protected\n")
    git(repo, "add", path)
    assert gate._requires_strict(
        gate._scope(repo, base, git(repo, "write-tree")), EXPIRED
    )


@pytest.mark.parametrize("mutation", ["delete", "rename", "symlink", "executable"])
def test_path_or_mode_changes_cannot_hide_symlinks(repository, mutation):
    repo, base, _ = repository
    path = repo / ALLOWED
    if mutation == "delete":
        git(repo, "rm", "-f", ALLOWED)
    elif mutation == "rename":
        git(repo, "mv", ALLOWED, ALLOWED + ".renamed")
    elif mutation == "symlink":
        path.unlink()
        path.symlink_to("elsewhere")
        git(repo, "add", ALLOWED)
    else:
        git(repo, "update-index", "--chmod=+x", ALLOWED)
    if mutation == "symlink":
        with pytest.raises(gate.CannotRun, match="unsupported"):
            gate._scope(repo, base, git(repo, "write-tree"))
    else:
        assert ALLOWED in gate._scope(repo, base, git(repo, "write-tree"))


def test_missing_or_mutable_object_is_not_a_baseline(repository):
    repo, base, candidate = repository
    for bad_base in ("HEAD", "0" * 40, candidate):
        with pytest.raises(gate.CannotRun):
            gate._scope(repo, bad_base, candidate)
    assert gate._scope(repo, base, base) == []


def test_equal_counts_cannot_hide_new_finding(capsys):
    assert gate._compare(
        report(dead_definitions=["old:one"]), report(dead_definitions=["new:two"])
    )
    assert "NEW dead_definitions: new:two" in capsys.readouterr().out


def test_existing_absolute_violation_still_fails():
    existing = report(never_executed=["module:handler"])
    assert gate._compare(existing, existing)


def test_coverage_mismatch_cannot_pass():
    after = report()
    after["coverage"] = True
    with pytest.raises(gate.CannotRun, match="coverage"):
        gate._compare(report(), after)


@pytest.mark.parametrize(
    "text", ["not tab separated", "dead_definitions\tx.py\t# owner=@x\n"]
)
def test_malformed_obligations_are_never_accepted(text):
    with pytest.raises(gate.CannotRun):
        gate._deferrals(text, date(2026, 10, 8))


def test_expiry_transition_is_evaluated_at_current_date():
    assert gate._deferrals(EXPIRED, date(2026, 10, 1)) == []
    assert len(gate._deferrals(EXPIRED, date(2026, 10, 2))) == 1


@pytest.fixture
def comparison(repository, monkeypatch, tmp_path):
    repo, base, candidate = repository
    analyzer = tmp_path / "analyzer.py"
    analyzer.write_text("# detector\n")
    snapshot = tmp_path / "snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "scripts/liveness_deferred.tsv").write_text(EXPIRED)

    @contextmanager
    def materialize(*args):
        yield snapshot

    monkeypatch.setattr(gate, "_snapshot", materialize)
    monkeypatch.setattr(gate, "_remote_main", lambda _: base)
    monkeypatch.setattr(gate._strict_gate(), "_find_analyzer", lambda: analyzer)
    monkeypatch.setattr(gate, "_scan", lambda *_: report(dead_definitions=["old:one"]))
    return repo, base, candidate, analyzer


def test_unchanged_expiry_is_visible_not_clean_approval(comparison, capsys):
    assert gate.run(*comparison[:3]) == 0
    output = capsys.readouterr().out
    assert "UNRESOLVED EXPIRY:" in output
    assert "review-by=2026-10-01" in output
    assert "not strict/release liveness approval" in output


@pytest.mark.parametrize("heads", [("stale",), (None, "moved")])
def test_stale_or_moving_remote_main_fails(comparison, monkeypatch, heads):
    sequence = iter(comparison[1] if head is None else head for head in heads)
    monkeypatch.setattr(gate, "_remote_main", lambda _: next(sequence))
    with pytest.raises(gate.CannotRun, match="main"):
        gate.run(*comparison[:3])


def test_missing_analyzer_fails_closed(comparison, monkeypatch):
    monkeypatch.setattr(gate._strict_gate(), "_find_analyzer", lambda: None)
    with pytest.raises(gate.CannotRun, match="analyzer"):
        gate.run(*comparison[:3])


def test_analyzer_change_fails_closed(comparison, monkeypatch):
    def scan(*_):
        comparison[3].write_text("# changed detector\n")
        return report()

    monkeypatch.setattr(gate, "_scan", scan)
    with pytest.raises(gate.CannotRun, match="analyzer changed"):
        gate.run(*comparison[:3])


def test_midnight_requires_recomputing_both_snapshots(comparison, monkeypatch):
    dates = iter([date(2026, 10, 8), date(2026, 10, 9)])

    class Clock:
        today = staticmethod(lambda: next(dates))

    monkeypatch.setattr(gate, "date", Clock)
    with pytest.raises(gate.CannotRun, match="date changed"):
        gate.run(*comparison[:3])


@pytest.mark.parametrize(
    "result",
    [
        subprocess.CompletedProcess([], 2, "", "detector failed"),
        subprocess.CompletedProcess([], 0, "{}", ""),
        subprocess.CompletedProcess([], 0, "not json", ""),
    ],
)
def test_failed_or_incomplete_analyzer_cannot_pass(monkeypatch, tmp_path, result):
    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **kw: result)
    with pytest.raises(gate.CannotRun):
        gate._scan(tmp_path, tmp_path / "analyzer.py")


def test_default_strict_gate_still_rejects_expiry(monkeypatch):
    strict = gate._strict_gate()
    monkeypatch.setattr(strict, "_find_analyzer", lambda: Path("detector"))
    monkeypatch.setattr(strict, "_import_analyzer", lambda _: object())
    monkeypatch.setattr(strict, "repo_root", lambda: str(gate.REPO))
    monkeypatch.setattr(strict, "_run_census", lambda _: ({}, {}, False))
    monkeypatch.setattr(strict, "_enforce_diff", lambda *_: False)
    monkeypatch.setattr(
        strict.liveness_deferred,
        "load_entries",
        lambda: strict.liveness_deferred.parse_entries(EXPIRED),
    )
    assert strict.main([]) == 1


def test_source_cli_does_not_offer_release_or_census_bypass():
    with pytest.raises(SystemExit) as exc:
        gate.main(["--release"])
    assert exc.value.code == 2


def test_documentation_repair_is_eligible(repository):
    repo, _, _ = repository
    path = repo / "docs/scaling/scale_claims.md"
    path.parent.mkdir(parents=True)
    path.write_text("before\n")
    git(repo, "add", "docs/scaling/scale_claims.md")
    git(repo, "commit", "-qm", "documentation fixture")
    base = git(repo, "rev-parse", "HEAD")
    path.write_text("after\n")
    git(repo, "add", "docs/scaling/scale_claims.md")
    gate._scope(repo, base, git(repo, "write-tree"))


def test_candidate_cannot_change_expiry_obligations(comparison, monkeypatch, tmp_path):
    paths = []
    for index, text in enumerate(
        [EXPIRED, EXPIRED.replace("2026-10-01", "2027-10-01")]
    ):
        root = tmp_path / f"side-{index}"
        (root / "scripts").mkdir(parents=True)
        (root / "scripts/liveness_deferred.tsv").write_text(text)
        paths.append(root)
    sequence = iter(paths)

    @contextmanager
    def snapshot(*_):
        yield next(sequence)

    monkeypatch.setattr(gate, "_snapshot", snapshot)
    with pytest.raises(gate.CannotRun, match="obligations changed"):
        gate.run(*comparison[:3])


def test_scanner_failure_propagates_instead_of_accepting_baseline(
    comparison, monkeypatch
):
    def broken(*_):
        raise gate.CannotRun("detector failed")

    monkeypatch.setattr(gate, "_scan", broken)
    with pytest.raises(gate.CannotRun, match="detector failed"):
        gate.run(*comparison[:3])


def test_content_identity_catches_replacement_with_same_detector_label():
    before = report(facade_handlers=["gateway:handler@0"])
    after = report(facade_handlers=["gateway:handler@0"])
    before["semantic"] = {"agent_utilities/gateway.py": ["old-hash"]}
    after["semantic"] = {"agent_utilities/gateway.py": ["new-hash"]}
    assert gate._compare(before, after)


def test_staged_capture_preserves_alternate_git_index(
    repository, monkeypatch, tmp_path
):
    repo, base, candidate = repository
    alternate = tmp_path / "alternate-index"
    monkeypatch.setenv("GIT_INDEX_FILE", str(alternate))
    git(repo, "read-tree", base)
    assert gate._staged_tree(repo) == git(repo, "rev-parse", f"{base}^{{tree}}")
    assert gate._staged_tree(repo) != candidate


@pytest.mark.parametrize(
    "path",
    [
        "scripts/check_liveness_source.py",
        ".config/pre-commit.yaml",
        "agent_utilities/governance/merge_queue.py",
        "agent_utilities/retrieval/budget.py",
    ],
)
def test_approved_policy_and_unrelated_runtime_use_source_comparison(path):
    assert not gate._requires_strict([path], EXPIRED)


def test_deferred_rename_or_deletion_stays_strict(repository):
    repo, _, _ = repository
    original = "agent_utilities/deferred.py"
    (repo / "agent_utilities").mkdir()
    (repo / original).write_text("class Deferred: pass\n")
    git(repo, "add", original)
    git(repo, "commit", "-qm", "deferred fixture")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "mv", original, "agent_utilities/moved.py")
    assert gate._requires_strict(
        gate._scope(repo, base, git(repo, "write-tree")), EXPIRED
    )


def test_present_unpaired_coverage_is_not_silently_dropped(comparison):
    (comparison[0] / "coverage.json").write_text("{}")
    with pytest.raises(gate.CannotRun, match="coverage"):
        gate.run(*comparison[:3])


def test_hook_cannot_override_frozen_candidate(monkeypatch):
    assert gate.main(["--hook", "commit", "--candidate", "0" * 40]) == 2


def test_release_routes_exact_snapshot_to_strict_gate(comparison, monkeypatch):
    roots = []

    def strict(root):
        roots.append(root)
        return 1

    monkeypatch.setattr(gate, "_strict_snapshot", strict)
    assert gate.run(*comparison[:3], release=True) == 1
    assert len(roots) == 1
    assert (roots[0] / "scripts/liveness_deferred.tsv").read_text() == EXPIRED


def test_normal_hook_configuration_and_release_expiry_stay_enforced():
    import yaml

    config = yaml.safe_load((gate.REPO / ".config/pre-commit.yaml").read_text())
    hooks = {h["id"]: h for repo in config["repos"] for h in repo["hooks"]}
    assert hooks["guardrail-liveness"]["entry"].endswith(
        "check_liveness_source.py --hook commit"
    )
    assert hooks["guardrail-liveness"]["stages"] == ["pre-commit"]
    assert hooks["guardrail-liveness-strict"]["entry"].endswith(
        "scripts/check_liveness.py"
    )
    release = yaml.safe_load((gate.REPO / ".github/workflows/release.yml").read_text())
    test_step = next(
        step
        for step in release["jobs"]["gates"]["steps"]
        if step.get("name") == "Test suite"
    )
    assert test_step["run"] == "python3 -m pytest -q -n auto --dist loadfile"
    assert (
        "test_real_liveness_deferred_tsv_is_well_formed_and_not_stale"
        in (gate.REPO / "tests/gates/test_liveness_deferred.py").read_text()
    )


@pytest.fixture
def push_adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(gate.REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "source_push", gate.REPO / "scripts/pre_push.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.liveness = gate
    return module


def test_native_push_checks_branch_and_reachable_tag(
    comparison, monkeypatch, push_adapter
):
    repo, base, _, _ = comparison
    calls = []
    monkeypatch.setattr(
        gate,
        "run",
        lambda repo, base, candidate, release=False: (
            calls.append((candidate, release)) or int(release)
        ),
    )
    payload = f"refs/heads/topic {base} refs/heads/topic {base}\nrefs/tags/v9 {base} refs/tags/v9 {'0' * 40}\n"
    assert push_adapter.check_refs(repo, "origin", payload) == 1
    assert calls == [(base, False), (base, True)]


def test_native_push_checks_second_candidate(comparison, monkeypatch, push_adapter):
    repo, base, _, _ = comparison
    (repo / ALLOWED).write_text("# second candidate\n")
    git(repo, "add", ALLOWED)
    git(repo, "commit", "-qm", "second candidate")
    second = git(repo, "rev-parse", "HEAD")
    calls = []
    monkeypatch.setattr(
        gate,
        "run",
        lambda repo, base, candidate, release=False: (
            calls.append(candidate) or int(candidate == second)
        ),
    )
    payload = f"refs/heads/a {base} refs/heads/a {base}\nrefs/heads/b {second} refs/heads/b {base}\n"
    assert push_adapter.check_refs(repo, "origin", payload) == 1
    assert calls == [base, second]


def test_native_push_preserves_normal_driver_and_input(monkeypatch, push_adapter):
    import io
    import sys

    payload = b"original Git input\n"
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
    monkeypatch.setattr(push_adapter, "check_refs", lambda *a: 0)
    calls = []
    monkeypatch.setattr(push_adapter.os, "access", lambda *a: True)

    def delegate(argv, **kwargs):
        calls.append((argv, kwargs["input"]))
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(push_adapter.subprocess, "run", delegate)
    args = [
        "/driver/python",
        "hook-impl",
        "--hook-type=pre-push",
        "--",
        "origin",
        "fixture",
    ]
    assert push_adapter.main(args) == 7
    assert calls == [(["/driver/python", "-m", "pre_commit", *args[1:]], payload)]


def test_native_hook_install_is_idempotent_and_preserves_generated_driver(
    tmp_path, monkeypatch, push_adapter
):
    path = tmp_path / "pre-push"
    body = '# ID: 138fd403232d2ddd5efb44317e38bf03\nexec "$INSTALL_PYTHON" -mpre_commit "${ARGS[@]}"\nexec pre-commit "${ARGS[@]}"\n'
    path.write_text(body)
    monkeypatch.setattr(gate, "_git", lambda *a: str(path))
    push_adapter.install(tmp_path)
    installed = path.read_text()
    push_adapter.install(tmp_path)
    assert path.read_text() == installed
    assert 'scripts/pre_push.py "$INSTALL_PYTHON" "${ARGS[@]}"' in installed
    assert 'scripts/pre_push.py "" "${ARGS[@]}"' in installed


@pytest.mark.parametrize(
    "payload", ["malformed", "x not-a-sha refs/heads/a " + "0" * 40]
)
def test_native_push_ref_input_fails_closed(comparison, push_adapter, payload):
    with pytest.raises(gate.CannotRun):
        push_adapter.check_refs(comparison[0], "origin", payload)
