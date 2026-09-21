from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts._clone_scanner_config import (
    CloneScannerConfigError,
    is_excluded_path,
    is_jscpd_diff_path,
    load_clone_scanner_config,
)
from scripts._git_subprocess_env import sanitized_git_env

ROOT = Path(__file__).parents[3]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scanner_config():
    return load_clone_scanner_config(ROOT / "pyproject.toml")


def test_scanner_versions_and_complementary_scope_are_centralized(scanner_config):
    assert scanner_config.dupehound_version == "0.1.2"
    assert scanner_config.jscpd_version == "5.0.16"
    assert scanner_config.dupehound_version_output == "dupehound 0.1.2"
    assert scanner_config.jscpd_version_output == "cpd 5.0.16"
    assert scanner_config.jscpd_format_names_arg == (
        "docker:Containerfile,Dockerfile;makefile:GNUmakefile,Makefile"
    )
    assert scanner_config.jscpd_format_exts_arg == (
        "asciidoc:adoc,asciidoc;cypher:cypher;dhall:dhall;"
        "graphql:gql;mermaid:mermaid,mmd;nginx:conf,nginx;promql:promql"
    )
    assert {"python", "rust", "text", "typescript", "tsx", "javascript"}.issubset(
        scanner_config.jscpd_diff_formats
    )


def test_dupehound_selects_only_supported_nonexcluded_changes(scanner_config):
    dupehound = _load_script("check_dupehound")
    paths = dupehound.select_supported_paths(
        [
            "agent_utilities/core.py",
            "agent_utilities/core.rs",
            "docs/examples/core.py",
            "tests/fixtures/core.py",
            "build/generated.py",
            "uv.lock",
            "README.md",
        ],
        scanner_config,
    )

    assert paths == ["agent_utilities/core.py", "agent_utilities/core.rs"]


def test_dupehound_mirrors_v012_test_scope_and_preserves_dot_dirs(scanner_config):
    dupehound = _load_script("check_dupehound")

    assert dupehound.is_dupehound_test_path("tests/test_core.py")
    assert dupehound.is_dupehound_test_path(".github/test_workflow.py")
    assert not dupehound.is_dupehound_test_path(".github/workflows/build.py")
    assert not dupehound.is_dupehound_test_path("src/testimonials.py")
    assert dupehound.select_supported_paths(
        ["tests/test_core.py", ".github/workflows/build.py"], scanner_config
    ) == [".github/workflows/build.py"]


def test_dupehound_command_pins_test_and_diff_contract(scanner_config):
    dupehound = _load_script("check_dupehound")

    command = dupehound._command("/opt/dupehound", scanner_config, "origin/main")

    assert command[:2] == ["/opt/dupehound", "check"]
    assert "--exclude-tests" in command
    assert "--include-tests" not in command
    assert command[command.index("--diff") + 1] == "origin/main"
    assert command[-1] == str(ROOT)
    assert command.count("--exclude") == len(scanner_config.exclusions)


def test_jscpd_selection_covers_mixed_language_and_excludes_samples(scanner_config):
    assert is_jscpd_diff_path("templates/index.html", scanner_config)
    assert is_jscpd_diff_path("docs/design.adoc", scanner_config)
    assert is_jscpd_diff_path("queries/report.gql", scanner_config)
    assert is_jscpd_diff_path("deploy/nginx.conf", scanner_config)
    assert not is_jscpd_diff_path("docs/readme.mdx", scanner_config)
    assert is_jscpd_diff_path("deploy/values.yaml", scanner_config)
    assert is_jscpd_diff_path(".github/workflows/ci.yml", scanner_config)
    assert is_jscpd_diff_path("Dockerfile", scanner_config)
    assert is_jscpd_diff_path("GNUmakefile", scanner_config)
    assert is_jscpd_diff_path("agent_utilities/core.py", scanner_config)
    assert is_excluded_path("docs/examples/index.html", scanner_config.exclusions)
    assert is_excluded_path("tests/fixtures/payload.json", scanner_config.exclusions)
    assert is_excluded_path(".venv/lib/python.py", scanner_config.exclusions)
    assert is_excluded_path("generated/client.py", scanner_config.exclusions)
    assert is_excluded_path(
        "agent_utilities/knowledge_graph/ontology/connector_manifests/"
        "github-agent/connector_manifest.yml",
        scanner_config.exclusions,
    )
    assert is_excluded_path("assets/bundle.min.js", scanner_config.exclusions)
    assert not is_excluded_path(
        "docs/architecture/code_intelligence.md", scanner_config.exclusions
    )


def test_jscpd_keeps_supported_dot_directories_but_prunes_git(tmp_path, scanner_config):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    (root / ".git" / "config").write_text("[core]\n")

    targets = jscpd._repo_scan_targets(root, scanner_config.prune_directories)
    files = {
        path.relative_to(root).as_posix()
        for path in jscpd._iter_files(targets, scanner_config.prune_directories)
    }

    assert root / ".github" in targets
    assert root / ".git" not in targets
    assert ".github/workflows/ci.yml" in files
    assert ".git/config" not in files


def test_jscpd_deduplicates_symlinked_scan_targets(tmp_path, scanner_config):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# guide\n")
    alias = root / "repo-alias"
    alias.symlink_to(root, target_is_directory=True)

    targets = jscpd._expand_roots([root], scanner_config.prune_directories)

    assert alias not in targets
    assert root / "docs" in targets
    assert len({target.resolve() for target in targets}) == len(targets)


def test_dupehound_json_parser_accepts_versioned_finding_shape():
    dupehound = _load_script("check_dupehound")
    findings = dupehound.parse_result(
        '{"schema_version": 1, "findings": [{'
        '"file": "src/new.py", "line": 4, "name": "new", '
        '"similarity": 0.91, "original_file": "src/old.py", '
        '"original_line": 4, "original_name": "old"}]}'
    )

    assert findings[0]["file"] == "src/new.py"
    assert findings[0]["original_name"] == "old"


def test_dupehound_json_parser_fails_closed_on_missing_findings():
    dupehound = _load_script("check_dupehound")

    with pytest.raises(SystemExit) as raised:
        dupehound.parse_result('{"schema_version": 1}')

    assert raised.value.code == 2


def test_scanner_config_rejects_unknown_keys(tmp_path):
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    changed = source.replace(
        "[tool.agent_utilities.clone_scanners]\n",
        "[tool.agent_utilities.clone_scanners]\nunknown_scanner_key = true\n",
        1,
    )
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(changed, encoding="utf-8")

    with pytest.raises(CloneScannerConfigError, match="unknown_scanner_key"):
        load_clone_scanner_config(config_path)


def test_jscpd_report_parser_fails_closed_on_malformed_duplicate():
    jscpd = _load_script("check_duplication")

    with pytest.raises(SystemExit) as raised:
        jscpd._validate_report({"duplicates": [{}]}, Path("/tmp/jscpd-report.json"))

    assert raised.value.code == 2


def test_jscpd_report_parser_fails_closed_without_statistics():
    jscpd = _load_script("check_duplication")

    with pytest.raises(SystemExit) as raised:
        jscpd._validate_report({"duplicates": []}, Path("/tmp/jscpd-report.json"))

    assert raised.value.code == 2


def test_jscpd_report_parser_rejects_identical_file_and_range_self_pair(tmp_path):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    root.mkdir()
    report = _valid_jscpd_report(root)
    report["duplicates"][0]["secondFile"] = report["duplicates"][0]["firstFile"].copy()

    with pytest.raises(SystemExit) as raised:
        jscpd._validate_report(
            report,
            tmp_path / "report.json",
            roots=[root],
            formats=["python"],
        )

    assert raised.value.code == 2


def test_jscpd_report_parser_keeps_different_range_intra_file_clone(tmp_path):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    root.mkdir()
    report = _valid_jscpd_report(root)
    clone = report["duplicates"][0]
    clone["secondFile"]["name"] = clone["firstFile"]["name"]
    clone["secondFile"]["startLoc"] = {"line": 4}
    clone["secondFile"]["endLoc"] = {"line": 4}

    jscpd._validate_report(
        report,
        tmp_path / "report.json",
        roots=[root],
        formats=["python"],
    )

    keys = jscpd._clone_keys(report, root)
    assert len(keys) == 1
    _format, _digest, locations = next(iter(keys))
    assert {location[0] for location in locations} == {"src/first.py"}
    assert {location[1][1:] for location in locations} == {(2, 2), (4, 4)}
    assert (
        jscpd._format_clone_pair(locations) == "src/first.py:2-2  <->  src/first.py:4-4"
    )


def test_jscpd_clone_keys_distinguish_same_file_ranges(tmp_path):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    root.mkdir()
    report = _valid_jscpd_report(root)
    clone = report["duplicates"][0]
    same_file = root / "src" / "same.py"
    clone["firstFile"]["name"] = str(same_file)
    clone["secondFile"]["name"] = str(same_file)
    clone["secondFile"]["startLoc"] = {"line": 4}
    clone["secondFile"]["endLoc"] = {"line": 4}
    another = deepcopy(clone)
    another["firstFile"]["startLoc"] = {"line": 8}
    another["firstFile"]["endLoc"] = {"line": 8}
    another["secondFile"]["startLoc"] = {"line": 10}
    another["secondFile"]["endLoc"] = {"line": 10}
    report["duplicates"].append(another)
    report["statistics"]["total"]["clones"] = 2

    jscpd._validate_report(
        report,
        tmp_path / "report.json",
        roots=[root],
        formats=["python"],
    )

    assert len(jscpd._clone_keys(report, root)) == 2


def test_jscpd_clone_pair_renderer_fails_closed_on_missing_location():
    jscpd = _load_script("check_duplication")

    with pytest.raises(SystemExit) as raised:
        jscpd._format_clone_pair((("src/only.py", None),))

    assert raised.value.code == 2


def _valid_jscpd_report(root: Path) -> dict:
    return {
        "duplicates": [
            {
                "format": "python",
                "fragment": "return value",
                "lines": 1,
                "tokens": 2,
                "firstFile": {
                    "name": str(root / "src" / "first.py"),
                    "startLoc": {"line": 2},
                    "endLoc": {"line": 2},
                },
                "secondFile": {
                    "name": str(root / "src" / "second.py"),
                    "startLoc": {"line": 4},
                    "endLoc": {"line": 4},
                },
            }
        ],
        "statistics": {
            "total": {
                "clones": 1,
                "sources": 2,
                "duplicatedLines": 2,
                "lines": 8,
                "percentage": 25.0,
            }
        },
    }


def test_jscpd_report_parser_rejects_clone_outside_scan_root(tmp_path):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    root.mkdir()
    report = _valid_jscpd_report(root)
    report["duplicates"][0]["secondFile"]["name"] = str(tmp_path / "outside.py")

    with pytest.raises(SystemExit) as raised:
        jscpd._validate_report(
            report,
            tmp_path / "report.json",
            roots=[root],
            formats=["python"],
        )

    assert raised.value.code == 2


def test_jscpd_report_parser_rejects_inconsistent_clone_count(tmp_path):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    root.mkdir()
    report = _valid_jscpd_report(root)
    report["statistics"]["total"]["clones"] = 0

    with pytest.raises(SystemExit) as raised:
        jscpd._validate_report(report, tmp_path / "report.json", roots=[root])

    assert raised.value.code == 2


def test_jscpd_report_loader_rejects_symlinked_report(tmp_path):
    jscpd = _load_script("check_duplication")
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    report = out_dir / "jscpd-report.json"
    report.symlink_to(outside)

    with pytest.raises(SystemExit) as raised:
        jscpd._load_report(report, out_dir=out_dir)

    assert raised.value.code == 2


def test_jscpd_report_loader_rejects_report_outside_output_directory(tmp_path):
    jscpd = _load_script("check_duplication")
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    outside = tmp_path / "jscpd-report.json"
    outside.write_text("{}")

    with pytest.raises(SystemExit) as raised:
        jscpd._load_report(outside, out_dir=out_dir)

    assert raised.value.code == 2


def test_jscpd_report_parser_rejects_relative_clone_path_when_roots_bound(tmp_path):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    root.mkdir()
    report = _valid_jscpd_report(root)
    report["duplicates"][0]["firstFile"]["name"] = "src/first.py"

    with pytest.raises(SystemExit) as raised:
        jscpd._validate_report(
            report,
            tmp_path / "report.json",
            roots=[root],
            formats=["python"],
        )

    assert raised.value.code == 2


def test_jscpd_report_parser_normalizes_virtual_format_path_suffix(tmp_path):
    jscpd = _load_script("check_duplication")
    root = tmp_path / "repo"
    root.mkdir()
    report = _valid_jscpd_report(root)
    report["duplicates"][0]["format"] = "markdown"
    report["duplicates"][0]["firstFile"]["name"] = f"{root / 'first.md'}:markdown:2-2"
    report["duplicates"][0]["secondFile"]["name"] = f"{root / 'second.md'}:markdown:4-4"

    jscpd._validate_report(
        report,
        tmp_path / "report.json",
        roots=[root / "first.md", root / "second.md"],
        formats=["markdown"],
    )
    keys = jscpd._clone_keys(report, root)
    assert len(keys) == 1
    format_name, _digest, paths = next(iter(keys))
    assert format_name == "markdown"
    assert {path for path, _range in paths} == {"first.md", "second.md"}
    assert all(range_key is None for _path, range_key in paths)


def test_jscpd_cleanup_fails_closed_when_worktree_remains(tmp_path, monkeypatch):
    jscpd = _load_script("check_duplication")
    worktree = tmp_path / "throwaway"
    worktree.mkdir()
    monkeypatch.setattr(
        jscpd.subprocess,
        "run",
        lambda *args, **kwargs: jscpd.subprocess.CompletedProcess(
            args[0], 0, stdout="", stderr=""
        ),
    )

    with pytest.raises(SystemExit) as raised:
        jscpd._remove_throwaway_worktree(worktree)

    assert raised.value.code == 2
    assert worktree.exists()


def _fixture_commit_env() -> dict[str, str]:
    """Identity for the FIXTURE's own setup commits, passed as
    ``GIT_AUTHOR_*``/``GIT_COMMITTER_*`` env vars — deliberately NOT via
    ``git config`` (local or global). A local ``git config user.name=...``
    would persist in the throwaway repo's own ``.git/config`` and give the
    later commit-tree-under-test call an identity source that a real,
    freshly-checked-out CI clone never has, silently defeating the whole
    point of this test (confirmed: an earlier draft that used local `git
    config` for fixture setup made the "fails without identity" case below
    pass for the wrong reason on every host)."""
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }


def _init_throwaway_git_repo(root: Path) -> tuple[str, str]:
    """A tiny two-commit repo with NO identity anywhere in its git config
    (local, global, or system) — only env-var identity on the fixture's own
    setup commits (see :func:`_fixture_commit_env`) — so the later
    ``commit-tree`` call under test genuinely has no identity source to fall
    back on, matching a freshly-checked-out CI clone."""
    env = _fixture_commit_env()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "a.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "a.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=root, check=True, env=env)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    (root / "a.txt").write_text("head\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "a.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "head"], cwd=root, check=True, env=env)
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return base_sha, head_sha


def _empty_home_git_env(empty_home: Path) -> dict[str, str]:
    """A git environment with no inherited identity or config source."""
    env = sanitized_git_env()
    env["HOME"] = str(empty_home)
    env["XDG_CONFIG_HOME"] = str(empty_home / ".config")
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def test_jscpd_gate_identity_is_neutral_and_never_claude():
    """The identity CX-DUP-ENFORCE's `commit-tree` call carries must be a
    dedicated, neutral identity — never `claude` (AGENTS.md/CLAUDE.md: never
    commit as user.name `claude`) — and must be wired as `git -c` global
    overrides (not `GIT_AUTHOR_*`/`GIT_COMMITTER_*` env vars, which
    `sanitized_git_env` deliberately strips and which would leak into every
    OTHER git call this process makes if set as env instead)."""
    duplication = _load_script("check_duplication")

    identity = duplication._JSCPD_GATE_IDENTITY
    assert identity == (
        "-c",
        "user.name=jscpd-gate",
        "-c",
        "user.email=jscpd-gate@example.invalid",
    )
    assert all("claude" not in value.lower() for value in identity)


def test_jscpd_gate_commit_tree_fails_without_identity_under_empty_home(tmp_path):
    """Reproduces D-CIP's `clone-scanners 10` failure directly: with NO
    gitconfig identity anywhere (empty HOME, no XDG config dir) and
    GIT_AUTHOR_*/GIT_COMMITTER_* stripped (sanitized_git_env's documented
    behavior), a bare `git commit-tree` — the pre-fix call shape — fails
    with "Author identity unknown". This is the regression this gate's fix
    (below) closes; keeping this assertion proves the fix is actually
    necessary, not incidental."""
    repo = tmp_path / "repo"
    repo.mkdir()
    base_sha, _head_sha = _init_throwaway_git_repo(repo)
    tree_sha = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    env = _empty_home_git_env(empty_home)

    broken = subprocess.run(
        ["git", "commit-tree", tree_sha, "-p", base_sha, "-m", "no identity"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )
    assert broken.returncode != 0
    assert "identity unknown" in (broken.stderr or "").lower()


def test_jscpd_gate_commit_tree_succeeds_with_explicit_identity_under_empty_home(
    tmp_path,
):
    """The fix: CX-DUP-ENFORCE's actual call shape (`git -c user.name=... -c
    user.email=... commit-tree ...`) succeeds under the exact same empty-HOME,
    no-inherited-author environment the test above proves is otherwise
    broken — i.e. the gate's own throwaway commit-tree call no longer
    depends on host gitconfig existing at all."""
    duplication = _load_script("check_duplication")

    repo = tmp_path / "repo"
    repo.mkdir()
    base_sha, head_sha = _init_throwaway_git_repo(repo)
    tree_sha = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    env = _empty_home_git_env(empty_home)

    fixed = subprocess.run(
        [
            "git",
            *duplication._JSCPD_GATE_IDENTITY,
            "commit-tree",
            tree_sha,
            "-p",
            base_sha,
            "-p",
            head_sha,
            "-m",
            "CX-DUP-ENFORCE throwaway snapshot — not a ref, do not reuse",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )
    assert fixed.returncode == 0, fixed.stderr
    synth_sha = fixed.stdout.strip()
    assert len(synth_sha) == 40

    verify = subprocess.run(
        ["git", "cat-file", "-t", synth_sha],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert verify.stdout.strip() == "commit"
