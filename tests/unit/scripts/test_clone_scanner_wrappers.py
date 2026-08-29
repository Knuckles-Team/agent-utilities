from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from scripts._clone_scanner_config import (
    CloneScannerConfigError,
    is_excluded_path,
    is_jscpd_diff_path,
    load_clone_scanner_config,
)

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
    assert paths == frozenset({"first.md", "second.md"})


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
