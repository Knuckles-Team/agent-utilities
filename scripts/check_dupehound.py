#!/usr/bin/env python3
"""Fail-closed, change-scoped wrapper for the pinned ``dupehound`` binary.

``dupehound check`` already has the right semantics for this repository: with
no ``--diff`` it compares the staged index (or, when there is no index delta,
the working tree) against ``HEAD``; with ``--diff REV`` it compares the
merge-base of ``REV`` and ``HEAD`` to the current commit.  This wrapper owns
the surrounding contract:

* select supported-language, non-test changes before launching the binary, so
  docs, generated output, fixtures, and lockfiles do not make a function gate
  run (dupehound v0.1.2's ``check`` path always skips test paths);
* pass the central threshold/exclusion policy explicitly;
* verify the exact binary version and the JSON result shape; and
* return exit 2 for a missing, drifted, malformed, or otherwise unusable
  scanner instead of turning an unavailable check into a false green.

The wrapper never installs a tool, writes a baseline, or edits source. The
all-format jscpd census and its block-level differential companion live in
``scripts/check_duplication.py``. jscpd deliberately retains code formats so
it can detect copied blocks inside otherwise different functions; the scanners
run at different hook stages rather than deleting that complementary coverage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _clone_scanner_config import (  # noqa: E402
    CloneScannerConfig,
    CloneScannerConfigError,
    is_excluded_path,
    load_clone_scanner_config,
)
from _git_subprocess_env import (  # noqa: E402
    sanitized_git_env,
    strip_inherited_git_repository_env,
)

strip_inherited_git_repository_env()

_CONFIG_PATH = _ROOT / "pyproject.toml"
EXPECTED_JSON_SCHEMA_VERSION = 1
# Mirrors dupehound v0.1.2's Lang::from_path map; unsupported extensions are
# never allowed to make a function gate run and then silently report clean.
SUPPORTED_SUFFIXES = frozenset(
    {
        ".c",
        ".c++",
        ".cc",
        ".cjs",
        ".cpp",
        ".cts",
        ".cs",
        ".cxx",
        ".go",
        ".h",
        ".hh",
        ".h++",
        ".hpp",
        ".hxx",
        ".java",
        ".js",
        ".jsx",
        ".mjs",
        ".mts",
        ".php",
        ".py",
        ".pyi",
        ".rb",
        ".rs",
        ".swift",
        ".ts",
        ".tsx",
    }
)
# dupehound v0.1.2's ``check`` implementation unconditionally skips paths that
# match its test-path classifier.  ``--include-tests`` only changes the scan
# score policy and cannot make ``check`` inspect these files, so keep this
# mirror explicit rather than claiming coverage the pinned binary does not
# provide.
_TEST_DIRECTORY_NAMES = frozenset({"tests", "test", "__tests__", "testdata", "spec"})
_REQUIRED_FINDING_FIELDS = (
    "file",
    "line",
    "name",
    "similarity",
    "original_file",
    "original_line",
    "original_name",
)


def _die(message: str) -> NoReturn:
    print(f"dupehound gate: CANNOT RUN: {message}", file=sys.stderr)
    raise SystemExit(2)


def _config() -> CloneScannerConfig:
    try:
        return load_clone_scanner_config(_CONFIG_PATH)
    except CloneScannerConfigError as exc:
        _die(str(exc))


def _setting(name: str, default: str) -> str:
    """Read a live process override through the repository config boundary."""

    try:
        from agent_utilities.core.config import setting

        value = setting(name, default, cast=str)
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        _die(f"could not read repository setting {name}: {exc}")
    return str(value or default).strip()


def _resolve_dupehound(config: CloneScannerConfig) -> str:
    """Resolve an already-installed binary without package-index access."""

    configured = _setting("DUPEHOUND_BIN", "")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            _die(f"DUPEHOUND_BIN points to a non-file path: {candidate}")
        return str(candidate)
    for candidate in (
        Path.home() / ".local/bin/dupehound",
        Path("/usr/local/bin/dupehound"),
    ):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("dupehound")
    if found:
        return found
    _die(
        "`dupehound` not found. Looked at $DUPEHOUND_BIN, "
        "~/.local/bin/dupehound, /usr/local/bin/dupehound and $PATH. "
        f"Install the pinned v{config.dupehound_version} binary before running "
        "this hook; the hook never installs dependencies."
    )


def _check_version(executable: str, config: CloneScannerConfig) -> None:
    try:
        result = subprocess.run(
            [executable, "--version"],
            cwd=str(_ROOT),
            env=sanitized_git_env(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        _die(f"could not run `{executable} --version`: {exc}")
    if result.returncode != 0:
        _die(
            f"`{executable} --version` exited {result.returncode}: "
            f"{(result.stderr or '').strip()[:400]}"
        )
    got = (result.stdout or "").strip()
    if got != config.dupehound_version_output:
        _die(
            f"version drift: want {config.dupehound_version_output!r}, got "
            f"{got!r}; install the centrally pinned dupehound build"
        )


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(_ROOT),
            env=sanitized_git_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, UnicodeError) as exc:
        _die(f"could not execute git ({' '.join(args)}): {exc}")
    if result.returncode != 0:
        _die(f"`git {' '.join(args)}` failed: {(result.stderr or '').strip()}")
    return (result.stdout or "").strip()


def changed_paths(base_ref: str | None = None) -> list[str]:
    """Return changed paths for staged/worktree or commit-range semantics."""

    if base_ref:
        output = _git(
            [
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
                f"{base_ref}...HEAD",
            ]
        )
    else:
        staged = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
        if staged:
            output = staged
        else:
            output = _git(["diff", "--name-only", "--diff-filter=ACMR", "HEAD"])
            untracked = _git(["ls-files", "--others", "--exclude-standard"])
            if untracked:
                output = "\n".join(filter(None, (output, untracked)))
    return sorted(
        {line.replace("\\", "/") for line in output.splitlines() if line.strip()}
    )


def _normalize_path(path: str) -> str:
    """Normalize a git-relative path without destroying a leading dot-dir."""

    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_dupehound_test_path(path: str) -> bool:
    """Mirror dupehound v0.1.2's test-path classifier for scope reporting."""

    normalized = _normalize_path(path).casefold()
    parts = normalized.split("/")
    filename = parts[-1] if parts else normalized
    if (
        filename.endswith(
            (
                "_test.go",
                "_test.py",
                "_test.rs",
                "_spec.rb",
                "_test.rb",
                "tests.swift",
                "test.swift",
                "spec.swift",
                "test.php",
                "tests.php",
            )
        )
        or filename in {"tests.rs", "test.rs"}
        or filename.startswith("test_")
        or ".test." in filename
        or ".spec." in filename
        or filename.endswith(("test.java", "tests.java"))
        or filename.startswith("conftest.")
    ):
        return True
    return any(part in _TEST_DIRECTORY_NAMES for part in parts)


def select_supported_paths(paths: list[str], config: CloneScannerConfig) -> list[str]:
    """Keep only supported, non-excluded, non-test paths for the function gate.

    The pinned dupehound ``check`` command skips test paths before extracting
    functions.  Filtering them here makes a test-only change report as outside
    this gate instead of producing a misleading clean result.
    """

    selected = []
    for path in paths:
        normalized = _normalize_path(path)
        if (
            Path(normalized).suffix.lower() in SUPPORTED_SUFFIXES
            and not is_excluded_path(normalized, config.exclusions)
            and not is_dupehound_test_path(normalized)
        ):
            selected.append(normalized)
    return sorted(set(selected))


def _command(
    executable: str, config: CloneScannerConfig, base_ref: str | None
) -> list[str]:
    command = [
        executable,
        "check",
        "--json",
        "--threshold",
        str(config.dupehound_threshold),
        "--min-tokens",
        str(config.dupehound_min_tokens),
        # v0.1.2's check path skips tests unconditionally.  State that policy
        # explicitly; --include-tests does not alter check.rs's classifier.
        "--exclude-tests",
    ]
    for pattern in config.exclusions:
        command.extend(("--exclude", pattern))
    if base_ref:
        command.extend(("--diff", base_ref))
    command.append(str(_ROOT))
    return command


def parse_result(output: str) -> list[dict[str, Any]]:
    """Parse and validate dupehound's versioned JSON findings payload."""

    findings = _findings_document(output)
    for index, finding in enumerate(findings):
        _validate_finding(index, finding)
    return findings


def _findings_document(output: str) -> list[dict[str, Any]]:
    try:
        document = json.loads(output)
    except json.JSONDecodeError as exc:
        _die(f"dupehound returned invalid JSON: {exc}")
    if not isinstance(document, dict):
        _die("dupehound JSON result is not an object")
    if document.get("schema_version") != EXPECTED_JSON_SCHEMA_VERSION:
        _die(
            "dupehound JSON schema drift: expected version "
            f"{EXPECTED_JSON_SCHEMA_VERSION}, got {document.get('schema_version')!r}"
        )
    findings = document.get("findings")
    if not isinstance(findings, list):
        _die("dupehound JSON result has no findings array")
    return findings


def _nonempty_string_fields(finding: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(
        isinstance(finding[field], str) and bool(finding[field]) for field in fields
    )


def _positive_int_fields(finding: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(
        not isinstance(finding[field], bool)
        and isinstance(finding[field], int)
        and finding[field] > 0
        for field in fields
    )


def _validate_finding(index: int, finding: object) -> None:
    if not isinstance(finding, dict):
        _die(f"dupehound finding {index} is not an object")
    _require_finding_fields(index, finding)


def _require_finding_fields(index: int, finding: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_FINDING_FIELDS if field not in finding]
    if missing:
        _die(f"dupehound finding {index} is missing {', '.join(missing)}")
    if not _nonempty_string_fields(finding, ("file", "original_file")):
        _die(f"dupehound finding {index} has a non-string file path")
    if not _nonempty_string_fields(finding, ("name", "original_name")):
        _die(f"dupehound finding {index} has a non-string function name")
    if not _positive_int_fields(finding, ("line", "original_line")):
        _die(f"dupehound finding {index} has a non-integer line")
    similarity = finding["similarity"]
    valid_similarity = (
        not isinstance(similarity, bool)
        and isinstance(similarity, (int, float))
        and 0.0 <= float(similarity) <= 1.0
    )
    if not valid_similarity:
        _die(f"dupehound finding {index} has an invalid similarity")


def _execute_check(
    executable: str, config: CloneScannerConfig, base_ref: str | None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            _command(executable, config, base_ref),
            cwd=str(_ROOT),
            env=sanitized_git_env(),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _die("dupehound timed out after 900s")
    except (OSError, UnicodeError) as exc:
        _die(f"could not execute {executable}: {exc}")


def _validated_findings(
    result: subprocess.CompletedProcess[str],
) -> list[dict[str, Any]]:
    if result.returncode not in (0, 1):
        _die(
            f"dupehound exited {result.returncode}: "
            f"{(result.stderr or '').strip()[:1000]}"
        )
    findings = parse_result(result.stdout or "")
    if result.returncode == 0 and findings:
        _die("dupehound returned findings with exit 0")
    if result.returncode == 1 and not findings:
        _die("dupehound returned exit 1 without findings")
    return findings


def _print_findings(findings: list[dict[str, Any]]) -> None:
    print(f"dupehound gate: FAIL — {len(findings)} new duplicate function(s)")
    for finding in findings:
        print(
            f"  {finding['file']}:{finding['line']} "
            f"{finding['name']}() is a "
            f"{float(finding['similarity']) * 100:.0f}% duplicate of "
            f"{finding['original_file']}:{finding['original_line']} "
            f"{finding['original_name']}() — reuse it"
        )


def run(base_ref: str | None = None) -> int:
    config = _config()
    paths = changed_paths(base_ref)
    if not select_supported_paths(paths, config):
        print(
            "dupehound gate: no changed supported-language production function; "
            "dupehound v0.1.2 excludes test paths and configured exclusions"
        )
        return 0
    executable = _resolve_dupehound(config)
    _check_version(executable, config)
    findings = _validated_findings(_execute_check(executable, config, base_ref))
    if findings:
        _print_findings(findings)
        return 1
    print("dupehound gate: PASS — no new duplicate functions")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        "--diff",
        dest="base_ref",
        default=None,
        help="compare the current commit with this revision (CI/PR semantics)",
    )
    return run(parser.parse_args().base_ref)


if __name__ == "__main__":
    sys.exit(main())
