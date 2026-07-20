#!/usr/bin/env python3
"""Reject host-specific data and local identities from tracked public artifacts.

Identifiers are derived in memory from the current account, home, checkout and
host. Findings report only ``file:line`` and a category; the sensitive matched
value is never written or printed. Machine paths and persisted path fields are
checked independently, so the gate remains useful in clean CI environments.
"""

from __future__ import annotations

import getpass
import os
import pwd
import re
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_TEXT_SUFFIXES = frozenset({".md", ".json", ".yaml", ".yml", ".toml"})
_SOURCE_SUFFIXES = frozenset(
    {".js", ".md", ".ps1", ".py", ".rs", ".sh", ".ts", ".yaml", ".yml"}
)
_GENERIC_IDENTIFIERS = frozenset(
    {
        "admin",
        "agent",
        "apps",
        "build",
        "developer",
        "home",
        "localhost",
        "maintainer",
        "maintainers",
        "root",
        "runner",
        "service",
        "user",
        "workspace",
    }
)
_HOME_PATH_PATTERN = (
    r"(?:(?<![A-Za-z0-9_.-])/home/[A-Za-z0-9_.-]+(?:/|\b)|"
    r"(?<![A-Za-z0-9_.-])/Users/[A-Za-z0-9_.-]+(?:/|\b)|"
    r"(?<![A-Za-z0-9_.-])/mnt/[A-Za-z]/Users/[A-Za-z0-9_.-]+(?:/|\b)|"
    r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+(?:[\\/]|\b))"
)
_HOME_PATH_RE = re.compile(_HOME_PATH_PATTERN, re.IGNORECASE)
_PERSISTED_FIELD_RE = re.compile(
    r"[\"']?(?P<field>workspace_path|source_path|skill_path|local_path|source_file|"
    r"eg_ledger_path)[\"']?\s*[:=]\s*(?P<value>.+)",
    re.IGNORECASE,
)
_NEUTRAL_URI_RE = re.compile(
    r"^[\s\"']*(?:repo|skill|connector|design)://", re.IGNORECASE
)
_INTERNAL_ENDPOINT_RE = re.compile(
    r"(?i)\b(?:[A-Za-z0-9-]+\.)+(?:arpa|internal)\b|"
    r"\b(?:[A-Za-z0-9-]+\.)*svc\.cluster\.local\b|"
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
_SOURCE_INTERNAL_URL_RE = re.compile(
    r"(?i)https?://[^\s\"'<>]*(?:\.(?:arpa|internal|corp|lan)\b|"
    r"\.svc\.cluster\.local\b)"
)
_PRIVATE_KEY_LINE_RE = re.compile(r"^\s*-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----\s*$")
_CREDENTIAL_URI_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@")
_HOST_IDENTITY_RE = re.compile(r"(?i)\bssh://(?!\$\{)[^\s/@]+@")
_MACHINE_HOST_ID_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:rw?|host)[0-9]{3,}(?![a-z0-9])"
)
_NEUTRAL_AUTHOR_NAME = "repository maintainers"
_NEUTRAL_AUTHOR_EMAIL_SUFFIX = "@example.invalid"
_SCAN_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".acp-sessions",
        ".benchmarks",
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".pytest_tmp",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "site",
        "target",
        "venv",
        "workspace",
    }
)
_MAX_SCAN_FILES = 500_000


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    category: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.category}"


def _identifier_from_path(value: str) -> set[str]:
    identifiers: set[str] = set()
    normalized = value.replace("\\", "/")
    for pattern in (r"/home/([^/]+)", r"/Users/([^/]+)"):
        identifiers.update(re.findall(pattern, normalized, flags=re.IGNORECASE))
    return identifiers


def derive_local_identifiers(root: Path = ROOT) -> frozenset[str]:
    candidates = {
        getpass.getuser(),
        pwd.getpwuid(os.getuid()).pw_name,
        socket.gethostname(),
        socket.gethostname().split(".", 1)[0],
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
        os.environ.get("USERNAME", ""),
        Path.home().name,
    }
    candidates.update(_identifier_from_path(str(Path.home())))
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        candidates.update(_identifier_from_path(result.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    for command in (
        ["git", "config", "--get", "user.name"],
        ["git", "config", "--get", "user.email"],
        ["git", "log", "-1", "--format=%an%n%ae"],
    ):
        try:
            result = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            for value in result.stdout.splitlines():
                candidates.add(value.strip())
        except OSError:
            pass
    return frozenset(
        value.casefold()
        for value in candidates
        if value
        and len(value) >= 4
        and value.casefold() not in _GENERIC_IDENTIFIERS
    )


def _is_deployment_doc(path: Path) -> bool:
    value = path.as_posix().casefold()
    return value.startswith("docs/recipes/") or any(
        marker in value
        for marker in (
            "deploy",
            "runbook",
            "configuration",
            "workspace-config",
            "mcp_auth",
            "secrets-auth",
        )
    )


def classify_line(
    line: str,
    *,
    identifiers: frozenset[str],
    deployment_doc: bool,
) -> frozenset[str]:
    categories: set[str] = set()
    persisted = _PERSISTED_FIELD_RE.search(line)
    if persisted and not _NEUTRAL_URI_RE.search(persisted.group("value")):
        value = persisted.group("value").strip(" \t,;)}]\"'").casefold()
        field = persisted.group("field")
        runtime_relative = field.isupper() and not re.match(
            r"^(?:[a-z]:|[/\\]|~)", value, re.IGNORECASE
        )
        if value not in {"", "none", "null", "unset"} and not runtime_relative:
            categories.add("persisted machine path")
    if "persisted machine path" not in categories and _HOME_PATH_RE.search(line):
        categories.add("machine-specific home path")
    folded = line.casefold()
    if any(re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", folded) for value in identifiers):
        categories.add("local account or host identifier")
    if _MACHINE_HOST_ID_RE.search(line):
        categories.add("machine-specific host identifier")
    if deployment_doc and _INTERNAL_ENDPOINT_RE.search(line):
        categories.add("hard-coded internal endpoint")
    if deployment_doc and _CREDENTIAL_URI_RE.search(line):
        categories.add("credential-bearing URI")
    if deployment_doc and _HOST_IDENTITY_RE.search(line):
        categories.add("hard-coded remote account")
    return frozenset(categories)


def classify_changed_source_line(
    line: str, *, identifiers: frozenset[str]
) -> frozenset[str]:
    """Classify changed source without applying public-doc path heuristics.

    Source code legitimately manipulates path-shaped values, so the generic
    ``source_path = ...`` rule would be noisy here. Concrete account paths,
    environment endpoints, credential-bearing URLs, and local identities are
    never legitimate package defaults and are checked for every changed source,
    skill, script, and test instead.
    """

    categories: set[str] = set()
    if _HOME_PATH_RE.search(line):
        categories.add("machine-specific home path in changed source")
    folded = line.casefold()
    if any(
        re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", folded)
        for value in identifiers
    ):
        categories.add("local account or host identifier in changed source")
    if _SOURCE_INTERNAL_URL_RE.search(line):
        categories.add("hard-coded internal endpoint in changed source")
    if _CREDENTIAL_URI_RE.search(line):
        categories.add("credential-bearing URI in changed source")
    if _PRIVATE_KEY_LINE_RE.fullmatch(line):
        categories.add("private key material in changed source")
    return frozenset(categories)


def _is_public_artifact(name: str) -> bool:
    path = Path(name)
    if path.suffix.casefold() not in _TEXT_SUFFIXES:
        return False
    if "skills" in path.parts:
        return False
    return (
        path.parts[0] == "docs"
        or path.parts[0] == ".github"
        or len(path.parts) == 1
        or path.suffix.casefold() == ".toml"
    )


def _filesystem_files(root: Path) -> list[Path]:
    """Enumerate a bounded no-Git source snapshot without following links."""

    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True):
        current = Path(directory)
        traversable: list[str] = []
        for name in sorted(directory_names):
            if name in _SCAN_EXCLUDED_DIRECTORIES or name.endswith(".egg-info"):
                continue
            metadata = (current / name).lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if stat.S_ISDIR(metadata.st_mode):
                traversable.append(name)
        directory_names[:] = traversable
        for name in sorted(file_names):
            path = current / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            files.append(path)
            if len(files) > _MAX_SCAN_FILES:
                raise RuntimeError("privacy source inventory exceeds its file bound")
    return files


def _git_file_names(root: Path, command: list[str]) -> list[str] | None:
    """Return Git inventory names, or ``None`` for an immutable no-Git snapshot."""

    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return [name for name in result.stdout.splitlines() if name]


def _tracked_artifacts(root: Path) -> list[Path]:
    names = _git_file_names(
        root,
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
    )
    candidates = (
        _filesystem_files(root) if names is None else [root / name for name in names]
    )
    return [
        path
        for path in candidates
        if _is_public_artifact(path.relative_to(root).as_posix())
    ]


def _changed_source_artifacts(root: Path) -> list[Path]:
    """Return changed/untracked source paths for the source privacy boundary."""

    names: set[str] = set()
    for command in (
        ["git", "diff", "--name-only", "--diff-filter=ACMR"],
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        values = _git_file_names(root, command)
        if values is None:
            names.clear()
            candidates = _filesystem_files(root)
            break
        names.update(values)
    else:
        candidates = [root / name for name in names]
    return sorted(
        (
            path
            for path in candidates
            if path.suffix.casefold() in _SOURCE_SUFFIXES
            and _is_runtime_source_path(path.relative_to(root))
        ),
        key=lambda path: path.as_posix(),
    )


def _is_runtime_source_path(path: Path) -> bool:
    """Scope the changed-source gate to shipped runtime/deployment material.

    Adversarial tests and the privacy scanners themselves intentionally contain
    synthetic bad values. Public docs are already scanned in full by
    ``_tracked_artifacts``; bundled skills live under the runtime package and are
    included here.
    """

    if not path.parts:
        return False
    return path.parts[0].casefold() in {
        "agent_utilities",
        "deploy",
        "docker",
        "helm",
        "k8s",
    }


def _is_bundled_connector_profile(path: Path) -> bool:
    parts = tuple(part.casefold() for part in path.parts)
    return (
        parts[:4]
        == (
            "agent_utilities",
            "protocols",
            "source_connectors",
            "profiles",
        )
        and path.suffix.casefold() in {".py", ".json", ".yaml", ".yml"}
    )


def _author_metadata_lines(path: Path, lines: list[str]) -> list[int]:
    """Return non-neutral package-author lines without returning their values."""
    if path.suffix.casefold() != ".toml":
        return []
    violations: list[int] = []
    in_project_authors = False
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == "[[project.authors]]":
            in_project_authors = True
            continue
        if in_project_authors and stripped.startswith("["):
            in_project_authors = False
        folded = stripped.casefold()
        if re.match(r"authors\s*=", folded):
            if (
                _NEUTRAL_AUTHOR_NAME not in folded
                or _NEUTRAL_AUTHOR_EMAIL_SUFFIX not in folded
            ):
                violations.append(number)
        elif in_project_authors and re.match(r"name\s*=", folded):
            if _NEUTRAL_AUTHOR_NAME not in folded:
                violations.append(number)
        elif in_project_authors and re.match(r"email\s*=", folded):
            if _NEUTRAL_AUTHOR_EMAIL_SUFFIX not in folded:
                violations.append(number)
    return violations


def scan(root: Path = ROOT) -> list[Violation]:
    identifiers = derive_local_identifiers(root)
    violations: list[Violation] = []
    for path in _tracked_artifacts(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        deployment_doc = _is_deployment_doc(relative)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number in _author_metadata_lines(path, lines):
            violations.append(
                Violation(
                    relative.as_posix(), number, "non-neutral package author identity"
                )
            )
        for number, line in enumerate(lines, 1):
            for category in classify_line(
                line,
                identifiers=identifiers,
                deployment_doc=deployment_doc,
            ):
                violations.append(
                    Violation(relative.as_posix(), number, category)
                )
    for path in _changed_source_artifacts(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_bundled_connector_profile(relative):
            violations.append(
                Violation(
                    relative.as_posix(),
                    1,
                    "bundled environment-specific connector profile",
                )
            )
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, 1):
            for category in classify_changed_source_line(
                line, identifiers=identifiers
            ):
                violations.append(
                    Violation(relative.as_posix(), number, category)
                )
    return violations


def main() -> int:
    violations = scan()
    if violations:
        print("Tracked artifact privacy gate FAILED:")
        for violation in violations:
            print(f"  - {violation.render()}")
        print("Matched values are intentionally suppressed.")
        return 1
    print("Tracked artifact privacy gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
