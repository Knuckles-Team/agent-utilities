#!/usr/bin/env python3
"""Reject local identity, machine-path, endpoint, and secret material in fleet changes.

The default scope is each package's tracked and untracked changes, which avoids
rewriting intentional public attribution in historical license metadata. Findings
contain only package-relative locations and rule identifiers; matched content is
never printed. Use ``--all`` for a complete source-tree audit and
``--deny-identifier`` to add deployment-specific identifiers without persisting
them in this repository.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_TEXT_SUFFIXES = {
    "",
    ".env",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
_SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-wsl",
    "__pycache__",
    "htmlcov",
    "node_modules",
}
_SAFE_PATH_IDENTIFIERS = {
    "$env:username",
    "%username%",
    "<user>",
    "<username>",
    "agent-user",
    "runner",
    "user",
    "username",
}
_RULE_DEFINITION_PATHS: dict[str, frozenset[str]] = {
    # The fleet-owned scanner intentionally carries synthetic credential-shaped
    # fixtures so it can recognize them. Exempt only that one rule at that exact
    # canonical path; endpoint, key, path, and deployment-specific identifier
    # checks still apply to the scanner source itself.
    "credential_token_material": frozenset({"scripts/security_sanitizer.py"}),
    # This protocol checker defines the local-path rejection expression used by
    # its own static gate. Exempt only that definition from the matching rule;
    # every other privacy rule still scans the file.
    "local_user_path": frozenset({"scripts/check_epistemic_operations_protocol.py"}),
}
_INTERNAL_ENDPOINT_PATTERN = re.compile(
    r"https?://(?!host\.docker\.internal(?:[/:]|$))[^\s\"'<>]*"
    r"(?:\.kob\.[A-Za-z0-9.-]+|\.arpa|\.internal|\.corp|\.lan)"
    r"(?:[/:]|$)",
    re.IGNORECASE,
)
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("internal_endpoint", _INTERNAL_ENDPOINT_PATTERN),
    (
        "private_key_material",
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    ),
    (
        "credential_token_material",
        re.compile(
            r"\b(?:sk-lf|pk-lf|ghp|github_pat|glpat|xox[baprs])-"
            r"[A-Za-z0-9_-]{8,}\b"
        ),
    ),
    (
        "node_tls_bypass",
        re.compile(r"NODE_TLS_REJECT_UNAUTHORIZED[\"']?\s*[:=]\s*[\"']?0\b"),
    ),
    (
        "workspace_absolute_path",
        re.compile(
            r"(?:[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+[\\/]+(?:Workspace|AppData)"
            r"|/mnt/[a-z]/users/[^/\s]+/(?:workspace|AppData)"
            r"|/home/[^/\s]+/(?:workspace|\.config)/)",
            re.IGNORECASE,
        ),
    ),
)
_PATH_IDENTITY_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]+Users[\\/]+([^\\/\s\"'`]+)",
        re.IGNORECASE,
    ),
    re.compile(r"/mnt/[a-z]/users/([^/\s\"'`]+)", re.IGNORECASE),
    re.compile(r"/home/([^/\s\"'`]+)/(?:workspace|\.config)/", re.IGNORECASE),
)
_SYNTHETIC_TEST_ENDPOINT_SUFFIXES = (".arpa", ".internal")


@dataclass(frozen=True, order=True)
class Finding:
    package: str
    path: str
    line: int
    rule: str


def _git_paths(package: Path, *, all_files: bool) -> list[Path]:
    if all_files:
        commands = [
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        ]
    else:
        commands = [
            ["git", "diff", "--name-only", "-z", "--diff-filter=ACMR"],
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
        ]
    paths: set[Path] = set()
    command_succeeded = False
    for command in commands:
        result = subprocess.run(  # noqa: S603 - fixed git argv, no shell
            command,
            cwd=package,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            continue
        command_succeeded = True
        paths.update(
            Path(value.decode("utf-8", errors="replace"))
            for value in result.stdout.split(b"\0")
            if value
        )
    if command_succeeded:
        return sorted(paths)
    for directory, names, filenames in os.walk(package):
        names[:] = sorted(name for name in names if name not in _SKIP_PARTS)
        base = Path(directory)
        paths.update(
            (base / filename).relative_to(package) for filename in sorted(filenames)
        )
    return sorted(paths)


def _path_identity_finding(line: str) -> bool:
    for pattern in _PATH_IDENTITY_PATTERNS:
        for match in pattern.finditer(line):
            if match.group(1).strip().lower() not in _SAFE_PATH_IDENTIFIERS:
                return True
    return False


def _synthetic_test_endpoint(relative: Path, line: str) -> bool:
    """Recognize only non-organizational endpoint fixtures under tests."""

    if not relative.parts or relative.parts[0] != "tests":
        return False
    matches = list(_INTERNAL_ENDPOINT_PATTERN.finditer(line))
    if not matches:
        return False
    hosts = [urlparse(match.group(0)).hostname or "" for match in matches]
    return all(
        host.casefold().endswith(_SYNTHETIC_TEST_ENDPOINT_SUFFIXES) for host in hosts
    )


def scan_package(
    package: Path,
    *,
    all_files: bool,
    denied_identifiers: tuple[str, ...],
) -> list[Finding]:
    findings: list[Finding] = []
    denied = tuple(
        re.compile(re.escape(value), re.IGNORECASE)
        for value in denied_identifiers
        if value.strip()
    )
    for relative in _git_paths(package, all_files=all_files):
        if any(part in _SKIP_PARTS for part in relative.parts):
            continue
        if relative.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        path = package / relative
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            for rule, pattern in _RULES:
                if relative.as_posix() in _RULE_DEFINITION_PATHS.get(rule, frozenset()):
                    continue
                if rule == "internal_endpoint" and _synthetic_test_endpoint(
                    relative, line
                ):
                    continue
                if rule == "workspace_absolute_path" and not _path_identity_finding(
                    line
                ):
                    continue
                if pattern.search(line):
                    findings.append(
                        Finding(package.name, relative.as_posix(), number, rule)
                    )
            if relative.as_posix() not in _RULE_DEFINITION_PATHS.get(
                "local_user_path", frozenset()
            ) and _path_identity_finding(line):
                findings.append(
                    Finding(
                        package.name,
                        relative.as_posix(),
                        number,
                        "local_user_path",
                    )
                )
            if any(pattern.search(line) for pattern in denied):
                findings.append(
                    Finding(
                        package.name,
                        relative.as_posix(),
                        number,
                        "denied_identifier",
                    )
                )
    return findings


def _default_agents_root() -> Path:
    return Path(__file__).resolve().parents[2] / "agents"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-root", type=Path, default=_default_agents_root())
    parser.add_argument("--all", action="store_true", help="scan complete source trees")
    parser.add_argument("--require-fleet", action="store_true")
    parser.add_argument("--deny-identifier", action="append", default=[])
    args = parser.parse_args()
    if not args.agents_root.is_dir():
        if args.require_fleet:
            print("fleet privacy policy: agent checkout missing")
            return 2
        print("fleet privacy policy: skipped (agent checkout not present)")
        return 0

    environment_denied = tuple(
        value.strip()
        for value in os.environ.get("FLEET_DENY_IDENTIFIERS", "").split(",")
        if value.strip()
    )
    denied = tuple(args.deny_identifier) + environment_denied
    findings: list[Finding] = []
    packages = [
        path
        for path in sorted(args.agents_root.iterdir())
        if path.is_dir() and (path / ".git").exists()
    ]
    for package in packages:
        findings.extend(
            scan_package(
                package,
                all_files=args.all,
                denied_identifiers=denied,
            )
        )
    for finding in sorted(set(findings)):
        print(f"{finding.package}/{finding.path}:{finding.line}: {finding.rule}")
    if findings:
        print(f"fleet privacy policy: FAIL ({len(set(findings))} finding(s))")
        return 1
    scope = "all source" if args.all else "changed source"
    print(f"fleet privacy policy: PASS ({len(packages)} package(s), {scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
