#!/usr/bin/env python3
"""Fail when an ecosystem package defaults outbound TLS verification off.

The scanner is intentionally lexical and dependency-free so it can run before
installing the 68-package connector fleet. It reports package-relative paths
only; machine paths, endpoints, and matched source text are never emitted.
Explicit test fixtures are excluded, but production code, generators,
examples, environment templates, skills, and documentation are all governed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._git_scan import repo_root_of  # noqa: E402

_SUFFIXES = {
    ".env",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".yaml",
    ".yml",
}
_SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "fixtures",
    "htmlcov",
    "node_modules",
    "tests",
}
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "python_false_default",
        re.compile(
            r"\b(?:verify|verify_ssl|ssl_verify|tls_verify)\s*"
            r"(?::\s*bool(?:\s*\|\s*None)?)?\s*=\s*False\b"
        ),
    ),
    (
        "environment_false_default",
        re.compile(
            r"(?:SSL_VERIFY|VERIFY_SSL)[^\r\n]{0,100}"
            r"(?:default[^\r\n]{0,20})?(?:False|false|off)\b"
        ),
    ),
    (
        "setting_false_default",
        re.compile(
            r"(?:setting|environ\.get|getenv)\([^\r\n)]*"
            r"(?:SSL_VERIFY|VERIFY_SSL)[^\r\n)]*,\s*[\"']?(?:False|false|0)"
        ),
    ),
    (
        "node_tls_disabled",
        re.compile(r"NODE_TLS_REJECT_UNAUTHORIZED[\"']?\s*[:=]\s*[\"']?0\b"),
    ),
    (
        "ssl_context_bypass",
        re.compile(
            r"\b(?:CERT_NONE|_create_unverified_context)\b"
            r"|\bcheck_hostname\s*=\s*False\b"
        ),
    ),
    (
        "insecure_cli_bypass",
        re.compile(r"(?:^|[\s`\"'])--insecure(?:[\s`\"']|$)"),
    ),
)


@dataclass(frozen=True, order=True)
class Finding:
    package: str
    path: str
    line: int
    rule: str


def _candidate(path: Path) -> bool:
    if any(part in _SKIP_PARTS for part in path.parts):
        return False
    return path.name == ".env.example" or path.suffix.lower() in _SUFFIXES


def scan_package(package_root: Path) -> list[Finding]:
    """Return privacy-safe TLS-policy findings for one package.

    BUG-043/BUG-174: git sets ``GIT_DIR``/``GIT_INDEX_FILE`` in the
    environment of every hook subprocess, under which ``git -C
    <package_root> ls-files`` silently reverts to ambient-work-tree-root-
    relative output -- or, when ``package_root`` belongs to a DIFFERENT
    repository than the ambient one (the common case here: this scans
    OTHER checked-out packages), returns the AMBIENT repo's own file list
    entirely, mislabeled as this package's (confirmed empirically, see
    ``scripts/_git_scan.py``). Anchor at ``package_root``'s own repository
    root -- found by pure filesystem ``.git`` inspection, never by asking
    git itself -- with a pathspec scoped back down to ``package_root``, and
    reconstruct every path from that anchor: immune to the ambient env
    either way. ``--others`` is preserved deliberately (unlike the tracked-
    only ``scripts/_git_scan.py`` helper): this gate must also catch a
    brand-new untracked file before its first commit.
    """

    findings: list[Finding] = []
    anchor = repo_root_of(package_root) or package_root
    try:
        rel = package_root.relative_to(anchor)
        pathspec = [rel.as_posix() + "/"] if str(rel) != "." else ["."]
    except ValueError:
        anchor = package_root
        pathspec = ["."]
    result = subprocess.run(  # noqa: S603 - fixed git argv, no shell
        [
            "git",
            "-C",
            str(anchor),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *pathspec,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        relatives = sorted(
            (anchor / Path(value.decode("utf-8", errors="replace"))).relative_to(
                package_root
            )
            for value in result.stdout.split(b"\0")
            if value
        )
    else:  # unit-test/non-git fallback, with aggressive directory pruning
        relatives = []
        for directory, names, filenames in os.walk(package_root):
            names[:] = sorted(name for name in names if name not in _SKIP_PARTS)
            base = Path(directory)
            relatives.extend(
                (base / filename).relative_to(package_root)
                for filename in sorted(filenames)
            )

    for relative in relatives:
        path = package_root / relative
        if not _candidate(relative):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            for rule, pattern in _RULES:
                if pattern.search(line):
                    findings.append(
                        Finding(package_root.name, relative.as_posix(), number, rule)
                    )
    return findings


def scan_fleet(agents_root: Path) -> list[Finding]:
    """Scan every checked-out Git package under the ecosystem agents root."""

    findings: list[Finding] = []
    if not agents_root.is_dir():
        return findings
    for package in sorted(agents_root.iterdir()):
        if package.is_dir() and (package / ".git").exists():
            findings.extend(scan_package(package))
    return sorted(findings)


def _default_agents_root() -> Path:
    return Path(__file__).resolve().parents[2] / "agents"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-root", type=Path, default=_default_agents_root())
    parser.add_argument(
        "--require-fleet",
        action="store_true",
        help="fail when the agents checkout is absent",
    )
    args = parser.parse_args()
    if not args.agents_root.is_dir():
        if args.require_fleet:
            print("fleet TLS policy: agent checkout missing")
            return 2
        print("fleet TLS policy: skipped (agent checkout not present)")
        return 0

    findings = scan_fleet(args.agents_root)
    for finding in findings:
        print(f"{finding.package}/{finding.path}:{finding.line}: {finding.rule}")
    if findings:
        print(f"fleet TLS policy: FAIL ({len(findings)} finding(s))")
        return 1
    package_count = sum(
        1
        for path in args.agents_root.iterdir()
        if path.is_dir() and (path / ".git").exists()
    )
    print(f"fleet TLS policy: PASS ({package_count} package(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
