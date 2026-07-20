#!/usr/bin/env python3
"""Fail closed when an Agent Utilities wheel contains local or non-runtime data."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

_BYTECODE_SUFFIXES = {".pyc", ".pyo"}
_RUNTIME_STATE_SEGMENTS = frozenset({".agent_data"})
_RUNTIME_SCRIPT_PACKAGES = frozenset({"certification", "release", "scale"})
_REQUIRED_RUNTIME_MEMBERS = frozenset(
    {
        "agent_utilities/observability/langfuse_trust.py",
        "agent_utilities/security/persistence_privacy.py",
    }
)
_GENERIC_IDENTIFIERS = {
    "admin",
    "agent",
    "apps",
    "build",
    "developer",
    "example",
    "home",
    "maintainer",
    "operator",
    "path",
    "repository automation",
    "root",
    "runner",
    "sample",
    "sandbox",
    "service",
    "test",
    "unknown",
    "user",
    "workspace",
}
_MACHINE_PATHS = (
    re.compile(rb"(?i)[A-Z]:\\\\Users\\\\(?P<account>[A-Za-z0-9._-]+)"),
    re.compile(rb"(?<![A-Za-z0-9._-])/mnt/[A-Za-z]/Users/(?P<account>[A-Za-z0-9._-]+)"),
    re.compile(rb"(?<![A-Za-z0-9._-])/Users/(?P<account>[A-Za-z0-9._-]+)"),
    re.compile(rb"(?<![A-Za-z0-9._-])/home/(?P<account>[A-Za-z0-9._-]+)"),
)


def _path_identifiers(value: str) -> set[str]:
    normalized = value.replace("\\", "/")
    identifiers: set[str] = set()
    for pattern in (r"/home/([^/]+)", r"/Users/([^/]+)"):
        identifiers.update(re.findall(pattern, normalized, flags=re.IGNORECASE))
    return identifiers


def derive_local_identifiers(root: Path | None = None) -> frozenset[str]:
    """Derive local account/build identities in memory without returning paths."""
    checkout = (root or Path(__file__).resolve().parent.parent).resolve()
    candidates = {
        getpass.getuser(),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
        os.environ.get("USERNAME", ""),
        Path.home().name,
    }
    candidates.update(_path_identifiers(str(checkout)))
    candidates.update(_path_identifiers(str(Path.home())))
    for command in (
        ["git", "config", "--get", "user.name"],
        ["git", "config", "--get", "user.email"],
    ):
        try:
            result = subprocess.run(
                command,
                cwd=checkout,
                check=False,
                capture_output=True,
                text=True,
            )
            candidates.update(result.stdout.splitlines())
        except OSError:
            pass
    return frozenset(
        value.strip().casefold()
        for value in candidates
        if len(value.strip()) >= 4
        and value.strip().casefold() not in _GENERIC_IDENTIFIERS
        and not value.strip().casefold().endswith("@example.invalid")
    )


def _is_runtime_member(path: PurePosixPath) -> bool:
    if not path.parts:
        return False
    top = path.parts[0]
    if (
        top == "agent_utilities"
        or top.startswith("agent_utilities-")
        and top.endswith((".dist-info", ".data"))
    ):
        return True
    # These bounded packages are part of the installed release runtime. Console
    # entry points import the release/certification/scale modules and the promoter
    # validates schemas from ``deploy.release``. No other top-level source tooling
    # is admitted to the wheel.
    if top == "scripts":
        return (
            path.parts == ("scripts", "__init__.py")
            or len(path.parts) >= 3
            and path.parts[1] in _RUNTIME_SCRIPT_PACKAGES
        )
    if top == "deploy":
        return path.parts == ("deploy", "__init__.py") or (
            len(path.parts) >= 3 and path.parts[1] == "release"
        )
    return False


def _contains_machine_path(payload: bytes) -> bool:
    for pattern in _MACHINE_PATHS:
        for match in pattern.finditer(payload):
            account = match.group("account").decode("utf-8", errors="ignore").casefold()
            if account and account not in _GENERIC_IDENTIFIERS:
                return True
    return False


def wheel_privacy_findings(
    wheel: Path,
    *,
    local_identifiers: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Return stable finding categories without exposing matched content."""
    findings: set[str] = set()
    identities = local_identifiers or derive_local_identifiers()
    identity_patterns = tuple(
        re.compile(
            rb"(?<![A-Za-z0-9_.-])"
            + re.escape(identity.encode("utf-8"))
            + rb"(?![A-Za-z0-9_.-])",
            re.IGNORECASE,
        )
        for identity in identities
    )
    with zipfile.ZipFile(wheel) as archive:
        member_names = {member.filename.rstrip("/") for member in archive.infolist()}
        if not _REQUIRED_RUNTIME_MEMBERS.issubset(member_names):
            findings.add("missing-required-runtime-member")
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                findings.add("unsafe-member-path")
            if not _is_runtime_member(path) or "tests" in path.parts:
                findings.add("non-runtime-content")
            if _RUNTIME_STATE_SEGMENTS.intersection(path.parts):
                findings.add("runtime-state-content")
            if "__pycache__" in path.parts or path.suffix.lower() in _BYTECODE_SUFFIXES:
                findings.add("bytecode-content")
            if member.is_dir():
                continue
            payload = archive.read(member)
            if _contains_machine_path(payload):
                findings.add("machine-path-content")
            if any(pattern.search(payload) for pattern in identity_patterns):
                findings.add("local-identity-content")
    return tuple(sorted(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    findings = wheel_privacy_findings(args.wheel)
    if findings:
        print(f"wheel_privacy_gate=failed findings={','.join(findings)}")
        return 1
    print("wheel_privacy_gate=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
