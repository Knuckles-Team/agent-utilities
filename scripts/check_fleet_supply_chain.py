#!/usr/bin/env python3
"""Fail closed on high-risk source supply-chain drift.

The checker deliberately has no project-runtime dependencies.  In normal mode it
inspects only Git-tracked and untracked/non-ignored source files.  An explicit
source-snapshot mode inventories the exact provider roots declared by the
repository-manager workspace without requiring copied Git object databases.  It
never follows repository symlinks and emits repository-relative locations so
diagnostics do not disclose developer filesystem layouts.

Examples::

    python3 scripts/check_fleet_supply_chain.py .
    python3 scripts/check_fleet_supply_chain.py --fleet-root ../
    python3 scripts/check_fleet_supply_chain.py \
        --source-snapshot-root ../agents \
        --snapshot-workspace ../agents/repository-manager/repository_manager/workspace.yml

The fleet mode discovers nested Git repositories and is intended for a checkout
managed by repository-manager.  It does not fetch, install, build, or execute any
project code.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import stat
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass

MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_REPOSITORIES = 1_000
MAX_DISCOVERY_DEPTH = 5
EXPECTED_SNAPSHOT_PROVIDERS = 65
MAX_SNAPSHOT_ENTRIES = 200_000
MAX_SNAPSHOT_FILES = 100_000
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024 * 1024
MAX_SNAPSHOT_DEPTH = 32
MAX_WORKSPACE_BYTES = 2 * 1024 * 1024
SKIP_DIRECTORIES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".worktrees",
        "__pycache__",
        "dist",
        "node_modules",
        "target",
        "vendor",
    }
)

ACTION_SHA_RE = re.compile(r"^[^/@\s]+/[^@\s]+@[0-9a-fA-F]{40}$")
CONTAINER_DIGEST_RE = re.compile(r"@sha256:[0-9a-fA-F]{64}(?:$|\s)")
EXTERNAL_USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", re.MULTILINE)
NETWORK_TO_SHELL_RE = re.compile(
    r"(?:curl|wget)\b[^\n|]*\|\s*(?:/bin/)?(?:ba|z|k)?sh\b", re.IGNORECASE
)
POWERSHELL_NETWORK_TO_EXPRESSION_RE = re.compile(
    r"(?:irm|iwr|Invoke-RestMethod|Invoke-WebRequest)\b[^\n|]*\|\s*"
    r"(?:iex|Invoke-Expression)\b",
    re.IGNORECASE,
)
SHELL_DYNAMIC_EXPRESSION_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:eval\b|Invoke-Expression\b|iex\b)", re.IGNORECASE
)
VCS_REVISION_RE = re.compile(r"@[0-9a-fA-F]{40}(?:$|[#&])")
SECRET_NAME_RE = re.compile(
    r"(?:^|_)(?:ACCESS_KEY|ACCESS_TOKEN|API_KEY|AUTH_TOKEN|CLIENT_SECRET|"
    r"PASSWORD|PASSWD|PRIVATE_KEY|SECRET|TOKEN)$",
    re.IGNORECASE,
)
COMPOSE_IMAGE_RE = re.compile(r"^\s*image:\s*(.+?)\s*(?:#.*)?$", re.MULTILINE)
REQUIRED_IMAGE_DIGEST_VARIABLE_RE = re.compile(
    r"^\$\{[A-Z][A-Z0-9_]*_IMAGE_DIGEST:\?[^}\r\n]{1,128}\}$"
)
DOCKER_FROM_RE = re.compile(
    r"^\s*FROM(?:\s+--platform=\S+)?\s+(\S+)(?:\s+AS\s+(\S+))?",
    re.IGNORECASE | re.MULTILINE,
)
DOCKER_COPY_FROM_RE = re.compile(r"\bCOPY\s+--from=([^\s]+)", re.IGNORECASE)
DOCKER_ARG_RE = re.compile(
    r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(\S+))?", re.MULTILINE
)
LOCK_FILES = frozenset(
    {
        "Cargo.lock",
        "Pipfile.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "yarn.lock",
    }
)
SENSITIVE_FILE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".pypirc",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
SENSITIVE_FILE_SUFFIXES = frozenset(
    {".cer", ".jks", ".key", ".keystore", ".p12", ".pem", ".pfx"}
)
SECRET_MATERIAL_PATTERN = (
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----|"
    r"(^|[^A-Za-z0-9])((AKIA|ASIA)[A-Z0-9]{16}|"
    r"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"sk-lf-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{32,}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,})"
)
SECRET_MATERIAL_RE = re.compile(SECRET_MATERIAL_PATTERN)
TOKEN_VALUE_RE = re.compile(
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|"
    r"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"sk-lf-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{32,}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,}"
)
SYNTHETIC_TOKEN_MARKERS = frozenset(
    {
        "change-me",
        "changeme",
        "dummy",
        "example",
        "fake",
        "placeholder",
        "redacted",
        "replace-me",
        "sample",
        "set-me",
        "test",
        "your-",
        "xxxxx",
    }
)
MAX_SECRET_SCAN_BYTES = 16 * 1024 * 1024
_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_WORKSPACE_KEY_RE = re.compile(r"^( *)([A-Za-z0-9_-]+):\s*(?:#.*)?$")
_WORKSPACE_URL_RE = re.compile(
    r"^( *)-\s+url:\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s#]+))\s*(?:#.*)?$"
)
_PROVIDER_WORKSPACE_CONTEXT = (
    "subdirectories",
    "agent-packages",
    "subdirectories",
    "agents",
    "repositories",
)


@dataclass(frozen=True, order=True)
class Finding:
    repository: str
    path: str
    line: int
    rule: str
    message: str

    def render(self) -> str:
        location = f"{self.repository}/{self.path}"
        if self.line:
            location += f":{self.line}"
        return f"{location}: {self.rule}: {self.message}"


@dataclass
class SnapshotBudget:
    """Bounded aggregate inventory counters for one provider snapshot."""

    entries: int = 0
    files: int = 0
    bytes: int = 0


def _repository_label(repository: pathlib.Path, fleet_root: pathlib.Path) -> str:
    try:
        relative = repository.relative_to(fleet_root)
    except ValueError:
        return repository.name
    value = relative.as_posix()
    return repository.name if value == "." else value


def _discover_repositories(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    root = root.resolve()
    if (root / ".git").exists():
        return (root,)
    repositories: list[pathlib.Path] = []
    pending: list[tuple[pathlib.Path, int]] = [(root, 0)]
    while pending:
        current_path, depth = pending.pop()
        if (current_path / ".git").exists():
            repositories.append(current_path)
            if len(repositories) > MAX_REPOSITORIES:
                raise RuntimeError("repository discovery exceeds the safe bound")
            continue
        if depth >= MAX_DISCOVERY_DEPTH:
            continue
        try:
            children = sorted(
                (
                    pathlib.Path(entry.path)
                    for entry in os.scandir(current_path)
                    if entry.name not in SKIP_DIRECTORIES
                    and entry.is_dir(follow_symlinks=False)
                ),
                reverse=True,
            )
        except OSError:
            raise RuntimeError("repository discovery is unavailable") from None
        pending.extend((child, depth + 1) for child in children)
    return tuple(sorted(repositories))


def _workspace_provider_names(workspace: pathlib.Path) -> tuple[str, ...]:
    """Read the provider branch from the bounded repository-manager workspace."""

    try:
        metadata = workspace.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("snapshot workspace must be a regular file")
        if metadata.st_size > MAX_WORKSPACE_BYTES:
            raise RuntimeError("snapshot workspace exceeds the safe bound")
        text = workspace.read_text(encoding="utf-8")
        if len(text.encode("utf-8")) > MAX_WORKSPACE_BYTES:
            raise RuntimeError("snapshot workspace exceeds the safe bound")
    except RuntimeError:
        raise
    except (OSError, UnicodeError):
        raise RuntimeError("snapshot workspace is unavailable") from None

    stack: list[tuple[int, str]] = []
    providers: list[str] = []
    for line in text.splitlines():
        if "\t" in line:
            raise RuntimeError("snapshot workspace contains invalid indentation")
        key_match = _WORKSPACE_KEY_RE.fullmatch(line)
        if key_match is not None:
            indentation = len(key_match.group(1))
            while stack and stack[-1][0] >= indentation:
                stack.pop()
            stack.append((indentation, key_match.group(2)))
            continue
        url_match = _WORKSPACE_URL_RE.fullmatch(line)
        if url_match is None:
            continue
        context = tuple(value for _, value in stack)
        if context != _PROVIDER_WORKSPACE_CONTEXT:
            continue
        url = next(value for value in url_match.groups()[1:] if value is not None)
        if len(url) > 2_048 or any(ord(character) < 32 for character in url):
            raise RuntimeError("snapshot workspace contains an invalid provider URL")
        name = url.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        if _PROVIDER_NAME_RE.fullmatch(name) is None:
            raise RuntimeError("snapshot workspace contains an invalid provider name")
        providers.append(name)

    if len(providers) != EXPECTED_SNAPSHOT_PROVIDERS:
        raise RuntimeError("snapshot workspace provider count is not exact")
    if len(providers) != len(set(providers)):
        raise RuntimeError("snapshot workspace contains duplicate providers")
    return tuple(sorted(providers))


def resolve_snapshot_repositories(
    providers_root: pathlib.Path, workspace: pathlib.Path
) -> tuple[pathlib.Path, tuple[pathlib.Path, ...]]:
    """Resolve exactly the direct provider roots declared by the workspace."""

    try:
        root_metadata = providers_root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise RuntimeError("source snapshot root must be a directory")
        root = providers_root.resolve(strict=True)
        expected_workspace = (
            root / "repository-manager" / "repository_manager" / "workspace.yml"
        )
        if workspace.resolve(strict=True) != expected_workspace.resolve(strict=True):
            raise RuntimeError("snapshot workspace is not authoritative")
    except RuntimeError:
        raise
    except OSError:
        raise RuntimeError("source snapshot root is unavailable") from None

    providers = _workspace_provider_names(workspace)
    expected = set(providers)
    direct_directories: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name in SKIP_DIRECTORIES:
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    direct_directories.add(entry.name)
    except OSError:
        raise RuntimeError("source snapshot membership is unavailable") from None
    if direct_directories != expected:
        raise RuntimeError("source snapshot provider membership is not exact")

    repositories: list[pathlib.Path] = []
    for name in providers:
        repository = root / name
        try:
            metadata = repository.lstat()
        except OSError:
            raise RuntimeError("source snapshot provider is unavailable") from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("source snapshot provider must be a direct directory")
        repositories.append(repository)
    return root, tuple(repositories)


def _snapshot_source_files(
    repository: pathlib.Path, budget: SnapshotBudget
) -> tuple[pathlib.Path, ...]:
    """Inventory one source-only provider tree without following special entries."""

    paths: list[pathlib.Path] = []
    pending: list[tuple[pathlib.Path, int]] = [(repository, 0)]
    while pending:
        current, depth = pending.pop()
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            raise RuntimeError("source snapshot inventory is unavailable") from None
        directories: list[tuple[pathlib.Path, int]] = []
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                raise RuntimeError("source snapshot entry is unavailable") from None
            budget.entries += 1
            if budget.entries > MAX_SNAPSHOT_ENTRIES:
                raise RuntimeError("source snapshot entry count exceeds the safe bound")
            entry_depth = depth + 1
            if entry_depth > MAX_SNAPSHOT_DEPTH:
                raise RuntimeError("source snapshot depth exceeds the safe bound")
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError("source snapshot contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                if entry.name not in SKIP_DIRECTORIES:
                    directories.append((pathlib.Path(entry.path), entry_depth))
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("source snapshot contains a special file")
            if metadata.st_size > MAX_SOURCE_BYTES:
                raise RuntimeError("source snapshot file exceeds the safe bound")
            budget.files += 1
            budget.bytes += metadata.st_size
            if budget.files > MAX_SNAPSHOT_FILES:
                raise RuntimeError("source snapshot file count exceeds the safe bound")
            if budget.bytes > MAX_SNAPSHOT_BYTES:
                raise RuntimeError("source snapshot bytes exceed the safe bound")
            paths.append(pathlib.Path(entry.path))
        pending.extend(reversed(directories))
    return tuple(paths)


def _source_files(repository: pathlib.Path) -> tuple[pathlib.Path, ...]:
    command = [
        "git",
        "-C",
        str(repository),
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("Git source inventory is unavailable") from None
    paths: list[pathlib.Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            relative = pathlib.PurePosixPath(os.fsdecode(raw))
        except UnicodeError:
            raise RuntimeError(
                "Git source inventory contains an invalid path"
            ) from None
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Git source inventory contains an unsafe path")
        # ``git ls-files`` is authoritative for the source inventory. Avoid a
        # per-path stat here: on WSL/9P-backed workspaces thousands of metadata
        # round trips can stall an otherwise source-only gate. Relevant files
        # are bounded and validated immediately before they are read.
        paths.append(repository.joinpath(*relative.parts))
    return tuple(paths)


def _read_source(path: pathlib.Path) -> str:
    try:
        if path.is_symlink():
            raise RuntimeError("source file must not be a symlink")
        size = path.stat().st_size
        if size > MAX_SOURCE_BYTES:
            raise RuntimeError("source file exceeds the safe inspection bound")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise RuntimeError("source file is unreadable") from None


def _read_snapshot_bytes(path: pathlib.Path) -> bytes:
    """Read one already-inventoried regular file without following a symlink."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("source snapshot entry is not a regular file")
            if metadata.st_size > MAX_SOURCE_BYTES:
                raise RuntimeError("source snapshot file exceeds the safe bound")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_SOURCE_BYTES:
                    raise RuntimeError("source snapshot file exceeds the safe bound")
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    except RuntimeError:
        raise
    except OSError:
        raise RuntimeError("source snapshot file is unreadable") from None
    return b"".join(chunks)


def _read_snapshot_source(path: pathlib.Path) -> str:
    try:
        return _read_snapshot_bytes(path).decode("utf-8")
    except UnicodeError:
        raise RuntimeError("source snapshot file is unreadable") from None


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _is_workflow(relative: pathlib.PurePath) -> bool:
    parts = relative.parts
    return (
        len(parts) >= 3
        and parts[-3:-1] == (".github", "workflows")
        and relative.suffix.casefold() in {".yml", ".yaml"}
    )


def _workflow_findings(
    label: str, relative: pathlib.PurePath, text: str
) -> list[Finding]:
    findings: list[Finding] = []
    display = relative.as_posix()
    for match in EXTERNAL_USES_RE.finditer(text):
        reference = match.group(1)
        valid = reference.startswith("./")
        if reference.startswith("docker://"):
            valid = bool(
                CONTAINER_DIGEST_RE.search(reference.removeprefix("docker://") + " ")
            )
        elif not valid:
            valid = bool(ACTION_SHA_RE.fullmatch(reference))
        if not valid:
            findings.append(
                Finding(
                    label,
                    display,
                    _line_number(text, match.start()),
                    "SC-GHA-001",
                    "external action or reusable workflow is not pinned to a full commit SHA",
                )
            )

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.search(r"\buses:\s*actions/checkout@[0-9a-fA-F]{40}\b", line):
            continue
        indentation = len(line) - len(line.lstrip())
        block: list[str] = []
        for candidate in lines[index + 1 :]:
            candidate_indent = len(candidate) - len(candidate.lstrip())
            if candidate.strip() and (
                candidate_indent < indentation
                or (
                    candidate_indent == indentation
                    and candidate.lstrip().startswith("-")
                )
            ):
                break
            block.append(candidate)
        if not any(
            re.match(r"^\s*persist-credentials:\s*false\s*(?:#.*)?$", candidate)
            for candidate in block
        ):
            findings.append(
                Finding(
                    label,
                    display,
                    index + 1,
                    "SC-GHA-007",
                    "checkout must explicitly disable persisted credentials",
                )
            )

    permissions = re.search(r"^permissions:\s*(.*)$", text, re.MULTILINE)
    if permissions is None:
        findings.append(
            Finding(
                label,
                display,
                1,
                "SC-GHA-002",
                "top-level token permissions are not declared",
            )
        )
    else:
        inline = permissions.group(1).strip().casefold()
        if inline == "write-all" or "write-all" in inline:
            findings.append(
                Finding(
                    label,
                    display,
                    _line_number(text, permissions.start()),
                    "SC-GHA-003",
                    "top-level write-all token permissions are forbidden",
                )
            )
        elif re.search(r"\bwrite\b", inline):
            findings.append(
                Finding(
                    label,
                    display,
                    _line_number(text, permissions.start()),
                    "SC-GHA-004",
                    "repository-wide write permission must be scoped to the publishing job",
                )
            )
        start = permissions.end()
        for line_number, line in enumerate(
            text[start:].splitlines(), _line_number(text, start) + 1
        ):
            if line and not line[0].isspace() and not line.lstrip().startswith("#"):
                break
            if re.match(
                r"^\s{1,2}[a-z-]+:\s*write\s*(?:#.*)?$",
                line,
                re.IGNORECASE,
            ):
                findings.append(
                    Finding(
                        label,
                        display,
                        line_number,
                        "SC-GHA-004",
                        "repository-wide write permission must be scoped to the publishing job",
                    )
                )

    forbidden = (
        (
            r"^\s*pull_request_target:\s*",
            "SC-GHA-005",
            "pull_request_target requires an explicit security exception",
        ),
        (
            r"\bsecrets:\s*inherit\b",
            "SC-GHA-006",
            "blanket reusable-workflow secret inheritance is forbidden",
        ),
        (
            r"\bpermissions:\s*write-all\b",
            "SC-GHA-003",
            "write-all token permissions are forbidden",
        ),
        (
            r"\bpersist-credentials:\s*true\b",
            "SC-GHA-007",
            "checkout credentials must not be persisted explicitly",
        ),
        (
            r"ACTIONS_ALLOW_UNSECURE_COMMANDS",
            "SC-GHA-008",
            "insecure Actions command compatibility is forbidden",
        ),
    )
    for pattern, rule, message in forbidden:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            findings.append(
                Finding(
                    label, display, _line_number(text, match.start()), rule, message
                )
            )
    for match in NETWORK_TO_SHELL_RE.finditer(text):
        findings.append(
            Finding(
                label,
                display,
                _line_number(text, match.start()),
                "SC-GHA-009",
                "network response is executed by a shell",
            )
        )
    return findings


def _is_precommit(relative: pathlib.PurePath) -> bool:
    return relative.name == ".pre-commit-config.yaml"


def _precommit_findings(
    label: str, relative: pathlib.PurePath, text: str
) -> list[Finding]:
    findings: list[Finding] = []
    display = relative.as_posix()
    pending: tuple[int, str] | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        repository_match = re.match(r"^\s*-\s*repo:\s*([^\s#]+)", line)
        if repository_match:
            if pending is not None:
                findings.append(
                    Finding(
                        label,
                        display,
                        pending[0],
                        "SC-HOOK-001",
                        "external pre-commit hook repository has no immutable revision",
                    )
                )
            repository = repository_match.group(1).strip("\"'")
            pending = (
                None if repository in {"local", "meta"} else (line_number, repository)
            )
            continue
        if pending is None:
            continue
        revision_match = re.match(r"^\s*rev:\s*([^\s#]+)", line)
        if revision_match:
            revision = revision_match.group(1).strip("\"'")
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                findings.append(
                    Finding(
                        label,
                        display,
                        line_number,
                        "SC-HOOK-001",
                        "external pre-commit hook is not pinned to a full commit SHA",
                    )
                )
            pending = None
    if pending is not None:
        findings.append(
            Finding(
                label,
                display,
                pending[0],
                "SC-HOOK-001",
                "external pre-commit hook repository has no immutable revision",
            )
        )

    for match in re.finditer(
        r"additional_dependencies:\s*\[([^\]]*)\]", text, re.DOTALL
    ):
        for raw_dependency in match.group(1).split(","):
            dependency = raw_dependency.strip().strip("\"'")
            if not dependency:
                continue
            python_pin = re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9_,.-]+\])?==[^\s]+",
                dependency,
            )
            npm_pin = re.fullmatch(
                r"@[^/\s]+/[^@\s]+@[0-9][0-9A-Za-z.+_-]*", dependency
            )
            vcs_pin = re.search(r"@[0-9a-fA-F]{40}(?:#|$)", dependency)
            if not (python_pin or npm_pin or vcs_pin):
                findings.append(
                    Finding(
                        label,
                        display,
                        _line_number(text, match.start()),
                        "SC-HOOK-002",
                        "pre-commit additional dependency is not pinned exactly",
                    )
                )
    return findings


def _is_dockerfile(relative: pathlib.PurePath) -> bool:
    name = relative.name.casefold()
    return (
        name == "dockerfile"
        or name.startswith("dockerfile.")
        or name.endswith(".dockerfile")
    )


def _is_compose(relative: pathlib.PurePath) -> bool:
    name = relative.name.casefold()
    return relative.suffix.casefold() in {".yml", ".yaml"} and (
        "compose" in name or name.endswith(".stack.yml") or name.endswith(".stack.yaml")
    )


def _is_installer(relative: pathlib.PurePath) -> bool:
    name = relative.name.casefold()
    return name == "promote_local_release.py" or name in {
        "bootstrap.ps1",
        "bootstrap.sh",
        "install.ps1",
        "install.sh",
        "setup.ps1",
        "setup.sh",
    }


def _installer_findings(
    label: str, relative: pathlib.PurePath, text: str
) -> list[Finding]:
    """Reject network execution and expression evaluation in bootstrap code."""

    findings: list[Finding] = []
    display = relative.as_posix()
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if NETWORK_TO_SHELL_RE.search(
            line
        ) or POWERSHELL_NETWORK_TO_EXPRESSION_RE.search(line):
            findings.append(
                Finding(
                    label,
                    display,
                    line_number,
                    "SC-INSTALL-001",
                    "installer executes an unverified network response",
                )
            )
        if SHELL_DYNAMIC_EXPRESSION_RE.search(line):
            findings.append(
                Finding(
                    label,
                    display,
                    line_number,
                    "SC-INSTALL-002",
                    "installer uses dynamic expression evaluation instead of argument-safe execution",
                )
            )
    return findings


def _docker_findings(
    label: str, relative: pathlib.PurePath, text: str
) -> list[Finding]:
    findings: list[Finding] = []
    display = relative.as_posix()
    arguments = {
        match.group(1): match.group(2) for match in DOCKER_ARG_RE.finditer(text)
    }
    stages: set[str] = set()
    for match in DOCKER_FROM_RE.finditer(text):
        image, stage = match.group(1), match.group(2)
        if stage:
            stages.add(stage.casefold())
        valid = image.casefold() == "scratch" or image.casefold() in stages
        variable_names = re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", image)
        if len(variable_names) == 1:
            name = variable_names[0]
            default = arguments.get(name)
            if name in arguments and default is None:
                valid = image in {f"${name}", f"${{{name}}}"}
            elif name in arguments and default is not None:
                resolved = image.replace(f"${{{name}}}", default).replace(
                    f"${name}", default
                )
                valid = bool(CONTAINER_DIGEST_RE.search(resolved + " "))
        elif not valid:
            valid = bool(CONTAINER_DIGEST_RE.search(image + " "))
        if not valid:
            findings.append(
                Finding(
                    label,
                    display,
                    _line_number(text, match.start()),
                    "SC-CTR-001",
                    "external base image is not pinned to sha256",
                )
            )

    for match in DOCKER_COPY_FROM_RE.finditer(text):
        source = match.group(1)
        valid = (
            source.isdigit()
            or source.casefold() in stages
            or source.startswith("$")
            or bool(CONTAINER_DIGEST_RE.search(source + " "))
        )
        if not valid:
            findings.append(
                Finding(
                    label,
                    display,
                    _line_number(text, match.start()),
                    "SC-CTR-002",
                    "external COPY --from image is not pinned to sha256",
                )
            )

    for match in NETWORK_TO_SHELL_RE.finditer(text):
        findings.append(
            Finding(
                label,
                display,
                _line_number(text, match.start()),
                "SC-CTR-003",
                "network response is executed by a shell during the build",
            )
        )
    for match in re.finditer(
        r"^\s*ADD\s+https?://", text, re.IGNORECASE | re.MULTILINE
    ):
        findings.append(
            Finding(
                label,
                display,
                _line_number(text, match.start()),
                "SC-CTR-004",
                "remote ADD bypasses explicit digest verification",
            )
        )
    for match in re.finditer(
        r"^\s*(?:ARG|ENV)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE
    ):
        if SECRET_NAME_RE.search(match.group(1)):
            findings.append(
                Finding(
                    label,
                    display,
                    _line_number(text, match.start()),
                    "SC-CTR-005",
                    "secret-shaped build ARG/ENV can persist in image metadata; use a secret mount",
                )
            )
    return findings


def _compose_findings(
    label: str, relative: pathlib.PurePath, text: str
) -> list[Finding]:
    findings: list[Finding] = []
    display = relative.as_posix()
    for match in COMPOSE_IMAGE_RE.finditer(text):
        image = match.group(1).strip().strip("\"'")
        if image.endswith((":local", ":dev-local")):
            continue
        if image.startswith("${"):
            valid = ":?" in image and "@sha256" in image
            valid = valid or bool(REQUIRED_IMAGE_DIGEST_VARIABLE_RE.fullmatch(image))
            valid = valid or bool(CONTAINER_DIGEST_RE.search(image + " "))
            valid = valid or bool(re.search(r":-[^}]+:(?:dev-)?local}", image))
        else:
            valid = bool(CONTAINER_DIGEST_RE.search(image + " "))
        if not valid:
            findings.append(
                Finding(
                    label,
                    display,
                    _line_number(text, match.start()),
                    "SC-CTR-006",
                    "runtime image must be a digest or a required immutable image variable",
                )
            )
    dangerous = (
        (
            r"^\s*privileged:\s*true\s*$",
            "privileged containers require a reviewed exception",
        ),
        (
            r"^\s*(?:network_mode|pid|ipc):\s*host\s*$",
            "host namespace sharing requires a reviewed exception",
        ),
        (
            r"^\s*security_opt:.*unconfined",
            "unconfined container security profiles are forbidden",
        ),
        (r"/var/run/docker\.sock", "mounting the Docker control socket is forbidden"),
    )
    for pattern, message in dangerous:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            findings.append(
                Finding(
                    label,
                    display,
                    _line_number(text, match.start()),
                    "SC-CTR-007",
                    message,
                )
            )
    return findings


def _dependency_findings(
    label: str,
    repository: pathlib.Path,
    sources: tuple[pathlib.Path, ...],
    *,
    source_reader: Callable[[pathlib.Path], str] = _read_source,
) -> list[Finding]:
    names = {path.name for path in sources if path.parent == repository}
    findings: list[Finding] = []
    if "pyproject.toml" in names and not names.intersection(
        {"Pipfile.lock", "poetry.lock", "uv.lock"}
    ):
        findings.append(
            Finding(
                label,
                "pyproject.toml",
                1,
                "SC-DEP-001",
                "Python dependency manifest has no committed resolver lock",
            )
        )
    if "package.json" in names and not names.intersection(
        {"package-lock.json", "pnpm-lock.yaml", "yarn.lock"}
    ):
        findings.append(
            Finding(
                label,
                "package.json",
                1,
                "SC-DEP-002",
                "Node dependency manifest has no committed resolver lock",
            )
        )
    if "Cargo.toml" in names and "Cargo.lock" not in names:
        findings.append(
            Finding(
                label,
                "Cargo.toml",
                1,
                "SC-DEP-003",
                "Rust application/workspace manifest has no committed Cargo.lock",
            )
        )

    pyproject = repository / "pyproject.toml"
    if "pyproject.toml" in names:
        try:
            pyproject_text = source_reader(pyproject)
            document = tomllib.loads(pyproject_text)
        except tomllib.TOMLDecodeError:
            findings.append(
                Finding(
                    label,
                    "pyproject.toml",
                    1,
                    "SC-DEP-004",
                    "Python dependency manifest is invalid",
                )
            )
        except RuntimeError as error:
            findings.append(
                Finding(label, "pyproject.toml", 0, "SC-SRC-001", str(error))
            )
        else:
            dependency_values: list[str] = []
            project = document.get("project", {})
            if isinstance(project, dict):
                dependencies = project.get("dependencies", [])
                if isinstance(dependencies, list):
                    dependency_values.extend(
                        item for item in dependencies if isinstance(item, str)
                    )
                optional = project.get("optional-dependencies", {})
                if isinstance(optional, dict):
                    for values in optional.values():
                        if isinstance(values, list):
                            dependency_values.extend(
                                item for item in values if isinstance(item, str)
                            )
            groups = document.get("dependency-groups", {})
            if isinstance(groups, dict):
                for values in groups.values():
                    if isinstance(values, list):
                        dependency_values.extend(
                            item for item in values if isinstance(item, str)
                        )

            for dependency in dependency_values:
                casefolded = dependency.casefold()
                unsafe_vcs = "git+" in casefolded and (
                    "git+https://" not in casefolded
                    or VCS_REVISION_RE.search(dependency) is None
                )
                unsafe_archive = re.search(
                    r"\s@\s*https?://", dependency, re.IGNORECASE
                ) and (" @ https://" not in casefolded or "#sha256=" not in casefolded)
                if not unsafe_vcs and not unsafe_archive:
                    continue
                offset = pyproject_text.find(dependency)
                findings.append(
                    Finding(
                        label,
                        "pyproject.toml",
                        _line_number(pyproject_text, max(offset, 0)),
                        "SC-DEP-004",
                        "direct network dependency must use HTTPS and an immutable revision or SHA-256 digest",
                    )
                )
    return findings


def _secret_content_finding(
    label: str, path: str, line: int, content: str
) -> Finding | None:
    """Classify one text line with the shared credential-material policy."""

    private_key = bool(re.search(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", content))
    tokens = TOKEN_VALUE_RE.findall(content)

    def token_payload(token: str) -> str:
        return re.sub(
            r"^(?:AKIA|ASIA|github_pat_|gh[pousr]_|sk-lf-|sk-|xox[baprs]-)",
            "",
            token.casefold(),
        )

    credible_token = any(
        not any(marker in token.casefold() for marker in SYNTHETIC_TOKEN_MARKERS)
        and len(set(token_payload(token))) > 4
        for token in tokens
    )
    if not private_key and not credible_token:
        return None
    return Finding(
        label,
        pathlib.PurePosixPath(path).as_posix(),
        line,
        "SC-SEC-002",
        "tracked source resembles live credential or private-key material",
    )


def _secret_findings(
    label: str, repository: pathlib.Path, sources: tuple[pathlib.Path, ...]
) -> list[Finding]:
    findings: list[Finding] = []
    for path in sources:
        relative = path.relative_to(repository)
        name = relative.name.casefold()
        if (
            name in SENSITIVE_FILE_NAMES
            or relative.suffix.casefold() in SENSITIVE_FILE_SUFFIXES
        ):
            findings.append(
                Finding(
                    label,
                    relative.as_posix(),
                    0,
                    "SC-SEC-001",
                    "tracked credential, private-key, or environment trust file is forbidden",
                )
            )

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "grep",
                "-I",
                "-n",
                "-E",
                "-e",
                SECRET_MATERIAL_PATTERN,
                "--",
                ".",
            ],
            check=False,
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{label}: Git secret scan exceeded the safe time bound"
        ) from None
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError(f"{label}: Git secret scan is unavailable") from None
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"{label}: Git secret scan failed")
    if len(result.stdout) > MAX_SECRET_SCAN_BYTES:
        raise RuntimeError("Git secret scan output exceeds the safe bound")
    for raw in result.stdout.splitlines():
        parts = raw.split(b":", 2)
        if len(parts) < 3:
            raise RuntimeError("Git secret scan returned an invalid location")
        try:
            path = os.fsdecode(parts[0])
            line = int(parts[1])
            content = parts[2].decode("utf-8", errors="replace")
        except (UnicodeError, ValueError):
            raise RuntimeError("Git secret scan returned an invalid location") from None
        finding = _secret_content_finding(label, path, line, content)
        if finding is not None:
            findings.append(finding)
    return findings


def _snapshot_secret_findings(
    label: str, repository: pathlib.Path, sources: tuple[pathlib.Path, ...]
) -> list[Finding]:
    """Apply the Git secret policy to a bounded no-Git source inventory."""

    findings: list[Finding] = []
    matched_bytes = 0
    for path in sources:
        relative = path.relative_to(repository)
        name = relative.name.casefold()
        if (
            name in SENSITIVE_FILE_NAMES
            or relative.suffix.casefold() in SENSITIVE_FILE_SUFFIXES
        ):
            findings.append(
                Finding(
                    label,
                    relative.as_posix(),
                    0,
                    "SC-SEC-001",
                    "tracked credential, private-key, or environment trust file is forbidden",
                )
            )
        content = _read_snapshot_bytes(path)
        if b"\0" in content:
            continue
        text = content.decode("utf-8", errors="replace")
        for line, value in enumerate(text.splitlines(), start=1):
            if SECRET_MATERIAL_RE.search(value) is None:
                continue
            matched_bytes += len(relative.as_posix().encode("utf-8"))
            matched_bytes += len(value.encode("utf-8")) + 32
            if matched_bytes > MAX_SECRET_SCAN_BYTES:
                raise RuntimeError("source snapshot secret scan exceeds the safe bound")
            finding = _secret_content_finding(label, relative.as_posix(), line, value)
            if finding is not None:
                findings.append(finding)
    return findings


def _inspect_source_inventory(
    repository: pathlib.Path,
    fleet_root: pathlib.Path,
    sources: tuple[pathlib.Path, ...],
    *,
    source_reader: Callable[[pathlib.Path], str],
    secret_scanner: Callable[
        [str, pathlib.Path, tuple[pathlib.Path, ...]], list[Finding]
    ],
    allow_missing: bool,
) -> tuple[list[Finding], int]:
    label = _repository_label(repository, fleet_root)
    findings = _dependency_findings(
        label, repository, sources, source_reader=source_reader
    )
    findings.extend(secret_scanner(label, repository, sources))
    inspected = 0
    for path in sources:
        relative = path.relative_to(repository)
        if not (
            _is_workflow(relative)
            or _is_precommit(relative)
            or _is_dockerfile(relative)
            or _is_compose(relative)
            or _is_installer(relative)
        ):
            continue
        if not path.exists():
            # A tracked path deleted in the working tree is not executable
            # source. This commonly occurs while consolidating workflows.
            if allow_missing:
                continue
            raise RuntimeError("source snapshot entry became unavailable")
        try:
            text = source_reader(path)
        except RuntimeError as error:
            findings.append(
                Finding(label, relative.as_posix(), 0, "SC-SRC-001", str(error))
            )
            continue
        inspected += 1
        if _is_workflow(relative):
            findings.extend(_workflow_findings(label, relative, text))
        if _is_precommit(relative):
            findings.extend(_precommit_findings(label, relative, text))
        if _is_dockerfile(relative):
            findings.extend(_docker_findings(label, relative, text))
        if _is_compose(relative):
            findings.extend(_compose_findings(label, relative, text))
        if _is_installer(relative):
            findings.extend(_installer_findings(label, relative, text))
    return findings, inspected


def inspect_repository(
    repository: pathlib.Path, fleet_root: pathlib.Path
) -> tuple[list[Finding], int]:
    """Inspect the unchanged Git-backed source inventory."""

    return _inspect_source_inventory(
        repository,
        fleet_root,
        _source_files(repository),
        source_reader=_read_source,
        secret_scanner=_secret_findings,
        allow_missing=True,
    )


def inspect_snapshot_repository(
    repository: pathlib.Path,
    fleet_root: pathlib.Path,
    budget: SnapshotBudget,
) -> tuple[list[Finding], int]:
    """Inspect one workspace-authorized source snapshot provider."""

    return _inspect_source_inventory(
        repository,
        fleet_root,
        _snapshot_source_files(repository, budget),
        source_reader=_read_snapshot_source,
        secret_scanner=_snapshot_secret_findings,
        allow_missing=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="single repository root (default: current directory)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fleet-root",
        help="discover and inspect nested Git repositories below this directory",
    )
    mode.add_argument(
        "--source-snapshot-root",
        help="inspect exactly the workspace-declared direct provider source roots",
    )
    parser.add_argument(
        "--snapshot-workspace",
        help="authoritative repository-manager workspace for source-snapshot mode",
    )
    arguments = parser.parse_args(argv)
    snapshot_mode = arguments.source_snapshot_root is not None
    if snapshot_mode != (arguments.snapshot_workspace is not None):
        parser.error(
            "--source-snapshot-root and --snapshot-workspace must be used together"
        )
    try:
        if snapshot_mode:
            fleet_root, repositories = resolve_snapshot_repositories(
                pathlib.Path(arguments.source_snapshot_root),
                pathlib.Path(arguments.snapshot_workspace),
            )
            budget = SnapshotBudget()
        else:
            fleet_root = pathlib.Path(arguments.fleet_root or arguments.root).resolve()
            repositories = _discover_repositories(fleet_root)
            if not repositories:
                raise RuntimeError("no Git repositories were discovered")
        findings: list[Finding] = []
        inspected = 0
        for repository in repositories:
            if snapshot_mode:
                repository_findings, repository_inspected = inspect_snapshot_repository(
                    repository, fleet_root, budget
                )
            else:
                repository_findings, repository_inspected = inspect_repository(
                    repository, fleet_root
                )
            findings.extend(repository_findings)
            inspected += repository_inspected
    except RuntimeError as error:
        print(f"supply-chain: FAILED - {error}", file=sys.stderr)
        return 2
    if findings:
        for finding in sorted(set(findings)):
            print(f"  FAIL {finding.render()}")
        print(
            f"supply-chain: FAILED ({len(set(findings))} findings across "
            f"{len(repositories)} repositories)",
            file=sys.stderr,
        )
        return 1
    print(
        f"supply-chain: clean ({len(repositories)} repositories, "
        f"{inspected} workflow/container assets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
