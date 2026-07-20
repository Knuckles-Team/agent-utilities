#!/usr/bin/env python3
"""Run the complete source-only release gate set from one immutable tree state.

The runner deliberately has no command-line escape hatch for ad-hoc commands.
Every command comes from the reviewed manifest, executes as an argv array without
a shell, and is confined to source inspection.  Command output is bounded and
discarded so host paths, identities, endpoints, and credentials cannot enter the
retained evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

MANIFEST_SCHEMA: Final = "source-freeze-gates/1"
EVIDENCE_SCHEMA: Final = "source-freeze-evidence/1"
CANONICAL_MANIFEST_SHA256: Final = (
    "ec025ae174e5e334499a9b6344d9b33aaa1c349efd204e49f6d83c6c5be1fc4d"
)
REPOSITORY_IDS: Final = (
    "agent-utilities",
    "epistemic-graph",
    "langfuse-agent",
    "provider-fleet",
)
EXPECTED_GATES: Final = tuple(f"G-{number:02d}" for number in range(1, 40))
EVIDENCE_CLASSES: Final = (
    "local-source",
    "exact-artifact",
    "external",
    "terminal",
)
_ID = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
_REPO_TOKEN = re.compile(r"^\{repo:([a-z][a-z0-9-]{2,63})\}(.*)$")
_LOCAL_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/]|/(?:home|users|mnt|tmp|var|opt|root)(?:/|\b)|file://)"
)
_URL = re.compile(r"(?i)^[a-z][a-z0-9+.-]*://")
_SCRIPT_NAMES = re.compile(
    r"^(?:check_[a-z0-9_]+|audit_fleet_dependencies|docs_contract|"
    r"security_contract|security_sanitizer|verify_api_integration)\.py$"
)
_FORBIDDEN_ARGUMENTS: Final = frozenset(
    {
        "--build",
        "--fix",
        "--generate",
        "--install",
        "--live",
        "--network",
        "--output",
        "--rendered",
        "--serve",
        "--start",
        "--write",
    }
)
_EXCLUDED_DIRECTORIES: Final = frozenset(
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
        ".worktrees",
        "__pycache__",
        "node_modules",
        "security-results",
        "venv",
        "workspace",
    }
)
_EXCLUDED_FILES: Final = frozenset({".coverage", ".DS_Store", "coverage.xml"})
_MAX_FILES: Final = 500_000
_MAX_BYTES: Final = 20 * 1024 * 1024 * 1024
_MAX_OUTPUT_BYTES: Final = 8 * 1024 * 1024
_MAX_EVIDENCE_BYTES: Final = 1024 * 1024
_SUPPORTS_ATOMIC_EVIDENCE: Final = (
    os.link in os.supports_dir_fd
    and os.link in os.supports_follow_symlinks
    and os.unlink in os.supports_dir_fd
)


class GateError(RuntimeError):
    """A privacy-safe, stable source-freeze failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Command:
    identifier: str
    repository: str
    argv: tuple[str, ...]
    timeout_seconds: int
    covers: tuple[str, ...]


@dataclass(frozen=True)
class Gate:
    identifier: str
    evidence_classes: tuple[str, ...]
    scope: tuple[str, ...]
    command_ids: tuple[str, ...]


@dataclass
class EvidenceTarget:
    parent_descriptor: int
    parent_device: int
    parent_inode: int
    name: str

    def close(self) -> None:
        os.close(self.parent_descriptor)


@dataclass(frozen=True)
class Manifest:
    digest: str
    commands: tuple[Command, ...]
    gates: tuple[Gate, ...]


def _expect_mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GateError(code)
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise GateError(code)


def _safe_text(value: Any, code: str, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
        or _LOCAL_PATH.search(value)
    ):
        raise GateError(code)
    return value


def _safe_identifier(value: Any, code: str) -> str:
    text = _safe_text(value, code, maximum=64)
    if _ID.fullmatch(text) is None:
        raise GateError(code)
    return text


def _safe_string_list(value: Any, code: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise GateError(code)
    values = tuple(_safe_text(item, code) for item in value)
    if len(values) != len(set(values)):
        raise GateError(code)
    return values


def _validate_script_argv(argv: tuple[str, ...], repositories: set[str]) -> None:
    if len(argv) < 2 or argv[0] != "{python}":
        raise GateError("manifest-command-executable")
    script = PurePosixPath(argv[1])
    allowed_parent = script.parts[:-1] in {
        ("scripts",),
        ("scripts", "deployment"),
        ("scripts", "release"),
        ("scripts", "security"),
    }
    if (
        script.is_absolute()
        or ".." in script.parts
        or not allowed_parent
        or _SCRIPT_NAMES.fullmatch(script.name) is None
    ):
        raise GateError("manifest-command-script")
    for token in argv[2:]:
        if token.casefold() in _FORBIDDEN_ARGUMENTS or _URL.search(token):
            raise GateError("manifest-command-forbidden")
        match = _REPO_TOKEN.fullmatch(token)
        if match is None:
            if "{" in token or "}" in token:
                raise GateError("manifest-command-placeholder")
            continue
        repository, suffix = match.groups()
        if repository not in repositories:
            raise GateError("manifest-command-repository")
        if suffix:
            if not suffix.startswith("/"):
                raise GateError("manifest-command-placeholder")
            relative = PurePosixPath(suffix[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise GateError("manifest-command-placeholder")


def load_manifest(path: Path) -> Manifest:
    """Load and strictly validate the complete current source-gate manifest."""

    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise GateError("manifest-type")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or not stat.S_ISREG(opened.st_mode)
            ):
                raise GateError("manifest-race")
            raw = b""
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                raw += chunk
                if len(raw) > 1024 * 1024:
                    raise GateError("manifest-bound")
        finally:
            os.close(descriptor)
        value = json.loads(raw)
    except GateError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError("manifest-unreadable") from exc
    root = _expect_mapping(value, "manifest-root")
    _expect_keys(
        root,
        {"schema", "repositories", "commands", "gates"},
        "manifest-root-fields",
    )
    if root["schema"] != MANIFEST_SCHEMA:
        raise GateError("manifest-schema")

    repositories_value = root["repositories"]
    if not isinstance(repositories_value, list):
        raise GateError("manifest-repositories")
    repositories: list[str] = []
    for item in repositories_value:
        entry = _expect_mapping(item, "manifest-repository")
        _expect_keys(entry, {"id", "kind"}, "manifest-repository-fields")
        identifier = _safe_identifier(entry["id"], "manifest-repository-id")
        if entry["kind"] not in {"repository", "fleet"}:
            raise GateError("manifest-repository-kind")
        repositories.append(identifier)
    if tuple(repositories) != REPOSITORY_IDS or len(repositories) != len(
        set(repositories)
    ):
        raise GateError("manifest-repository-set")
    repository_set = set(repositories)

    commands_value = root["commands"]
    if not isinstance(commands_value, list) or not commands_value:
        raise GateError("manifest-commands")
    commands: list[Command] = []
    for item in commands_value:
        entry = _expect_mapping(item, "manifest-command")
        _expect_keys(
            entry,
            {"id", "repository", "argv", "timeout_seconds", "covers"},
            "manifest-command-fields",
        )
        identifier = _safe_identifier(entry["id"], "manifest-command-id")
        repository = _safe_identifier(
            entry["repository"], "manifest-command-repository"
        )
        if repository not in repository_set:
            raise GateError("manifest-command-repository")
        argv = _safe_string_list(entry["argv"], "manifest-command-argv")
        _validate_script_argv(argv, repository_set)
        timeout = entry["timeout_seconds"]
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or not 5 <= timeout <= 600
        ):
            raise GateError("manifest-command-timeout")
        covers = _safe_string_list(entry["covers"], "manifest-command-covers")
        if any(gate not in EXPECTED_GATES for gate in covers):
            raise GateError("manifest-command-gate")
        commands.append(Command(identifier, repository, argv, timeout, covers))
    command_ids = [command.identifier for command in commands]
    if len(command_ids) != len(set(command_ids)):
        raise GateError("manifest-command-duplicate")

    gates_value = root["gates"]
    if not isinstance(gates_value, list):
        raise GateError("manifest-gates")
    gates: list[Gate] = []
    for item in gates_value:
        entry = _expect_mapping(item, "manifest-gate")
        _expect_keys(
            entry,
            {"id", "evidence_classes", "scope", "command_ids", "rationale"},
            "manifest-gate-fields",
        )
        identifier = _safe_text(entry["id"], "manifest-gate-id", maximum=4)
        evidence_classes = _safe_string_list(
            entry["evidence_classes"], "manifest-gate-evidence-classes"
        )
        if any(value not in EVIDENCE_CLASSES for value in evidence_classes):
            raise GateError("manifest-gate-evidence-class")
        expected_order = tuple(
            value for value in EVIDENCE_CLASSES if value in evidence_classes
        )
        if evidence_classes != expected_order:
            raise GateError("manifest-gate-evidence-order")
        if "terminal" in evidence_classes and evidence_classes != ("terminal",):
            raise GateError("manifest-gate-terminal")
        scope = _safe_string_list(entry["scope"], "manifest-gate-scope")
        if any(repository not in repository_set for repository in scope):
            raise GateError("manifest-gate-scope")
        ids_value = entry["command_ids"]
        if not isinstance(ids_value, list):
            raise GateError("manifest-gate-commands")
        ids = tuple(
            _safe_identifier(command_id, "manifest-gate-command-id")
            for command_id in ids_value
        )
        if len(ids) != len(set(ids)):
            raise GateError("manifest-gate-command-duplicate")
        _safe_text(entry["rationale"], "manifest-gate-rationale")
        if "local-source" in evidence_classes and not ids:
            raise GateError("manifest-gate-command-missing")
        if "local-source" not in evidence_classes and ids:
            raise GateError("manifest-nonlocal-command")
        gates.append(Gate(identifier, evidence_classes, scope, ids))

    gate_ids = tuple(gate.identifier for gate in gates)
    if gate_ids != EXPECTED_GATES or len(gate_ids) != len(set(gate_ids)):
        raise GateError("manifest-gate-set")
    command_by_id = {command.identifier: command for command in commands}
    references: dict[str, set[str]] = {
        identifier: set() for identifier in command_by_id
    }
    for gate in gates:
        for command_id in gate.command_ids:
            if command_id not in command_by_id:
                raise GateError("manifest-command-unlisted")
            references[command_id].add(gate.identifier)
    for command in commands:
        if not references[command.identifier]:
            raise GateError("manifest-command-orphan")
        if references[command.identifier] != set(command.covers):
            raise GateError("manifest-command-coverage")
        command_scope = {command.repository}
        command_scope.update(
            match.group(1)
            for token in command.argv
            if (match := _REPO_TOKEN.fullmatch(token)) is not None
        )
        for gate in gates:
            if gate.identifier in references[
                command.identifier
            ] and not command_scope <= set(gate.scope):
                raise GateError("manifest-command-scope")
    for gate in gates:
        if "local-source" not in gate.evidence_classes:
            continue
        covered_scope: set[str] = set()
        for command_id in gate.command_ids:
            command = command_by_id[command_id]
            covered_scope.add(command.repository)
            covered_scope.update(
                match.group(1)
                for token in command.argv
                if (match := _REPO_TOKEN.fullmatch(token)) is not None
            )
        if covered_scope != set(gate.scope):
            raise GateError("manifest-gate-scope-coverage")

    return Manifest(hashlib.sha256(raw).hexdigest(), tuple(commands), tuple(gates))


def load_canonical_manifest(path: Path) -> Manifest:
    """Load the reviewed adjacent catalog and require its source-pinned digest."""

    manifest = load_manifest(path)
    if manifest.digest != CANONICAL_MANIFEST_SHA256:
        raise GateError("manifest-pin")
    return manifest


def parse_repository_roots(values: Sequence[str]) -> dict[str, Path]:
    """Resolve exactly one explicit, non-symlink root for each manifest alias."""

    roots: dict[str, Path] = {}
    for value in values:
        identifier, separator, raw_path = value.partition("=")
        if not separator or identifier not in REPOSITORY_IDS or identifier in roots:
            raise GateError("repository-argument")
        path = Path(raw_path)
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise GateError("repository-unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise GateError("repository-type")
        roots[identifier] = resolved
    if tuple(sorted(roots)) != tuple(sorted(REPOSITORY_IDS)):
        raise GateError("repository-set")
    if len(set(roots.values())) != len(roots):
        raise GateError("repository-duplicate")
    required_entries = {
        "agent-utilities": ("pyproject.toml", "agent_utilities", "scripts"),
        "epistemic-graph": ("Cargo.toml", "crates", "scripts"),
        "langfuse-agent": ("pyproject.toml", "langfuse_agent", "scripts"),
        "provider-fleet": (
            "agents",
            "skills",
            "agents/repository-manager/repository_manager/workspace.yml",
        ),
    }
    for identifier, entries in required_entries.items():
        if any(not (roots[identifier] / entry).exists() for entry in entries):
            raise GateError("repository-membership")
    expected_members = {
        "agent-utilities": roots["provider-fleet"] / "agent-utilities",
        "epistemic-graph": roots["provider-fleet"] / "epistemic-graph",
        "langfuse-agent": roots["provider-fleet"] / "agents" / "langfuse-agent",
    }
    try:
        if any(
            candidate.resolve(strict=True) != roots[identifier]
            for identifier, candidate in expected_members.items()
        ):
            raise GateError("repository-membership")
    except OSError as exc:
        raise GateError("repository-membership") from exc
    return roots


def _excluded_directory(name: str) -> bool:
    return name in _EXCLUDED_DIRECTORIES or name.endswith(".egg-info")


def _excluded_file(name: str) -> bool:
    return name in _EXCLUDED_FILES or name.endswith((".pyc", ".pyo"))


def source_tree_digest(root: Path) -> str:
    """Return a path-independent digest of every non-generated source entry."""

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for directory, directory_names, file_names in os.walk(root, topdown=True):
        current = Path(directory)
        if current != root:
            relative_directory = current.relative_to(root).as_posix().encode("utf-8")
            try:
                directory_metadata = current.lstat()
            except OSError as exc:
                raise GateError("source-tree-read") from exc
            if not stat.S_ISDIR(directory_metadata.st_mode):
                raise GateError("source-tree-race")
            digest.update(b"D")
            digest.update(len(relative_directory).to_bytes(8, "big"))
            digest.update(relative_directory)
            digest.update(directory_metadata.st_mode.to_bytes(8, "big"))
        traversable: list[str] = []
        symlink_directories: list[str] = []
        for name in sorted(directory_names):
            if _excluded_directory(name):
                continue
            try:
                metadata = (Path(directory) / name).lstat()
            except OSError as exc:
                raise GateError("source-tree-read") from exc
            if stat.S_ISLNK(metadata.st_mode):
                symlink_directories.append(name)
            elif stat.S_ISDIR(metadata.st_mode):
                traversable.append(name)
            else:
                raise GateError("source-tree-special-file")
        directory_names[:] = traversable
        for name in sorted([*file_names, *symlink_directories]):
            if name == ".git" or _excluded_file(name):
                continue
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix().encode("utf-8")
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise GateError("source-tree-read") from exc
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.readlink(path).encode("utf-8")
                except OSError as exc:
                    raise GateError("source-tree-read") from exc
                digest.update(b"L")
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
                file_count += 1
                byte_count += len(target)
                if file_count > _MAX_FILES or byte_count > _MAX_BYTES:
                    raise GateError("source-tree-bound")
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise GateError("source-tree-special-file")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags)
                try:
                    opened = os.fstat(descriptor)
                    if (
                        opened.st_dev != metadata.st_dev
                        or opened.st_ino != metadata.st_ino
                        or opened.st_size != metadata.st_size
                    ):
                        raise GateError("source-tree-race")
                    digest.update(b"F")
                    digest.update(metadata.st_mode.to_bytes(8, "big"))
                    while True:
                        chunk = os.read(descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        byte_count += len(chunk)
                finally:
                    os.close(descriptor)
                after = path.lstat()
            except GateError:
                raise
            except OSError as exc:
                raise GateError("source-tree-read") from exc
            if (
                after.st_dev != metadata.st_dev
                or after.st_ino != metadata.st_ino
                or after.st_size != metadata.st_size
                or after.st_mtime_ns != metadata.st_mtime_ns
            ):
                raise GateError("source-tree-race")
            file_count += 1
            if file_count > _MAX_FILES or byte_count > _MAX_BYTES:
                raise GateError("source-tree-bound")
    return digest.hexdigest()


def digest_repositories(roots: Mapping[str, Path]) -> dict[str, str]:
    return {
        identifier: source_tree_digest(roots[identifier])
        for identifier in REPOSITORY_IDS
        if identifier in roots
    }


def aggregate_digest(digests: Mapping[str, str]) -> str:
    canonical = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _resolve_token(token: str, roots: Mapping[str, Path]) -> str:
    if token == "{python}":
        return sys.executable
    match = _REPO_TOKEN.fullmatch(token)
    if match is None:
        return token
    identifier, suffix = match.groups()
    root = roots[identifier]
    if not suffix:
        return str(root)
    target = root.joinpath(*PurePosixPath(suffix[1:]).parts)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise GateError("command-path-escape") from exc
    return str(target)


def _resolve_command(command: Command, roots: Mapping[str, Path]) -> tuple[str, ...]:
    argv = tuple(_resolve_token(token, roots) for token in command.argv)
    script = roots[command.repository] / command.argv[1]
    try:
        metadata = script.lstat()
        script.resolve(strict=True).relative_to(roots[command.repository])
    except (OSError, ValueError) as exc:
        raise GateError("command-script-unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GateError("command-script-type")
    return argv


_PROCESS_GUARD = r'''"""Source-freeze process guard; generated outside every source root."""
import os
import pathlib
import shutil
import socket
import sys

_roots = tuple(
    pathlib.Path(item).resolve()
    for item in os.environ["SOURCE_FREEZE_PROTECTED_ROOTS"].split(os.pathsep)
    if item
)
_write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_mutations = {
    "os.chdir", "os.chmod", "os.chown", "os.fchdir", "os.link", "os.mkdir",
    "os.remove", "os.rename", "os.rmdir", "os.symlink", "os.truncate",
    "os.unlink", "os.utime",
}
_expected_tools = {
    "git": pathlib.Path(os.environ["SOURCE_FREEZE_GIT"]).resolve(),
    "rg": pathlib.Path(os.environ["SOURCE_FREEZE_RG"]).resolve(),
}

def _text(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        raise PermissionError("source-freeze process denied")
    return os.fsdecode(value)

def _inside(value):
    try:
        path = pathlib.Path(_text(value)).resolve(strict=False)
    except (OSError, ValueError):
        return False
    return any(path == root or root in path.parents for root in _roots)

def _safe_path(value):
    try:
        path = pathlib.Path(_text(value))
    except (OSError, ValueError):
        return False
    if ".." in path.parts:
        return False
    return _inside(path if path.is_absolute() else pathlib.Path.cwd() / path)

def _resolved_executable(value):
    raw = _text(value)
    if os.path.dirname(raw):
        candidate = raw
    else:
        candidate = shutil.which(raw, path=os.environ["PATH"])
    if not candidate:
        raise PermissionError("source-freeze process denied")
    return pathlib.Path(candidate).resolve(strict=True)

def _git_allowed(values):
    position = 1
    while position < len(values) and values[position] == "-C":
        if position + 1 >= len(values) or not _safe_path(values[position + 1]):
            return False
        position += 2
    if position >= len(values) or values[position].startswith("-"):
        return False
    subcommand = values[position]
    arguments = values[position + 1:]
    if subcommand == "rev-parse":
        return arguments == ["--path-format=absolute", "--git-common-dir"]
    if subcommand == "ls-files":
        return all(
            value in {"--cached", "--others", "--exclude-standard", "-z"}
            for value in arguments
        )
    if subcommand == "diff":
        return all(
            value in {"--name-only", "--cached", "-z", "--diff-filter=ACMR"}
            for value in arguments
        )
    if subcommand == "config":
        return arguments in (["--get", "user.name"], ["--get", "user.email"])
    if subcommand == "log":
        return arguments == ["-1", "--format=%an%n%ae"]
    if subcommand != "grep":
        return False
    position = 0
    pattern_seen = False
    while position < len(arguments):
        value = arguments[position]
        if value in {"-n", "-E", "-I"}:
            position += 1
            continue
        if value == "-e":
            if position + 1 >= len(arguments):
                return False
            pattern_seen = True
            position += 2
            continue
        if value == "--":
            return pattern_seen and all(_safe_path(item) for item in arguments[position + 1:])
        if value.startswith("-") or pattern_seen:
            return False
        pattern_seen = True
        position += 1
    return pattern_seen

def _rg_allowed(values):
    position = 1
    while position < len(values):
        value = values[position]
        if value in {"--files", "-n", "-F", "--no-heading", "--color=never"}:
            position += 1
            continue
        if value in {"-e", "--glob"}:
            if position + 1 >= len(values):
                return False
            position += 2
            continue
        if value.startswith("-") or not _safe_path(value):
            return False
        position += 1
    return True

def _audit(event, args):
    if event.startswith("socket.") and event != "socket.gethostname":
        raise PermissionError("source-freeze network denied")
    if event == "open":
        _path, mode, flags = args
        writing = (
            isinstance(mode, str) and any(marker in mode for marker in "wax+")
        ) or (isinstance(flags, int) and bool(flags & _write_flags))
        if writing:
            raise PermissionError("source-freeze write denied")
    if event in _mutations:
        raise PermissionError("source-freeze write denied")
    if event in {"os.fork", "os.forkpty", "os.posix_spawn", "os.posix_spawnp", "os.system"}:
        raise PermissionError("source-freeze process denied")
    if event == "ctypes.dlopen" and args == (None,):
        return
    if event.startswith(("os.exec", "os.spawn", "pty.spawn", "ctypes.")):
        raise PermissionError("source-freeze process denied")
    if event != "subprocess.Popen":
        return
    executable, command, cwd, environment = args
    if environment is not None or not isinstance(command, (list, tuple)):
        raise PermissionError("source-freeze process denied")
    values = [_text(value) for value in command]
    if len(values) < 2 or (cwd is not None and not _inside(cwd)):
        raise PermissionError("source-freeze process denied")
    name = pathlib.Path(values[0]).name.casefold().removesuffix(".exe")
    if name not in _expected_tools:
        raise PermissionError("source-freeze process denied")
    if _resolved_executable(executable) != _expected_tools[name]:
        raise PermissionError("source-freeze process denied")
    allowed = _git_allowed(values) if name == "git" else _rg_allowed(values)
    if not allowed:
        raise PermissionError("source-freeze process denied")

sys.addaudithook(_audit)

def _deny(*args, **kwargs):
    raise PermissionError("source-freeze network denied")

socket.create_connection = _deny
'''

_BOOTSTRAP = r"""
import runpy

for _entry in os.environ["SOURCE_FREEZE_SITE_PATHS"].split(os.pathsep):
    if _entry:
        _dependency_root = pathlib.Path(_entry).resolve(strict=True)
        if not _dependency_root.is_dir():
            raise PermissionError("source-freeze dependency root denied")
        sys.path.append(str(_dependency_root))
_repository = pathlib.Path(os.environ["SOURCE_FREEZE_COMMAND_ROOT"]).resolve(strict=True)
_script = pathlib.Path(sys.argv[1])
if not _script.is_absolute():
    _script = _repository / _script
_script = _script.resolve(strict=True)
_script.relative_to(_repository)
sys.path[:0] = [str(_script.parent), str(_repository)]
sys.argv = [str(_script), *sys.argv[2:]]
runpy.run_path(str(_script), run_name="__main__")
"""


def _resolve_reviewed_tool(name: str) -> Path:
    candidate = shutil.which(name)
    if candidate is None:
        raise GateError("reviewed-tool-unavailable")
    try:
        resolved = Path(candidate).resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise GateError("reviewed-tool-unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
    ):
        raise GateError("reviewed-tool-untrusted")
    return resolved


def _regular_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise GateError("reviewed-tool-untrusted")
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(descriptor)
    except GateError:
        raise
    except OSError as exc:
        raise GateError("reviewed-tool-unavailable") from exc
    return digest.hexdigest()


def _prepare_guard(guard_directory: Path) -> dict[str, Path]:
    sources = {name: _resolve_reviewed_tool(name) for name in ("git", "rg")}
    bin_directory = guard_directory / "bin"
    bin_directory.mkdir(mode=0o700)
    tools: dict[str, Path] = {}
    for name, source in sources.items():
        target = bin_directory / name
        shutil.copyfile(source, target, follow_symlinks=False)
        target.chmod(0o500)
        tools[name] = target
    (guard_directory / "bootstrap.py").write_text(
        _PROCESS_GUARD + _BOOTSTRAP, encoding="utf-8"
    )
    return tools


def _site_package_paths() -> tuple[str, ...]:
    import site
    import sysconfig

    candidates: list[str | None] = []
    executable = Path(sys.executable)
    venv_prefix = next(
        (
            prefix
            for prefix in (executable.parent.parent, executable.parent)
            if (prefix / "pyvenv.cfg").is_file()
        ),
        None,
    )
    if venv_prefix is not None:
        scheme = "nt_venv" if os.name == "nt" else "posix_venv"
        if scheme not in sysconfig.get_scheme_names():
            scheme = "venv"
        prefix = str(venv_prefix)
        variables = {
            "base": prefix,
            "platbase": prefix,
            "installed_base": prefix,
            "installed_platbase": prefix,
            "prefix": prefix,
            "exec_prefix": prefix,
        }
        venv_paths = sysconfig.get_paths(scheme=scheme, vars=variables)
        candidates.extend((venv_paths.get("purelib"), venv_paths.get("platlib")))
    candidates.extend(
        (
            *site.getsitepackages(),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        )
    )
    paths: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = str(Path(candidate).resolve(strict=True))
        except OSError:
            continue
        if resolved not in paths:
            paths.append(resolved)
    return tuple(paths)


def _command_environment(
    guard_directory: Path,
    repository: Path,
    roots: Mapping[str, Path],
    tools: Mapping[str, Path],
) -> dict[str, str]:
    return {
        "GIT_ASKPASS": "/bin/false",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str(guard_directory),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NO_COLOR": "1",
        "PATH": str(guard_directory / "bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYDANTIC_DISABLE_PLUGINS": "__all__",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SOURCE_DATE_EPOCH": "0",
        "SOURCE_FREEZE_COMMAND_ROOT": str(repository),
        "SOURCE_FREEZE_GIT": str(tools["git"]),
        "SOURCE_FREEZE_PROTECTED_ROOTS": os.pathsep.join(
            [
                *(str(roots[identifier]) for identifier in REPOSITORY_IDS),
                str(guard_directory),
            ]
        ),
        "SOURCE_FREEZE_RG": str(tools["rg"]),
        "SOURCE_FREEZE_SITE_PATHS": os.pathsep.join(_site_package_paths()),
        "TEMP": str(guard_directory),
        "TMP": str(guard_directory),
        "TMPDIR": str(guard_directory),
        "TZ": "UTC",
    }


def _open_directory_no_symlinks(path: Path) -> int:
    if (
        os.name != "posix"
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or not path.is_absolute()
        or ".." in path.parts
    ):
        raise GateError("evidence-parent")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open("/", flags)
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise GateError("evidence-parent") from exc


def _validate_evidence_target(path: Path, roots: Mapping[str, Path]) -> EvidenceTarget:
    if not path.name or path.name in {".", ".."}:
        raise GateError("evidence-target")
    parent_descriptor = _open_directory_no_symlinks(path.parent)
    try:
        metadata = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise GateError("evidence-parent-security")
        target = path
        for root in roots.values():
            try:
                target.relative_to(root)
            except ValueError:
                continue
            raise GateError("evidence-inside-source")
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise GateError("evidence-output-exists")
        return EvidenceTarget(
            parent_descriptor,
            metadata.st_dev,
            metadata.st_ino,
            path.name,
        )
    except Exception:
        os.close(parent_descriptor)
        raise


def _write_exclusive(target: EvidenceTarget, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(payload) > _MAX_EVIDENCE_BYTES:
        raise GateError("evidence-bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    temporary_name = f".source-freeze-{secrets.token_hex(16)}.tmp"
    temporary_created = False
    published = False
    try:
        parent = os.fstat(target.parent_descriptor)
        if (
            parent.st_dev != target.parent_device
            or parent.st_ino != target.parent_inode
            or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
            or parent.st_mode & 0o077
        ):
            raise GateError("evidence-parent-race")
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=target.parent_descriptor,
        )
        temporary_created = True
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise GateError("evidence-output-security")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise GateError("evidence-write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if not _SUPPORTS_ATOMIC_EVIDENCE:
            raise GateError("evidence-platform")
        os.link(
            temporary_name,
            target.name,
            src_dir_fd=target.parent_descriptor,
            dst_dir_fd=target.parent_descriptor,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary_name, dir_fd=target.parent_descriptor)
        temporary_created = False
        os.fsync(target.parent_descriptor)
    except FileExistsError as exc:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=target.parent_descriptor)
            except OSError:
                pass
        raise GateError("evidence-output-exists") from exc
    except GateError:
        if published:
            try:
                os.unlink(target.name, dir_fd=target.parent_descriptor)
            except OSError:
                pass
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=target.parent_descriptor)
            except OSError:
                pass
        raise
    except OSError as exc:
        if published:
            try:
                os.unlink(target.name, dir_fd=target.parent_descriptor)
            except OSError:
                pass
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=target.parent_descriptor)
            except OSError:
                pass
        raise GateError("evidence-write") from exc


@dataclass
class _StreamDigest:
    count: int = 0


def _drain_stream(
    stream: Any, state: _StreamDigest, output_limit: threading.Event
) -> None:
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            state.count += len(chunk)
            if state.count > _MAX_OUTPUT_BYTES:
                output_limit.set()
    finally:
        stream.close()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    for process_signal, pause in ((signal.SIGTERM, 0.25), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(process.pid, process_signal)
        except ProcessLookupError:
            break
        if process.poll() is None:
            try:
                process.wait(timeout=max(pause, 0.01))
            except subprocess.TimeoutExpired:
                pass
        elif pause:
            time.sleep(pause)
    if process.poll() is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired as exc:
            raise GateError("command-process-cleanup") from exc


def _run_bounded(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    if os.name != "posix":
        raise GateError("command-platform")
    stdout_state = _StreamDigest()
    stderr_state = _StreamDigest()
    output_limit = threading.Event()
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "exit_code": -1,
            "termination": "launch-error",
        }
    if process.stdout is None or process.stderr is None:
        _terminate_process_group(process)
        raise GateError("command-capture")
    readers = (
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_state, output_limit),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_state, output_limit),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout_seconds
    termination = "exited"
    while process.poll() is None:
        if output_limit.is_set():
            termination = "output-limit"
            break
        if time.monotonic() >= deadline:
            termination = "timeout"
            break
        time.sleep(0.02)
    _terminate_process_group(process)
    for reader in readers:
        reader.join(timeout=2)
    if any(reader.is_alive() for reader in readers):
        raise GateError("command-capture-cleanup")
    if output_limit.is_set():
        termination = "output-limit"
    return {
        "exit_code": process.returncode if process.returncode is not None else -1,
        "termination": termination,
    }


def _command_roots(command: Command, roots: Mapping[str, Path]) -> dict[str, Path]:
    identifiers = {command.repository}
    identifiers.update(
        match.group(1)
        for token in command.argv
        if (match := _REPO_TOKEN.fullmatch(token)) is not None
    )
    return {
        identifier: roots[identifier]
        for identifier in REPOSITORY_IDS
        if identifier in identifiers
    }


def _bootstrap_command(argv: tuple[str, ...], guard_directory: Path) -> tuple[str, ...]:
    return (
        argv[0],
        "-S",
        "-B",
        str(guard_directory / "bootstrap.py"),
        *argv[1:],
    )


def _runner_isolated() -> bool:
    """Return whether interpreter startup excluded environment and site code."""

    return bool(
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and sys.flags.ignore_environment
        and sys.flags.safe_path
    )


def execute_manifest(
    manifest: Manifest,
    roots: Mapping[str, Path],
    evidence_path: Path,
) -> dict[str, Any]:
    """Execute all source gates serially and publish privacy-safe evidence."""

    target = _validate_evidence_target(evidence_path, roots)
    try:
        before = digest_repositories(roots)
        command_results: list[dict[str, Any]] = []
        tool_evidence: list[dict[str, str]] = []
        passed = True
        with tempfile.TemporaryDirectory(prefix="source-freeze-") as guard:
            guard_directory = Path(guard)
            tools = _prepare_guard(guard_directory)
            tool_evidence = [
                {"id": name, "sha256": _regular_file_digest(tools[name])}
                for name in ("git", "rg")
            ]
            for command in manifest.commands:
                argv = _bootstrap_command(
                    _resolve_command(command, roots), guard_directory
                )
                command_roots = _command_roots(command, roots)
                command_before = digest_repositories(command_roots)
                expected_before = {
                    identifier: before[identifier] for identifier in command_roots
                }
                if command_before != expected_before:
                    result = {
                        "exit_code": -1,
                        "termination": "source-changed",
                    }
                else:
                    result = _run_bounded(
                        argv,
                        cwd=roots[command.repository],
                        environment=_command_environment(
                            guard_directory,
                            roots[command.repository],
                            roots,
                            tools,
                        ),
                        timeout_seconds=command.timeout_seconds,
                    )
                command_after = digest_repositories(command_roots)
                command_unchanged = command_before == command_after == expected_before
                status = (
                    "passed"
                    if result["exit_code"] == 0
                    and result["termination"] == "exited"
                    and command_unchanged
                    else "failed"
                )
                termination = (
                    result["termination"] if command_unchanged else "source-changed"
                )
                command_results.append(
                    {
                        "id": command.identifier,
                        "status": status,
                        "exit_code": result["exit_code"],
                        "termination": termination,
                        "source_digest_before": aggregate_digest(command_before),
                        "source_digest_after": aggregate_digest(command_after),
                    }
                )
                if status != "passed":
                    passed = False
                    break
        after = digest_repositories(roots)
        unchanged = before == after
        passed = passed and unchanged and len(command_results) == len(manifest.commands)
        passed_commands = {
            result["id"] for result in command_results if result["status"] == "passed"
        }
        gate_results = []
        for gate in manifest.gates:
            if "local-source" in gate.evidence_classes:
                source_status = (
                    "passed"
                    if passed
                    and all(
                        command_id in passed_commands for command_id in gate.command_ids
                    )
                    else "failed"
                )
            else:
                source_status = "not-applicable"
            gate_results.append(
                {
                    "id": gate.identifier,
                    "required_evidence": list(gate.evidence_classes),
                    "source_status": source_status,
                    "remaining_evidence": [
                        value
                        for value in gate.evidence_classes
                        if value != "local-source"
                    ],
                }
            )
        evidence: dict[str, Any] = {
            "schema": EVIDENCE_SCHEMA,
            "status": "passed" if passed else "failed",
            "manifest_sha256": manifest.digest,
            "source_digest_before": aggregate_digest(before),
            "source_digest_after": aggregate_digest(after),
            "tools": tool_evidence,
            "repositories": [
                {
                    "id": identifier,
                    "sha256_before": before[identifier],
                    "sha256_after": after[identifier],
                }
                for identifier in REPOSITORY_IDS
            ],
            "commands": command_results,
            "gates": gate_results,
        }
        _write_exclusive(target, evidence)
        if not unchanged:
            raise GateError("source-tree-edited")
        if not passed:
            raise GateError("source-command-failed")
        return evidence
    finally:
        target.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="ID=ROOT",
        help="explicit repository root; repeat for all four manifest aliases",
    )
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if not _runner_isolated():
            raise GateError("runner-isolation")
        manifest = load_canonical_manifest(
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "release"
            / "source-freeze-gates.json"
        )
        roots = parse_repository_roots(arguments.repo)
        evidence = execute_manifest(manifest, roots, arguments.evidence)
    except GateError as exc:
        print(f"source-freeze gate: FAIL ({exc.code})", file=sys.stderr)
        return 1
    print(
        "source-freeze gate: PASS "
        f"({len(evidence['commands'])} commands, {len(evidence['gates'])} gates, "
        f"digest={evidence['source_digest_after']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
