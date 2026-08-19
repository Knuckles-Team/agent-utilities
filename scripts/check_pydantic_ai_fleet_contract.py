#!/usr/bin/env python3
"""Check fleet Pydantic-AI resolution and editable-AU metadata parity.

The expected version is read from AU's one runtime contract constant rather
than copied into this gate. Callers pass the lock/manifest paths they own or
want to audit; directory expansion and printed findings are bounded so a
workspace-wide check cannot become an unreviewable output or memory sink.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONTRACT_SYMBOL = "_PYDANTIC_AI_CONTRACT_VERSION"
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[A-Za-z0-9.-]+)?$")
_PYDANTIC_REQUIREMENT_RE = re.compile(
    r"pydantic-ai-slim(?:\[[^\]\r\n]+\])?"
    r"(?:\s*(?:===|==|!=|~=|>=|<=|>|<)\s*[0-9A-Za-z.+!-]+"
    r"(?:\s*,\s*(?:===|==|!=|~=|>=|<=|>|<)\s*[0-9A-Za-z.+!-]+)*)?",
    re.IGNORECASE,
)
_SUPPORTED_FILENAMES = {"uv.lock", "pyproject.toml", "requirements.txt"}
_SIBLINGS_MARKER = ".uv-workspace-siblings"
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_PEP503_NAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_PEP503_SEPARATORS_RE = re.compile(r"[-_.]+")
_LOCAL_SOURCE_FIELDS = frozenset({"editable", "directory", "path", "virtual"})
_PYPROJECT_SOURCE_KEYS = frozenset(
    {
        "branch",
        "editable",
        "extra",
        "git",
        "index",
        "lfs",
        "marker",
        "package",
        "path",
        "rev",
        "subdirectory",
        "tag",
        "url",
        "workspace",
    }
)
_LOCK_SOURCE_KEYS = frozenset(
    {
        "branch",
        "directory",
        "editable",
        "git",
        "index",
        "path",
        "registry",
        "rev",
        "subdirectory",
        "tag",
        "url",
        "virtual",
    }
)
_PYPROJECT_REMOTE_SOURCE_FIELDS = frozenset(
    {"branch", "git", "index", "lfs", "rev", "subdirectory", "tag", "url", "workspace"}
)
_LOCK_REMOTE_SOURCE_FIELDS = frozenset(
    {"branch", "git", "index", "registry", "rev", "subdirectory", "tag", "url"}
)
_PYPROJECT_PRIMARY_SOURCE_FIELDS = frozenset(
    {"git", "index", "path", "url", "workspace"}
)
_LOCK_PRIMARY_SOURCE_FIELDS = frozenset(
    {
        "directory",
        "editable",
        "git",
        "index",
        "path",
        "registry",
        "url",
        "virtual",
    }
)
_SOURCE_SELECTOR_FIELDS = frozenset({"branch", "rev", "tag"})
_STRING_SOURCE_FIELDS = frozenset(
    {
        "branch",
        "extra",
        "git",
        "index",
        "marker",
        "registry",
        "rev",
        "subdirectory",
        "tag",
        "url",
    }
)


@dataclass(frozen=True)
class Finding:
    """One bounded, human-readable parity failure."""

    path: str
    kind: str
    detail: str


@dataclass(frozen=True)
class ScanResult:
    """Bounded scan outcome returned by both the API and CLI."""

    expected_version: str
    files_scanned: int
    findings: tuple[Finding, ...]
    truncated_findings: int = 0
    truncated_files: int = 0

    @property
    def ok(self) -> bool:
        return (
            not self.findings
            and not self.truncated_findings
            and not self.truncated_files
        )


def default_contract_source() -> Path:
    """Return the AU source file containing the canonical version literal."""

    return (
        Path(__file__).resolve().parents[1] / "agent_utilities/mcp/protocol_compat.py"
    )


def read_contract_version(contract_source: Path | None = None) -> str:
    """Read and validate AU's single Pydantic-AI contract literal via AST."""

    source = contract_source or default_contract_source()
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"cannot read AU contract source {source}: {exc}") from exc

    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == _CONTRACT_SYMBOL
            for target in targets
        ):
            continue
        try:
            version = ast.literal_eval(value)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                f"AU contract {_CONTRACT_SYMBOL} is not a literal"
            ) from exc
        if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
            raise RuntimeError(
                f"AU contract {_CONTRACT_SYMBOL} is not a valid version: {version!r}"
            )
        return version

    raise RuntimeError(f"AU contract source {source} has no {_CONTRACT_SYMBOL} literal")


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_strings(child)


def _requirement_strings(
    path: Path,
    text: str,
    parsed: Mapping[str, Any] | None = None,
) -> Iterable[str]:
    """Yield Pydantic requirement strings from manifests without broad matching."""

    if path.name == "pyproject.toml":
        if parsed is None:
            return
        for value in _iter_strings(parsed):
            if "pydantic-ai-slim" in value.lower():
                yield value
        return

    for match in _PYDANTIC_REQUIREMENT_RE.finditer(text):
        yield match.group(0)


def _check_manifest(
    path: Path,
    text: str,
    expected: str,
    parsed: Mapping[str, Any] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    if path.name == "pyproject.toml":
        if parsed is None:
            try:
                parsed = tomllib.loads(text)
            except tomllib.TOMLDecodeError as exc:
                # A malformed manifest must fail closed; treating it as having no
                # Pydantic requirement would let stale fleet metadata pass.
                return [Finding(str(path), "invalid-manifest", str(exc))]
        if not isinstance(parsed, Mapping):
            # A malformed manifest must fail closed; treating it as having no
            # Pydantic requirement would let stale fleet metadata pass.
            return [
                Finding(str(path), "invalid-manifest", "TOML document is not a table")
            ]
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.version import InvalidVersion, Version

        expected_version = Version(expected)
    except (
        ImportError
    ) as exc:  # pragma: no cover - AU's base dependency includes packaging
        raise RuntimeError("packaging is required for the fleet parity gate") from exc

    seen: set[str] = set()
    for raw in _requirement_strings(path, text, parsed):
        if raw in seen or "pydantic-ai-slim" not in raw.lower():
            continue
        seen.add(raw)
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            findings.append(Finding(str(path), "invalid-requirement", repr(raw)))
            continue
        if not requirement.specifier:
            continue
        try:
            supported = requirement.specifier.contains(
                expected_version, prereleases=True
            )
        except InvalidVersion:
            supported = False
        if not supported:
            findings.append(
                Finding(
                    str(path),
                    "manifest-version-mismatch",
                    f"{raw!r} does not admit the AU contract {expected}",
                )
            )
    return findings


def _check_lock(path: Path, parsed: Mapping[str, Any], expected: str) -> list[Finding]:
    findings: list[Finding] = []
    versions: list[str] = []
    packages = parsed.get("package")
    if packages is None:
        packages = []
    if not isinstance(packages, list):
        return [
            Finding(str(path), "invalid-lock", "package must be an array of tables")
        ]

    for package in packages:
        if not isinstance(package, Mapping):
            findings.append(
                Finding(str(path), "invalid-lock", "package entry must be a table")
            )
            continue
        name = package.get("name")
        if not isinstance(name, str) or name.lower() != "pydantic-ai-slim":
            continue
        version = package.get("version")
        if not isinstance(version, str):
            findings.append(
                Finding(
                    str(path),
                    "missing-resolution",
                    "pydantic-ai-slim has no locked version",
                )
            )
            continue
        versions.append(version)
        if version != expected:
            findings.append(
                Finding(
                    str(path),
                    "resolved-version-mismatch",
                    f"uv.lock resolves pydantic-ai-slim {version}; expected {expected}",
                )
            )

    # uv stores dependency metadata in more than the legacy [manifest] table.
    # Inspect the parsed document (comments remain inert) so a dependency named
    # outside that table cannot make a missing package resolution pass.
    lock_mentions_pydantic = any(
        "pydantic-ai-slim" in value.lower() for value in _iter_strings(parsed)
    )
    if not versions and lock_mentions_pydantic:
        findings.append(
            Finding(
                str(path),
                "missing-resolution",
                "lock metadata mentions pydantic-ai-slim but uv.lock has no package resolution",
            )
        )
    return findings


def _editable_is_current(path: Path, editable: str, au_root: Path) -> bool:
    # Absolute paths and arbitrary relative paths make locks depend on a
    # particular checkout/worktree. The two forms below are the only generated
    # AU lock shapes: AU's own lock and a sibling consumer's lock.
    parts = editable.split("/")
    if (
        len(parts) == 2
        and parts[0] == _SIBLINGS_MARKER
        and _pep503_normalize(parts[1]) == "agent-utilities"
    ):
        return True
    return editable == "." and (
        path.parent.name == "agent-utilities" or path.parent.resolve() == au_root
    )


def _pep503_normalize(value: object) -> str | None:
    """Normalize a project/distribution name using PEP 503 rules."""

    if not isinstance(value, str) or _PEP503_NAME_RE.fullmatch(value) is None:
        return None
    return _PEP503_SEPARATORS_RE.sub("-", value).lower()


def _local_path_reason(
    value: object,
    declared_name: object,
    *,
    allow_self: bool,
) -> str | None:
    """Return why a local source path is outside the canonical two shapes."""

    if not isinstance(value, str):
        return "path value must be a string"
    if not value:
        return "path value must not be empty"
    if "\\" in value:
        return "backslash path separators are not allowed"
    if value == ".":
        return None if allow_self else "self path is only valid for the project itself"
    if value.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(value):
        return "absolute paths are not allowed"

    parts = value.split("/")
    if len(parts) != 2 or parts[0] != _SIBLINGS_MARKER:
        return f"path must be exactly {_SIBLINGS_MARKER}/<package-name> or ."
    package_name = parts[1]
    if _PEP503_NAME_RE.fullmatch(package_name) is None:
        return "sibling path must contain one valid package-name component"
    normalized_declared = _pep503_normalize(declared_name)
    normalized_component = _pep503_normalize(package_name)
    if normalized_declared is None or normalized_component != normalized_declared:
        return (
            "sibling path component must PEP-503-normalize to the declared "
            f"package name {declared_name!r}"
        )
    return None


def _source_finding(
    path: Path, package_name: object, kind: str, detail: str
) -> Finding:
    return Finding(str(path), kind, f"{package_name!r}: {detail}")


def _check_source_value(
    path: Path,
    package_name: object,
    source_path: object,
    source_key: str,
    au_root: Path,
    findings: list[Finding],
    *,
    allow_self: bool,
) -> None:
    if not isinstance(source_path, str):
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                f"{source_key} source must be a string",
            )
        )
        return

    reason = _local_path_reason(
        source_path,
        package_name,
        allow_self=allow_self,
    )
    if reason is not None:
        findings.append(
            _source_finding(
                path,
                package_name,
                "nested-editable-source",
                f"local {source_key} source {source_path!r} is not canonical: {reason}",
            )
        )
    if (
        _pep503_normalize(package_name) == "agent-utilities"
        and not (source_path == "." and allow_self)
        and not _editable_is_current(path, source_path, au_root)
    ):
        findings.append(
            _source_finding(
                path,
                package_name,
                "stale-editable-au",
                f"local {source_key} source {source_path!r} is not a generated AU workspace path",
            )
        )


def _check_source_shape(
    path: Path,
    package_name: object,
    source: Mapping[str, Any],
    *,
    lock: bool,
) -> list[Finding]:
    """Require one uv source kind and validate its dependent selectors."""

    findings: list[Finding] = []
    primary_fields = (
        _LOCK_PRIMARY_SOURCE_FIELDS if lock else _PYPROJECT_PRIMARY_SOURCE_FIELDS
    )
    present_primary = set(source).intersection(primary_fields)
    if not present_primary:
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                "source table must declare exactly one primary source kind",
            )
        )
    elif len(present_primary) > 1:
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                f"source kinds are mutually exclusive: {sorted(present_primary)!r}",
            )
        )

    if "workspace" in source and (
        not isinstance(source["workspace"], bool) or source["workspace"] is not True
    ):
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                "workspace source must be exactly boolean true",
            )
        )

    for key in present_primary.difference(_LOCAL_SOURCE_FIELDS, {"workspace"}):
        value = source[key]
        if isinstance(value, str) and not value.strip():
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    f"{key} source must not be empty",
                )
            )

    selectors = set(source).intersection(_SOURCE_SELECTOR_FIELDS)
    selector_values = selectors.union({"subdirectory"}.intersection(source))
    for key in selector_values:
        value = source[key]
        if isinstance(value, str) and not value.strip():
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    f"{key} source must not be empty",
                )
            )
    if len(selectors) > 1:
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                f"git selectors are mutually exclusive: {sorted(selectors)!r}",
            )
        )
    if selectors and "git" not in source:
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                "branch, tag, and rev require a git source",
            )
        )
    if "lfs" in source and "git" not in source:
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                "lfs requires a git source",
            )
        )
    if "subdirectory" in source and not {"git", "url"}.intersection(source):
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                "subdirectory requires a git or url source",
            )
        )
    return findings


def _check_source_mapping(
    path: Path,
    package_name: object,
    source: object,
    au_root: Path,
    *,
    lock: bool,
    project_name: object | None = None,
) -> list[Finding]:
    """Validate one parsed uv source table and its local path, if any."""

    findings: list[Finding] = []
    if isinstance(source, list):
        if lock:
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    "uv.lock source must be a table, not a list",
                )
            )
            return findings
        if not source:
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    "tool.uv.sources alternatives must not be empty",
                )
            )
            return findings
        for alternative in source:
            if not isinstance(alternative, Mapping):
                findings.append(
                    _source_finding(
                        path,
                        package_name,
                        "malformed-source",
                        "each source alternative must be a TOML table",
                    )
                )
                continue
            findings.extend(
                _check_source_mapping(
                    path,
                    package_name,
                    alternative,
                    au_root,
                    lock=False,
                    project_name=project_name,
                )
            )
        return findings

    if not isinstance(source, Mapping):
        findings.append(
            _source_finding(
                path, package_name, "malformed-source", "source must be a TOML table"
            )
        )
        return findings

    if _pep503_normalize(package_name) is None:
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                "source name must be an ASCII PEP 503 distribution name",
            )
        )

    allowed_keys = _LOCK_SOURCE_KEYS if lock else _PYPROJECT_SOURCE_KEYS
    unknown_keys = set(source) - allowed_keys
    if unknown_keys:
        findings.append(
            _source_finding(
                path,
                package_name,
                "malformed-source",
                f"unknown source field(s): {sorted(unknown_keys)!r}",
            )
        )
        return findings

    # TOML arrays/tables are not scalar uv source fields. Reject them even for
    # remote sources so a malformed value cannot be silently ignored by this gate.
    if any(isinstance(value, (Mapping, list, tuple)) for value in source.values()):
        findings.append(
            _source_finding(
                path, package_name, "malformed-source", "source values must be scalar"
            )
        )
        return findings

    findings.extend(_check_source_shape(path, package_name, source, lock=lock))

    for key in _STRING_SOURCE_FIELDS:
        if key in source and not isinstance(source[key], str):
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    f"{key} source must be a string",
                )
            )
    bool_fields = {"lfs", "package"}
    if not lock:
        bool_fields.add("editable")
    for key in bool_fields.intersection(source):
        if not isinstance(source[key], bool):
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    f"{key} source must be boolean",
                )
            )
    local_keys = set(source).intersection(_LOCAL_SOURCE_FIELDS)
    if lock:
        remote_keys = set(source).intersection(_LOCK_REMOTE_SOURCE_FIELDS)
        if local_keys and remote_keys:
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    "local and remote source fields cannot be combined",
                )
            )
        elif len(local_keys) > 1:
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    "a lock source may contain only one local path field",
                )
            )
        elif local_keys:
            source_key = next(iter(local_keys))
            _check_source_value(
                path,
                package_name,
                source[source_key],
                source_key,
                au_root,
                findings,
                allow_self=(
                    _pep503_normalize(package_name) is not None
                    and _pep503_normalize(package_name)
                    == _pep503_normalize(project_name)
                ),
            )
        return findings

    # In pyproject.toml, a local source is expressed as path plus optional
    # metadata modifiers. Marker/extra/package are modifiers, not alternate
    # remote sources, and are therefore valid alongside path. A path combined
    # with an actual remote source would make the selected identity ambiguous.
    if local_keys:
        remote_keys = set(source).intersection(_PYPROJECT_REMOTE_SOURCE_FIELDS)
        if local_keys and remote_keys:
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    "local and remote source fields cannot be combined",
                )
            )
        if "path" not in source:
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    "local source is missing path",
                )
            )
        elif "editable" in source and not isinstance(source["editable"], bool):
            findings.append(
                _source_finding(
                    path,
                    package_name,
                    "malformed-source",
                    "editable source flag must be boolean",
                )
            )
        if (
            "path" in source
            and not remote_keys
            and not ("editable" in source and not isinstance(source["editable"], bool))
        ):
            _check_source_value(
                path,
                package_name,
                source["path"],
                "path",
                au_root,
                findings,
                allow_self=(
                    _pep503_normalize(package_name) is not None
                    and _pep503_normalize(package_name)
                    == _pep503_normalize(project_name)
                ),
            )
    return findings


def _adjacent_project_name(path: Path) -> object | None:
    """Read the lock root's project name for self-editable identity binding."""

    manifest = path.with_name("pyproject.toml")
    try:
        parsed = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None
    project = parsed.get("project") if isinstance(parsed, Mapping) else None
    return project.get("name") if isinstance(project, Mapping) else None


def _check_local_sources(
    path: Path, document: Mapping[str, Any], au_root: Path
) -> list[Finding]:
    """Inspect only parsed uv source tables; comments and unrelated strings are inert."""

    findings: list[Finding] = []
    if path.name == "uv.lock":
        project_name = _adjacent_project_name(path)
        packages = document.get("package")
        if packages is None:
            return findings
        if not isinstance(packages, list):
            return [
                _source_finding(
                    path,
                    "package",
                    "malformed-source",
                    "package must be an array of tables",
                )
            ]
        for package in packages:
            if not isinstance(package, Mapping):
                findings.append(
                    _source_finding(
                        path,
                        "package",
                        "malformed-source",
                        "package entry must be a table",
                    )
                )
                continue
            if "source" not in package:
                continue
            package_name = package.get("name")
            if not isinstance(package_name, str) or not package_name:
                findings.append(
                    _source_finding(
                        path,
                        "package",
                        "malformed-source",
                        "a package source requires a string package name",
                    )
                )
                continue
            findings.extend(
                _check_source_mapping(
                    path,
                    package_name,
                    package["source"],
                    au_root,
                    lock=True,
                    project_name=project_name,
                )
            )
        return findings

    tool = document.get("tool")
    if tool is None:
        return findings
    if not isinstance(tool, Mapping):
        return [
            _source_finding(
                path, "tool", "malformed-source", "tool must be a TOML table"
            )
        ]
    uv = tool.get("uv")
    if uv is None:
        return findings
    if not isinstance(uv, Mapping):
        return [
            _source_finding(
                path, "uv", "malformed-source", "tool.uv must be a TOML table"
            )
        ]
    sources = uv.get("sources")
    if sources is None:
        return findings
    if not isinstance(sources, Mapping):
        return [
            _source_finding(
                path,
                "sources",
                "malformed-source",
                "tool.uv.sources must be a TOML table",
            )
        ]
    project = document.get("project")
    project_name = project.get("name") if isinstance(project, Mapping) else None
    for package_name, source in sources.items():
        findings.extend(
            _check_source_mapping(
                path,
                package_name,
                source,
                au_root,
                lock=False,
                project_name=project_name,
            )
        )
    return findings


def _candidate_files(paths: Iterable[Path], max_files: int) -> tuple[list[Path], int]:
    candidates: list[Path] = []
    omitted = 0
    for supplied in paths:
        if len(candidates) >= max_files:
            return candidates, 1
        if supplied.is_file():
            candidates.append(supplied)
            continue
        if not supplied.is_dir():
            candidates.append(supplied)
            continue
        # Walk incrementally rather than materializing/sorting an entire
        # workspace tree. Once the bound is reached, stop traversal and report
        # an omission sentinel; bounded work matters as much as bounded output.
        for root, directories, filenames in os.walk(supplied):
            directories[:] = sorted(
                directory for directory in directories if not directory.startswith(".")
            )
            for filename in sorted(filenames):
                if not (
                    filename in _SUPPORTED_FILENAMES
                    or filename.startswith("requirements")
                    or filename.startswith("Dockerfile")
                    or filename.endswith(".Dockerfile")
                ):
                    continue
                if len(candidates) >= max_files:
                    return candidates, 1
                candidates.append(Path(root) / filename)
    if len(candidates) > max_files:
        omitted = len(candidates) - max_files
        candidates = candidates[:max_files]
    return candidates, omitted


def scan_paths(
    paths: Iterable[Path],
    *,
    contract_source: Path | None = None,
    max_files: int = 256,
    max_findings: int = 100,
    max_bytes: int = 2_000_000,
) -> ScanResult:
    """Scan supplied lock/manifest paths with bounded files, bytes, and output."""

    if max_files < 1 or max_findings < 1 or max_bytes < 1:
        raise ValueError("max_files, max_findings, and max_bytes must be positive")
    resolved_contract_source = contract_source or default_contract_source()
    expected = read_contract_version(resolved_contract_source)
    au_root = resolved_contract_source.resolve().parents[2]
    candidates, omitted = _candidate_files(paths, max_files)
    raw_findings: list[Finding] = []
    files_scanned = 0
    for path in candidates:
        try:
            if path.stat().st_size > max_bytes:
                raw_findings.append(
                    Finding(str(path), "file-too-large", f"exceeds {max_bytes} bytes")
                )
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raw_findings.append(
                Finding(str(path), "unreadable", f"{type(exc).__name__}: {exc}")
            )
            continue
        files_scanned += 1
        document: Mapping[str, Any] | None = None
        if path.name in {"uv.lock", "pyproject.toml"}:
            try:
                parsed = tomllib.loads(text)
            except tomllib.TOMLDecodeError as exc:
                kind = "invalid-lock" if path.name == "uv.lock" else "invalid-manifest"
                raw_findings.append(Finding(str(path), kind, str(exc)))
                continue
            if not isinstance(parsed, Mapping):
                kind = "invalid-lock" if path.name == "uv.lock" else "invalid-manifest"
                raw_findings.append(
                    Finding(str(path), kind, "TOML document is not a table")
                )
                continue
            document = parsed

        if path.name == "uv.lock":
            assert document is not None
            raw_findings.extend(_check_lock(path, document, expected))
        else:
            raw_findings.extend(_check_manifest(path, text, expected, document))
        if document is not None:
            raw_findings.extend(_check_local_sources(path, document, au_root))

    truncated = max(0, len(raw_findings) - max_findings)
    return ScanResult(
        expected_version=expected,
        files_scanned=files_scanned,
        findings=tuple(raw_findings[:max_findings]),
        truncated_findings=truncated,
        truncated_files=omitted,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="lock/manifest files or bounded directories to scan",
    )
    parser.add_argument("--contract-source", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=256)
    parser.add_argument("--max-findings", type=int, default=100)
    args = parser.parse_args(argv)
    result = scan_paths(
        args.paths,
        contract_source=args.contract_source,
        max_files=args.max_files,
        max_findings=args.max_findings,
    )
    print(
        f"pydantic-ai contract expected={result.expected_version} files={result.files_scanned}"
    )
    for finding in result.findings:
        print(f"FAIL {finding.path}: {finding.kind}: {finding.detail}")
    if result.truncated_findings:
        print(
            f"FAIL output bounded at {args.max_findings} findings; {result.truncated_findings} more omitted"
        )
    if result.truncated_files:
        print(
            f"FAIL file scan bounded at {args.max_files}; {result.truncated_files} files omitted"
        )
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
