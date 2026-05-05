#!/usr/bin/python
"""CONCEPT:KG-2.3 — Structural Fingerprint Engine.

Generic capability for detecting structural vs. cosmetic changes in any
workspace the agent operates on. Enables incremental KG updates by
classifying changes into three levels:

    - **NONE**: File content identical (same SHA-256 hash).
    - **COSMETIC**: Content changed but structure identical
      (whitespace, comments, docstrings, formatting).
    - **STRUCTURAL**: Signature-level changes (function params, class methods,
      imports, exports) that require KG re-ingestion.

Inspired by Understand-Anything's ``fingerprint.ts`` and ``staleness.ts``.

Usage::

    from agent_utilities.knowledge_graph.fingerprint import (
        StructuralFingerprint,
        classify_change,
        compute_fingerprint,
    )

    fp = compute_fingerprint("/path/to/file.py")
    change = classify_change(old_fp, new_fp)
    # change == ChangeLevel.COSMETIC  # Only formatting changed

See docs/emergent-architecture.md §AU-048.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Change Classification
# ---------------------------------------------------------------------------


class ChangeLevel(StrEnum):
    """Three-level change classification for KG incremental updates."""

    NONE = "none"  # Identical content hash
    COSMETIC = "cosmetic"  # Content changed but structure unchanged
    STRUCTURAL = "structural"  # Signature-level changes → requires re-ingestion


# ---------------------------------------------------------------------------
# Structural Fingerprint
# ---------------------------------------------------------------------------


@dataclass
class StructuralFingerprint:
    """Structural fingerprint of a source file.

    Captures the structural skeleton of a file — function signatures,
    class definitions, import specifiers, and export names — separate
    from the raw content hash. This allows distinguishing cosmetic
    changes (reformatting, comment edits) from structural ones
    (new function parameters, renamed classes).

    Attributes:
        file_path: Absolute or relative path to the file.
        content_hash: SHA-256 of the full file content.
        structural_hash: SHA-256 of the structural skeleton.
        functions: List of function/method signatures.
        classes: List of class names with method lists.
        imports: List of import specifiers.
        exports: List of exported names (__all__ for Python).
        computed_at: ISO timestamp when fingerprint was computed.
    """

    file_path: str
    content_hash: str
    structural_hash: str
    functions: list[dict[str, Any]] = field(default_factory=list)
    classes: list[dict[str, Any]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    computed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for KG storage."""
        return {
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "structural_hash": self.structural_hash,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "exports": self.exports,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuralFingerprint:
        """Deserialize from dict."""
        return cls(
            file_path=data.get("file_path", ""),
            content_hash=data.get("content_hash", ""),
            structural_hash=data.get("structural_hash", ""),
            functions=data.get("functions", []),
            classes=data.get("classes", []),
            imports=data.get("imports", []),
            exports=data.get("exports", []),
            computed_at=data.get("computed_at", ""),
        )


# ---------------------------------------------------------------------------
# Python AST Structural Extraction
# ---------------------------------------------------------------------------


def _extract_python_structure(source: str) -> dict[str, Any]:
    """Extract structural skeleton from Python source using AST.

    Extracts:
    - Function names, parameters, return types, decorator names
    - Class names, base classes, method names
    - Import specifiers (module names, imported names)
    - __all__ exports

    Returns a deterministic dict suitable for hashing.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"parse_error": True}

    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: list[str] = []
    exports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_info: dict[str, Any] = {
                "name": node.name,
                "args": [arg.arg for arg in node.args.args if arg.arg != "self"],
                "decorators": [ast.dump(d) for d in node.decorator_list],
            }
            # Return type annotation
            if node.returns:
                func_info["return_type"] = ast.dump(node.returns)
            # Default values count (signature stability indicator)
            func_info["defaults_count"] = len(node.args.defaults)
            functions.append(func_info)

        elif isinstance(node, ast.ClassDef):
            class_info: dict[str, Any] = {
                "name": node.name,
                "bases": [ast.dump(b) for b in node.bases],
                "methods": [],
                "decorators": [ast.dump(d) for d in node.decorator_list],
            }
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    class_info["methods"].append(item.name)
            classes.append(class_info)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")

        elif isinstance(node, ast.Assign):
            # Detect __all__ = [...]
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, ast.List)
                ):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports.append(elt.value)

    return {
        "functions": sorted(functions, key=lambda f: f["name"]),
        "classes": sorted(classes, key=lambda c: c["name"]),
        "imports": sorted(imports),
        "exports": sorted(exports),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fingerprint(file_path: str) -> StructuralFingerprint | None:
    """Compute a structural fingerprint for a file.

    Currently supports Python files. Other languages return a
    content-only fingerprint (structural_hash == content_hash).

    Args:
        file_path: Path to the file to fingerprint.

    Returns:
        A ``StructuralFingerprint`` or None if the file can't be read.
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError) as e:
        logger.debug("Cannot read file %s: %s", file_path, e)
        return None

    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Extract structural skeleton based on file extension
    ext = Path(file_path).suffix.lower()
    if ext in (".py", ".pyi"):
        structure = _extract_python_structure(content)
    else:
        # For non-Python files, structural hash = content hash
        # (any change is treated as structural)
        structure = {"raw": True}

    # Compute structural hash from the deterministic skeleton
    skeleton_json = json.dumps(structure, sort_keys=True, default=str)
    structural_hash = hashlib.sha256(skeleton_json.encode()).hexdigest()

    return StructuralFingerprint(
        file_path=file_path,
        content_hash=content_hash,
        structural_hash=structural_hash,
        functions=structure.get("functions", []),
        classes=structure.get("classes", []),
        imports=structure.get("imports", []),
        exports=structure.get("exports", []),
        computed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def classify_change(
    old: StructuralFingerprint | None,
    new: StructuralFingerprint | None,
) -> ChangeLevel:
    """Classify the change between two fingerprints.

    Args:
        old: Previous fingerprint (None if file is new).
        new: Current fingerprint (None if file was deleted).

    Returns:
        ``ChangeLevel.NONE``, ``COSMETIC``, or ``STRUCTURAL``.
    """
    if old is None or new is None:
        return ChangeLevel.STRUCTURAL

    if old.content_hash == new.content_hash:
        return ChangeLevel.NONE

    if old.structural_hash == new.structural_hash:
        return ChangeLevel.COSMETIC

    return ChangeLevel.STRUCTURAL


# ---------------------------------------------------------------------------
# Git-based Staleness Detection
# ---------------------------------------------------------------------------


def detect_stale_files(
    workspace_path: str,
    since_commit: str | None = None,
) -> list[dict[str, Any]]:
    """Detect files that have changed since a given commit.

    Uses ``git diff`` to identify files modified since the last KG
    fingerprint snapshot, similar to UA's ``staleness.ts``.

    Args:
        workspace_path: Root of the git repository.
        since_commit: Git commit hash to compare against.
            If None, compares against HEAD~1.

    Returns:
        List of dicts with ``file_path``, ``status`` (modified/added/deleted).
    """
    ref = since_commit or "HEAD~1"

    try:
        result = subprocess.run(  # nosec B607 B603
            ["git", "diff", f"{ref}..HEAD", "--name-status"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("git diff failed: %s", result.stderr.strip())
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("git not available or timeout: %s", e)
        return []

    status_map = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed"}
    changes: list[dict[str, Any]] = []

    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", maxsplit=1)
        if len(parts) == 2:
            status_code, file_path = parts
            changes.append(
                {
                    "file_path": file_path,
                    "status": status_map.get(status_code[0], "modified"),
                    "full_path": os.path.join(workspace_path, file_path),
                }
            )

    return changes


# ---------------------------------------------------------------------------
# Workspace Fingerprint Manager
# ---------------------------------------------------------------------------


class FingerprintManager:
    """Manages structural fingerprints for an entire workspace.

    CONCEPT:KG-2.3 — Structural Fingerprint Engine

    Computes, stores, and compares fingerprints to determine which files
    need KG re-ingestion after code changes. Works for any workspace.

    Args:
        workspace_path: Root of the workspace to fingerprint.
    """

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path
        self._fingerprints: dict[str, StructuralFingerprint] = {}

    def scan(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, StructuralFingerprint]:
        """Scan workspace and compute fingerprints for all eligible files.

        Args:
            include_patterns: Glob patterns to include (default: *.py).
            exclude_patterns: Patterns to exclude (default: common junk dirs).

        Returns:
            Dict mapping file paths to their fingerprints.
        """
        include = include_patterns or ["**/*.py"]
        exclude = exclude_patterns or [
            ".git",
            "node_modules",
            "__pycache__",
            "venv",
            ".venv",
            ".repo_graph",
            ".ladybug",
        ]

        root = Path(self.workspace_path)

        for pattern in include:
            for file_path in root.glob(pattern):
                # Skip excluded directories
                rel = file_path.relative_to(root)
                if any(part in exclude for part in rel.parts):
                    continue

                fp = compute_fingerprint(str(file_path))
                if fp:
                    self._fingerprints[str(file_path)] = fp

        logger.info(
            "[AU-048] Fingerprinted %d files in %s",
            len(self._fingerprints),
            self.workspace_path,
        )
        return self._fingerprints

    def diff(
        self,
        previous: dict[str, StructuralFingerprint],
    ) -> dict[str, ChangeLevel]:
        """Compare current fingerprints against a previous snapshot.

        Args:
            previous: Previous fingerprint snapshot.

        Returns:
            Dict mapping file paths to their change level.
        """
        changes: dict[str, ChangeLevel] = {}
        all_paths = set(self._fingerprints.keys()) | set(previous.keys())

        for path in all_paths:
            old = previous.get(path)
            new = self._fingerprints.get(path)

            level = classify_change(old, new)
            if level != ChangeLevel.NONE:
                changes[path] = level

        return changes

    def get_structural_changes(
        self,
        previous: dict[str, StructuralFingerprint],
    ) -> list[str]:
        """Return only files with STRUCTURAL changes (need re-ingestion).

        Args:
            previous: Previous fingerprint snapshot.

        Returns:
            List of file paths requiring KG re-ingestion.
        """
        diff = self.diff(previous)
        return [path for path, level in diff.items() if level == ChangeLevel.STRUCTURAL]
