#!/usr/bin/python
from __future__ import annotations

"""Confined physical artifact staging for approved evolution proposals.

This compatibility facade may update an already-selected artifact beneath one
configured staging/workspace root.  It never follows a path outside that root,
never executes or commits graph-carried data, and never logs identifiers,
content, or machine paths.  Reviewable branch publication belongs to the
governed ``ChangePublisher`` seam.
"""

import ast
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.core.config import AgentConfig
from agent_utilities.security.persistence_privacy import sanitize_for_persistence

logger = logging.getLogger(__name__)

_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
_MAX_DESCRIPTION_BYTES = 32 * 1024
_SAFE_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_SAFE_PYTHON_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


def _bounded_text(value: Any, *, limit: int = _MAX_DESCRIPTION_BYTES) -> str:
    rendered = str(value or "")
    if "\x00" in rendered or len(rendered.encode("utf-8")) > limit:
        raise ValueError("artifact text exceeds its safety boundary")
    sanitized, _report = sanitize_for_persistence(rendered)
    if sanitized != rendered:
        raise ValueError("artifact text contains prohibited identifying material")
    return rendered


def _atomic_write(target: Path, content: str) -> None:
    payload = content.encode("utf-8")
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds the write-size boundary")
    if target.is_symlink():
        raise PermissionError("symbolic-link artifact targets are not permitted")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temp, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("atomic artifact write did not make progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temp, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temp.unlink(missing_ok=True)


class PhysicalDistillationEngine:
    """Stage bounded artifact updates beneath an explicit trusted root."""

    def __init__(self, workspace_root: str | None = None) -> None:
        configured = str(
            workspace_root or AgentConfig().evolution_staging_root or ""
        ).strip()
        if not configured:
            raise ValueError(
                "PhysicalDistillationEngine requires an explicit evolution staging root"
            )
        configured_root = Path(configured).expanduser()
        if configured_root.is_symlink():
            raise ValueError("evolution staging root cannot be a symbolic link")
        root = configured_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("evolution staging root is not a directory")
        self.workspace_root = root

    def _target(
        self,
        raw: str,
        *,
        expected_name: str | None = None,
        expected_suffix: str | None = None,
        must_exist: bool,
    ) -> Path:
        candidate = Path(str(raw or "")).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        lexical = Path(os.path.abspath(candidate))
        try:
            relative = lexical.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError(
                "artifact target is outside the staging root"
            ) from exc
        cursor = self.workspace_root
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise PermissionError("symbolic-link artifact paths are not permitted")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError(
                "artifact target is outside the staging root"
            ) from exc
        if expected_name is not None and resolved.name != expected_name:
            raise ValueError("artifact target has an unexpected filename")
        if expected_suffix is not None and resolved.suffix.lower() != expected_suffix:
            raise ValueError("artifact target has an unexpected file type")
        if must_exist and (not resolved.is_file() or resolved.is_symlink()):
            raise FileNotFoundError("artifact target is unavailable")
        if not must_exist:
            parent = resolved.parent.resolve(strict=False)
            try:
                parent.relative_to(self.workspace_root)
            except ValueError as exc:
                raise PermissionError(
                    "artifact parent is outside the staging root"
                ) from exc
        return resolved

    def distill_skill(
        self,
        skill_id: str,
        new_name: str,
        new_description: str,
        artifact_path: str,
        tags: list[str] | None = None,
        requires: list[str] | None = None,
    ) -> bool:
        """Update one existing staged ``SKILL.md`` after structural/privacy checks."""

        del skill_id
        try:
            name = str(new_name or "").strip()
            if not _SAFE_SKILL_NAME.fullmatch(name):
                raise ValueError("skill name is invalid")
            description = _bounded_text(new_description)
            normalized_tags = self._safe_string_list(tags)
            normalized_requires = self._safe_string_list(requires)

            selected = self._target(str(artifact_path), must_exist=False)
            if selected.is_dir():
                selected = selected / "SKILL.md"
            target = self._target(
                str(selected), expected_name="SKILL.md", must_exist=True
            )
            source = target.read_text(encoding="utf-8")
            if len(source.encode("utf-8")) > _MAX_ARTIFACT_BYTES:
                raise ValueError("existing skill artifact is too large")
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", source, re.DOTALL)
            if match:
                parsed = yaml.safe_load(match.group(1)) or {}
                if not isinstance(parsed, dict):
                    raise ValueError("skill frontmatter must be a mapping")
                frontmatter: dict[str, Any] = dict(parsed)
                body = match.group(2)
            else:
                frontmatter = {}
                body = source
            frontmatter["name"] = name
            frontmatter["description"] = description
            if tags is not None:
                frontmatter["tags"] = normalized_tags
            if requires is not None:
                frontmatter["requires"] = normalized_requires
            header = yaml.safe_dump(
                frontmatter,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            ).strip()
            _atomic_write(target, f"---\n{header}\n---\n\n{body.lstrip()}")
            logger.info("Approved skill artifact staged")
            return True
        except Exception as exc:  # noqa: BLE001 - compatibility API returns bool
            logger.warning(
                "Skill artifact staging rejected (exception_type=%s)",
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _safe_string_list(values: list[str] | None) -> list[str]:
        if values is None:
            return []
        if len(values) > 64:
            raise ValueError("artifact list exceeds its item boundary")
        normalized = [str(value or "").strip() for value in values]
        if any(not _SAFE_TAG.fullmatch(value) for value in normalized):
            raise ValueError("artifact list contains an invalid item")
        return normalized

    def distill_mcp_tool(
        self,
        tool_name: str,
        new_description: str,
        file_path: str,
        function_name: str,
    ) -> bool:
        """Update a Python docstring without allowing string-literal injection."""

        del tool_name
        try:
            if not _SAFE_PYTHON_NAME.fullmatch(str(function_name or "")):
                raise ValueError("function name is invalid")
            description = _bounded_text(new_description)
            target = self._target(file_path, expected_suffix=".py", must_exist=True)
            source = target.read_text(encoding="utf-8")
            if len(source.encode("utf-8")) > _MAX_ARTIFACT_BYTES:
                raise ValueError("existing source artifact is too large")
            tree = ast.parse(source)

            selected: ast.FunctionDef | ast.AsyncFunctionDef | None = None
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == function_name
                ):
                    if selected is not None:
                        raise ValueError("function name is ambiguous")
                    selected = node
            if selected is None or not selected.body:
                raise ValueError("function was not found")

            lines = source.splitlines(keepends=True)
            literal = repr(description)
            first = selected.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
                and first.end_lineno is not None
            ):
                indent = " " * first.col_offset
                lines[first.lineno - 1 : first.end_lineno] = [f"{indent}{literal}\n"]
            else:
                indent = " " * first.col_offset
                lines.insert(first.lineno - 1, f"{indent}{literal}\n")
            updated = "".join(lines)
            ast.parse(updated)
            _atomic_write(target, updated)
            logger.info("Approved MCP tool artifact staged")
            return True
        except Exception as exc:  # noqa: BLE001 - compatibility API returns bool
            logger.warning(
                "MCP tool artifact staging rejected (exception_type=%s)",
                type(exc).__name__,
            )
            return False

    def distill_system_prompt(self, file_path: str, new_content: str) -> bool:
        """Stage one bounded prompt file beneath the configured root."""

        try:
            content = _bounded_text(new_content, limit=_MAX_ARTIFACT_BYTES)
            target = self._target(file_path, must_exist=False)
            if target.suffix.lower() not in {".md", ".json", ".yaml", ".yml", ".txt"}:
                raise ValueError("prompt artifact type is not permitted")
            _atomic_write(target, content)
            logger.info("Approved prompt artifact staged")
            return True
        except Exception as exc:  # noqa: BLE001 - compatibility API returns bool
            logger.warning(
                "Prompt artifact staging rejected (exception_type=%s)",
                type(exc).__name__,
            )
            return False

    def commit_distilled_changes(
        self,
        file_paths: list[str],
        concept_id: str = "AU-AHE.optimization.gitops-commit-automation",
    ) -> bool:
        """Refuse legacy auto-commit; publication requires ActionPolicy approval."""

        del file_paths, concept_id
        logger.warning(
            "Direct distillation commits are retired; use governed ChangePublisher"
        )
        return False
