#!/usr/bin/python
from __future__ import annotations

"""Permission Policy Engine — Deny/Allow Rules for Tool and File Access.

CONCEPT:AU-OS.governance.permission-policy — Permission Policy Engine

Version-controlled permission rules that restrict tool and file access
for all agents in the project. Loaded from ``.agents/permissions.json``
and enforced via PRE_TOOL_USE hooks.

Policy format::

    {
      "version": 1,
      "deny_paths": ["*.env", "secrets/**", ".ssh/**"],
      "deny_tools": ["run_command"],
      "allow_paths": ["src/**", "tests/**"],
      "deny_write_paths": ["pyproject.toml", "LICENSE"],
      "rules": [
        {"tool": "run_command", "deny_args": {"pattern": "rm -rf"}}
      ]
    }
"""

import fnmatch
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..capabilities.hooks import HookInput, HookResult

logger = logging.getLogger(__name__)

__all__ = [
    "PermissionPolicy",
    "PermissionPolicyEngine",
    "create_permission_hook",
    "PolicyViolation",
]


class PolicyViolation(Exception):
    """Raised when a tool call violates the permission policy."""

    def __init__(self, rule: str, detail: str = "") -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"Policy violation [{rule}]: {detail}")


@dataclass
class PermissionRule:
    """A single permission rule with optional argument matching."""

    tool: str = ""
    deny_args: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class PermissionPolicy:
    """Complete permission policy for a project.

    CONCEPT:AU-OS.governance.permission-policy — Permission Policy Engine
    """

    version: int = 1
    deny_paths: list[str] = field(default_factory=list)
    allow_paths: list[str] = field(default_factory=list)
    deny_write_paths: list[str] = field(default_factory=list)
    deny_tools: list[str] = field(default_factory=list)
    rules: list[PermissionRule] = field(default_factory=list)

    @classmethod
    def load(cls, workspace: str | Path) -> PermissionPolicy:
        """Load policy from .agents/permissions.json."""
        path = Path(workspace) / ".agents" / "permissions.json"
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rules = [PermissionRule(**r) for r in data.get("rules", [])]
            return cls(
                version=data.get("version", 1),
                deny_paths=data.get("deny_paths", []),
                allow_paths=data.get("allow_paths", []),
                deny_write_paths=data.get("deny_write_paths", []),
                deny_tools=data.get("deny_tools", []),
                rules=rules,
            )
        except Exception as e:
            logger.warning("[ECO-4.5] Failed to load permissions: %s", e)
            return cls()

    def save(self, workspace: str | Path) -> None:
        """Save policy to .agents/permissions.json."""
        path = Path(workspace) / ".agents" / "permissions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "deny_paths": self.deny_paths,
            "allow_paths": self.allow_paths,
            "deny_write_paths": self.deny_write_paths,
            "deny_tools": self.deny_tools,
            "rules": [
                {"tool": r.tool, "deny_args": r.deny_args, "description": r.description}
                for r in self.rules
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class PermissionPolicyEngine:
    """Evaluate tool calls against the permission policy.

    CONCEPT:AU-OS.governance.permission-policy — Permission Policy Engine

    Usage::

        engine = PermissionPolicyEngine("/my/project")
        engine.check_tool("run_command", {"command": "rm -rf /"})
        # Raises PolicyViolation if denied
    """

    def __init__(self, workspace: str | Path = ".") -> None:
        self.workspace = Path(workspace).resolve()
        self.policy = PermissionPolicy.load(self.workspace)

    def reload(self) -> None:
        """Reload policy from disk."""
        self.policy = PermissionPolicy.load(self.workspace)

    def check_tool(self, tool_name: str, args: dict[str, Any]) -> None:
        """Check if a tool call is permitted. Raises PolicyViolation if denied."""
        # Check tool deny list
        if tool_name in self.policy.deny_tools:
            raise PolicyViolation("deny_tool", f"Tool '{tool_name}' is denied")

        # Check custom rules
        for rule in self.policy.rules:
            if rule.tool and rule.tool == tool_name:
                for key, pattern in rule.deny_args.items():
                    val = str(args.get(key, ""))
                    if re.search(pattern, val):
                        raise PolicyViolation(
                            "deny_args",
                            f"Tool '{tool_name}' arg '{key}' matches deny pattern '{pattern}'",
                        )

        # Check file paths in arguments
        for key in (
            "file_path",
            "path",
            "target_file",
            "filename",
            "source",
            "destination",
        ):
            if key in args:
                self._check_path(str(args[key]), tool_name)

    def check_file_read(self, file_path: str) -> None:
        """Check if reading a file is permitted."""
        self._check_path(file_path, "read")

    def check_file_write(self, file_path: str) -> None:
        """Check if writing a file is permitted."""
        self._check_path(file_path, "write")

        # Additional write-specific deny paths
        rel = self._relative_path(file_path)
        for pattern in self.policy.deny_write_paths:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(
                Path(rel).name, pattern
            ):
                raise PolicyViolation("deny_write_path", f"Write to '{rel}' is denied")

    def _check_path(self, file_path: str, operation: str) -> None:
        """Check a file path against deny/allow lists."""
        rel = self._relative_path(file_path)

        # Check deny paths
        for pattern in self.policy.deny_paths:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(
                Path(rel).name, pattern
            ):
                raise PolicyViolation(
                    "deny_path", f"Access to '{rel}' is denied ({operation})"
                )

        # If allow_paths is set, check that path matches at least one
        if self.policy.allow_paths:
            if not any(
                fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(Path(rel).name, p)
                for p in self.policy.allow_paths
            ):
                raise PolicyViolation(
                    "allow_path", f"Path '{rel}' not in allow list ({operation})"
                )

    def _relative_path(self, file_path: str) -> str:
        """Convert to workspace-relative path for matching."""
        fp = Path(file_path)
        try:
            return str(fp.resolve().relative_to(self.workspace))
        except ValueError:
            return str(fp)

    def as_hook(self):
        """Return a PRE_TOOL_USE hook callable."""

        async def _permission_hook(input: HookInput) -> HookResult | None:
            from ..capabilities.hooks import HookEvent
            from ..capabilities.hooks import HookResult as HR

            if input.event != HookEvent.PRE_TOOL_USE:
                return None

            tool_name = getattr(input, "tool_name", "")
            tool_args = getattr(input, "tool_args", {}) or {}

            try:
                self.check_tool(tool_name, tool_args)
            except PolicyViolation as e:
                logger.warning("[ECO-4.5] %s", e)
                return HR(modify_result=f"DENIED: {e}")

            return None

        return _permission_hook


def create_permission_hook(workspace: str | Path = "."):
    """Convenience: create a permission policy hook callable."""
    return PermissionPolicyEngine(workspace=workspace).as_hook()
