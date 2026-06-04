#!/usr/bin/python
from __future__ import annotations

"""Lint Enforcement Hook — Deterministic Code Quality via PRE_TOOL_USE.

CONCEPT:ECO-4.11 — Deterministic Lint Enforcement Hook

For automated checks like linting and formatting, hooks enforce the rules
deterministically and produce more consistent results than relying on the
agent to remember an instruction.  This hook intercepts file-write tool
calls, runs configured linters, and surfaces violations immediately.

No LLM involvement — pure subprocess execution for determinism.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..capabilities.hooks import HookInput, HookResult

logger = logging.getLogger(__name__)

__all__ = ["LintEnforcementHook", "LintConfig", "create_lint_hook"]


@dataclass
class LintRule:
    """A single linter configuration."""

    name: str
    command: list[str]
    extensions: list[str]
    fix_command: list[str] | None = None
    enabled: bool = True


@dataclass
class LintConfig:
    """Lint configuration loaded from .agents/lint.yaml or AGENTS.md."""

    rules: list[LintRule] = field(default_factory=list)
    auto_fix: bool = False
    fail_on_error: bool = False

    @classmethod
    def default(cls) -> LintConfig:
        """Default linter config for Python projects."""
        rules = []
        if shutil.which("ruff"):
            rules.append(
                LintRule(
                    name="ruff-check",
                    command=["ruff", "check", "--no-fix"],
                    extensions=[".py"],
                    fix_command=["ruff", "check", "--fix"],
                )
            )
            rules.append(
                LintRule(
                    name="ruff-format",
                    command=["ruff", "format", "--check"],
                    extensions=[".py"],
                    fix_command=["ruff", "format"],
                )
            )
        if shutil.which("mypy"):
            rules.append(
                LintRule(
                    name="mypy",
                    command=["mypy", "--no-error-summary"],
                    extensions=[".py"],
                )
            )
        return cls(rules=rules)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LintConfig:
        """Load config from .agents/lint.yaml."""
        import yaml

        p = Path(path)
        if not p.is_file():
            return cls.default()
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            rules = [LintRule(**r) for r in data.get("rules", [])]
            return cls(
                rules=rules or cls.default().rules,
                auto_fix=data.get("auto_fix", False),
                fail_on_error=data.get("fail_on_error", False),
            )
        except Exception as e:
            logger.warning("[ECO-4.3] Failed to load lint config: %s", e)
            return cls.default()


@dataclass
class LintResult:
    """Result of running a linter on a file."""

    rule_name: str
    file_path: str
    passed: bool
    output: str = ""
    fixed: bool = False


class LintEnforcementHook:
    """Deterministic lint enforcement via PRE_TOOL_USE hook.

    CONCEPT:ECO-4.11 — Deterministic Lint Enforcement Hook

    Usage::

        hook = LintEnforcementHook(workspace_path="/my/project")
        # Register as a POST_TOOL_USE hook to check files after writes
        hooks_capability.hooks.append(hook.as_hook())
    """

    def __init__(
        self,
        workspace_path: str | Path = ".",
        config: LintConfig | None = None,
        timeout: int = 30,
    ) -> None:
        self.workspace_path = Path(workspace_path).resolve()
        self.config = config or self._load_config()
        self.timeout = timeout

    def _load_config(self) -> LintConfig:
        yaml_path = self.workspace_path / ".agents" / "lint.yaml"
        if yaml_path.is_file():
            return LintConfig.from_yaml(yaml_path)
        return LintConfig.default()

    def lint_file(self, file_path: str | Path) -> list[LintResult]:
        """Run all applicable linters on a single file."""
        fp = Path(file_path)
        suffix = fp.suffix.lower()
        results: list[LintResult] = []

        for rule in self.config.rules:
            if not rule.enabled or suffix not in rule.extensions:
                continue
            result = self._run_linter(rule, str(fp))
            results.append(result)

            # Auto-fix if configured and linter failed
            if not result.passed and self.config.auto_fix and rule.fix_command:
                fix_result = self._run_fix(rule, str(fp))
                if fix_result:
                    result = LintResult(
                        rule_name=rule.name,
                        file_path=str(fp),
                        passed=True,
                        output="Auto-fixed",
                        fixed=True,
                    )
                    results[-1] = result

        return results

    def _run_linter(self, rule: LintRule, file_path: str) -> LintResult:
        """Execute a linter subprocess."""
        cmd = rule.command + [file_path]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.workspace_path),
            )
            return LintResult(
                rule_name=rule.name,
                file_path=file_path,
                passed=proc.returncode == 0,
                output=(proc.stdout + proc.stderr).strip()[:2000],
            )
        except subprocess.TimeoutExpired:
            return LintResult(
                rule_name=rule.name,
                file_path=file_path,
                passed=False,
                output=f"Timeout after {self.timeout}s",
            )
        except FileNotFoundError:
            return LintResult(
                rule_name=rule.name,
                file_path=file_path,
                passed=True,
                output=f"Linter '{rule.command[0]}' not found, skipping",
            )

    def _run_fix(self, rule: LintRule, file_path: str) -> bool:
        """Run auto-fix command."""
        if not rule.fix_command:
            return False
        try:
            proc = subprocess.run(
                rule.fix_command + [file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.workspace_path),
            )
            return proc.returncode == 0
        except Exception:
            return False

    def as_hook(self):
        """Return a callable for HooksCapability POST_TOOL_USE registration."""

        async def _lint_hook(input: HookInput) -> HookResult | None:
            from ..capabilities.hooks import HookEvent
            from ..capabilities.hooks import HookResult as HR

            if input.event != HookEvent.POST_TOOL_USE:
                return None

            # Check if tool wrote a file
            file_path = _extract_written_file(input)
            if not file_path:
                return None

            results = self.lint_file(file_path)
            failures = [r for r in results if not r.passed]
            if not failures:
                return None

            report = "\n".join(f"[{r.rule_name}] {r.output[:500]}" for r in failures)
            logger.info("[ECO-4.3] Lint violations in %s:\n%s", file_path, report)

            return HR(
                modify_result=f"\n\n⚠️ Lint violations detected:\n{report}"
                if not self.config.fail_on_error
                else None,
            )

        return _lint_hook


def create_lint_hook(workspace_path: str | Path = ".", **kw: Any):
    """Convenience: create a lint enforcement hook callable."""
    return LintEnforcementHook(workspace_path=workspace_path, **kw).as_hook()


def _extract_written_file(input: Any) -> str | None:
    """Extract file path from a tool call that wrote a file."""
    try:
        args = getattr(input, "tool_args", {}) or {}
        for key in ("file_path", "path", "target_file", "filename"):
            if key in args:
                return str(args[key])
        tool_name = getattr(input, "tool_name", "")
        if "write" in tool_name or "create" in tool_name or "edit" in tool_name:
            for v in args.values():
                if isinstance(v, str) and ("/" in v or "\\" in v):
                    return v
    except Exception:
        pass
    return None
