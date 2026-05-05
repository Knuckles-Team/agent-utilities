#!/usr/bin/python
"""CONCEPT:OS-5.0 — Watchdog File Trigger System.

Monitors the project workspace for file changes and maps them to
autonomous graph execution queries.  This enables the agent ecosystem
to react to environmental changes without manual user intervention.

Architecture:
    - Uses ``watchdog`` for filesystem monitoring when available,
      with a polling fallback for environments without inotify.
    - Maps file patterns to predefined maintenance queries.
    - Rate-limits triggers to prevent cascading re-execution.
    - Can discover installed Python packages via ``systems-manager``
      for dependency drift detection.

Integrates with:
    - AU-030 (CognitiveScheduler): Triggers run at LOW priority
    - AU-019 (ResourceOptimizer): Budget-capped execution
    - AU-032 (AgentRegistry): Monitors registry directory changes
    - ``systems-manager``: Package audit via pip inspection

See docs/watchdog-triggers.md §AU-036.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import subprocess  # nosec B404
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Rate limiting: minimum seconds between triggers for the same pattern
MIN_TRIGGER_INTERVAL = 30


@dataclass
class TriggerRule:
    """A rule mapping a file pattern to an autonomous query.

    Attributes:
        pattern: Glob pattern to match (e.g., ``pyproject.toml``).
        query: The graph query to execute when the pattern is matched.
        priority: Execution priority (LOW, MEDIUM, HIGH).
        cooldown: Minimum seconds between triggers for this rule.
    """

    pattern: str
    query: str
    priority: str = "LOW"
    cooldown: int = MIN_TRIGGER_INTERVAL


# Default trigger rules — covers the most common maintenance scenarios
DEFAULT_TRIGGERS: list[TriggerRule] = [
    TriggerRule(
        pattern="pyproject.toml",
        query=(
            "The pyproject.toml was modified. Run a dependency audit: "
            "check for version conflicts, missing dependencies, and "
            "update the lockfile if needed."
        ),
        priority="MEDIUM",
        cooldown=60,
    ),
    TriggerRule(
        pattern="mcp_config.json",
        query=(
            "The MCP configuration was modified. Reload MCP servers, "
            "validate the config structure, and check that all referenced "
            "servers are reachable."
        ),
        priority="HIGH",
        cooldown=30,
    ),
    TriggerRule(
        pattern="requirements*.txt",
        query=(
            "A requirements file was modified. Verify all listed packages "
            "are compatible and run a security audit against known CVEs."
        ),
        priority="MEDIUM",
        cooldown=60,
    ),
    TriggerRule(
        pattern="tests/**/*.py",
        query=(
            "A test file was modified. Run the affected test suite and "
            "report any failures."
        ),
        priority="LOW",
        cooldown=45,
    ),
    TriggerRule(
        pattern="*.py",
        query=(
            "A Python source file was created or modified. Run pre-commit "
            "checks (ruff, bandit) on the changed file."
        ),
        priority="LOW",
        cooldown=30,
    ),
]


@dataclass
class FileWatcher:
    """CONCEPT:OS-5.0 — Watchdog-triggered autonomous graph execution.

    Monitors a project directory for file changes and maps them to
    graph execution queries via configurable trigger rules.

    Args:
        project_root: Root directory to watch.
        triggers: List of ``TriggerRule`` instances.
        registry_path: Optional registry directory to also monitor.
    """

    project_root: str = ""
    triggers: list[TriggerRule] = field(default_factory=lambda: list(DEFAULT_TRIGGERS))
    registry_path: str | None = None

    _last_triggered: dict[str, float] = field(default_factory=dict, repr=False)
    _pending_queries: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def check_file_change(self, filepath: str) -> dict[str, Any] | None:
        """Evaluate a file change against trigger rules.

        Args:
            filepath: Absolute or relative path of the changed file.

        Returns:
            A trigger dict with ``query``, ``priority``, and ``filepath``
            if a rule matches and the cooldown has elapsed, else ``None``.
        """
        rel_path = filepath
        if self.project_root and filepath.startswith(self.project_root):
            rel_path = os.path.relpath(filepath, self.project_root)

        for rule in self.triggers:
            if fnmatch.fnmatch(rel_path, rule.pattern) or fnmatch.fnmatch(
                os.path.basename(filepath), rule.pattern
            ):
                now = time.time()
                last = self._last_triggered.get(rule.pattern, 0)
                if now - last < rule.cooldown:
                    logger.debug(
                        f"[AU-036] Trigger for '{rule.pattern}' on cooldown "
                        f"({int(rule.cooldown - (now - last))}s remaining)"
                    )
                    return None

                self._last_triggered[rule.pattern] = now
                trigger = {
                    "query": rule.query,
                    "priority": rule.priority,
                    "filepath": filepath,
                    "pattern": rule.pattern,
                    "timestamp": now,
                }
                self._pending_queries.append(trigger)
                logger.info(
                    f"[AU-036] File trigger matched: '{rel_path}' → "
                    f"'{rule.pattern}' (priority: {rule.priority})"
                )
                return trigger

        return None

    def drain_pending(self) -> list[dict[str, Any]]:
        """Drain and return all pending triggered queries.

        Returns:
            List of trigger dicts, cleared from the internal queue.
        """
        pending = list(self._pending_queries)
        self._pending_queries.clear()
        return pending

    def add_trigger(self, pattern: str, query: str, **kwargs: Any) -> None:
        """Add a custom trigger rule at runtime.

        Args:
            pattern: Glob pattern to match.
            query: Query string for graph execution.
            **kwargs: Additional ``TriggerRule`` fields.
        """
        self.triggers.append(TriggerRule(pattern=pattern, query=query, **kwargs))

    # ── Package Audit (via systems-manager) ─────────────────────────

    @staticmethod
    def audit_installed_packages() -> dict[str, Any]:
        """Check installed Python packages for version drift and security issues.

        Uses ``pip list --outdated --format=json`` for package staleness
        and ``pip audit`` (if available) for known vulnerabilities.

        Returns:
            Dict with ``outdated`` (list of package dicts) and
            ``vulnerabilities`` (list of advisory dicts).
        """
        result: dict[str, Any] = {"outdated": [], "vulnerabilities": []}

        # Check for outdated packages
        try:
            proc = subprocess.run(  # nosec B603 B607
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                result["outdated"] = json.loads(proc.stdout)[:20]  # Cap at 20
                logger.info(
                    f"[AU-036] Found {len(result['outdated'])} outdated packages"
                )
        except Exception as e:
            logger.debug(f"[AU-036] pip outdated check failed: {e}")

        # Check for vulnerabilities (requires pip-audit)
        try:
            proc = subprocess.run(  # nosec B603 B607
                ["pip-audit", "--format=json", "--progress-spinner=off"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                audit_data = json.loads(proc.stdout)
                vulns = audit_data.get("vulnerabilities", audit_data)
                if isinstance(vulns, list):
                    result["vulnerabilities"] = vulns[:20]
        except FileNotFoundError:
            logger.debug("[AU-036] pip-audit not installed — skipping vuln scan")
        except Exception as e:
            logger.debug(f"[AU-036] pip-audit check failed: {e}")

        return result

    @staticmethod
    def list_installed_packages() -> list[dict[str, str]]:
        """List all installed Python packages with versions.

        Returns:
            List of dicts with ``name`` and ``version`` keys.
        """
        try:
            proc = subprocess.run(  # nosec B603 B607
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return json.loads(proc.stdout)
        except Exception as e:
            logger.debug(f"[AU-036] pip list failed: {e}")
        return []

    # ── Polling-based watcher (fallback when watchdog is unavailable) ─

    def poll_directory(
        self, snapshot: dict[str, float] | None = None
    ) -> tuple[list[str], dict[str, float]]:
        """Poll the project directory for changes since the last snapshot.

        Args:
            snapshot: Previous {filepath: mtime} snapshot. If None, builds
                      a fresh baseline.

        Returns:
            Tuple of (list of changed files, new snapshot).
        """
        root = Path(self.project_root or os.getcwd())
        current: dict[str, float] = {}
        changed: list[str] = []

        for path in root.rglob("*"):
            if path.is_file() and not any(part.startswith(".") for part in path.parts):
                try:
                    fpath = str(path)
                    current[fpath] = path.stat().st_mtime
                    if snapshot and fpath in snapshot:
                        if current[fpath] > snapshot[fpath]:
                            changed.append(fpath)
                    elif snapshot and fpath not in snapshot:
                        changed.append(fpath)  # New file
                except OSError:
                    pass

        return changed, current
