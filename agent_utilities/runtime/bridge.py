"""CONCEPT:ORCH-1.46 — Action dispatcher: the bidirectional bridge from a typed Action to a
backend operation and back to a typed Observation.

The dispatcher is the one place that knows *how* each action kind executes. It is deliberately
backend-agnostic: it drives a :class:`~.workspace.WorkspaceBackend` (shell exec) and does file
operations host-side on ``backend.root`` (for the Docker backend that path is the container's
bind-mount, so a host-side write is visible in-container and vice versa). Shell state that must
persist across commands — the working directory — is captured via a marker file the wrapped
command writes ``pwd`` into, and read back into :class:`WorkspaceState`.

Why host-side file ops instead of a UDS helper bridge (as ``rlm/sandboxes/docker_backend.py``
does for snippet exec): the workspace bind-mount already gives a shared filesystem, so file
read/write/edit need no in-container shim — only shell commands cross the boundary. This keeps
the protocol small and the failure modes few.
"""

from __future__ import annotations

import difflib
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .events import (
    AgentFinishAction,
    BrowseAction,
    CmdOutputObservation,
    CmdRunAction,
    ErrorObservation,
    FileContentObservation,
    FileEditAction,
    FileEditObservation,
    FileReadAction,
    FileWriteAction,
    FileWriteObservation,
    NullObservation,
    PortExposeAction,
    PortObservation,
    TestResultObservation,
    TestRunAction,
)

if TYPE_CHECKING:
    from .browser_tier import BrowserDriver
    from .events import Action, Observation
    from .workspace import WorkspaceBackend

# Marker file the cwd-capturing wrapper writes ``pwd`` into (lives at the workspace root).
_CWD_MARKER = ".au_workspace_cwd"
# Observations truncate raw output to keep KG provenance and the event stream bounded.
_MAX_RAW = 16_384


@dataclass
class WorkspaceState:
    """Mutable shell state the workspace carries across actions."""

    cwd: str
    env: dict[str, str] = field(default_factory=dict)


def _wrap_for_cwd(command: str, cwd: str, marker_ctx_path: str) -> str:
    """Wrap a command so it runs in ``cwd`` and records the resulting cwd to a marker file.

    The marker lets a ``cd`` inside ``command`` persist to the next action — the workspace's
    stateful-shell guarantee — without a long-lived PTY.
    """
    return (
        f"cd {shlex.quote(cwd)} 2>/dev/null || true\n"
        f"{command}\n"
        f"__au_rc=$?\n"
        f"pwd > {shlex.quote(marker_ctx_path)} 2>/dev/null || true\n"
        f"exit $__au_rc\n"
    )


_PYTEST_SUMMARY = re.compile(
    r"(?:(\d+) failed)?[,\s]*(?:(\d+) passed)?[,\s]*(?:(\d+) errors?)?",
)


def _parse_pytest(output: str) -> tuple[int, int, int]:
    """Extract (passed, failed, errors) from a pytest summary line. Best-effort."""
    passed = failed = errors = 0
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            for n, kind in re.findall(r"(\d+)\s+(passed|failed|errors?)", line):
                if kind == "passed":
                    passed = int(n)
                elif kind == "failed":
                    failed = int(n)
                else:
                    errors = int(n)
            if passed or failed or errors:
                break
    return passed, failed, errors


class ActionDispatcher:
    """Execute one Action against a backend, returning the matching Observation."""

    async def dispatch(
        self,
        action: Action,
        backend: WorkspaceBackend,
        state: WorkspaceState,
        browser: BrowserDriver | None = None,
    ) -> Observation:
        try:
            if isinstance(action, CmdRunAction):
                return await self._cmd(action, backend, state)
            if isinstance(action, FileReadAction):
                return self._read(action, backend)
            if isinstance(action, FileWriteAction):
                return self._write(action, backend)
            if isinstance(action, FileEditAction):
                return self._edit(action, backend)
            if isinstance(action, TestRunAction):
                return await self._test(action, backend, state)
            if isinstance(action, PortExposeAction):
                return self._port(action, backend)
            if isinstance(action, BrowseAction):
                return await self._browse(action, browser)
            if isinstance(action, AgentFinishAction):
                return NullObservation()
        except Exception as exc:  # noqa: BLE001 - any backend/FS error -> typed observation
            return ErrorObservation(
                message=str(exc), action_kind=getattr(action, "kind", "")
            )
        return ErrorObservation(
            message=f"unknown action kind {getattr(action, 'kind', '?')!r}",
            action_kind=getattr(action, "kind", ""),
        )

    # ── shell ───────────────────────────────────────────────────────────────
    async def _cmd(
        self, action: CmdRunAction, backend: WorkspaceBackend, state: WorkspaceState
    ) -> CmdOutputObservation:
        cwd = action.cwd or state.cwd
        marker_ctx = f"{backend.workdir.rstrip('/')}/{_CWD_MARKER}"
        script = _wrap_for_cwd(action.command, cwd, marker_ctx)
        code, out, err = await backend.exec_shell(
            script, dict(state.env), action.timeout
        )
        self._sync_cwd(backend, state)
        return CmdOutputObservation(
            exit_code=code, stdout=out[:_MAX_RAW], stderr=err[:_MAX_RAW], cwd=state.cwd
        )

    def _sync_cwd(self, backend: WorkspaceBackend, state: WorkspaceState) -> None:
        marker = backend.root / _CWD_MARKER
        try:
            if marker.exists():
                new_cwd = marker.read_text().strip()
                if new_cwd:
                    state.cwd = new_cwd
        except OSError:
            pass

    # ── files (host-side on the bind-mount / workspace root) ──────────────────
    def _host_path(self, backend: WorkspaceBackend, path: str) -> Path:
        """Translate an execution-context path to the host path under ``backend.root``."""
        workdir = backend.workdir.rstrip("/")
        if path.startswith(workdir + "/"):
            rel = path[len(workdir) + 1 :]
            return backend.root / rel
        if path == workdir:
            return backend.root
        p = Path(path)
        if p.is_absolute():
            # Local backend: absolute host path is usable as-is; Docker: only the mount is reachable.
            return p if backend.name == "local" else backend.root / p.name
        return backend.root / path

    def _read(
        self, action: FileReadAction, backend: WorkspaceBackend
    ) -> FileContentObservation:
        text = self._host_path(backend, action.path).read_text(errors="replace")
        if action.start is not None or action.end is not None:
            lines = text.splitlines()
            start = (action.start or 1) - 1
            end = action.end if action.end is not None else len(lines)
            text = "\n".join(lines[start:end])
        return FileContentObservation(path=action.path, content=text)

    def _write(
        self, action: FileWriteAction, backend: WorkspaceBackend
    ) -> FileWriteObservation:
        hp = self._host_path(backend, action.path)
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(action.content)
        return FileWriteObservation(
            path=action.path, bytes_written=len(action.content.encode())
        )

    def _edit(self, action: FileEditAction, backend: WorkspaceBackend) -> Observation:
        hp = self._host_path(backend, action.path)
        original = hp.read_text(errors="replace")
        if action.old not in original:
            return ErrorObservation(
                message=f"old string not found in {action.path}",
                action_kind="file_edit",
            )
        count = original.count(action.old)
        if count > 1 and not action.replace_all:
            return ErrorObservation(
                message=f"old string is not unique in {action.path} ({count} matches); "
                "set replace_all or provide more context",
                action_kind="file_edit",
            )
        updated = original.replace(
            action.old, action.new, -1 if action.replace_all else 1
        )
        hp.write_text(updated)
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{action.path}",
                tofile=f"b/{action.path}",
            )
        )
        return FileEditObservation(
            path=action.path,
            diff=diff[:_MAX_RAW],
            applied=True,
            replacements=count if action.replace_all else 1,
        )

    # ── tests ────────────────────────────────────────────────────────────────
    async def _test(
        self, action: TestRunAction, backend: WorkspaceBackend, state: WorkspaceState
    ) -> TestResultObservation:
        target = action.selector or ""
        if action.framework == "pytest":
            cmd = (
                f"python -m pytest {target} -q --no-header -p no:cacheprovider".strip()
            )
        else:
            cmd = action.selector or "make test"
        cwd = action.cwd or state.cwd
        marker_ctx = f"{backend.workdir.rstrip('/')}/{_CWD_MARKER}"
        code, out, err = await backend.exec_shell(
            _wrap_for_cwd(cmd, cwd, marker_ctx), dict(state.env), action.timeout
        )
        combined = (out + "\n" + err).strip()
        passed, failed, errors = _parse_pytest(combined)
        summary = f"{passed} passed, {failed} failed, {errors} errors (exit {code})"
        return TestResultObservation(
            passed=passed,
            failed=failed,
            errors=errors,
            exit_code=code,
            report=summary,
            raw=combined[-_MAX_RAW:],
        )

    # ── ports ────────────────────────────────────────────────────────────────
    def _port(
        self, action: PortExposeAction, backend: WorkspaceBackend
    ) -> PortObservation:
        url = backend.exposed_url(action.port)
        return PortObservation(port=action.port, url=url)

    # ── browser (optional tier, ECO-4.44) ─────────────────────────────────────
    async def _browse(
        self, action: BrowseAction, browser: BrowserDriver | None
    ) -> Observation:
        from .browser_tier import NullBrowserDriver

        driver = browser if browser is not None else NullBrowserDriver()
        return await driver.browse(action)
