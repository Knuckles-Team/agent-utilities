"""CONCEPT:AU-ORCH.reactive.action-dispatcher — Action dispatcher: the bidirectional bridge from a typed Action to a
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
import os
import re
import shlex
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .events import (
    AgentFinishAction,
    BrowseAction,
    CmdOutputObservation,
    CmdRunAction,
    ComputerUseAction,
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
    from .computer_use_tier import ComputerUseDriver
    from .events import Action, Observation
    from .workspace import WorkspaceBackend

# Marker file the cwd-capturing wrapper writes ``pwd`` into (lives at the workspace root).
_CWD_MARKER = ".au_workspace_cwd"
# Observations truncate raw output to keep KG provenance and the event stream bounded.
_MAX_RAW = 16_384
_MAX_FILE_BYTES = 1024 * 1024


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
        computer_use: ComputerUseDriver | None = None,
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
            if isinstance(action, ComputerUseAction):
                return await self._computer_use(action, computer_use)
            if isinstance(action, AgentFinishAction):
                return NullObservation()
        except Exception as exc:  # noqa: BLE001 - any backend/FS error -> typed observation
            import logging

            logging.getLogger(__name__).warning(
                "Workspace action failed (exception_type=%s)", type(exc).__name__
            )
            return ErrorObservation(
                message="workspace action failed",
                action_kind=getattr(action, "kind", ""),
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
    def _workspace_parts(self, backend: WorkspaceBackend, path: str) -> tuple[str, ...]:
        """Translate an execution-context path to safe relative components."""
        if (
            not isinstance(path, str)
            or not path
            or len(path) > 4_096
            or any(ord(character) < 32 for character in path)
        ):
            raise ValueError("invalid workspace path")
        workdir = backend.workdir.rstrip("/")
        if path.startswith(workdir + "/"):
            rel = path[len(workdir) + 1 :]
        elif path == workdir:
            rel = ""
        else:
            parsed = Path(path)
            rel = parsed.name if parsed.is_absolute() else path
        parts = tuple(part for part in rel.split("/") if part)
        if not parts or any(part in {".", ".."} for part in parts):
            raise PermissionError("workspace path is not a file")
        return parts

    def _workspace_parent(
        self,
        backend: WorkspaceBackend,
        path: str,
        *,
        create: bool,
    ) -> tuple[int, str]:
        """Open a no-follow parent directory and return ``(fd, leaf)``.

        Every component is resolved relative to an already-open directory
        descriptor. A sandbox process can therefore neither swap a parent for
        a symlink nor redirect the final open outside the bind root.
        """
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        if not nofollow or not directory or os.open not in os.supports_dir_fd:
            raise RuntimeError("secure workspace file operations are unavailable")
        parts = self._workspace_parts(backend, path)
        root_fd = os.open(
            backend.root.resolve(),
            os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
        current = root_fd
        try:
            for component in parts[:-1]:
                try:
                    next_fd = os.open(
                        component,
                        os.O_RDONLY | directory | nofollow,
                        dir_fd=current,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=0o700, dir_fd=current)
                    next_fd = os.open(
                        component,
                        os.O_RDONLY | directory | nofollow,
                        dir_fd=current,
                    )
                if current != root_fd:
                    os.close(current)
                current = next_fd
            if current == root_fd:
                return os.dup(root_fd), parts[-1]
            result = current
            current = -1
            return result, parts[-1]
        finally:
            if current not in {-1, root_fd}:
                os.close(current)
            os.close(root_fd)

    @staticmethod
    def _require_regular_file(descriptor: int) -> None:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PermissionError("workspace target must be a regular file")

    @staticmethod
    def _write_descriptor(descriptor: int, content: bytes) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("workspace write did not make progress")
            view = view[written:]
        os.fsync(descriptor)

    @staticmethod
    def _read_descriptor(descriptor: int, limit: int) -> bytes:
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        result = b"".join(chunks)
        if len(result) > limit:
            raise ValueError("workspace read exceeds the file boundary")
        return result

    def _read(
        self, action: FileReadAction, backend: WorkspaceBackend
    ) -> FileContentObservation:
        parent_fd, leaf = self._workspace_parent(backend, action.path, create=False)
        try:
            descriptor = os.open(
                leaf,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            try:
                self._require_regular_file(descriptor)
                if os.fstat(descriptor).st_size > _MAX_FILE_BYTES:
                    raise ValueError("workspace read exceeds the file boundary")
                raw = self._read_descriptor(descriptor, _MAX_FILE_BYTES)
                text = raw.decode("utf-8", errors="replace")
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_fd)
        if action.start is not None or action.end is not None:
            lines = text.splitlines()
            start = (action.start or 1) - 1
            end = action.end if action.end is not None else len(lines)
            text = "\n".join(lines[start:end])
        return FileContentObservation(path=action.path, content=text)

    def _write(
        self, action: FileWriteAction, backend: WorkspaceBackend
    ) -> FileWriteObservation:
        content = action.content.encode("utf-8")
        if len(content) > _MAX_FILE_BYTES:
            raise ValueError("workspace write exceeds the file boundary")
        parent_fd, leaf = self._workspace_parent(backend, action.path, create=True)
        try:
            descriptor = os.open(
                leaf,
                os.O_WRONLY
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                0o600,
                dir_fd=parent_fd,
            )
            try:
                self._require_regular_file(descriptor)
                self._write_descriptor(descriptor, content)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_fd)
        return FileWriteObservation(
            path=action.path, bytes_written=len(content)
        )

    def _edit(self, action: FileEditAction, backend: WorkspaceBackend) -> Observation:
        parent_fd, leaf = self._workspace_parent(backend, action.path, create=False)
        try:
            descriptor = os.open(
                leaf,
                os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            try:
                self._require_regular_file(descriptor)
                if os.fstat(descriptor).st_size > _MAX_FILE_BYTES:
                    raise ValueError("workspace edit exceeds the file boundary")
                raw = self._read_descriptor(descriptor, _MAX_FILE_BYTES)
                original = raw.decode("utf-8", errors="replace")
                if action.old not in original:
                    return ErrorObservation(
                        message="old string not found",
                        action_kind="file_edit",
                    )
                count = original.count(action.old)
                if count > 1 and not action.replace_all:
                    return ErrorObservation(
                        message="old string is not unique; provide more context",
                        action_kind="file_edit",
                    )
                updated = original.replace(
                    action.old, action.new, -1 if action.replace_all else 1
                )
                encoded = updated.encode("utf-8")
                if len(encoded) > _MAX_FILE_BYTES:
                    raise ValueError("workspace edit exceeds the file boundary")
                self._write_descriptor(descriptor, encoded)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_fd)
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

    # ── computer-use (optional tier, ECO-4.93) ─────────────────────────────────
    async def _computer_use(
        self, action: ComputerUseAction, computer_use: ComputerUseDriver | None
    ) -> Observation:
        from .computer_use_tier import NullComputerUseDriver

        driver = computer_use if computer_use is not None else NullComputerUseDriver()
        return await driver.run(action)
