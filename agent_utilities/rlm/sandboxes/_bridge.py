"""CONCEPT:AU-ORCH.sandbox.shared-host-helper-bridge — Shared host-helper bridge for isolated sandbox backends.

An isolated backend (a ``--network none`` container or a microVM guest) runs
the snippet in a *separate* address space, but the RLM host helpers (``rlm_query``,
``graph_query``, … and ``FINAL_VAR``) are async methods bound to the orchestrator process. The
bridge lets the isolated child reach them: the host runs a Unix-socket server bound to a socket
*inside the shared run dir*; the child connects over the filesystem socket and issues framed
JSON RPCs. Async helpers are awaited host-side; ``FINAL_VAR`` round-trips and mutates the host
``vars`` directly. The child's only egress is that one socket.

The bridge is shared by confined backends so host callbacks use one bounded wire format. Two
child-side forms share that format:

* :func:`make_runner_script` — a self-contained script string for backends whose child cannot
  import ``agent_utilities`` (a container / microVM guest runs ``python <runner>``);
* :func:`run_child` — the importable equivalent for a confined child that can import the
  package, so it needs no injected script.

Wire format (both directions): ``struct.pack(">I", len) + json_bytes``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import secrets
import stat
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .base import HELPER_NAMES

logger = logging.getLogger(__name__)

# Only JSON-able namespace values cross the boundary; live refs can't be serialized — reaching
# them is exactly what the helper bridge is for.
JSONABLE = (str, int, float, bool, type(None), list, dict)
MAX_BRIDGE_REQUEST_BYTES = 1024 * 1024
MAX_BRIDGE_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_BYTES = 16 * 1024 * 1024
MAX_USER_CODE_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_RPC_ARGS = 64
MAX_JSON_ITEMS = 10_000
MAX_JSON_DEPTH = 16
BRIDGE_IO_TIMEOUT_SECONDS = 30.0
HELPER_TIMEOUT_SECONDS = 60.0


def new_bridge_token() -> str:
    """Return a per-run capability token for the host-helper socket."""
    return secrets.token_urlsafe(32)


def _bounded_json_shape(value: Any) -> bool:
    """Reject deeply nested or item-heavy JSON before helper dispatch."""
    stack: list[tuple[Any, int]] = [(value, 0)]
    items = 0
    while stack:
        current, depth = stack.pop()
        items += 1
        if items > MAX_JSON_ITEMS or depth > MAX_JSON_DEPTH:
            return False
        if isinstance(current, dict):
            if not all(isinstance(key, str) for key in current):
                return False
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
        elif not isinstance(current, (str, int, float, bool, type(None))):
            return False
    return True


def _write_private(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short sandbox artifact write")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_bounded_file(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("sandbox artifact exceeds configured limit")
        payload = bytearray()
        while len(payload) <= limit:
            chunk = os.read(fd, min(64 * 1024, limit + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > limit:
            raise ValueError("sandbox artifact exceeds configured limit")
        return bytes(payload)
    finally:
        os.close(fd)


def classify_helpers(helpers: Mapping[str, object]) -> tuple[list[str], list[str]]:
    """Split helpers into (async, sync) so the child's shims match the call sites.

    A helper the RLM glue ``await``s (``rlm_query`` etc.) must be an async shim; ``FINAL_VAR``
    and other plain calls must be sync. Coroutine-ness is detected host-side (bound ``async
    def`` methods report correctly).
    """
    async_names: list[str] = []
    sync_names: list[str] = []
    for name, fn in helpers.items():
        if name not in HELPER_NAMES or not callable(fn):
            continue
        (async_names if inspect.iscoroutinefunction(fn) else sync_names).append(name)
    return async_names, sync_names


def write_inputs(
    run_dir: Path,
    code: str,
    *,
    vars_payload: Mapping[str, Any],
    tool_sources: Mapping[str, str],
    helpers: Mapping[str, object],
    bridge_token: str,
    runner_data_dir: str | None = "/data",
) -> None:
    """Write ``context.json`` + ``usercode.py`` (and, for script-based children, ``runner.py``).

    ``runner_data_dir`` is the path the *child* sees the run dir at (``/data`` for a bind-mounted
    container/guest). Pass ``None`` when the child calls :func:`run_child` directly instead of
    running an injected script.
    """
    if not bridge_token or len(bridge_token) > 256:
        raise ValueError("invalid sandbox bridge capability")
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("sandbox run directory is not a private directory")
    os.chmod(run_dir, 0o700)
    async_helpers, sync_helpers = classify_helpers(helpers)
    ctx = {
        "vars": {k: v for k, v in vars_payload.items() if isinstance(v, JSONABLE)},
        "async_helpers": async_helpers,
        "sync_helpers": sync_helpers,
        "bridge_token": bridge_token,
    }
    if not _bounded_json_shape(ctx):
        raise ValueError("sandbox context exceeds structural limits")
    context_payload = json.dumps(ctx, separators=(",", ":")).encode("utf-8")
    code_payload = "\n".join([*tool_sources.values(), code]).encode("utf-8")
    if (
        len(context_payload) > MAX_CONTEXT_BYTES
        or len(code_payload) > MAX_USER_CODE_BYTES
    ):
        raise ValueError("sandbox input exceeds configured limits")
    _write_private(run_dir / "context.json", context_payload)
    _write_private(run_dir / "usercode.py", code_payload)
    if runner_data_dir is not None:
        _write_private(
            run_dir / "runner.py", make_runner_script(runner_data_dir).encode("utf-8")
        )


def read_result(run_dir: Path) -> tuple[str, str | None, bool]:
    """Read the child's ``result.json`` → ``(stdout, error, wrote_result)``.

    ``wrote_result=False`` (missing/corrupt file) means the child died without completing — an
    irreversible failure the caller surfaces as :class:`SandboxFatalError`.
    """
    result_path = run_dir / "result.json"
    if result_path.exists() and not result_path.is_symlink():
        try:
            data = json.loads(
                _read_bounded_file(result_path, MAX_RESULT_BYTES).decode("utf-8")
            )
            if not isinstance(data, dict) or not _bounded_json_shape(data):
                return "", None, False
            stdout = data.get("stdout", "")
            error = data.get("error")
            if not isinstance(stdout, str) or not isinstance(error, (str, type(None))):
                return "", None, False
            return stdout, error, True
        except Exception as exc:  # noqa: BLE001 - corrupt result == failed run
            logger.warning("sandbox result unreadable (%s)", type(exc).__name__)
    return "", None, False


async def start_bridge(
    sock_path: Path,
    helpers: Mapping[str, Callable[..., Any]],
    bridge_token: str,
) -> asyncio.AbstractServer:
    """Host-side UDS server dispatching one framed-JSON request to the matching host helper."""
    if not bridge_token or len(bridge_token) > 256:
        raise ValueError("invalid sandbox bridge capability")
    if sock_path.exists() or sock_path.is_symlink():
        raise ValueError("sandbox bridge path already exists")
    concurrency = asyncio.Semaphore(16)

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        async with concurrency:
            try:
                raw_len = await asyncio.wait_for(
                    reader.readexactly(4), timeout=BRIDGE_IO_TIMEOUT_SECONDS
                )
                n = struct.unpack(">I", raw_len)[0]
                if not 1 <= n <= MAX_BRIDGE_REQUEST_BYTES:
                    raise ValueError("invalid bridge frame length")
                raw_request = await asyncio.wait_for(
                    reader.readexactly(n), timeout=BRIDGE_IO_TIMEOUT_SECONDS
                )
                req = json.loads(raw_request)
                if not isinstance(req, dict) or not _bounded_json_shape(req):
                    raise ValueError("invalid bridge request")
                if not secrets.compare_digest(str(req.get("token", "")), bridge_token):
                    raise PermissionError("invalid bridge capability")
                helper_name = req.get("helper")
                args = req.get("args", [])
                kwargs = req.get("kwargs", {})
                if (
                    not isinstance(helper_name, str)
                    or helper_name not in HELPER_NAMES
                    or not isinstance(args, list)
                    or not isinstance(kwargs, dict)
                    or len(args) > MAX_RPC_ARGS
                    or len(kwargs) > MAX_RPC_ARGS
                ):
                    raise ValueError("invalid bridge invocation")
                fn = helpers.get(helper_name)
                if fn is None or not callable(fn):
                    raise PermissionError("helper is not available")

                async def invoke() -> Any:
                    if inspect.iscoroutinefunction(fn):
                        return await fn(*args, **kwargs)
                    result = await asyncio.to_thread(fn, *args, **kwargs)
                    return await result if inspect.isawaitable(result) else result

                result = await asyncio.wait_for(
                    invoke(), timeout=HELPER_TIMEOUT_SECONDS
                )
                if not _bounded_json_shape(result):
                    raise ValueError("helper result is not bounded JSON")
                resp = {"ok": True, "result": result}
            except Exception as exc:  # noqa: BLE001 - stable error types only
                resp = {"ok": False, "error": f"bridge_error:{type(exc).__name__}"}
            try:
                data = json.dumps(resp, separators=(",", ":")).encode()
                if len(data) > MAX_BRIDGE_RESPONSE_BYTES:
                    data = b'{"ok":false,"error":"bridge_error:ResponseTooLarge"}'
                writer.write(struct.pack(">I", len(data)) + data)
                await asyncio.wait_for(
                    writer.drain(), timeout=BRIDGE_IO_TIMEOUT_SECONDS
                )
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

    server = await asyncio.start_unix_server(handle, path=str(sock_path), backlog=16)
    with contextlib.suppress(OSError):
        os.chmod(sock_path, 0o600)
    return server


def run_child(data_dir: str, sock_path: str) -> None:
    """Importable child runner (for forked children that inherit the package).

    Reads ``{data_dir}/context.json`` + ``usercode.py``, wires helper shims to the bridge at
    ``sock_path``, execs the user code in an async wrapper, writes ``{data_dir}/result.json``.
    Mirrors :func:`make_runner_script` exactly — keep the two in lockstep (one is importable,
    one is a no-import string for containers/guests).
    """
    import asyncio as _asyncio
    import socket
    import sys

    os.umask(0o077)

    class _BoundedOutput:
        def __init__(self, limit: int) -> None:
            self.limit = limit
            self.parts: list[str] = []
            self.size = 0
            self.truncated = False

        def write(self, value: Any) -> int:
            text = str(value)
            encoded = text.encode("utf-8", errors="replace")
            remaining = max(0, self.limit - self.size)
            if remaining:
                accepted = encoded[:remaining].decode("utf-8", errors="ignore")
                self.parts.append(accepted)
                self.size += len(accepted.encode("utf-8"))
            if len(encoded) > remaining:
                self.truncated = True
            return len(text)

        def flush(self) -> None:
            return None

        def getvalue(self) -> str:
            suffix = "\n[output truncated]" if self.truncated else ""
            return "".join(self.parts) + suffix

    def _recvn(s: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("bridge closed")
            buf += chunk
        return buf

    def _bridge_call(name: str, args: tuple, kwargs: dict) -> Any:
        if name not in HELPER_NAMES or not _bounded_json_shape(
            {"args": list(args), "kwargs": kwargs}
        ):
            raise ValueError("invalid bridge invocation")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(BRIDGE_IO_TIMEOUT_SECONDS)
        s.connect(sock_path)
        try:
            payload = json.dumps(
                {
                    "token": bridge_token,
                    "helper": name,
                    "args": list(args),
                    "kwargs": kwargs,
                },
                separators=(",", ":"),
            ).encode()
            if not 1 <= len(payload) <= MAX_BRIDGE_REQUEST_BYTES:
                raise ValueError("bridge request exceeds configured limit")
            s.sendall(struct.pack(">I", len(payload)) + payload)
            n = struct.unpack(">I", _recvn(s, 4))[0]
            if not 1 <= n <= MAX_BRIDGE_RESPONSE_BYTES:
                raise ValueError("bridge response exceeds configured limit")
            resp = json.loads(_recvn(s, n))
        finally:
            s.close()
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "bridge error"))
        return resp["result"]

    def _make_shim(name: str, is_async: bool) -> Callable[..., Any]:
        if is_async:

            async def ashim(*a: Any, **k: Any) -> Any:
                return _bridge_call(name, a, k)

            return ashim

        def shim(*a: Any, **k: Any) -> Any:
            return _bridge_call(name, a, k)

        return shim

    data_path = Path(data_dir)
    ctx = json.loads(
        _read_bounded_file(data_path / "context.json", MAX_CONTEXT_BYTES).decode(
            "utf-8"
        )
    )
    code = _read_bounded_file(data_path / "usercode.py", MAX_USER_CODE_BYTES).decode(
        "utf-8"
    )
    if not isinstance(ctx, dict) or not _bounded_json_shape(ctx):
        raise ValueError("invalid sandbox context")
    bridge_token = str(ctx.get("bridge_token", ""))
    if not bridge_token:
        raise ValueError("sandbox bridge capability missing")
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    ns.update(ctx.get("vars", {}))
    for name in ctx.get("async_helpers", []):
        if name in HELPER_NAMES:
            ns[name] = _make_shim(name, True)
    for name in ctx.get("sync_helpers", []):
        if name in HELPER_NAMES:
            ns[name] = _make_shim(name, False)

    wrapped = "async def __main__():\n"
    for line in code.splitlines():
        wrapped += "    " + line + "\n"
    wrapped += "    return None\n"

    buf = _BoundedOutput(MAX_RESULT_BYTES // 4)
    old = sys.stdout
    sys.stdout = buf
    error: str | None = None
    try:
        exec(wrapped, ns)  # nosec B102 - RLM REPL, restricted namespace
        _asyncio.run(ns["__main__"]())
    except Exception as exc:  # noqa: BLE001 - stable type only
        error = f"execution_error:{type(exc).__name__}"
        buf.write(f"\n{error}")
    finally:
        sys.stdout = old
    _write_private(
        data_path / "result.json",
        json.dumps(
            {"stdout": buf.getvalue(), "error": error}, separators=(",", ":")
        ).encode("utf-8"),
    )


def make_runner_script(data_dir: str = "/data") -> str:
    """Return a self-contained child-runner script (no ``agent_utilities`` import).

    For backends whose child cannot import the package (a container / microVM guest runs
    ``python <runner>``). Mirrors :func:`run_child`; keep both in lockstep.
    """
    return _RUNNER_TEMPLATE.replace("{{DATA_DIR}}", data_dir)


_RUNNER_TEMPLATE = r"""
import asyncio, json, os, socket, struct, sys

DATA = "{{DATA_DIR}}"
SOCK = DATA + "/bridge.sock"
MAX_REQUEST = 1048576
MAX_RESPONSE = 8388608
MAX_CONTEXT = 16777216
MAX_CODE = 2097152
MAX_RESULT = 2097152
SOCKET_TIMEOUT = 30.0
os.umask(0o077)

class _BoundedOutput:
    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.size = 0
        self.truncated = False
    def write(self, value):
        text = str(value)
        encoded = text.encode("utf-8", errors="replace")
        remaining = max(0, self.limit - self.size)
        if remaining:
            accepted = encoded[:remaining].decode("utf-8", errors="ignore")
            self.parts.append(accepted)
            self.size += len(accepted.encode("utf-8"))
        if len(encoded) > remaining:
            self.truncated = True
        return len(text)
    def flush(self):
        pass
    def getvalue(self):
        return "".join(self.parts) + ("\n[output truncated]" if self.truncated else "")

def _read_bounded(path, limit):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        size = os.fstat(fd).st_size
        if size > limit:
            raise ValueError("sandbox artifact too large")
        chunks = []
        total = 0
        while total <= limit:
            chunk = os.read(fd, min(65536, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > limit:
            raise ValueError("sandbox artifact too large")
        return b"".join(chunks)
    finally:
        os.close(fd)

def _write_private(path, payload):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short sandbox artifact write")
            view = view[written:]
    finally:
        os.close(fd)

def _recvn(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge closed")
        buf += chunk
    return buf

def _bridge_call(name, args, kwargs):
    if name not in HELPERS or len(args) > 64 or len(kwargs) > 64:
        raise ValueError("invalid bridge invocation")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(SOCKET_TIMEOUT)
    s.connect(SOCK)
    try:
        payload = json.dumps({"token": TOKEN, "helper": name, "args": list(args), "kwargs": kwargs}, separators=(",", ":")).encode()
        if not 1 <= len(payload) <= MAX_REQUEST:
            raise ValueError("bridge request too large")
        s.sendall(struct.pack(">I", len(payload)) + payload)
        n = struct.unpack(">I", _recvn(s, 4))[0]
        if not 1 <= n <= MAX_RESPONSE:
            raise ValueError("bridge response too large")
        resp = json.loads(_recvn(s, n))
    finally:
        s.close()
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "bridge error"))
    return resp["result"]

def _make_shim(name, is_async):
    if is_async:
        async def shim(*a, **k):
            return _bridge_call(name, a, k)
    else:
        def shim(*a, **k):
            return _bridge_call(name, a, k)
    return shim

def main():
    global TOKEN, HELPERS
    ctx = json.loads(_read_bounded(DATA + "/context.json", MAX_CONTEXT))
    code = _read_bounded(DATA + "/usercode.py", MAX_CODE).decode("utf-8")
    if not isinstance(ctx, dict):
        raise ValueError("invalid sandbox context")
    TOKEN = str(ctx.get("bridge_token", ""))
    if not TOKEN:
        raise ValueError("sandbox bridge capability missing")
    HELPERS = set(ctx.get("async_helpers", [])) | set(ctx.get("sync_helpers", []))

    ns = {"__builtins__": __builtins__}
    ns.update(ctx.get("vars", {}))
    for name in ctx.get("async_helpers", []):
        ns[name] = _make_shim(name, True)
    for name in ctx.get("sync_helpers", []):
        ns[name] = _make_shim(name, False)

    wrapped = "async def __main__():\n"
    for line in code.splitlines():
        wrapped += "    " + line + "\n"
    wrapped += "    return None\n"

    buf = _BoundedOutput(MAX_RESULT // 4)
    old = sys.stdout
    sys.stdout = buf
    error = None
    try:
        exec(wrapped, ns)
        asyncio.run(ns["__main__"]())
    except Exception as e:
        error = "execution_error:" + type(e).__name__
        buf.write("\n" + error)
    finally:
        sys.stdout = old
    payload = json.dumps({"stdout": buf.getvalue(), "error": error}, separators=(",", ":")).encode()
    if len(payload) > MAX_RESULT:
        raise ValueError("sandbox result too large")
    result_path = DATA + "/result.json"
    _write_private(result_path, payload)
    try:
        owner = os.stat(DATA)
        os.chown(result_path, owner.st_uid, owner.st_gid)
    except OSError:
        pass

main()
"""
