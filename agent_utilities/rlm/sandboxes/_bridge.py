"""CONCEPT:AU-ORCH.sandbox.shared-host-helper-bridge — Shared host-helper bridge for isolated sandbox backends.

An isolated backend (a ``--network none`` container, a forked process, a microVM guest) runs
the snippet in a *separate* address space, but the RLM host helpers (``rlm_query``,
``graph_query``, … and ``FINAL_VAR``) are async methods bound to the orchestrator process. The
bridge lets the isolated child reach them: the host runs a Unix-socket server bound to a socket
*inside the shared run dir*; the child connects over the filesystem socket and issues framed
JSON RPCs. Async helpers are awaited host-side; ``FINAL_VAR`` round-trips and mutates the host
``vars`` directly. The child's only egress is that one socket.

This was originally inline in ``docker_backend.py``; it is factored here so every isolated
warm-fork rung (``forkserver``, ``container_fork``, ``firecracker``) serves host callbacks the
*same* way. Two child-side forms share one wire format:

* :func:`make_runner_script` — a self-contained script string for backends whose child cannot
  import ``agent_utilities`` (a container / microVM guest runs ``python <runner>``);
* :func:`run_child` — the importable equivalent for a child that *can* import the package (a
  ``forkserver`` fork inherits the parent's loaded modules), so it needs no injected script.

Wire format (both directions): ``struct.pack(">I", len) + json_bytes``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Only JSON-able namespace values cross the boundary; live refs can't be serialized — reaching
# them is exactly what the helper bridge is for.
JSONABLE = (str, int, float, bool, type(None), list, dict)


def classify_helpers(helpers: Mapping[str, object]) -> tuple[list[str], list[str]]:
    """Split helpers into (async, sync) so the child's shims match the call sites.

    A helper the RLM glue ``await``s (``rlm_query`` etc.) must be an async shim; ``FINAL_VAR``
    and other plain calls must be sync. Coroutine-ness is detected host-side (bound ``async
    def`` methods report correctly).
    """
    async_names: list[str] = []
    sync_names: list[str] = []
    for name, fn in helpers.items():
        (async_names if inspect.iscoroutinefunction(fn) else sync_names).append(name)
    return async_names, sync_names


def write_inputs(
    run_dir: Path,
    code: str,
    *,
    vars_payload: Mapping[str, Any],
    tool_sources: Mapping[str, str],
    helpers: Mapping[str, object],
    runner_data_dir: str | None = "/data",
) -> None:
    """Write ``context.json`` + ``usercode.py`` (and, for script-based children, ``runner.py``).

    ``runner_data_dir`` is the path the *child* sees the run dir at (``/data`` for a bind-mounted
    container/guest). Pass ``None`` to skip emitting ``runner.py`` — used by the ``forkserver``
    rung, whose child calls :func:`run_child` directly instead of running an injected script.
    """
    async_helpers, sync_helpers = classify_helpers(helpers)
    ctx = {
        "vars": {k: v for k, v in vars_payload.items() if isinstance(v, JSONABLE)},
        "async_helpers": async_helpers,
        "sync_helpers": sync_helpers,
    }
    (run_dir / "context.json").write_text(json.dumps(ctx, default=str))
    (run_dir / "usercode.py").write_text("\n".join([*tool_sources.values(), code]))
    if runner_data_dir is not None:
        (run_dir / "runner.py").write_text(make_runner_script(runner_data_dir))


def read_result(run_dir: Path) -> tuple[str, str | None, bool]:
    """Read the child's ``result.json`` → ``(stdout, error, wrote_result)``.

    ``wrote_result=False`` (missing/corrupt file) means the child died without completing — an
    irreversible failure the caller surfaces as :class:`SandboxFatalError`.
    """
    result_path = run_dir / "result.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            return data.get("stdout", ""), data.get("error"), True
        except Exception as e:  # noqa: BLE001 - corrupt result == failed run
            logger.warning("sandbox result.json unreadable: %s", e)
    return "", None, False


async def start_bridge(
    sock_path: Path, helpers: Mapping[str, Callable[..., Any]]
) -> asyncio.AbstractServer:
    """Host-side UDS server dispatching one framed-JSON request to the matching host helper."""

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            n = struct.unpack(">I", await reader.readexactly(4))[0]
            req = json.loads(await reader.readexactly(n))
            fn = helpers.get(req["helper"])
            if fn is None:
                resp = {"ok": False, "error": f"unknown helper {req['helper']!r}"}
            else:
                result = fn(*req.get("args", []), **req.get("kwargs", {}))
                if inspect.isawaitable(result):
                    result = await result
                resp = {"ok": True, "result": result}
        except Exception as e:  # noqa: BLE001 - report bridge/helper errors to the child
            resp = {"ok": False, "error": str(e)}
        try:
            data = json.dumps(resp, default=str).encode()
            writer.write(struct.pack(">I", len(data)) + data)
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    return await asyncio.start_unix_server(handle, path=str(sock_path))


def run_child(data_dir: str, sock_path: str) -> None:
    """Importable child runner (for forked children that inherit the package).

    Reads ``{data_dir}/context.json`` + ``usercode.py``, wires helper shims to the bridge at
    ``sock_path``, execs the user code in an async wrapper, writes ``{data_dir}/result.json``.
    Mirrors :func:`make_runner_script` exactly — keep the two in lockstep (one is importable,
    one is a no-import string for containers/guests).
    """
    import asyncio as _asyncio
    import io
    import socket
    import sys
    import traceback

    def _recvn(s: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("bridge closed")
            buf += chunk
        return buf

    def _bridge_call(name: str, args: tuple, kwargs: dict) -> Any:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
        try:
            payload = json.dumps(
                {"helper": name, "args": list(args), "kwargs": kwargs}
            ).encode()
            s.sendall(struct.pack(">I", len(payload)) + payload)
            n = struct.unpack(">I", _recvn(s, 4))[0]
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

    ctx = json.loads((Path(data_dir) / "context.json").read_text())
    code = (Path(data_dir) / "usercode.py").read_text()
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    ns.update(ctx.get("vars", {}))
    for name in ctx.get("async_helpers", []):
        ns[name] = _make_shim(name, True)
    for name in ctx.get("sync_helpers", []):
        ns[name] = _make_shim(name, False)

    wrapped = "async def __main__():\n"
    for line in code.splitlines():
        wrapped += "    " + line + "\n"
    wrapped += "    return None\n"

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    error: str | None = None
    try:
        exec(wrapped, ns)  # nosec B102 - RLM REPL, restricted namespace
        _asyncio.run(ns["__main__"]())
    except Exception as e:  # noqa: BLE001 - surface to the model, keep the loop alive
        traceback.print_exc(file=buf)
        error = str(e)
    finally:
        sys.stdout = old
    (Path(data_dir) / "result.json").write_text(
        json.dumps({"stdout": buf.getvalue(), "error": error})
    )


def make_runner_script(data_dir: str = "/data") -> str:
    """Return a self-contained child-runner script (no ``agent_utilities`` import).

    For backends whose child cannot import the package (a container / microVM guest runs
    ``python <runner>``). Mirrors :func:`run_child`; keep both in lockstep.
    """
    return _RUNNER_TEMPLATE.replace("{{DATA_DIR}}", data_dir)


_RUNNER_TEMPLATE = r"""
import asyncio, io, json, socket, struct, sys, traceback

DATA = "{{DATA_DIR}}"
SOCK = DATA + "/bridge.sock"

def _recvn(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge closed")
        buf += chunk
    return buf

def _bridge_call(name, args, kwargs):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK)
    try:
        payload = json.dumps({"helper": name, "args": list(args), "kwargs": kwargs}).encode()
        s.sendall(struct.pack(">I", len(payload)) + payload)
        n = struct.unpack(">I", _recvn(s, 4))[0]
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
    ctx = json.load(open(DATA + "/context.json"))
    code = open(DATA + "/usercode.py").read()

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

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    error = None
    try:
        exec(wrapped, ns)
        asyncio.run(ns["__main__"]())
    except Exception as e:
        traceback.print_exc(file=buf)
        error = str(e)
    finally:
        sys.stdout = old
    json.dump({"stdout": buf.getvalue(), "error": error}, open(DATA + "/result.json", "w"))

main()
"""
