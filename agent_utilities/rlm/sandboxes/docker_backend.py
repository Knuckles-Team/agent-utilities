"""CONCEPT:ORCH-1.38 — Docker/Podman sandbox: full-isolation escalation tier with a host bridge.

This is where code that monty can't run (third-party libs, classes, full stdlib) goes. The
previous ``_execute_container`` had three gaps this backend closes:

* **No resource limits / network isolation** → now ``--network none``, ``--memory``,
  ``--pids-limit``, ``--cpus``, ``--cap-drop ALL``, ``--security-opt no-new-privileges``, and
  a wall-clock timeout (kill on overrun).
* **JSON-string-only context** → now the whole JSON-able REPL namespace + tool sources are
  shipped in, and seeded as globals.
* **Could not serve the RLM host helpers** → now a per-run **UDS bridge**: the host runs an
  asyncio Unix-socket server bound to a socket *inside the bind-mounted run dir*, so a
  ``--network none`` container still reaches ``rlm_query`` etc. over the filesystem socket.
  Async helpers are awaited host-side and the result returned; ``FINAL_VAR`` round-trips and
  mutates the host ``vars`` directly. The container's only egress is that one socket.

The container script (:data:`_RUNNER_SCRIPT`) mirrors :class:`~.local_backend.LocalSandbox`
semantics: tool sources + user code are wrapped in an ``async def`` (so ``await`` works),
stdout is captured, and the helper shims (async for the async host helpers, sync otherwise)
match how the RLM glue calls them. The result (stdout + any in-sandbox error) comes back via a
``result.json`` in the run dir. A container that dies without writing it — daemon gone, OOM
kill, timeout — is an irreversible failure and raises :class:`SandboxFatalError` (ORCH-1.29).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import shutil
import struct
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

from ..telemetry import SandboxFatalError
from .base import Sandbox, SandboxCapabilities, SandboxEnv, SandboxResult

logger = logging.getLogger(__name__)

# JSON is the bridge + context wire format; only JSON-able namespace values cross into the
# container (live refs can't be serialized — that's what the helper bridge is for).
_JSONABLE = (str, int, float, bool, type(None), list, dict)

# The fixed in-container runner. Reads /data/context.json + /data/usercode.py, wires helper
# shims to the UDS bridge, execs the user code in an async wrapper, writes /data/result.json.
_RUNNER_SCRIPT = r"""
import asyncio, io, json, socket, struct, sys, traceback

SOCK = "/data/bridge.sock"

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
    ctx = json.load(open("/data/context.json"))
    code = open("/data/usercode.py").read()

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
    json.dump({"stdout": buf.getvalue(), "error": error}, open("/data/result.json", "w"))

main()
"""


class DockerSandbox(Sandbox):
    """Run a snippet in a resource-limited, network-isolated container with a host-helper bridge."""

    name = "docker"
    capabilities = SandboxCapabilities(
        host_callbacks=True,  # via the UDS bridge
        third_party_libs=True,
        classes=True,
        full_stdlib=True,
        network=False,  # --network none
        isolated=True,
        preference_rank=20,  # heavy: tried only when cheaper tiers can't
    )

    def __init__(
        self,
        image: str = "python:3.11-slim",
        *,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 256,
        timeout_secs: float = 120.0,
    ):
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout_secs = timeout_secs
        self._runtime: str | None | bool = None  # False = probed-unavailable

    def is_available(self) -> bool:
        return self._resolve_runtime() is not None

    def _resolve_runtime(self) -> str | None:
        """Find a working docker/podman CLI + daemon (cached). ``None`` if neither is usable."""
        if self._runtime is None:
            self._runtime = False
            for rt in ("docker", "podman"):
                if shutil.which(rt) is None:
                    continue
                # `<rt> info` returns 0 only with a reachable daemon.
                try:
                    ok = (
                        subprocess.run(
                            [rt, "info"], capture_output=True, timeout=10
                        ).returncode
                        == 0
                    )
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    self._runtime = rt
                    break
        return self._runtime if isinstance(self._runtime, str) else None

    async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:
        runtime = self._resolve_runtime()
        if runtime is None:
            # Router shouldn't route here if unavailable; if it did, the infra is gone.
            raise SandboxFatalError("docker/podman runtime unavailable")

        run_id = uuid.uuid4().hex[:12]
        tmpdir = Path(tempfile.mkdtemp(prefix=f"rlm-docker-{run_id}-"))
        sock_path = tmpdir / "bridge.sock"
        server: asyncio.AbstractServer | None = None
        try:
            self._write_inputs(tmpdir, code, env)
            server = await self._start_bridge(sock_path, env)
            # The container process may run under a different uid (rootless/userns-remapped
            # docker), so it needs traverse on the dir AND connect (write) on the socket file.
            os.chmod(tmpdir, 0o777)  # nosec B103 — throwaway sandbox dir; container uid differs
            with contextlib.suppress(FileNotFoundError):
                os.chmod(sock_path, 0o777)  # nosec B103 — bridge socket must accept the container uid

            stdout, error, wrote_result = await self._run_container(
                runtime, tmpdir, run_id
            )
            if not wrote_result:
                # No result.json => the container never completed its runner (OOM/timeout/daemon)
                raise SandboxFatalError(
                    f"container produced no result (likely OOM/timeout/daemon death): {stdout[:200]}"
                )
            return SandboxResult(updated_vars={}, stdout=stdout, error=error)
        except SandboxFatalError:
            raise
        except Exception as e:  # noqa: BLE001
            # Any other failure once we've committed to the container path (bridge/subprocess
            # error, resource exhaustion under load) is an irreversible infra failure, not a
            # code rejection — surface it as fatal (ORCH-1.29), never as a wrong-typed escape.
            raise SandboxFatalError(f"docker sandbox failed: {e}") from e
        finally:
            if server is not None:
                server.close()
                with contextlib.suppress(Exception):
                    await server.wait_closed()
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── inputs ───────────────────────────────────────────────────────────────
    def _write_inputs(self, tmpdir: Path, code: str, env: SandboxEnv) -> None:
        async_helpers, sync_helpers = self._classify_helpers(env.helpers)
        vars_payload = {k: v for k, v in env.vars.items() if isinstance(v, _JSONABLE)}
        ctx = {
            "vars": vars_payload,
            "async_helpers": async_helpers,
            "sync_helpers": sync_helpers,
        }
        (tmpdir / "context.json").write_text(json.dumps(ctx, default=str))
        usercode = "\n".join([*env.tool_sources.values(), code])
        (tmpdir / "usercode.py").write_text(usercode)
        (tmpdir / "runner.py").write_text(_RUNNER_SCRIPT)

    @staticmethod
    def _classify_helpers(
        helpers: Mapping[str, object],
    ) -> tuple[list[str], list[str]]:
        """Split helpers into (async, sync) so the in-container shims match the call sites.

        A helper the RLM glue ``await``s (``rlm_query`` etc.) must be an async shim; ``FINAL_VAR``
        and other plain calls must be sync. We detect coroutine functions on the host — bound
        ``async def`` methods report correctly.
        """
        async_names, sync_names = [], []
        for name, fn in helpers.items():
            if inspect.iscoroutinefunction(fn):
                async_names.append(name)
            else:
                sync_names.append(name)
        return async_names, sync_names

    # ── bridge ───────────────────────────────────────────────────────────────
    async def _start_bridge(
        self, sock_path: Path, env: SandboxEnv
    ) -> asyncio.AbstractServer:
        """UDS server dispatching one framed JSON request to the matching host helper."""

        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                n = struct.unpack(">I", await reader.readexactly(4))[0]
                req = json.loads(await reader.readexactly(n))
                fn = env.helpers.get(req["helper"])
                if fn is None:
                    resp = {"ok": False, "error": f"unknown helper {req['helper']!r}"}
                else:
                    result = fn(*req.get("args", []), **req.get("kwargs", {}))
                    if inspect.isawaitable(result):
                        result = await result
                    resp = {"ok": True, "result": result}
            except Exception as e:  # noqa: BLE001 - report bridge/helper errors to the container
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

    # ── container ──────────────────────────────────────────────────────────────
    async def _run_container(
        self, runtime: str, tmpdir: Path, run_id: str
    ) -> tuple[str, str | None, bool]:
        """Run the container with hard isolation + a timeout; return (stdout, error, wrote_result)."""
        name = f"rlm-{run_id}"
        argv = [
            runtime,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            self.memory,
            "--pids-limit",
            str(self.pids_limit),
            "--cpus",
            self.cpus,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{tmpdir}:/data:rw",
            self.image,
            "python",
            "/data/runner.py",
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            raw, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout_secs
            )
        except TimeoutError:
            await self._kill_container(runtime, name)
            with contextlib.suppress(Exception):
                await proc.wait()
            container_log = ""
        else:
            container_log = (raw or b"").decode(errors="replace")

        result_path = tmpdir / "result.json"
        if result_path.exists():
            try:
                data = json.loads(result_path.read_text())
                return data.get("stdout", ""), data.get("error"), True
            except Exception as e:  # noqa: BLE001 - corrupt result == failed run
                logger.warning("docker result.json unreadable: %s", e)
        return container_log, None, False

    @staticmethod
    async def _kill_container(runtime: str, name: str) -> None:
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                runtime,
                "kill",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
