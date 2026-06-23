"""CONCEPT:ORCH-1.38 — CPython-WASI sandbox: isolated full-stdlib compute, no daemon.

The previous "WASM sandbox" never loaded a ``python.wasm`` — it emulated three hardcoded
tasks. This backend runs a real **CPython-on-WASI** payload under ``wasmtime``: full standard
library, true OS-level isolation (the only filesystem access is a single preopened run dir; no
network, no env, no subprocess), and no container daemon. It sits between monty and Docker —
heavier than monty's in-process VM, far lighter than a container.

**v1 scope:** ``host_callbacks=False``. Wiring the RLM host helpers through WASI (a sync bridge
over imported functions) is deferred, so the router sends only *self-contained* full-Python
compute here — snippets that examine/transform seeded vars and communicate via stdout, not the
helper-driven glue that goes to monty/Docker. Because a WASI run has **no side effects** (no
helpers, no network, scratch-only fs), any infra-level failure (trap, epoch timeout, missing
payload) is a safe :class:`SandboxRejected` — the router simply escalates to Docker.

The payload is large (~25MB) and lives out-of-tree; resolve it via ``$RLM_WASM_PYTHON`` or the
platform cache (see :func:`_resolve_payload`). Absent payload or wasmtime ⇒ ``is_available()``
is False and the router never routes here. Module compilation is cached per backend instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

import platformdirs

from agent_utilities.core.config import setting

from .base import (
    Sandbox,
    SandboxCapabilities,
    SandboxEnv,
    SandboxRejected,
    SandboxResult,
)

logger = logging.getLogger(__name__)

# Only JSON-able namespace values can be seeded into the WASI process (no live refs cross the
# boundary — and v1 has no helper bridge to reach host capabilities anyway).
_JSONABLE = (str, int, float, bool, type(None), list, dict)

# In-WASI runner: seed vars, exec the snippet at module level (no host helpers ⇒ no ``await``
# needed), capture stdout, write the result. Mirrors LocalSandbox's "report, don't raise" on
# in-sandbox errors; plain-local assignments are not synced back (output is via stdout).
_WASM_RUNNER = r"""
import io, json, sys, traceback

def main():
    ctx = json.load(open("/data/context.json"))
    code = open("/data/usercode.py").read()
    ns = {"__builtins__": __builtins__}
    ns.update(ctx.get("vars", {}))
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    error = None
    try:
        exec(code, ns)
    except Exception as e:
        traceback.print_exc(file=buf)
        error = str(e)
    finally:
        sys.stdout = old
    json.dump({"stdout": buf.getvalue(), "error": error}, open("/data/result.json", "w"))

main()
"""


def _resolve_payload() -> Path | None:
    """Locate the CPython-WASI payload: ``$RLM_WASM_PYTHON`` first, then the platform cache.

    A **Wizer-preinitialized** payload (CONCEPT:ORCH-1.88, ``python-warm*.wasm``) is preferred
    when present: Wizer snapshots the CPython-WASI heap *after* the heavy imports run, so
    instantiating it skips the per-run ``import`` cost — the build-time analogue of the
    ``forkserver`` rung's runtime warm-fork, and it works on any platform incl. ARM. Build one
    with ``scripts/build_wasm_warm_payload.sh``. Falls back to a cold ``python-*.wasm``/
    ``python.wasm``. Returns ``None`` (backend unavailable) if nothing is provisioned.
    """
    env = setting("RLM_WASM_PYTHON")
    if env and Path(env).is_file():
        return Path(env)
    cache = Path(platformdirs.user_cache_dir("agent-utilities")) / "rlm-wasm"
    if cache.is_dir():
        # Warm (Wizer-preinitialized) payloads first, then cold payloads.
        ordered = (
            sorted(cache.glob("python-warm*.wasm"))
            + sorted(cache.glob("python-*.wasm"))
            + sorted(cache.glob("python.wasm"))
        )
        for candidate in ordered:
            if candidate.is_file():
                return candidate
    return None


class WasmSandbox(Sandbox):
    """Run a self-contained snippet in CPython-on-WASI under wasmtime (isolated, no daemon)."""

    name = "wasm"
    capabilities = SandboxCapabilities(
        host_callbacks=False,  # v1: no helper bridge — router sends self-contained compute only
        third_party_libs=False,  # only what's baked into the payload (stdlib)
        classes=True,
        full_stdlib=True,
        network=False,
        isolated=True,
        preference_rank=10,  # between monty (0) and docker (20)
    )

    def __init__(
        self,
        *,
        payload: str | os.PathLike[str] | None = None,
        memory_bytes: int = 1 << 30,  # 1 GiB cap on the WASI linear memory
        timeout_secs: float = 30.0,
    ):
        self._payload_override = Path(payload) if payload else None
        self.memory_bytes = memory_bytes
        self.timeout_secs = timeout_secs
        self._available: bool | None = None
        self._engine = None
        self._module = None

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._probe()
        return self._available

    def _probe(self) -> bool:
        try:
            import wasmtime  # noqa: F401
        except Exception:  # noqa: BLE001 - optional dependency
            return False
        return self._payload() is not None

    def _payload(self) -> Path | None:
        if self._payload_override is not None:
            return self._payload_override if self._payload_override.is_file() else None
        return _resolve_payload()

    def _compiled(self):
        """Compile (once) and cache the WASI module + an epoch-interruptible engine."""
        if self._module is None:
            import wasmtime

            cfg = wasmtime.Config()
            # Epoch interruption lets a timer thread trap a runaway snippet (wall-clock timeout).
            try:
                cfg.epoch_interruption = True
            except Exception:  # noqa: BLE001 - best-effort; timeout becomes a no-op if unset
                logger.debug(
                    "wasmtime epoch_interruption unavailable; timeout disabled"
                )
            self._engine = wasmtime.Engine(cfg)
            payload = self._payload()
            if payload is None:  # pragma: no cover - guarded by is_available
                raise SandboxRejected("wasm", "no python.wasm payload")
            logger.debug("compiling CPython-WASI payload %s (one-time)", payload)
            self._module = wasmtime.Module.from_file(self._engine, str(payload))
        return self._module, self._engine

    async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:
        if not self.is_available():
            # Router shouldn't route here; if it did, escalate (a WASI run has no side effects).
            raise SandboxRejected("wasm", "wasmtime or python.wasm payload unavailable")
        module, engine = self._compiled()
        loop = asyncio.get_running_loop()
        # The wasm run is blocking; offload it so the event loop keeps serving (this backend
        # has no host callbacks to service, but we still must not block other tasks).
        return await loop.run_in_executor(
            None, self._run_blocking, module, engine, code, env
        )

    def _run_blocking(
        self, module, engine, code: str, env: SandboxEnv
    ) -> SandboxResult:
        import wasmtime

        with tempfile.TemporaryDirectory(prefix="rlm-wasm-") as d:
            tmp = Path(d)
            vars_payload = {
                k: v for k, v in env.vars.items() if isinstance(v, _JSONABLE)
            }
            (tmp / "context.json").write_text(
                json.dumps({"vars": vars_payload}, default=str)
            )
            (tmp / "usercode.py").write_text(
                "\n".join([*env.tool_sources.values(), code])
            )
            (tmp / "runner.py").write_text(_WASM_RUNNER)

            cfg = wasmtime.WasiConfig()
            cfg.argv = ("python", "/data/runner.py")
            cfg.preopen_dir(
                str(tmp), "/data"
            )  # the ONLY fs access; no env, no network, no stdin
            cfg.stdout_file = str(tmp / "wasm_stdout.txt")
            cfg.stderr_file = str(tmp / "wasm_stderr.txt")

            store = wasmtime.Store(engine)
            store.set_wasi(cfg)
            with contextlib.suppress(Exception):
                store.set_limits(memory_size=self.memory_bytes)
            store.set_epoch_deadline(1)

            # Wall-clock timeout: bump the epoch after the budget to trap a runaway snippet.
            timer = threading.Timer(self.timeout_secs, engine.increment_epoch)
            timer.daemon = True
            timer.start()
            trapped: str | None = None
            try:
                instance = wasmtime.Linker(engine)
                instance.define_wasi()
                inst = instance.instantiate(store, module)
                start = inst.exports(store)["_start"]
                try:
                    start(store)
                except wasmtime.ExitTrap as e:
                    if e.code != 0:
                        trapped = f"wasi exit code {e.code}"
                except wasmtime.Trap as e:
                    # Epoch timeout or a VM-level crash — no result will exist.
                    trapped = str(e).splitlines()[0] if str(e) else "wasm trap"
            finally:
                timer.cancel()

            result_path = tmp / "result.json"
            if result_path.is_file():
                try:
                    data = json.loads(result_path.read_text())
                    return SandboxResult(
                        updated_vars={},
                        stdout=data.get("stdout", ""),
                        error=data.get("error"),
                    )
                except Exception as e:  # noqa: BLE001 - corrupt result == failed run
                    trapped = trapped or f"unreadable result.json: {e}"

            # No result.json ⇒ the runner never finished (trap/timeout). Safe to escalate: a
            # WASI run produced no side effects (no helpers, no network, scratch-only fs).
            raise SandboxRejected("wasm", trapped or "wasm produced no result")
