"""Live proof of B-19's redesign: the stdio JSON-RPC channel stays pure without
any process-wide ``builtins.print``/``warnings.showwarning`` monkeypatch.

Background (see the "Stdio JSON-RPC purity" note in
``agent_utilities/mcp/server_factory.py``): the old ``protect_stdio_jsonrpc()``
permanently replaced ``builtins.print``/``warnings.showwarning`` for the whole
process, with no teardown, and was the confirmed root cause of a cross-file test
pollution incident. It has been deleted. Stdout purity on the stdio transport is
now provided ONLY by the vendored MCP SDK's own ``mcp.server.stdio.stdio_server()``
(entered by ``mcp.run(transport="stdio")`` via FastMCP's ``run_stdio_async``),
which claims fd 1 at the OS level for the scope of serving and restores it on
exit — no Python-object patch, nothing to save/restore.

This test spawns a REAL fastmcp-4 stdio server subprocess (not mocked, not
in-process) whose one tool writes onto stdout through every channel a served
module could reach it by — a bare ``print()``, an explicit
``print(file=sys.stdout)``, a raw ``os.write(1, ...)`` (bypassing Python's
stdout object entirely), a child subprocess's inherited stdout, and a Python
warning — then proves, straight off the wire:

1. Every line the process ever wrote to its real stdout parses as a clean
   JSON-RPC frame (no corruption).
2. All five leak channels landed on stderr instead — including the fd-level
   and subprocess cases a ``builtins.print``/``contextlib.redirect_stdout``
   patch could never have caught, which is the whole reason this design chose
   fd-level ownership over a Python-object-level one.
3. Teardown is real: after the subprocess's ``mcp.run()`` returns (server
   shutdown), a print AFTER the protocol loop exited reaches the process's
   real stdout again — fd 1 was restored, not left diverted.

A real finding from building this proof, worth recording here: the fixture's
``print()`` calls below MUST pass ``flush=True``. Without it, the bytes sit in
Python's userspace ``BufferedWriter`` (piped stdout is block-buffered, not
line-buffered) until something flushes them — and if that flush happens to
land AFTER ``stdio_server()``'s ``finally`` has already restored fd 1 to the
real pipe (e.g. at interpreter shutdown once the client disconnects), the
buffered bytes leak onto real stdout even though the fd was correctly
diverted for the ENTIRE time they were written. This was reproduced directly:
the exact same fixture without ``flush=True`` leaked "LEAK bare-print" onto
real stdout during clean shutdown, after a request/response cycle that itself
stayed perfectly clean. It is precisely why the design keeps the static gate
(``scripts/check_no_stdout_writes.py``) as the PRIMARY defense for the served
package's OWN code, rather than treating the SDK's fd claim as sufficient on
its own — the fd claim reliably catches everything OUTSIDE our control (a
dependency, a C extension, a subprocess), all of which write unbuffered or are
flushed on their own exit well before ours; it is not a substitute for never
writing an unflushed ``print()`` into served code in the first place.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_PROTOCOL_VERSION = "2025-06-18"

_LEAKY_SERVER_SCRIPT = textwrap.dedent(
    """
    import os
    import subprocess
    import sys
    import warnings

    from fastmcp import FastMCP

    mcp = FastMCP("stdio-purity-fixture")


    @mcp.tool()
    def leak() -> str:
        \"\"\"Write onto stdout through every channel the fd-level design must catch.\"\"\"
        # flush=True: piped stdout is block-buffered, not line-buffered -- an
        # unflushed print()'s bytes can sit in the userspace buffer past the
        # point where fd 1 is diverted, and land on the real pipe only once
        # something later flushes them (see the module docstring's finding).
        print("LEAK bare-print", flush=True)
        print("LEAK explicit-stdout-print", file=sys.stdout, flush=True)
        os.write(1, b"LEAK raw-fd-write\\n")
        subprocess.run(
            [sys.executable, "-c", "print('LEAK child-subprocess-stdout')"],
            check=True,
        )
        warnings.warn("LEAK python-warning", UserWarning, stacklevel=1)
        return "leaked"


    if __name__ == "__main__":
        mcp.run(transport="stdio", show_banner=False)
        # Proves teardown: this runs only after mcp.run() returns, i.e. after
        # the SDK's stdio_server() has restored fd 1 — so it must land on the
        # REAL stdout the test harness reads, not stderr.
        print("POST-SHUTDOWN real-stdout-print")
    """
)


def _write(proc: subprocess.Popen, payload: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()


def _read_json_line(proc: subprocess.Popen) -> tuple[str, dict]:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line.strip():
        raise AssertionError(f"expected a JSON-RPC line on stdout, got: {line!r}")
    return line, json.loads(line)


@pytest.mark.timeout(60)
def test_stdio_server_stays_pure_and_captures_every_leak_channel(
    tmp_path: Path,
) -> None:
    server_script = tmp_path / "leaky_stdio_server.py"
    server_script.write_text(_LEAKY_SERVER_SCRIPT, encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(server_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    raw_stdout_lines: list[str] = []
    try:
        # ---- 1. initialize handshake ---------------------------------------
        _write(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "stdio-purity-test", "version": "0"},
                },
            },
        )
        init_line, init_resp = _read_json_line(proc)
        raw_stdout_lines.append(init_line)
        assert init_resp.get("id") == 1, init_resp
        assert "error" not in init_resp, init_resp

        _write(
            proc,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

        # ---- 2. call the leaky tool -----------------------------------------
        _write(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "leak", "arguments": {}},
            },
        )
        call_line, call_resp = _read_json_line(proc)
        raw_stdout_lines.append(call_line)
        assert call_resp.get("id") == 2, call_resp
        assert "error" not in call_resp, call_resp
        result = call_resp["result"]
        assert result.get("isError") in (None, False), result

        # ---- 3. shut the server down cleanly and drain both streams --------
        assert proc.stdin is not None
        proc.stdin.close()
        remaining_stdout, stderr_text = proc.communicate(timeout=30)
        raw_stdout_lines.append(remaining_stdout)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)

    assert proc.returncode == 0, (
        f"leaky stdio server exited {proc.returncode}; stderr:\n{stderr_text}"
    )

    # ---- Assertion A: EVERY non-empty line the process wrote to real stdout
    # over its whole life parses as clean JSON-RPC, or is the one deliberate
    # post-shutdown proof line. Zero tolerance for a stray byte from any leak
    # channel — that is the corruption this design exists to prevent.
    full_stdout = "".join(raw_stdout_lines)
    post_shutdown_marker = "POST-SHUTDOWN real-stdout-print"
    for line in full_stdout.splitlines():
        if not line.strip():
            continue
        if line.strip() == post_shutdown_marker:
            continue
        json.loads(line)  # raises if a leak corrupted the frame stream
        assert "LEAK" not in line, f"a leak channel reached real stdout: {line!r}"

    # ---- Assertion B: teardown proof — the post-mcp.run() print landed on
    # the REAL stdout stream (fd 1 was restored, not left diverted forever).
    assert post_shutdown_marker in full_stdout, (
        f"fd 1 was not restored after mcp.run() returned — expected the "
        f"post-shutdown print on real stdout, got:\n{full_stdout!r}"
    )

    # ---- Assertion C: every leak channel — INCLUDING the fd-level and
    # subprocess ones a builtins.print/contextlib.redirect_stdout patch could
    # never see — landed on stderr instead. This is the breadth argument for
    # choosing fd-level ownership over a Python-object-level patch.
    for expected in (
        "LEAK bare-print",
        "LEAK explicit-stdout-print",
        "LEAK raw-fd-write",
        "LEAK child-subprocess-stdout",
        "LEAK python-warning",
    ):
        assert expected in stderr_text, (
            f"expected {expected!r} on stderr, not found. stderr was:\n{stderr_text}"
        )
