#!/usr/bin/env python3
"""Dev harness: exercise `graph_orchestrate execute_agent` against a fleet server.

Reusable diagnostic for the authenticated spawned-agent path (ORCH-1.21 /
OS-5.32) — e.g. verifying a jwt-protected `*.arpa` server is reachable by a
spawned agent, or reproducing an execute_agent issue with a captured stderr log.

It launches a throwaway local `graph-os` (SSE, `KG_SERVED_PROFILE=0` so it accepts
local calls), drives one `execute_agent`, prints the result, and tails the
server's stderr. It carries **no credential-store path of its own**: the OIDC
service-account creds are loaded from ``/tmp/oidc.env`` (KEY=VALUE lines) or the
ambient environment. Populate ``/tmp/oidc.env`` first from the proper secret
store (OpenBao, or the session's MCP-server env) — see AGENTS.md → "Secrets &
credential retrieval". Credential access is human-gated by design; this harness
deliberately does not read any credential store itself.

Usage:
    python scripts/dev_execute_agent.py <agent_name> "<task>" [--port 8197]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import time

_OIDC_ENV = "/tmp/oidc.env"  # nosec B108 — dev-only cred handoff, caller-populated
_REQUIRED = (
    "MCP_CLIENT_AUTH",
    "OIDC_CLIENT_ID",
    "OIDC_CLIENT_SECRET",
    "OIDC_TOKEN_URL",
)
_VENV_PY = "/home/apps/workspace/.venv/bin/python"


def _load_creds() -> None:
    if os.path.exists(_OIDC_ENV):
        for line in open(_OIDC_ENV):
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "agent_name", help="KG Server/agent name (e.g. repository-manager-mcp)"
    )
    ap.add_argument("task", help="task string for the spawned agent")
    ap.add_argument("--port", type=int, default=8197)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--boot", type=int, default=32, help="seconds to wait for graph-os")
    args = ap.parse_args()

    _load_creds()
    missing = [k for k in _REQUIRED if not os.environ.get(k)]
    if missing:
        print(
            f"Missing required env vars {missing}. Populate {_OIDC_ENV} from the "
            "proper store first (see AGENTS 'Secrets & credential retrieval').",
            file=sys.stderr,
        )
        return 2

    graph_os = os.path.join(os.path.dirname(_VENV_PY), "graph-os")
    errlog = f"/tmp/dev_execute_agent_{args.port}.err"  # nosec B108
    env = os.environ.copy()
    env.update(
        {
            "KG_SERVED_PROFILE": "0",
            "GRAPH_PERSISTENCE_PATH": f"/tmp/dev_execute_agent_{args.port}_state",  # nosec B108
            "GRAPH_SERVICE_SOCKET": os.environ.get(
                "GRAPH_SERVICE_SOCKET",
                "/tmp/epistemic-graph.sock",  # nosec B108
            ),
            "GRAPH_BACKEND": os.environ.get("GRAPH_BACKEND", "fanout"),
        }
    )

    proc = subprocess.Popen(
        [
            graph_os,
            "--transport",
            "sse",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
        env=env,
        stdout=open(errlog, "wb"),
        stderr=subprocess.STDOUT,
    )
    print(
        f"graph-os pid {proc.pid} on :{args.port} (booting ~{args.boot}s) ...",
        flush=True,
    )
    time.sleep(args.boot)

    async def go() -> None:
        from fastmcp import Client

        async with Client(f"http://127.0.0.1:{args.port}/sse") as c:
            r = await c.call_tool(
                "graph_orchestrate",
                {
                    "action": "execute_agent",
                    "agent_name": args.agent_name,
                    "task": args.task,
                    "max_steps": args.max_steps,
                },
            )
            print("RESULT:", str(r)[:600], flush=True)

    try:
        asyncio.run(go())
    except Exception as e:  # noqa: BLE001
        print("CALL EXC:", type(e).__name__, str(e)[:400], flush=True)
    finally:
        proc.terminate()

    # Surface any crash/traceback from the server, masking token-ish strings.
    try:
        raw = open(errlog, encoding="utf-8", errors="replace").read()
    except OSError:
        raw = ""
    hits = [
        ln
        for ln in raw.splitlines()
        if re.search(
            r"traceback|error|exception|fatal|segmentation|anyio|taskgroup|cancel|killed",
            ln,
            re.I,
        )
    ]
    if hits:
        masked = re.sub(
            r"(Bearer|access_token|secret|token)\S*",
            r"\1 ***",
            "\n".join(hits[-40:]),
            flags=re.I,
        )
        print("===== server stderr (masked) =====", flush=True)
        print(masked, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
