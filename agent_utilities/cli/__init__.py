"""CONCEPT:OS-5.11 (+ OS-5.1/5.2 extension) — Unified dev-lifecycle CLI.

Assimilated from open-design's ``tools-dev``: one entry point with ``start/stop/status/logs/inspect/run``
subcommands, ``--namespace`` isolation (all state under ``$TMPDIR/agent-utilities/<namespace>/``), and
``--json`` for CI. The ``run`` subcommand mints a run-scoped tool token (OS-5.11) and injects it into
the run environment — the daemon as sole policy authority.

The lifecycle ops orchestrate the existing console-scripts (``graph-os-daemon``, ``graph-os``,
``mcp-multiplexer``); this module owns the namespace model + token minting (the testable core) and a
thin dispatcher.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.security.run_token import mint_token

COMPONENTS = ("daemon", "mcp", "gateway")


def runtime_dir(namespace: str) -> Path:
    """Namespaced runtime root (isolates parallel stacks; mirrors open-design's ``.tmp/<namespace>``)."""
    base = setting("AGENT_UTILITIES_RUNTIME_DIR") or os.path.join(
        tempfile.gettempdir(), "agent-utilities"
    )
    return Path(base) / namespace


def status(namespace: str) -> dict[str, Any]:
    """Report per-component lifecycle status for a namespace (pid-file based)."""
    root = runtime_dir(namespace)
    components: dict[str, Any] = {}
    for comp in COMPONENTS:
        pid_file = root / f"{comp}.pid"
        running = False
        pid = None
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)  # signal 0 = liveness probe
                running = True
            except (ValueError, OSError):
                running = False
        components[comp] = {"running": running, "pid": pid}
    return {"namespace": namespace, "runtime_dir": str(root), "components": components}


def run(namespace: str, agent: str, task: str, *, project: str = "") -> dict[str, Any]:
    """Mint a run-scoped token for a run and return the dispatch descriptor (OS-5.11)."""
    runtime_dir(namespace).mkdir(parents=True, exist_ok=True, mode=0o700)
    run_id = f"run:{namespace}:{agent}"
    token = mint_token(
        run_id,
        project=project or namespace,
        endpoints=("/api/proxy/*", "/api/artifacts/*", "/api/runs/*"),
        operations=("read", "write"),
        ttl_seconds=3600.0,
    )
    return {"run_id": run_id, "agent": agent, "task": task, "tool_token": token}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-utilities", description="agent-utilities dev lifecycle CLI"
    )
    p.add_argument("--namespace", default="default", help="isolated stack namespace")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("start", "stop", "status", "logs", "inspect"):
        sub.add_parser(cmd)
    run_p = sub.add_parser("run")
    run_p.add_argument("agent")
    run_p.add_argument("task")
    run_p.add_argument("--project", default="")

    # ── Claude Code harness (claude_harness package) ──
    # CONCEPT:OS-5.41 — the PreToolUse dynamic gate body (reads the event on stdin).
    sub.add_parser("harness-gate")
    # CONCEPT:OS-5.40 — write the governance-derived permission fence.
    hf = sub.add_parser("harness-fence")
    hf.add_argument(
        "--target", default=None, help="Claude config dir (default ~/.claude)."
    )
    hf.add_argument("--policy", default=None, help="ActionPolicy YAML override.")
    hf.add_argument("--dry-run", action="store_true")
    # CONCEPT:ECO-4.47 — drive the Loop engine unattended + write a morning summary.
    sr = sub.add_parser("sleep-run")
    sr.add_argument("--max-cycles", type=int, default=6)
    sr.add_argument("--max-topics", type=int, default=5)
    sr.add_argument("--workspace", default=None)
    sr.add_argument("--no-commit", action="store_true")
    return p


def _harness_gate() -> int:
    """PreToolUse gate body — read the event on stdin, print the verdict JSON."""
    from agent_utilities.claude_harness.pretooluse_gate import run as gate_run

    print(json.dumps(gate_run()))
    return 0


def _harness_fence(args: argparse.Namespace) -> dict[str, Any]:
    from agent_utilities.claude_harness.claude_fence import write_fence
    from agent_utilities.orchestration.action_policy import ActionPolicy

    target = args.target or str(Path.home() / ".claude")
    policy = ActionPolicy(policy_path=args.policy) if args.policy else ActionPolicy()
    return write_fence(target, policy, dry_run=args.dry_run)


def _sleep_run(args: argparse.Namespace) -> dict[str, Any]:
    from agent_utilities.claude_harness.overnight_runner import run_session

    return run_session(
        max_cycles=args.max_cycles,
        max_topics=args.max_topics,
        commit=not args.no_commit,
        workspace=args.workspace,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "harness-gate":
        # Prints ONLY the verdict JSON (Claude Code reads stdout); bypass the
        # generic envelope below.
        return _harness_gate()
    if args.command == "status":
        out = status(args.namespace)
    elif args.command == "run":
        out = run(args.namespace, args.agent, args.task, project=args.project)
    elif args.command == "harness-fence":
        out = _harness_fence(args)
    elif args.command == "sleep-run":
        out = _sleep_run(args)
    else:
        # start/stop/logs/inspect orchestrate the existing console-scripts; report intent + namespace.
        out = {
            "command": args.command,
            "namespace": args.namespace,
            "components": list(COMPONENTS),
        }
    print(json.dumps(out, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
