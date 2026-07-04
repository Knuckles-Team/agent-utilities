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

    # CONCEPT:OS-5.77 — the unified install path. `install` materializes every provider
    # contribution (skills + prompts + ontologies, incl. the hub's OWN) into the ONE XDG
    # data tree the runtime reads from, then (unless --no-toolkit) also installs the AU
    # skill toolkit into the calling agent tool(s) — the CONCEPT:OS-5.52 behavior.
    # `install-skills` is kept as a backward-compatible alias.
    def _add_install_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--tool",
            default=None,
            help="target one tool (e.g. claude, agent-utilities)",
        )
        parser.add_argument(
            "--path", default=None, help="explicit skills dir to install into"
        )
        parser.add_argument(
            "--layer",
            choices=["all", "atomic", "workflows"],
            default="all",
            help="which layer to install (default: all)",
        )
        parser.add_argument(
            "--skills", default="", help="comma-separated skill names (default: all)"
        )
        parser.add_argument(
            "--group",
            default=None,
            help="install only skills in this category/path part",
        )
        parser.add_argument(
            "--no-graphs",
            action="store_true",
            help="skip skill-graphs (the agent-utilities skill-graph is installed by default)",
        )
        parser.add_argument(
            "--force", action="store_true", help="overwrite existing skills"
        )
        parser.add_argument(
            "--symlink",
            action="store_true",
            help="symlink instead of copy (auto-updates)",
        )
        parser.add_argument(
            "--no-toolkit",
            action="store_true",
            help="only materialize the unified XDG tree; skip installing the skill "
            "toolkit into agent tools",
        )

    _add_install_args(
        sub.add_parser(
            "install",
            help="materialize all provider skills+prompts+ontologies into the unified "
            "XDG tree (+ the skill toolkit into agent tools)",
        )
    )
    _add_install_args(
        sub.add_parser(
            "install-skills",
            help="alias of `install` (backward-compatible)",
        )
    )

    # CONCEPT:ECO-4.42 — client-side chat/session ingestion for Claude + Antigravity
    # (and every other detected agent). `--upload` parses THIS host's local logs and
    # pushes them to a REMOTE engine via the graph-os `ingest_sessions` upload action
    # (the remote-engine path); default `collect` sinks into a local engine.
    ig = sub.add_parser(
        "ingest-sessions",
        help="parse local agent chat logs (claude/antigravity/...) and ingest them",
    )
    ig.add_argument(
        "--upload",
        action="store_true",
        help="push to a REMOTE engine via MCP (use when the engine is on another host)",
    )
    ig.add_argument(
        "--server", default="graph-os", help="remote MCP server name (mcp_config.json)"
    )
    ig.add_argument(
        "--url", default="", help="explicit remote MCP url (overrides --server)"
    )
    ig.add_argument("--tenant", default="", help="tenant scope for the rows")
    ig.add_argument(
        "--all", action="store_true", help="re-parse every file (default: changed only)"
    )

    # CONCEPT:OS-5.42 — atomic concept-ID reservation (offline/worktree entry point).
    cp = sub.add_parser("concept", help="reserve/list/release/reconcile concept ids")
    cp.add_argument(
        "concept_action",
        choices=["reserve", "release", "list", "reconcile", "resolve"],
    )
    cp.add_argument(
        "--ns", default="", help="pillar (e.g. KG-2) or package prefix (e.g. KEY)"
    )
    cp.add_argument(
        "--session", default="", help="claiming session id (default host:pid)"
    )
    cp.add_argument("--design-doc", default="", help="design-doc path to record")
    cp.add_argument(
        "--id", dest="concept_id", default="", help="concept id for release"
    )
    cp.add_argument(
        "--status", default="", help="filter for list (reserved/landed/expired)"
    )
    cp.add_argument("--ttl", type=int, default=86_400, help="reservation TTL seconds")
    cp.add_argument("--repo", default="", help="repo root (default agent-utilities)")
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


def _install(args: argparse.Namespace) -> dict[str, Any]:
    """Unified install (CONCEPT:OS-5.77) — materialize the XDG tree + the skill toolkit.

    1. Materialize every provider contribution (skills + prompts + ontologies, incl. the
       hub's OWN) into the one XDG data tree the runtime reads from
       (:func:`agent_utilities.core.unified_install.install_unified`, overwrite-on-reinstall).
    2. Unless ``--no-toolkit``, also install the AU skill toolkit into the detected agent
       tool(s) — the backward-compatible CONCEPT:OS-5.52 behavior.
    """
    from agent_utilities.core.unified_install import install_unified

    out: dict[str, Any] = {"unified_tree": install_unified(force=True)}
    if not getattr(args, "no_toolkit", False):
        out["skill_toolkit"] = _install_skills(args)
    return out


def _install_skills(args: argparse.Namespace) -> dict[str, Any]:
    """Install the agent-utilities skill toolkit into agent tool(s) (CONCEPT:OS-5.52).

    Thin delegate to the universal-skills installer (the single source of truth for
    skill discovery/placement). With no ``--tool``/``--path`` it installs into every
    detected agent tool AND the agent-utilities XDG home (so AU agents auto-load them);
    skill-graphs — including the ``agent-utilities`` platform graph — are included by
    default because that graph is what unlocks how to use everything else.
    """
    try:
        from universal_skills.core.skill_installer.scripts import install as inst
    except ImportError:
        return {
            "error": "universal-skills is not installed",
            "fix": "pip install universal-skills  (or: pip install 'agent-utilities[agent]')",
        }

    skill_names = [s for s in args.skills.split(",") if s] or None
    include_graphs = not args.no_graphs

    targets: dict[str, Path] = {}
    if args.path:
        targets["custom"] = Path(args.path).expanduser()
    elif args.tool:
        target = inst.TOOL_PATHS.get(args.tool.lower())
        if target is None:
            return {
                "error": f"unknown tool {args.tool!r}",
                "known_tools": sorted(inst.TOOL_PATHS),
            }
        targets[args.tool.lower()] = target
    else:
        targets = dict(inst.detect_present_tools())
        # Always include the agent-utilities XDG home (factory auto-load target).
        targets.setdefault("agent-utilities", inst.TOOL_PATHS["agent-utilities"])

    installed: dict[str, str] = {}
    seen: set[str] = set()
    for tool, target in targets.items():
        if str(target) in seen:
            continue
        seen.add(str(target))
        inst.install_skills(
            target,
            skill_names,
            args.group,
            args.force,
            include_graphs,
            symlink=args.symlink,
            layer=args.layer,
        )
        installed[tool] = str(target)
    return {"installed": installed, "layer": args.layer, "skill_graphs": include_graphs}


def _ingest_sessions(args: argparse.Namespace) -> dict[str, Any]:
    """Parse local agent chat logs and ingest them (CONCEPT:ECO-4.42).

    ``--upload`` parses THIS host's logs and pushes them to a remote engine over MCP
    (the remote-engine path — Claude + Antigravity + every other detected agent);
    otherwise it sinks into a local engine.
    """
    if args.upload:
        from agent_utilities.ingestion.collector import upload_local_sessions

        return upload_local_sessions(
            server=args.server,
            url=args.url,
            tenant_id=args.tenant,
            only_changed=not args.all,
        )
    from agent_utilities.ingestion.collector import collect_local_sessions

    return collect_local_sessions(only_changed=not args.all)


def _concept(args: argparse.Namespace) -> dict[str, Any]:
    """Concept-ID reservation — runs against the file ledger directly (no gateway)."""
    import socket

    from agent_utilities.governance import concept_allocator as ca

    repo_root = Path(args.repo).expanduser().resolve() if args.repo else ca.REPO_ROOT
    action = args.concept_action
    if action == "list":
        return {
            "reservations": ca.list_reservations(
                repo_root=repo_root, status=args.status or None
            )
        }
    if action == "reconcile":
        return ca.reconcile(repo_root=repo_root)
    if action == "resolve":
        # CONCEPT:OS-5.76 — canonicalize a flat/dotted id → its 3-level form + aliases.
        from agent_utilities.governance import concept_hierarchy as ch

        if not args.concept_id:
            return {"error": "resolve requires --id (a flat or dotted concept id)"}
        try:
            parsed = ch.parse_concept_id(args.concept_id)
        except ValueError as exc:
            return {"error": str(exc)}
        return {
            "raw": parsed.raw,
            "canonical": parsed.canonical,
            "namespace": parsed.namespace,
            "pillar": parsed.pillar,
            "concept": parsed.concept,
            "segment": parsed.segment,
            "is_project": parsed.is_project,
            "aliases": list(parsed.aliases),
            "flags": list(parsed.flags),
        }
    if action == "release":
        if not args.concept_id:
            return {"error": "release requires --id"}
        return {"released": ca.release_concept_id(args.concept_id, repo_root=repo_root)}
    # reserve
    if not args.ns:
        return {"error": "reserve requires --ns (e.g. KG-2 or KEY)"}
    sid = args.session or f"{socket.gethostname()}:{os.getpid()}"
    return ca.reserve_concept_id(
        args.ns,
        session_id=sid,
        design_doc=args.design_doc or None,
        ttl_seconds=int(args.ttl),
        repo_root=repo_root,
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
    elif args.command in ("install", "install-skills"):
        out = _install(args)
    elif args.command == "ingest-sessions":
        out = _ingest_sessions(args)
    elif args.command == "concept":
        out = _concept(args)
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
