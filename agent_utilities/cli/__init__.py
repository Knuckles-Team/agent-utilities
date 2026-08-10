"""CONCEPT:AU-OS.observability.run-wide-correlation-id (+ OS-5.1/5.2 extension) — Unified dev-lifecycle CLI.

Assimilated from open-design's ``tools-dev``: one entry point with ``start/stop/status/logs/inspect/run``
subcommands, ``--namespace`` isolation (all state under ``$TMPDIR/agent-utilities/<namespace>/``), and
``--json`` for CI. The ``run`` subcommand mints a run-scoped tool token (OS-5.11) and injects it into
the run environment — the daemon as sole policy authority.

The lifecycle ops orchestrate the existing console scripts (`graph-os-daemon`
and `graph-os`); this module owns the namespace model + token minting (the
testable core) and a thin dispatcher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    # CONCEPT:AU-OS.deployment.dynamic-two-fail-closed — the PreToolUse dynamic gate body (reads the event on stdin).
    sub.add_parser("harness-gate")
    # CONCEPT:AU-OS.deployment.governance-derived-claude-code — write the governance-derived permission fence.
    hf = sub.add_parser("harness-fence")
    hf.add_argument(
        "--target", default=None, help="Claude config dir (default ~/.claude)."
    )
    hf.add_argument("--policy", default=None, help="ActionPolicy YAML override.")
    hf.add_argument("--dry-run", action="store_true")
    # CONCEPT:AU-AHE.harness.overnight-loop-driver — drive the Loop engine unattended + write a morning summary.
    sr = sub.add_parser("sleep-run")
    sr.add_argument("--max-cycles", type=int, default=6)
    sr.add_argument("--max-topics", type=int, default=5)
    sr.add_argument("--workspace", default=None)
    sr.add_argument("--no-commit", action="store_true")

    # CONCEPT:AU-OS.governance.concept-2 — the unified install path. `install` materializes every provider
    # contribution (skills + prompts + ontologies, incl. the hub's OWN) into the ONE XDG
    # data tree the runtime reads from, then (unless --no-toolkit) also installs the AU
    # skill toolkit into the calling agent tool(s) — the CONCEPT:AU-OS.deployment.agent-factory-autoload behavior.
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
    # CONCEPT:AU-ECO.mcp.client-side-chat-session — client-side chat/session ingestion for Claude + Antigravity
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

    # CONCEPT:AU-OS.governance.atomic-concept-id-reservation — atomic concept-ID reservation (offline/worktree entry point).
    cp = sub.add_parser("concept", help="reserve/list/release/reconcile concept ids")
    cp.add_argument(
        "concept_action",
        choices=["reserve", "release", "list", "reconcile", "resolve"],
    )
    cp.add_argument(
        "--session", default="", help="claiming session id (default host:pid)"
    )
    cp.add_argument("--design-doc", default="", help="design-doc path to record")
    cp.add_argument(
        "--id",
        dest="concept_id",
        default="",
        help="canonical OKF-CIS concept id for reserve, release, or resolve",
    )
    cp.add_argument(
        "--status", default="", help="filter for list (reserved/landed/expired)"
    )
    cp.add_argument("--ttl", type=int, default=86_400, help="reservation TTL seconds")
    cp.add_argument("--repo", default="", help="repo root (default agent-utilities)")

    # CONCEPT:AU-OS.governance.lane-arbitration-classes — concurrent-lane arbitration entry point.
    # Nested subparsers (not one positional choice) so `lease` can take a
    # REMAINDER command without swallowing the other actions' own flags.
    lp = sub.add_parser(
        "lane", help="concurrent-lane arbitration: isolation, leases, guards"
    )
    lane_sub = lp.add_subparsers(dest="lane_action", required=True)

    def _lane_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        parser = lane_sub.add_parser(name, help=help_text)
        parser.add_argument("--path", default="", help="working tree (default: cwd)")
        return parser

    _lane_parser("status", "this lane's isolation, partitions, and live leases")
    _lane_parser("env", "shell exports that give this lane its own build/test state")
    _lane_parser(
        "classify", "the shared-resource -> arbitration-class table"
    ).add_argument(
        "--resource", default="", help="one resource instead of the whole table"
    )
    guard_p = _lane_parser("guard", "refuse a mutation that would destroy other work")
    guard_p.add_argument("--operation", default="edit", help="what is being attempted")
    guard_p.add_argument(
        "--reset", default="", help="path a global actor wants to reset or discard"
    )
    guard_p.add_argument(
        "--owner", default="", help="the lane that owns the reset target"
    )
    guard_p.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="`-- <command>` to run under the guard (lease held across the mutation)",
    )
    _lane_parser(
        "park", "clean this tree for a moment WITHOUT touching the shared refs/stash"
    )
    _lane_parser("unpark", "restore what `lane park` set aside")
    bind_cargo_p = _lane_parser(
        "bind-cargo",
        "write .cargo/config.toml so cargo PARTITION binds with no export needed",
    )
    bind_cargo_p.add_argument(
        "--force",
        action="store_true",
        help="append the partition block even if .cargo/config.toml already exists",
    )
    lease_p = _lane_parser("lease", "hold a LEASE-class resource, or report its holder")
    lease_p.add_argument("--resource", default="", help="LEASE-class resource name")
    lease_p.add_argument("--operation", default="", help="why the lease is being taken")
    lease_p.add_argument(
        "--ttl", dest="lease_ttl", type=int, default=1_800, help="lease TTL seconds"
    )
    lease_p.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="`-- <command>` to run while holding the lease (omit to just report)",
    )

    # CONCEPT:AU-OS.governance.serialized-merge-queue — continuous merge into main.
    # A sibling of `lane` and deliberately the same shape: this is the lane
    # *arbitration* surface for the one operation lanes cannot do independently.
    mq = sub.add_parser(
        "merge-queue", help="continuous merge into main through one serialized queue"
    )
    mq_sub = mq.add_subparsers(dest="mq_action", required=True)

    def _mq_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        parser = mq_sub.add_parser(name, help=help_text)
        parser.add_argument("--path", default="", help="working tree (default: cwd)")
        return parser

    enqueue_p = _mq_parser("enqueue", "offer this lane's branch for landing on main")
    enqueue_p.add_argument("--branch", default="", help="branch (default: this HEAD)")
    enqueue_p.add_argument("--base", default="main", help="branch to land onto")
    _mq_parser("status", "queue depth, order, recent outcomes, and the latency budget")
    withdraw_p = _mq_parser("withdraw", "pull a candidate back out of the queue")
    withdraw_p.add_argument("--branch", required=True, help="candidate to withdraw")
    withdraw_p.add_argument("--reason", default="", help="why it was withdrawn")
    run_p = _mq_parser("run", "drain a batch under the reconciliation-merge lease")
    run_p.add_argument("--base", default="main", help="branch to land onto")
    run_p.add_argument(
        "--batch-size", type=int, default=0, help="candidates per gate (0 = default)"
    )
    run_p.add_argument(
        "--no-prune", action="store_true", help="land but keep worktrees and branches"
    )
    _mq_parser("promotion", "how far the deployed ref lags main (merge != deploy)")

    # Self-composing graph-os entrypoint (`uvx agent-utilities graph-os`): runs the
    # SAME ``graph-os`` MCP server as the standalone console script, plus whatever
    # co-services the loaded AgentConfig says are configured (messaging, the KG
    # host daemon if this run would otherwise be unhosted) — see
    # ``agent_utilities.mcp.co_service_supervisor``. All graph-os flags
    # (--transport/--host/--port/...) pass straight through; graph-os parses them
    # itself, so nothing is declared here beyond a REMAINDER capture.
    gp = sub.add_parser(
        "graph-os",
        help="run graph-os (MCP server) + its configured co-services in one process",
    )
    gp.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="passthrough flags for graph-os, e.g. --transport stdio",
    )

    # Multi-backend deployment planner (CONCEPT: project the same self-composing
    # entrypoint onto in_process/container/kubernetes/native_shell) — see
    # ``agent_utilities.deployment.backends``. Plan-only for every backend except
    # in_process; never mutates a remote host/cluster from this command.
    dp = sub.add_parser(
        "deploy-plan",
        help="render a DeploymentPlan for graph-os on one backend (plan-only "
        "except in_process; never applies to a remote host/cluster)",
    )
    dp.add_argument(
        "--backend",
        required=True,
        choices=["in_process", "container", "kubernetes", "native_shell"],
    )
    dp.add_argument(
        "--target",
        default="this process",
        help="host alias / cluster / namespace this plan targets",
    )
    dp.add_argument(
        "--param",
        action="append",
        default=[],
        help="KEY=VALUE backend-specific override (repeatable), e.g. "
        "--param image=ghcr.io/org/agent-utilities:1.2.3 --param namespace=graphos",
    )
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
    """Unified install (CONCEPT:AU-OS.governance.concept-2) — materialize the XDG tree + the skill toolkit.

    1. Materialize every provider contribution (skills + prompts + ontologies, incl. the
       hub's OWN) into the one XDG data tree the runtime reads from
       (:func:`agent_utilities.core.unified_install.install_unified`, transactional
       content-addressed generations).
    2. Unless ``--no-toolkit``, also install the AU skill toolkit into the detected agent
       tool(s) — the CONCEPT:AU-OS.deployment.agent-factory-autoload behavior.
    """
    from agent_utilities.core.unified_install import install_unified

    out: dict[str, Any] = {"unified_tree": install_unified()}
    if not getattr(args, "no_toolkit", False):
        out["skill_toolkit"] = _install_skills(args)
    return out


def _install_skills(args: argparse.Namespace) -> dict[str, Any]:
    """Install the agent-utilities skill toolkit into agent tool(s) (CONCEPT:AU-OS.deployment.agent-factory-autoload).

    Thin delegate to the universal-skills installer (the single source of truth for
    skill discovery/placement). With no ``--tool``/``--path`` it installs into every
    detected external agent tool. Agent Utilities reads the provider-owned XDG
    generation written by :func:`install_unified`; it is never duplicated as a flat
    operator skill. Skill graphs are included by default.
    """
    try:
        from universal_skills.core import skill_installer as inst
    except ImportError:
        return {
            "error": "universal-skills is not installed",
            "fix": "pip install universal-skills  (or: pip install 'agent-utilities[agent-runtime]')",
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
        targets.pop("agent-utilities", None)

    installed: list[str] = []
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
        installed.append(tool)
    return {
        "installed_tools": sorted(installed),
        "installed_count": len(installed),
        "layer": args.layer,
        "skill_graphs": include_graphs,
        "path_free": True,
    }


def _ingest_sessions(args: argparse.Namespace) -> dict[str, Any]:
    """Parse local agent chat logs and ingest them (CONCEPT:AU-ECO.mcp.client-side-chat-session).

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
    """Same-host compatibility concept reservation against the file ledger.

    Separate-host callers must use graph-os' native concept authority; this
    legacy CLI path is intentionally not advertised as globally atomic.
    """
    import uuid

    from agent_utilities.governance import concept_allocator as ca

    repo_root = Path(args.repo).expanduser().resolve() if args.repo else None
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
        # CONCEPT:AU-OS.governance.concept-id-canonicalization — validate and project a canonical OKF-CIS id.
        from agent_utilities.governance import concept_hierarchy as ch

        if not args.concept_id:
            return {"error": "resolve requires --id"}
        try:
            parsed = ch.parse_okf_id(args.concept_id)
        except ValueError as exc:
            return {"error": str(exc)}
        return {
            "raw": parsed.raw,
            "canonical": parsed.canonical,
            "slug": parsed.slug,
            "pillar": parsed.pillar,
            "domain": parsed.domain,
            "concept": parsed.concept,
            "facets": list(parsed.facets),
            "path": parsed.path,
            "iri": parsed.iri,
        }
    if action == "release":
        if not args.concept_id:
            return {"error": "release requires --id"}
        return {"released": ca.release_concept_id(args.concept_id, repo_root=repo_root)}
    # reserve
    if not args.concept_id:
        return {"error": "reserve requires --id"}
    sid = args.session or f"session-{uuid.uuid4().hex}"
    return ca.reserve_concept_id(
        args.concept_id,
        session_id=sid,
        design_doc=args.design_doc or None,
        ttl_seconds=int(args.ttl),
        repo_root=repo_root,
    )


def _lane(args: argparse.Namespace) -> dict[str, Any]:
    """Lane arbitration — the operator/agent surface over ``governance.lanes``.

    Every action here exists so the *safe* path is the convenient one: you get
    your isolated paths from ``env``, you run a contended operation through
    ``lease``, and ``guard`` refuses the mutation that would eat someone's work.
    """
    import subprocess

    from agent_utilities.governance import lanes

    path = args.path or None
    action = args.lane_action
    if action == "status":
        return lanes.lane_report(path)
    if action == "env":
        parts = lanes.partitioned_paths(path)
        orphaned = lanes.orphaned_precommit_patches(path)
        return {
            "exports": {
                "CARGO_TARGET_DIR": str(parts.cargo_target_dir),
                "PYTEST_ADDOPTS": f"--basetemp={parts.pytest_basetemp}",
                "TMPDIR": str(parts.scratch_dir),
                "PRE_COMMIT_HOME": str(parts.precommit_home),
            },
            "stash_ref": parts.stash_ref,
            "note": (
                "never `git stash` — refs/stash is one ref shared by every "
                "worktree. To READ a pristine file while yours is dirty use "
                "`git show HEAD:<path>` (mutates nothing). To PARK work use a "
                f"scratch commit on your branch, or `lane park` -> {parts.stash_ref}"
            ),
            "precommit_home_note": (
                "PRE_COMMIT_HOME is also per-lane: a shared pre-commit store "
                "means a killed/OOMed/power-lost pre-commit orphans another "
                "lane's uncommitted work as an unreplayed patch file (D-OB-12), "
                "and the store's shared SQLite db.db raises `OperationalError: "
                "database is locked` under concurrent lanes. See D-ORC-37."
            ),
            "orphaned_precommit_patches": [
                p for p in orphaned if p["state"] in ("ORPHANED", "unknown")
            ],
        }
    if action == "park":
        return lanes.park_worktree(path)
    if action == "unpark":
        return lanes.unpark_worktree(path)
    if action == "bind-cargo":
        try:
            return lanes.write_cargo_partition_config(path, force=args.force)
        except lanes.LaneArbitrationError as exc:
            return {"written": False, "refused": str(exc), "exit_code": 1}
    if action == "classify":
        rules = lanes.resource_rules()
        if args.resource:
            return {
                "resource": args.resource,
                "class": lanes.resource_class(args.resource).value,
            }
        return {
            "resources": [
                {
                    "name": r.name,
                    "class": r.arbitration.value,
                    "mechanism": r.mechanism,
                    "evidence": r.evidence,
                }
                for r in rules
            ]
        }
    if action == "guard":
        command = [a for a in getattr(args, "command_args", []) if a != "--"]
        try:
            if args.reset:
                owner = args.owner or "unknown"
                if not command:
                    lanes.require_resettable_tree(
                        args.reset, operation=args.operation, owner=owner
                    )
                    return {"allowed": True, "target": args.reset}
                # The mutation itself runs INSIDE the guard, so the tree cannot go
                # dirty between the check and the command — the whole point of the
                # single choke point.
                with lanes.guarded_tree_mutation(
                    args.reset, operation=args.operation, owner=owner
                ) as scope:
                    completed = subprocess.run(command, check=False)  # noqa: S603
                return {
                    "allowed": True,
                    "target": str(scope.tree),
                    "command": command,
                    "exit_code": completed.returncode,
                }
            scope = lanes.require_mutable_tree(path, operation=args.operation)
            return {"allowed": True, "lane": scope.lane, "tree": str(scope.tree)}
        except lanes.LaneArbitrationError as exc:
            return {"allowed": False, "refused": str(exc), "exit_code": 1}
    # lease
    if not args.resource:
        return {"exit_code": 2, "error": "lease requires --resource"}
    if lanes.resource_class(args.resource) is not lanes.ArbitrationClass.LEASE:
        return {
            "exit_code": 2,
            "error": f"{args.resource} is not a LEASE-class resource",
        }
    command = [a for a in getattr(args, "command_args", []) if a != "--"]
    if not command:
        return {
            "resource": args.resource,
            "holder": lanes.lease_status(args.resource, path),
        }
    try:
        with lanes.hold_lease(
            args.resource,
            operation=args.operation,
            ttl_seconds=args.lease_ttl,
            path=path,
        ) as held:
            completed = subprocess.run(command, check=False)  # noqa: S603
        return {
            "resource": args.resource,
            "held_by": held["lane"],
            "command": command,
            "exit_code": completed.returncode,
        }
    except lanes.LeaseUnavailable as exc:
        return {"deferred": True, "holder": exc.holder, "exit_code": 75}


def _merge_queue(args: argparse.Namespace) -> dict[str, Any]:
    """The merge queue — the one path a lane's work takes into ``main``.

    Same contract as ``lane lease``: an unavailable lease is **exit 75**, so a
    shell or hook that chains with ``&&`` stops instead of proceeding. A rejected
    candidate is exit 1 with the failing checks, because a rejection is a result
    the lane must act on, not an error in the queue.
    """
    from agent_utilities.governance import lanes, merge_queue

    path = args.path or None
    action = args.mq_action
    try:
        if action == "enqueue":
            return merge_queue.enqueue(args.branch, base=args.base, path=path)
        if action == "status":
            return merge_queue.queue_report(path)
        if action == "withdraw":
            return merge_queue.withdraw(args.branch, reason=args.reason, path=path)
        if action == "promotion":
            return merge_queue.promotion_state(path)
        result = merge_queue.run_queue(
            base=args.base,
            batch_size=args.batch_size or merge_queue.DEFAULT_BATCH_SIZE,
            prune=not args.no_prune,
            path=path,
        )
        if result.get("rejected"):
            result["exit_code"] = 1
        return result
    except lanes.LeaseUnavailable as exc:
        return {"deferred": True, "holder": exc.holder, "exit_code": 75}
    except lanes.LaneArbitrationError as exc:
        return {"refused": str(exc), "exit_code": 1}


def _deploy_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Render a :class:`DeploymentPlan` for the chosen backend and print it.

    Delegates entirely to :mod:`agent_utilities.deployment.backends` — see that
    module's docstring for exactly which backends are live-capable
    (``in_process`` only) vs. plan-only (``container``/``kubernetes``/
    ``native_shell``, which this command never applies).
    """
    from agent_utilities.deployment.backends import get_backend

    overrides: dict[str, str] = {}
    for kv in args.param:
        if "=" in kv:
            key, _, value = kv.partition("=")
            overrides[key] = value

    backend = get_backend(args.backend)
    plan = backend.plan(target=args.target, **overrides)
    return {
        "backend": plan.backend,
        "target": plan.target,
        "live_capable": plan.live_capable,
        "composition": list(plan.composition.co_service_names()),
        "steps": [
            {
                "description": step.description,
                "fleet_call": (
                    {
                        "server": step.fleet_call.server,
                        "tool": step.fleet_call.tool,
                        "args": step.fleet_call.args,
                    }
                    if step.fleet_call is not None
                    else None
                ),
                "local_action": step.local_action,
            }
            for step in plan.steps
        ],
        "warnings": list(plan.warnings),
        "artifacts": dict(plan.artifacts),
    }


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if raw_argv[:1] == ["graph-os"]:
        # graph-os owns an entirely separate flag universe
        # (--transport/--host/--port/... via ``create_mcp_parser``) and OWNS
        # stdout for its whole lifetime under the stdio transport (it IS the
        # JSON-RPC channel) — dispatch directly from the raw argv, bypassing
        # this module's argparse (whose subparsers can't losslessly forward
        # arbitrary flags — https://bugs.python.org/issue17050) and the generic
        # JSON envelope below. graph-os re-parses ``sys.argv`` itself.
        from agent_utilities.mcp.kg_server import mcp_server

        mcp_server()
        return 0
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
    elif args.command == "install":
        out = _install(args)
    elif args.command == "ingest-sessions":
        out = _ingest_sessions(args)
    elif args.command == "concept":
        out = _concept(args)
    elif args.command == "lane":
        out = _lane(args)
    elif args.command == "merge-queue":
        out = _merge_queue(args)
    elif args.command == "deploy-plan":
        out = _deploy_plan(args)
    else:
        # start/stop/logs/inspect orchestrate the existing console-scripts; report intent + namespace.
        out = {
            "command": args.command,
            "namespace": args.namespace,
            "components": list(COMPONENTS),
        }
    print(json.dumps(out, indent=None if args.json else 2))
    # A refusal or a deferral must be actionable by a shell/hook, not just
    # readable — the guard is worthless if `&&` still proceeds after it.
    return int(out.get("exit_code", 0)) if isinstance(out, dict) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
