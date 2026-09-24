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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

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
                # R-07: portable liveness probe (never a signal on Windows).
                running = psutil.pid_exists(pid)
            except ValueError:
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

    # Lane arbitration, concept-ID reservation and the merge queue moved to
    # repository-manager with the rest of development governance (OQ-3):
    # `repository-manager-governance lane|concept …` and
    # `repository-manager --merge-queue …`.

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

    # CONCEPT:AU-KG.ingest.voice-model-acquisition — GOC-36 governed Piper voice-model
    # acquisition. Operator-driven by design (the lane doc's Authority and invariants:
    # acquisition/license decisions are reviewed actions, not agent-facing capability),
    # so a CLI entry point — not an MCP tool — is this package's live caller.
    vm = sub.add_parser(
        "voice-model",
        help="GOC-36 governed Piper voice-model acquisition, config pairing, and "
        "license-decision recording (quarantine only — no promotion authority)",
    )
    vm_sub = vm.add_subparsers(dest="voice_model_action", required=True)

    def _pinned_source_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--repo-id", required=True, help="Hugging Face 'owner/name' repository"
        )
        parser.add_argument(
            "--revision", required=True, help="exact 40-char pinned commit SHA"
        )
        parser.add_argument("--path", required=True, help="exact path within the repo")
        parser.add_argument(
            "--sha256", required=True, help="expected SHA-256 of the downloaded bytes"
        )
        parser.add_argument(
            "--byte-length",
            type=int,
            default=None,
            help="optional expected byte length (extra fail-closed check)",
        )

    acquire_p = vm_sub.add_parser(
        "acquire", help="fetch+verify+quarantine a pinned Piper .onnx model file"
    )
    _pinned_source_args(acquire_p)

    acquire_cfg_p = vm_sub.add_parser(
        "acquire-config",
        help="fetch+verify+pair-validate a pinned Piper .onnx.json/.json config",
    )
    _pinned_source_args(acquire_cfg_p)
    acquire_cfg_p.add_argument(
        "--model-manifest-id",
        required=True,
        help="manifest id (sha256) of a previously acquired model to pair with",
    )

    license_p = vm_sub.add_parser(
        "license", help="record a license/consent decision for an acquired asset"
    )
    license_p.add_argument("--asset-id", required=True, help="asset manifest id")
    license_p.add_argument("--declared-license", default="", help="e.g. 'MIT'")
    license_p.add_argument("--spdx-id", default="", help="SPDX identifier, if known")
    license_p.add_argument(
        "--gpl-flag",
        action="store_true",
        help="flag as GPL/copyleft — does not change the fail-closed default",
    )
    license_p.add_argument(
        "--counsel-decision",
        default="pending",
        choices=["approved", "blocked", "pending"],
    )
    license_p.add_argument("--reviewer", default="", help="reviewer identity")
    license_p.add_argument("--rationale", default="")

    status_p = vm_sub.add_parser(
        "status", help="whether an asset is ready for EG-registry promotion handoff"
    )
    status_p.add_argument("--asset-id", required=True, help="asset manifest id")

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


def _voice_model(args: argparse.Namespace) -> dict[str, Any]:
    """GOC-36 governed Piper voice-model acquisition CLI dispatch.

    Live caller for :mod:`agent_utilities.protocols.voice_supply_chain` (Wire-First):
    an operator runs ``agent-utilities voice-model <action>`` to acquire/verify a
    pinned asset, record a license decision, or check promotion-handoff readiness.
    Errors from the package's typed exceptions (digest mismatch, unsupported format,
    source-pin conflict) are reported as a JSON envelope, never a raw traceback —
    this command is meant to be scripted.
    """
    import asyncio

    from agent_utilities.protocols.voice_supply_chain import (
        acquisition as voice_acq,
    )
    from agent_utilities.protocols.voice_supply_chain import (
        license_registry as voice_lic,
    )
    from agent_utilities.protocols.voice_supply_chain.manifest import (
        VoiceLicenseDecision,
    )

    action = args.voice_model_action
    try:
        if action in ("acquire", "acquire-config"):
            source = voice_acq.PinnedVoiceSource(
                repo_id=args.repo_id,
                revision=args.revision,
                repo_path=args.path,
                expected_sha256=args.sha256,
                expected_byte_length=args.byte_length,
            )
            if action == "acquire":
                manifest = asyncio.run(voice_acq.acquire_voice_model(source))
                return {"model_manifest": manifest.model_dump(mode="json")}
            model_manifest = voice_acq.get_model_manifest(args.model_manifest_id)
            if model_manifest is None:
                return {
                    "error": f"no quarantined model manifest {args.model_manifest_id!r}"
                }
            config_manifest = asyncio.run(
                voice_acq.acquire_voice_config(source, model_manifest=model_manifest)
            )
            return {"config_manifest": config_manifest.model_dump(mode="json")}
        if action == "license":
            decision = voice_lic.record_license_decision(
                VoiceLicenseDecision(
                    asset_manifest_id=args.asset_id,
                    declared_license=args.declared_license,
                    spdx_id=args.spdx_id or None,
                    is_gpl_or_copyleft_flagged=args.gpl_flag,
                    counsel_decision=args.counsel_decision,
                    reviewer=args.reviewer,
                    rationale=args.rationale,
                )
            )
            return {"license_decision": decision.model_dump(mode="json")}
        if action == "status":
            status_manifest = voice_acq.get_model_manifest(args.asset_id)
            if status_manifest is None:
                return {"error": f"no quarantined manifest {args.asset_id!r}"}
            ready, reason = voice_lic.is_ready_for_promotion_handoff(status_manifest)
            return {
                "asset_id": args.asset_id,
                "ready_for_promotion_handoff": ready,
                "reason": reason,
            }
    except (
        voice_acq.UnsupportedVoiceAssetFormat,
        voice_acq.VoiceAssetDigestMismatch,
        voice_acq.VoiceSourcePinConflict,
        ValueError,
    ) as exc:
        return {"error": str(exc), "error_type": type(exc).__name__}
    return {"error": f"unknown voice-model action {action!r}"}


_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "status": lambda args: status(args.namespace),
    "run": lambda args: run(
        args.namespace, args.agent, args.task, project=args.project
    ),
    "harness-fence": _harness_fence,
    "sleep-run": _sleep_run,
    "install": _install,
    "ingest-sessions": _ingest_sessions,
    "deploy-plan": _deploy_plan,
    "voice-model": _voice_model,
}


def _default_command_output(args: argparse.Namespace) -> dict[str, Any]:
    """start/stop/logs/inspect orchestrate the existing console-scripts; report intent + namespace."""
    return {
        "command": args.command,
        "namespace": args.namespace,
        "components": list(COMPONENTS),
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
    handler = _COMMAND_HANDLERS.get(args.command)
    out = handler(args) if handler is not None else _default_command_output(args)
    print(json.dumps(out, indent=None if args.json else 2))
    # A refusal or a deferral must be actionable by a shell/hook, not just
    # readable — the guard is worthless if `&&` still proceeds after it.
    return int(out.get("exit_code", 0)) if isinstance(out, dict) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
