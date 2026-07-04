#!/usr/bin/python
"""``setup-config`` console entry — generate / validate / document the full config.

Wraps :mod:`agent_utilities.deployment.config_generator` so a deployment (or Claude
setting itself up) can produce a COMPLETE profile-seeded ``config.json``, validate a
config's completeness/health, or dump the grouped option reference — matching the
``graph_configure`` MCP actions and the ``agent-utilities-deployment`` skill.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config_generator import (
    PROFILES,
    config_doctor,
    config_reference,
    generate_mcp_config,
    write_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="setup-config",
        description="Generate, validate, and document the full agent-utilities config.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Write a complete config.json for a profile.")
    g.add_argument("--profile", choices=list(PROFILES), default="tiny")
    g.add_argument(
        "--out", default=None, help="Output path (default: XDG config.json)."
    )
    g.add_argument(
        "--with-secrets",
        action="store_true",
        help="Do NOT blank secret-like keys (use with care).",
    )

    d = sub.add_parser("doctor", help="Validate config completeness/health.")
    d.add_argument("--profile", choices=list(PROFILES), default=None)
    d.add_argument(
        "--config", default=None, help="config.json to check (default: live)."
    )

    sub.add_parser("reference", help="Print every option grouped by subsystem (JSON).")

    m = sub.add_parser(
        "mcp",
        help="Print the minimal mcp_config.json (graph-os + mcp-multiplexer) to register.",
    )
    m.add_argument("--profile", choices=list(PROFILES), default="tiny")
    m.add_argument(
        "--fleet",
        dest="fleet",
        action="store_true",
        default=True,
        help="Include mcp-multiplexer (the whole fleet). Default.",
    )
    m.add_argument(
        "--no-fleet",
        dest="fleet",
        action="store_false",
        help="Emit only graph-os (a single KG, no multiplexer).",
    )

    hf = sub.add_parser(
        "harness-fence",
        help="Write a governance-derived Claude Code permission fence (CONCEPT:AU-OS.deployment.governance-derived-claude-code).",
    )
    hf.add_argument(
        "--target",
        default=None,
        help="Claude config dir (default: ~/.claude). Writes settings.json + ../.claudeignore.",
    )
    hf.add_argument(
        "--policy", default=None, help="ActionPolicy YAML (default: shipped policy)."
    )
    hf.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the fence that would be written without touching disk.",
    )

    args = parser.parse_args(argv)

    if args.command == "generate":
        res = write_config(args.profile, args.out, redact_secrets=not args.with_secrets)
        print(json.dumps(res, indent=2))
        return 0
    if args.command == "doctor":
        res = config_doctor(args.profile, args.config)
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("healthy") else 1
    if args.command == "reference":
        print(json.dumps(config_reference(), indent=2, default=str))
        return 0
    if args.command == "mcp":
        print(json.dumps(generate_mcp_config(args.profile, fleet=args.fleet), indent=2))
        return 0
    if args.command == "harness-fence":
        from pathlib import Path

        from agent_utilities.claude_harness.claude_fence import write_fence
        from agent_utilities.orchestration.action_policy import ActionPolicy

        target = args.target or str(Path.home() / ".claude")
        policy = (
            ActionPolicy(policy_path=args.policy) if args.policy else ActionPolicy()
        )
        res = write_fence(target, policy, dry_run=args.dry_run)
        print(json.dumps(res, indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
