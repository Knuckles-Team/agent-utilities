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

from .config_generator import PROFILES, config_doctor, config_reference, write_config


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
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
