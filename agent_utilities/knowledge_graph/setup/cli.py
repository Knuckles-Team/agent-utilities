#!/usr/bin/python
"""``setup-databases`` console entry — provision Stardog + pg-age from the shell.

A thin argparse wrapper over
:func:`agent_utilities.knowledge_graph.setup.database_environment.setup_environment`
(and ``--verify`` over :func:`verify_postgres`) so the laptop/CI path matches the
``graph_configure`` MCP action and the ``database-environment-setup`` skill.
"""

from __future__ import annotations

import argparse
import json
import sys

from .database_environment import setup_environment, verify_postgres


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="setup-databases",
        description="Provision the agent-utilities database environment "
        "(Stardog + pg-age) from credentials.",
    )
    parser.add_argument(
        "--profile",
        choices=["dev", "prod"],
        default="dev",
        help="dev = local SPARQL (built-in /api/sparql); prod = push to Stardog.",
    )
    parser.add_argument(
        "--postgres-mode",
        choices=["managed_image", "existing"],
        default="managed_image",
        help="managed_image = combined pg-age-full image; existing = connect-only.",
    )
    parser.add_argument("--dsn", default=None, help="Postgres DSN (else GRAPH_DB_URI).")
    parser.add_argument(
        "--sparql-target",
        choices=["builtin", "fuseki", "stardog"],
        default=None,
        help="Override the SPARQL host (defaults from --profile).",
    )
    parser.add_argument(
        "--mirror",
        action="append",
        default=None,
        metavar="CONNECTION",
        help="Fanout mirror connection name (repeatable; KG-2.74).",
    )
    parser.add_argument(
        "--no-backfill", action="store_true", help="Skip the durable backfill step."
    )
    parser.add_argument(
        "--no-mirror-data",
        action="store_true",
        help="Don't register Stardog as a live data mirror (publish only the "
        "ontology). Default: mirror instance data for the Stardog target.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only probe Postgres for age/vector/pg_search and exit.",
    )
    args = parser.parse_args(argv)

    if args.verify:
        result = verify_postgres(args.dsn)
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") == "success" else 1

    report = setup_environment(
        profile=args.profile,
        postgres_mode=args.postgres_mode,
        dsn=args.dsn,
        sparql_target=args.sparql_target,
        mirror_targets=args.mirror,
        do_backfill=not args.no_backfill,
        mirror_data_to_stardog=False if args.no_mirror_data else None,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("status") == "success" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
