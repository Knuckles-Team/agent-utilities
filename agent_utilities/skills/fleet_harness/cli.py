"""Console-script entry point: `agent-utilities-validate-skill-fleet`.

    agent-utilities-validate-skill-fleet \
        --root /path/to/agent-utilities --root /path/to/epistemic-graph \
        [--root /path/to/more-repo ...] \
        [--functional] [--json-out PATH] [--table-out PATH]

Static layer always runs (no services required). Pass ``--functional`` to
also run the live layer against a reachable graph-os MCP endpoint (see
`functional_checks.py` for how the endpoint is resolved — it degrades to
SKIPPED-unreachable rather than hanging or failing the whole run).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agent_utilities.skills.fleet_harness.discovery import discover_skills
from agent_utilities.skills.fleet_harness.report import (
    build_report,
    write_json,
    write_table,
)
from agent_utilities.skills.fleet_harness.static_checks import run_static_checks

_DEFAULT_JSON_OUT = Path("reports/skill-validation-harness.json")
_DEFAULT_TABLE_OUT = Path("reports/skill-validation-harness.md")


def _default_roots() -> list[Path]:
    """Best-effort au + eg discovery for a standard ``agent-packages/`` checkout.

    Walks up from this file to the ``agent-utilities`` repo root, then looks
    for a sibling ``epistemic-graph`` checkout. Either miss is reported, not
    silently dropped — the caller should pass ``--root`` explicitly when the
    layout doesn't match (e.g. a lone worktree with no sibling checkouts).
    """
    here = Path(__file__).resolve()
    au_root: Path | None = None
    for candidate in here.parents:
        if (
            candidate / "pyproject.toml"
        ).is_file() and candidate.name == "agent-utilities":
            au_root = candidate
            break
    roots: list[Path] = []
    if au_root is not None:
        roots.append(au_root)
        eg_candidate = au_root.parent / "epistemic-graph"
        if eg_candidate.is_dir():
            roots.append(eg_candidate)
        else:
            print(
                f"note: no sibling epistemic-graph checkout at {eg_candidate} — "
                "pass --root explicitly to include it",
                file=sys.stderr,
            )
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        metavar="PATH",
        help="a repo root to scan for SKILL.md (repeatable). Default: au + sibling eg, if found.",
    )
    parser.add_argument(
        "--functional",
        action="store_true",
        help="also run the functional layer against a reachable graph-os MCP endpoint",
    )
    parser.add_argument("--json-out", type=Path, default=_DEFAULT_JSON_OUT)
    parser.add_argument("--table-out", type=Path, default=_DEFAULT_TABLE_OUT)
    parser.add_argument(
        "--fail-on-static",
        action="store_true",
        help="exit non-zero when any skill fails the static layer (CI gate mode)",
    )
    args = parser.parse_args(argv)

    roots = [Path(r) for r in args.roots] if args.roots else _default_roots()
    if not roots:
        print("error: no roots to scan — pass --root explicitly", file=sys.stderr)
        return 2

    records = discover_skills(roots)
    if not records:
        print(
            f"error: no SKILL.md discovered under {[str(r) for r in roots]}",
            file=sys.stderr,
        )
        return 2

    static_reports = run_static_checks(records)

    functional_results = None
    if args.functional:
        from agent_utilities.skills.fleet_harness.functional_checks import (
            run_functional_checks,
        )

        functional_results = asyncio.run(run_functional_checks(records))

    report = build_report(static_reports, functional_results)
    write_json(report, args.json_out)
    write_table(report, args.table_out)

    summary = report["summary"]
    print(
        f"Discovered {summary['total_skills']} skills across {len(roots)} repo(s): "
        f"{summary['static_pass']} PASS / {summary['static_fail']} FAIL (static layer)."
    )
    if summary["functional_tally"]:
        print(f"Functional layer: {summary['functional_tally']}")
    print(f"JSON:  {args.json_out}")
    print(f"Table: {args.table_out}")

    if args.fail_on_static and summary["static_fail"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
