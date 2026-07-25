#!/usr/bin/env python3
from __future__ import annotations

"""Apply (or, by default, simulate) the W2.8 ownership-grant plan (HG-4).

    python3 scripts/apply_graph_ownership_grants.py [--node-sample-limit N]
        [--ledger-sample-limit N] [--force-template] [--rollback-out PATH]
        [--apply-unambiguous]

**DEFAULT IS DRY-RUN.** Without ``--apply-unambiguous`` this script never calls a
single mutating RBAC method — it builds the ownership report (the SAME
live-probe-then-TEMPLATE-fallback ``scripts/graph_ownership_report.py`` uses),
computes the grant plan for every UNAMBIGUOUS, not-yet-covered graph (program
decision: auto-apply-UNAMBIGUOUS / hold-ambiguous — see
``agent_utilities/knowledge_graph/maintenance/graph_ownership{,_apply}.py``), and
prints:

* the plan (what WOULD be applied: one ``add_role`` + two ``add_grant`` — Read and
  Write — per uncovered UNAMBIGUOUS graph), and
* its exact rollback (the inverse ``remove_grant`` calls), also written to
  ``--rollback-out`` as JSON when given, else printed.

``--apply-unambiguous`` additionally REQUIRES a live, reachable engine — it refuses
to "apply" against a TEMPLATE fixture (that would apply nothing real while implying
otherwise) — and then actually issues the ``add_role``/``add_grant`` calls, in plan
order, stopping at the first failure. The printed/written rollback always reflects
exactly what was applied (a strict prefix of the plan on a partial failure), never
the full requested plan.

AMBIGUOUS graphs are never touched by this script under any flag combination — that
is the entire point of the disposition split (see the report's own "reasons"
column); a human resolves them and applies a targeted grant by hand (or via
:func:`graph_ownership_apply.plan_grants(report, include_ambiguous=True)` after
manual review, which this CLI deliberately does not expose as a flag).

W2.8 SCOPE NOTE: this task (epistemic-graph master program, W2.8) runs this script
DRY-RUN ONLY. ``--apply-unambiguous`` is implemented and unit-tested against a
fixture RBAC client (``tests/unit/knowledge_graph/test_graph_ownership_apply.py``),
but is never invoked here against any live store — HG-4 (a human gate) decides when
an operator runs it live, against which target, after reviewing the decision table.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.maintenance.graph_ownership_apply import (  # noqa: E402
    FixtureRbacAdminClient,
    GrantApplicationError,
    GrantPlanEntry,
    apply_plan,
    plan_grants,
    resolve_rbac_admin_client,
    rollback_to_json,
)
from scripts.graph_ownership_report import build_report  # noqa: E402

logger = logging.getLogger(__name__)


def _print_plan(plan: list[GrantPlanEntry]) -> None:
    if not plan:
        print("plan: empty (nothing UNAMBIGUOUS + uncovered to apply)")
        return
    graph_count = len({entry.graph for entry in plan})
    print(f"plan: {len(plan)} grant(s) across {graph_count} graph(s)")
    for entry in plan:
        print(
            f"  {entry.graph}: add_role({entry.role!r}) [idempotent] + "
            f"add_grant({entry.role!r}, {entry.resource!r}, {entry.action!r}, "
            f"{entry.effect!r})   # owner={entry.owner}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-sample-limit", type=int, default=50)
    parser.add_argument("--ledger-sample-limit", type=int, default=20)
    parser.add_argument(
        "--force-template",
        action="store_true",
        help="skip the live-engine probe and always generate the TEMPLATE fixture",
    )
    parser.add_argument(
        "--rollback-out",
        type=Path,
        default=None,
        help="write the rollback (inverse remove_grant) op list here as JSON",
    )
    parser.add_argument(
        "--apply-unambiguous",
        action="store_true",
        help=(
            "actually apply the UNAMBIGUOUS grant plan against a LIVE engine "
            "(default: dry-run only — never mutates anything)"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    report = build_report(
        node_sample_limit=args.node_sample_limit,
        ledger_sample_limit=args.ledger_sample_limit,
        force_template=args.force_template,
    )
    plan = plan_grants(
        report
    )  # UNAMBIGUOUS + not-already-covered only — never ambiguous

    print(f"mode: {report.mode}")
    _print_plan(plan)

    if args.apply_unambiguous:
        if report.mode != "live":
            print(
                "REFUSING --apply-unambiguous: the ownership report was generated "
                "in TEMPLATE mode (no live engine reachable) — there is nothing "
                "real to apply against. Re-run from an environment with a live, "
                "authenticated graph-os/engine connection.",
                file=sys.stderr,
            )
            return 2
        client = resolve_rbac_admin_client()
        dry_run = False
    else:
        # A plain, empty fixture client — dry-run never calls it (apply_plan's
        # dry_run branch short-circuits before any client method), it exists only
        # to satisfy the RbacAdminClient parameter.
        client = FixtureRbacAdminClient()
        dry_run = True

    failed = False
    try:
        applied, rollback = apply_plan(plan, client, dry_run=dry_run)
    except GrantApplicationError as exc:
        # A partial apply still carries the rollback for exactly what DID land
        # before the failure (see GrantApplicationError/apply_plan) — surface it
        # rather than stranding an un-rollbackable partial mutation.
        logger.error("grant application failed partway through (%s)", exc)
        print(f"APPLY FAILED partway through: {exc}", file=sys.stderr)
        applied, rollback = exc.applied, exc.rollback
        failed = True

    print(f"{'DRY-RUN' if dry_run else 'APPLIED'}: {len(applied)} grant(s)")
    for outcome in applied:
        role_note = " (new role)" if outcome.role_created else ""
        print(
            f"  {outcome.entry.graph}: {outcome.entry.role}{role_note} "
            f"{outcome.entry.action}/{outcome.entry.effect} -> {outcome.grant_result}"
        )

    rollback_ops = rollback_to_json(rollback)
    print(f"rollback: {len(rollback_ops)} inverse remove_grant op(s)")
    if args.rollback_out:
        args.rollback_out.parent.mkdir(parents=True, exist_ok=True)
        args.rollback_out.write_text(
            json.dumps(rollback_ops, indent=2), encoding="utf-8"
        )
        print(f"rollback written: {args.rollback_out}")
    else:
        print(json.dumps(rollback_ops, indent=2))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
