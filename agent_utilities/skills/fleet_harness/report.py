"""Render the static (+ optional functional) layer results.

Writes both a machine-readable JSON document and a human-readable table,
matching the shape used elsewhere in this repo's release/certification
tooling (a `generated_at` timestamp, a `summary` tally, and a `skills` list)
so the output is easy to diff run over run.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_utilities.skills.fleet_harness.functional_checks import FunctionalResult
from agent_utilities.skills.fleet_harness.static_checks import SkillStaticReport


def build_report(
    static_reports: list[SkillStaticReport],
    functional_results: list[FunctionalResult] | None = None,
) -> dict[str, Any]:
    functional_by_skill = {r.skill: r for r in (functional_results or [])}
    skills: list[dict[str, Any]] = []
    for report in static_reports:
        functional = functional_by_skill.get(report.record.relative_path)
        skills.append(
            {
                "repo": report.record.repo_name,
                "path": report.record.relative_path,
                "name": report.name,
                "static_status": report.status,
                "static_checks": [asdict(c) for c in report.checks],
                "functional_status": functional.status
                if functional
                else "SKIPPED-no-functional-run",
                "functional_detail": functional.detail if functional else None,
                "functional_referenced_tools": list(functional.referenced_tools)
                if functional
                else [],
            }
        )

    static_pass = sum(1 for r in static_reports if r.status == "PASS")
    static_fail = sum(1 for r in static_reports if r.status == "FAIL")
    functional_tally: dict[str, int] = {}
    for result in functional_results or []:
        functional_tally[result.status] = functional_tally.get(result.status, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_skills": len(static_reports),
            "static_pass": static_pass,
            "static_fail": static_fail,
            "functional_tally": functional_tally,
        },
        "skills": skills,
    }


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def render_table(report: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = report["summary"]
    lines.append("# Skill validation harness — results")
    lines.append("")
    lines.append(f"_Generated {report['generated_at']}_")
    lines.append("")
    lines.append(
        f"**Static layer:** {summary['total_skills']} skills discovered — "
        f"{summary['static_pass']} PASS / {summary['static_fail']} FAIL."
    )
    if summary["functional_tally"]:
        tally = ", ".join(
            f"{k}={v}" for k, v in sorted(summary["functional_tally"].items())
        )
        lines.append(f"**Functional layer:** {tally}")
    lines.append("")
    lines.append("| Repo | Skill path | Name | Static | Functional | First violation |")
    lines.append("|---|---|---|---|---|---|")
    for skill in report["skills"]:
        first_violation = ""
        failing = [c for c in skill["static_checks"] if c["status"] == "FAIL"]
        if failing:
            first_violation = f"`{failing[0]['rule']}`: {failing[0]['message']}"
        elif skill["functional_status"] == "FAIL":
            first_violation = skill["functional_detail"] or ""
        lines.append(
            f"| {skill['repo']} | {skill['path']} | {skill['name'] or '—'} | "
            f"{skill['static_status']} | {skill['functional_status']} | {first_violation} |"
        )

    fails = [s for s in report["skills"] if s["static_status"] == "FAIL"]
    if fails:
        lines.append("")
        lines.append("## Static FAIL detail")
        lines.append("")
        for skill in fails:
            lines.append(f"### {skill['repo']}/{skill['path']}")
            for check in skill["static_checks"]:
                if check["status"] == "FAIL":
                    lines.append(f"- `{check['rule']}`: {check['message']}")
            lines.append("")

    func_fails = [s for s in report["skills"] if s["functional_status"] == "FAIL"]
    if func_fails:
        lines.append("## Functional FAIL detail")
        lines.append("")
        for skill in func_fails:
            lines.append(
                f"- **{skill['repo']}/{skill['path']}**: {skill['functional_detail']}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_table(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_table(report), encoding="utf-8")
