#!/usr/bin/env python3
"""Fleet skill-shape repair (D-SNI-2 / D-FSR-1).

The ingester bug fixed in fix/servicenow-skill-ingestion (D-SNV-1 / D-SNI-1)
stopped NEW misclassification of atomic ``skill_type: skill`` SKILL.md files
as 0-step ``WorkflowDefinition`` nodes, but it is not retroactive: as of this
repair, only 1 of 327 on-disk fleet ``skill_type: skill`` files
(``servicenow-incident-management``, fixed by hand to reproduce the bug) has
the ``CallableResource(resource_type='AGENT_SKILL')`` shape the delegation
binder (``agent_runner._resolve_agent_from_kg``) requires to actually RUN a
skill. This script closes that gap for the rest of the fleet.

WHAT IT DOES
------------
1. Walks ``agent-packages/agents/*/**/SKILL.md`` (excluding vendored
   ``.venv``/``node_modules`` copies and stale ``build/``/``dist/`` artifact
   trees), keeping only files whose own frontmatter declares
   ``skill_type: skill`` — the same signal the D-SNV-1 ingester fix now reads.
2. Resolves each owning package's MCP server name from its own
   ``pyproject.toml`` ``[project.scripts]`` entry point (the
   ``<name>-mcp = "....mcp_server:mcp_server"`` convention every fleet
   package already follows — e.g. ``servicenow-api`` -> ``servicenow-mcp``).
3. For each file, checks the LIVE KG for an existing
   ``CallableResource(resource_type='AGENT_SKILL')`` of the same name:
     - absent                                -> CREATE (the repair)
     - present, same ``mcp_server``          -> already correct, no-op
     - present, DIFFERENT ``mcp_server``     -> NAME CONFLICT, skipped and
       reported (mirrors ``fleet_skill_harvest._local_provider_owns``: an
       existing resource is the stronger authority and is never silently
       overwritten by a same-named fleet file).
4. Creates the missing ones via the SAME canonical entrypoint the ServiceNow
   one-off fix used: ``ingest_runnable_skill()`` — nothing new, no shortcuts.

This is idempotent: ``ingest_runnable_skill`` upserts by deterministic id
(``resource:skill:<slug>``), so re-running with ``--execute`` after a partial
or full prior run only re-writes identical properties for anything already
landed and creates whatever is still missing. It runs INSIDE the graph-os pod
because that's where the live engine + verified process-authority session
live and where the NFS-mounted canonical checkout (read-only) is visible at
its real workspace path.

USAGE (inside the graph-os pod; confirm the pod name first):

    kubectl -n platform get pods | grep graph-os

    # 1. DRY RUN (no KG mutation) — always run this first.
    kubectl -n platform exec -i <pod> -c graph-os -- \\
        python3 - --dry-run < scripts/repair_fleet_skill_shapes.py

    # 2. EXECUTE (mutates live KG state — only after reviewing the dry-run plan).
    kubectl -n platform exec -i <pod> -c graph-os -- \\
        python3 - --execute < scripts/repair_fleet_skill_shapes.py

Prints a machine-readable JSON report between ``===REPORT_JSON_START===`` and
``===REPORT_JSON_END===`` markers (the engine/session layers log a lot of
their own noise to stdout/stderr around it).

VERIFIED (D-FSR-2, closes D-SNI-2) — executed against the live graph-os KG in 6
bounded ``--limit`` chunks (30/50/50/60/57, one 50-chunk interrupted by two
unrelated live pod restarts and safely resumed thanks to the MERGE-based
idempotency above): 267/267 creates succeeded, 0 failures. Final ``--dry-run``
re-measurement confirms ``to_create_count: 0`` (fully converged). Before: 77
``CallableResource(AGENT_SKILL)`` nodes; after: 344. Verified by delegation,
not node count — 3 cross-package ``delegation_probe.py --stop-after skill``
proofs (gitlab-issues/gitlab-mcp, mattermost-channel-messaging/mattermost-mcp,
jellyfin-library-catalog/jellyfin-mcp) all report ``runnable=True``. See
``reports/deferred/lane-fleet-skill-repair.md`` (D-FSR-1, D-FSR-2) for the full
before/after inventory, the 4 langfuse-agent name-conflict skips, and the
``graph_workflows execute`` silent-success risk the leftover 415 stale
``step_count=0`` ``WorkflowDefinition`` nodes pose (NOT deleted — deletion is
out of scope for this lane).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover — pyyaml is a base dep of the served image
    yaml = None


def _agent_packages_root() -> Path:
    """The ``agent-packages`` directory ``agents/`` lives under, as a sibling of
    agent-utilities. Works whether this script is running from agent-utilities'
    canonical checkout or one of its worktrees, by asking git for THIS repo's
    own common dir rather than assuming a fixed number of ``parents[...]`` hops
    (a worktree adds a level a canonical checkout does not have) — same idiom
    as ``scripts/rollout_lane_guard_hook.py::_workspace_root``.
    """
    here = Path(__file__).resolve().parent
    common_dir = Path(
        subprocess.run(
            [
                "git",
                "-C",
                str(here),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    ).resolve()
    au_canonical = common_dir.parent  # .git's parent is the canonical checkout
    return au_canonical.parent  # agent-packages


AGENTS_ROOT = _agent_packages_root() / "agents"
EXCLUDE_SEGMENTS = {".venv", "node_modules", "build", "dist"}

_SCRIPT_RE = re.compile(
    r'^([a-zA-Z0-9_-]+)\s*=\s*"([^"]*\.mcp_server:mcp_server)"', re.MULTILINE
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "step"


def skill_reference(name: str) -> str:
    """Mirrors skill_workflow_ingest.skill_reference exactly (kept local so this
    script has no import-time dependency on the package layout beyond stdlib +
    pyyaml, and can be piped into the pod as a standalone file)."""
    return f"skill://{_slug(name).replace('_', '-')}"


def resolve_mcp_server(pkg_dir: Path) -> str | None:
    """Find a package's own MCP server entry-point name from its pyproject.toml."""
    pt = pkg_dir / "pyproject.toml"
    if not pt.exists():
        return None
    text = pt.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[project\.scripts\](.*?)(\n\[|\Z)", text, re.DOTALL)
    scope = m.group(1) if m else text
    matches = _SCRIPT_RE.findall(scope)
    if not matches:
        return None
    for name, _target in matches:
        if name.endswith("-mcp"):
            return name
    return matches[0][0]


def parse_skill_md(skill_md: Path) -> tuple[dict[str, Any], str] | None:
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError:
        return None
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    fm_text, body = parts[1], parts[2]
    frontmatter: dict[str, Any] = {}
    if yaml is not None:
        try:
            frontmatter = yaml.safe_load(fm_text) or {}
        except Exception:
            frontmatter = {}
    return frontmatter, body.strip()


def discover_fleet_skill_files() -> list[Path]:
    return sorted(
        f
        for f in AGENTS_ROOT.rglob("SKILL.md")
        if not (EXCLUDE_SEGMENTS & set(f.parts)) and ".egg-info" not in str(f)
    )


def build_manifest() -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Returns (manifest, skipped_non_skill_paths, unresolved_server_pkgs)."""
    manifest: list[dict[str, Any]] = []
    skipped_non_skill: list[str] = []
    unresolved_server: list[str] = []
    server_cache: dict[str, str | None] = {}

    for f in discover_fleet_skill_files():
        parsed = parse_skill_md(f)
        if parsed is None:
            continue
        fm, body = parsed
        skill_type = str(fm.get("skill_type") or "").strip().lower()
        if skill_type != "skill":
            skipped_non_skill.append(str(f))
            continue

        rel = f.relative_to(AGENTS_ROOT)
        pkg = rel.parts[0]
        if pkg not in server_cache:
            server_cache[pkg] = resolve_mcp_server(AGENTS_ROOT / pkg)
        server = server_cache[pkg]
        if not server:
            unresolved_server.append(f"{pkg}: {f}")
            continue

        name = str(fm.get("name") or f.parent.name)
        description = str(fm.get("description") or "").strip()
        if not body:
            # ingest_runnable_skill requires a non-empty body; nothing to ingest.
            continue

        slug = skill_reference(name).removeprefix("skill://")
        manifest.append(
            {
                "name": name,
                "description": description,
                "instructions": body,
                "package": pkg,
                "skill_md": str(f),
                "mcp_server": server,
                "provider": f"mcp:{server}",
                "resource_id": f"resource:skill:{slug}",
            }
        )
    return manifest, skipped_non_skill, unresolved_server


def classify_against_live_kg(
    manifest: list[dict[str, Any]], existing_names: dict[str, str | None]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the manifest into (to_create, to_skip) against the live KG state.

    ``existing_names`` maps CallableResource(AGENT_SKILL).name -> its current
    mcp_server (None for an agent-utilities core skill that was never fleet-
    harvested).
    """
    to_create: list[dict[str, Any]] = []
    to_skip: list[dict[str, Any]] = []
    _ABSENT = object()
    for m in manifest:
        prior_server = existing_names.get(m["name"], _ABSENT)
        if prior_server is _ABSENT:
            to_create.append(m)
        elif prior_server == m["mcp_server"]:
            m = {**m, "_status": "already_correct"}
            to_skip.append(m)
        else:
            m = {**m, "_status": f"name_conflict_owned_by={prior_server!r}"}
            to_skip.append(m)
    return to_create, to_skip


def fetch_existing_agent_skill_names(engine: Any) -> dict[str, str | None]:
    rows = engine.backend.execute(
        "MATCH (r:CallableResource) WHERE r.resource_type = 'AGENT_SKILL' "
        "RETURN r.name AS name, r.mcp_server AS mcp_server"
    )
    return {r["name"]: r.get("mcp_server") for r in rows}


def fetch_stale_workflow_zero_step_count(engine: Any) -> int:
    rows = engine.backend.execute(
        "MATCH (w:WorkflowDefinition) WHERE w.step_count = 0 RETURN count(w) AS c"
    )
    return int(rows[0]["c"]) if rows else 0


def _remint_session() -> None:
    """(Re)mint the process-authority session and bind it as ambient.

    Under sustained engine contention a long batch (300+ skills, each several
    upserts/edges, with individual engine calls observed taking 2-13s — see
    the "slow engine call ... engine likely contended" warnings) can outlive
    the credential's TTL mid-run, raising SessionExpiredError on the NEXT
    engine call even though nothing about that call was itself wrong. Calling
    this again is cheap (a real OAuth2 client-credentials mint, but a fast
    one) and gives every subsequent call a fresh authority window.
    """
    from agent_utilities.knowledge_graph.core.session import set_session
    from agent_utilities.mcp.kg_server import _mint_process_session
    from agent_utilities.security.brain_context import set_actor

    session = _mint_process_session("streamable-http")
    set_actor(session.actor)
    set_session(session)


def mint_engine() -> Any:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    _remint_session()
    eng = IntelligenceGraphEngine.get_active()
    if eng is None:
        eng = IntelligenceGraphEngine.get_or_create(defer_background_start=True)
    return eng


def execute_creates(
    engine: Any, to_create: list[dict[str, Any]], *, limit: int | None = None
) -> dict[str, Any]:
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        ingest_runnable_skill,
    )

    created: list[str] = []
    failed: dict[str, str] = {}
    attempted = 0
    for m in to_create:
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        # Refresh authority before EVERY write, not just on failure: under
        # contention a call that is merely slow (not yet expired) can tip the
        # NEXT call over the TTL edge, so proactive refresh is cheaper than
        # discovering expiry mid-batch and losing the whole remaining run.
        _remint_session()
        try:
            resource_id = ingest_runnable_skill(
                engine,
                name=m["name"],
                description=m["description"],
                instructions=m["instructions"],
                provider=m["provider"],
                mcp_server=m["mcp_server"],
            )
            created.append(resource_id)
        except Exception as exc:  # noqa: BLE001 — one skill's failure must not
            # abort the batch; the exact exception is kept, never reduced to a
            # class name, so the failure is diagnosable per-skill afterward.
            kind = type(exc).__name__
            if kind in ("SessionExpiredError", "SessionRequiredError"):
                # One retry with a freshly-minted session before giving up on
                # this particular skill — the batch keeps moving either way.
                try:
                    _remint_session()
                    resource_id = ingest_runnable_skill(
                        engine,
                        name=m["name"],
                        description=m["description"],
                        instructions=m["instructions"],
                        provider=m["provider"],
                        mcp_server=m["mcp_server"],
                    )
                    created.append(resource_id)
                    continue
                except Exception as exc2:  # noqa: BLE001
                    failed[m["name"]] = f"{type(exc2).__name__}: {exc2} (after retry)"
                    continue
            failed[m["name"]] = f"{kind}: {exc}"
    return {
        "created": created,
        "created_count": len(created),
        "failed": failed,
        "attempted": attempted,
        "remaining": len(to_create) - attempted,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="plan only, no KG mutation"
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="actually create missing CallableResource nodes",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap how many creates this invocation attempts (chunked/resumable runs "
        "under engine contention); omit to attempt all",
    )
    args = p.parse_args()

    manifest, skipped_non_skill, unresolved_server = build_manifest()

    engine = mint_engine()
    existing_names = fetch_existing_agent_skill_names(engine)
    stale_zero_step = fetch_stale_workflow_zero_step_count(engine)

    to_create, to_skip = classify_against_live_kg(manifest, existing_names)

    from collections import Counter

    per_package = Counter(m["package"] for m in to_create)

    report: dict[str, Any] = {
        "mode": "execute" if args.execute else "dry_run",
        "on_disk_skill_type_skill_files": len(manifest),
        "skipped_non_skill_files": len(skipped_non_skill),
        "unresolved_server_files": unresolved_server,
        "pre_existing_callable_agent_skill_count": len(existing_names),
        "stale_workflow_step_count_0": stale_zero_step,
        "to_create_count": len(to_create),
        "to_create_per_package": dict(sorted(per_package.items())),
        "to_create_names": sorted(m["name"] for m in to_create),
        "to_skip_count": len(to_skip),
        "to_skip_detail": [
            {"name": m["name"], "package": m["package"], "status": m["_status"]}
            for m in to_skip
        ],
    }

    if args.execute:
        exec_report = execute_creates(engine, to_create, limit=args.limit)
        report["execute_result"] = exec_report
        # Re-count post-mutation to prove it landed. Best-effort: a read this far
        # into a long, contended batch can itself hit an expired/contended
        # authority — that must not swallow the execute_result we already have.
        try:
            _remint_session()
            post_existing = fetch_existing_agent_skill_names(engine)
            report["post_execute_callable_agent_skill_count"] = len(post_existing)
        except Exception as exc:  # noqa: BLE001
            report["post_execute_callable_agent_skill_count"] = (
                f"UNAVAILABLE ({type(exc).__name__}: {exc})"
            )

    print("===REPORT_JSON_START===")
    print(json.dumps(report, default=str))
    print("===REPORT_JSON_END===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
