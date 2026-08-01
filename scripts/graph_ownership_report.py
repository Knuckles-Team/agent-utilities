#!/usr/bin/env python3
from __future__ import annotations

"""Generate the W2.8 per-graph ownership decision table (the HG-4 artifact).

    python3 scripts/graph_ownership_report.py [--out PATH] [--node-sample-limit N]
        [--ledger-sample-limit N] [--force-template] [--quiet]

Tries a READ-ONLY live engine connection first — the SAME
``GraphComputeEngine`` process authority every AU tool/MCP server uses (see
``agent_utilities/knowledge_graph/maintenance/graph_ownership.py::LiveEngineCatalogClient``).
On ANY failure (unreachable engine, expired credentials, no engine configured, ...)
it falls back to a realistic TEMPLATE fixture modeling the confirmed real shape of
the legacy corpus (``reports/HANDOFF-2026-07-22.md`` §8: 82 ``code_<repo>`` legacy
graphs recovered pre-identity/RBAC + 3 named graphs already reachable) and marks the
report accordingly — it never fabricates a "live" result. Re-run this script from an
environment with a live, authenticated graph-os/engine connection to regenerate the
report against the real catalog before HG-4 acts on it.

This script is READ-ONLY end to end — it only calls ``list_graphs``/``rbac_policy``/
``sample_nodes``/``graph_ledger``. Grant application is a separate, explicit script:
``scripts/apply_graph_ownership_grants.py`` (dry-run by default).
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Run this checkout's code, not whatever `agent_utilities` happens to be
# editable-installed against (this repo is developed across concurrent git
# worktrees — see scripts/check_skill_validation_certification.py /
# scripts/generate_connector_capability_bundles.py for the same precedented fix).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _workspace_root() -> Path:
    """The shared WORKSPACE root ``reports/wave<N>/`` artifacts live under.

    Works whether this script is running from agent-utilities' canonical
    checkout or one of its worktrees, by asking git for THIS repo's own
    common dir rather than assuming a fixed number of ``parents[...]`` hops
    (a worktree adds a level a canonical checkout does not have) — same
    idiom as ``scripts/rollout_lane_guard_hook.py::_workspace_root``.
    """
    common_dir = Path(
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    ).resolve()
    au_canonical = common_dir.parent  # .git's parent is the canonical checkout
    return au_canonical.parent.parent  # agent-packages -> workspace root

from agent_utilities.knowledge_graph.maintenance.graph_ownership import (  # noqa: E402
    EngineCatalogClient,
    EngineUnreachableError,
    FixtureCatalogClient,
    OwnershipReport,
    build_ownership_report,
    check_invariant,
    render_markdown,
    resolve_catalog_client,
)

logger = logging.getLogger(__name__)

#: The program's shared cross-repo review-packet location (see
#: ``reports/epistemic-graph-master-plan-2026-07-23.md``): every wave drops its
#: HG-gated artifacts under ``reports/wave<N>/`` at the WORKSPACE root, not inside
#: any one repo's own git history — sibling wave-2 deliverables already live there
#: (e.g. ``reports/wave2/ADR-identity-and-agents-as-data.md``). Overridable via
#: ``--out`` for a local/test run.
DEFAULT_OUT = _workspace_root() / "reports" / "wave2" / "graph-ownership-report.md"

# Confirmed real legacy-corpus shape (reports/HANDOFF-2026-07-22.md §8): 82
# `code_<repo>` graphs recovered into the durable catalog before identity/RBAC
# existed. These slugs are real `workspace.yml` repo names, used ONLY to make the
# TEMPLATE-mode fixture representative of the real catalog's shape; they have no
# bearing on live-mode output (which reads the real catalog instead of this list).
_TEMPLATE_CODE_REPOS: tuple[str, ...] = (
    "2fauth",
    "adminer",
    "agent-terminal-ui",
    "agent-utilities",
    "agent-webui",
    "alpine",
    "ansible-tower-mcp",
    "archimate-mcp",
    "archivebox",
    "archivebox-api",
    "archivebox-mcp",
    "arr-mcp",
    "arr-stack",
    "atlassian-agent",
    "audio-transcriber",
    "audio-transcriber-mcp",
    "audiobookshelf",
    "audiobookshelf-mcp",
    "bazarr",
    "borgbackup",
    "caddy",
    "caddy-mcp",
    "calibre",
    "calibre-server",
    "camunda-mcp",
    "chaptarr",
    "ciso-assistant",
    "ciso-assistant-api",
    "clarity-api",
    "cloudflare-connector",
    "code-server",
    "container-manager-mcp",
    "data-science-mcp",
    "ddclient",
    "debian",
    "debian-dind",
    "docker-registry",
    "docker-registry-webui",
    "dockerhub-api",
    "documentdb",
    "documentdb-mcp",
    "dozzle",
    "drop",
    "egeria-mcp",
    "elasticsearch",
    "emerald-exchange",
    "epistemic-graph",
    "erpnext",
    "erpnext-agent",
    "erpnext-mcp",
    "eunomia",
    "fan-manager",
    "faster-whisper",
    "firefly-iii",
    "firefly-iii-mcp",
    "flaresolverr",
    "flux.2",
    "freshrss",
    "frigate",
    "genius-agent",
    "geniusbot",
    "github-agent",
    "github-mcp",
    "github-runner",
    "gitlab",
    "gitlab-api",
    "gitlab-cli-image",
    "gitlab-mcp",
    "gitlab-pipelines",
    "gitlab-release-cli-image",
    "gitlab-runner",
    "gitlab-runner-helper-image",
    "gluetun",
    "golang",
    "gosec",
    "gotenberg",
    "grafana",
    "gramps",
    "gramps-mcp",
    "gramps-postgres",
    "gramps-web",
    "hdhomerun-mcp",
)


def _build_template_catalog() -> FixtureCatalogClient:
    """A realistic TEMPLATE-mode fixture exercising every disposition branch.

    82 ``code_<repo>`` legacy graphs (a handful enriched with node samples), the 3
    graphs the 2026-07-22 recovery confirmed are already reachable (one already
    granted), plus a small set of deliberately-odd graphs so the AMBIGUOUS path is
    demonstrated too, not just the dominant UNAMBIGUOUS ``code_*`` case.
    """
    graphs: list[dict[str, Any]] = [
        {"name": "__commons__", "type": "Commons", "valid": True},
        {"name": "tenant__homelab__default", "type": "Global", "valid": True},
        {"name": "tenant__homelab____commons__", "type": "Commons", "valid": True},
    ]
    for repo in _TEMPLATE_CODE_REPOS:
        graphs.append({"name": f"code_{repo}", "type": "Agent", "valid": True})
    graphs.extend(
        [
            {"name": "agent:known-writer", "type": "Agent", "valid": True},
            {"name": "agent:cold-case", "type": "Agent", "valid": True},
            {"name": "scratch_import_2026", "type": "Agent", "valid": True},
            {"name": "misc_legacy_dump", "type": "Agent", "valid": True},
        ]
    )

    node_samples: dict[str, list[dict[str, Any]]] = {}
    # A handful of code_* graphs get a clean, corroborating source_system sample —
    # demonstrates "name convention + decisive matching sample".
    for repo in _TEMPLATE_CODE_REPOS[:5]:
        node_samples[f"code_{repo}"] = [
            {"source_system": f"code:{repo}"} for _ in range(10)
        ]
    # One code_* graph gets a genuinely conflicting sample (two systems collided) —
    # demonstrates "conflicting sampled values" forcing AMBIGUOUS even under a
    # high-confidence name hint.
    conflict_repo = _TEMPLATE_CODE_REPOS[5]
    node_samples[f"code_{conflict_repo}"] = [
        {"source_system": f"code:{conflict_repo}"} for _ in range(6)
    ] + [{"source_system": "code:unrelated-import"} for _ in range(5)]
    # tenant graph: matching tenant_id sample corroborates the naming convention.
    node_samples["tenant__homelab__default"] = [
        {"tenant_id": "homelab"} for _ in range(8)
    ]
    # agent:known-writer: a matching _owner_id sample -> UNAMBIGUOUS via
    # medium-confidence-name + corroborating sample.
    node_samples["agent:known-writer"] = [
        {"owner_id": "known-writer"} for _ in range(6)
    ]
    # agent:cold-case: no sample at all -> AMBIGUOUS (medium confidence alone is not
    # auto-appliable — "do not guess grants").
    # misc_legacy_dump: unrecognized name AND a conflicting sample -> AMBIGUOUS.
    node_samples["misc_legacy_dump"] = [{"owner_id": "alice"} for _ in range(3)] + [
        {"owner_id": "bob"} for _ in range(4)
    ]
    # scratch_import_2026: unrecognized name, no sample -> AMBIGUOUS.

    roles = [{"name": "owner:tenant-homelab", "parents": []}]
    grants = [
        {
            "role": "owner:tenant-homelab",
            "resource": {"Graph": "tenant__homelab__default"},
            "action": "Read",
            "effect": "Allow",
        },
        {
            "role": "owner:tenant-homelab",
            "resource": {"Graph": "tenant__homelab__default"},
            "action": "Write",
            "effect": "Allow",
        },
    ]

    return FixtureCatalogClient(
        graphs, grants=grants, roles=roles, node_samples=node_samples
    )


def _probe_live_client(config: Any = None) -> EngineCatalogClient:
    """Return a live client, having proven ``list_graphs`` actually works.

    Raises :class:`EngineUnreachableError` (or lets any other exception propagate)
    on failure — the caller decides the TEMPLATE fallback; this function never
    fabricates data itself.
    """
    client = resolve_catalog_client(config)
    client.list_graphs()
    return client


def build_report(
    *,
    node_sample_limit: int,
    ledger_sample_limit: int,
    force_template: bool,
) -> OwnershipReport:
    if not force_template:
        try:
            client = _probe_live_client()
        except EngineUnreachableError as exc:
            logger.warning(
                "live engine unreachable (%s); falling back to TEMPLATE mode", exc
            )
        except Exception as exc:  # noqa: BLE001 - a live probe must never crash the report
            logger.warning(
                "live engine probe failed (%s: %s); falling back to TEMPLATE mode",
                type(exc).__name__,
                exc,
            )
        else:
            return build_ownership_report(
                client,
                mode="live",
                source_note=(
                    "live epistemic-graph engine catalog (GraphComputeEngine "
                    "process authority)"
                ),
                node_sample_limit=node_sample_limit,
                ledger_sample_limit=ledger_sample_limit,
            )

    client = _build_template_catalog()
    return build_ownership_report(
        client,
        mode="template",
        source_note=(
            "TEMPLATE fixture modeling the confirmed real legacy-corpus shape "
            "(reports/HANDOFF-2026-07-22.md §8): 82 code_<repo> graphs + 3 "
            "named graphs — NOT the live catalog"
        ),
        node_sample_limit=node_sample_limit,
        ledger_sample_limit=ledger_sample_limit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="markdown output path"
    )
    parser.add_argument("--node-sample-limit", type=int, default=50)
    parser.add_argument("--ledger-sample-limit", type=int, default=20)
    parser.add_argument(
        "--force-template",
        action="store_true",
        help="skip the live-engine probe and always generate the TEMPLATE fixture",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    report = build_report(
        node_sample_limit=args.node_sample_limit,
        ledger_sample_limit=args.ledger_sample_limit,
        force_template=args.force_template,
    )
    markdown = render_markdown(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")

    # Report-only: this run never enforces the invariant (see graph_ownership.py's
    # invariant_enforced/KG_GRAPH_OWNERSHIP_ENFORCED) — it only surfaces the count.
    violations = check_invariant(report.dispositions, enforced=False)
    counts = report.counts
    print(f"mode: {report.mode}")
    print(f"wrote: {args.out}")
    print(
        f"graphs: {counts['total']}  unambiguous: {counts['unambiguous']}  "
        f"ambiguous: {counts['ambiguous']}  needs_grant: {counts['needs_grant']}  "
        f"already_covered: {counts['already_covered']}  "
        f"explicit_public: {counts['explicit_public']}"
    )
    print(
        f"invariant (report-only, KG_GRAPH_OWNERSHIP_ENFORCED unset): "
        f"{len(violations)} graph(s) without a covering grant or public flag"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
