#!/usr/bin/python
"""CONCEPT:AU-ECO.mcp.kg-skill-verb-coverage — kg-* skill ↔ graph-os verb coverage doctor.

The graph-os MCP surface — every tool in :data:`kg_server.REGISTERED_TOOLS` — must
be mirrored by a discoverable ``kg-*`` skill, so a new verb cannot ship without a
skill (operators can't discover it) and a skill cannot point at a dead verb. This
module computes that coverage; ``tests/unit/test_gateway_mcp_parity.py`` asserts it
(the third parity leg, alongside the tool⇄REST-route legs) and the
``kg-coverage-doctor`` skill runs it as a CLI (``python -m
agent_utilities.mcp.skill_coverage``).

Naming contract (user-locked): ``kg-<capability>`` where ``<capability>`` = the MCP
verb minus the ``graph_`` prefix with ``_``→``-`` (``graph_ontology`` →
``kg-ontology``). So the common case needs no configuration — the skill's slug alone
maps it to its verb. Two escape hatches for the surface that isn't 1:1:

* ``wraps: [verb, ...]`` frontmatter — a skill that fronts *several* verbs declares
  them explicitly (e.g. ``kg-ingest`` wraps ``graph_ingest`` + ``source_sync`` +
  ``source_drain`` + ``source_connector`` + ``document_process``; ``kg-ontology``
  wraps ``graph_ontology`` + every ``ontology_*`` + ``object_*`` verb; each
  ``kg-modality-*`` wraps its ``engine_*`` domains).
* ``tier: meta|surface`` frontmatter — a skill that is not a verb wrapper at all
  (``kg-mux-use``, ``kg-capability-builder``, the ``kg-webui-*`` skills). Exempt
  from both coverage and orphan checks.

Discovery is filesystem-based (walk each installed skill-provider package's dir for
``SKILL.md``) rather than pure entry-point metadata, because editable/worktree
installs routinely carry stale ``entry_points()`` metadata in this workspace — the
package is importable but its ``agent_utilities.skill_providers`` entry-point is not
yet re-registered. We union the live entry-point dirs with a direct import-path
resolve of the known provider modules so the gate is stable in dev and CI alike.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Verbs deliberately NOT surfaced as a kg-* skill. Keep this list tiny and
# justified — every entry weakens the gate. A new registered tool must either get a
# skill or be added here with a reason.
INTENTIONALLY_UNSKILLED: frozenset[str] = frozenset(
    {
        # ``quant`` is the emerald-exchange finance domain tool, not part of the
        # generic graph-os surface; it carries a pre-existing surface-parity waiver.
        "quant",
        # AU-P0-6: ``engine_rbac``/``engine_admin`` are the two newly-exposed
        # ADMIN-family low-level namespaces (RBAC policy administration; ops
        # backup/restore) — gated behind ``kg:admin`` (see
        # ``engine_tools.ADMIN_DOMAINS``/``_enforce_admin_scope``) BEFORE being
        # newly exposed, per the audit's explicit ordering. Their natural
        # wrapper is ``kg-modality-consensus`` (already wraps the sibling ADMIN
        # domains ``engine_consensus``/``engine_resharding``/``engine_tenants``),
        # which ships from the ``epistemic-graph`` package/repo — out of this
        # worktree's scope to edit. Waived here rather than left silently
        # uncovered; follow-up: extend that skill's `wraps:` (or add a
        # dedicated one) in the epistemic-graph repo.
        "engine_rbac",
        "engine_admin",
        # Seam 8 (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse) intent verbs — only present in
        # REGISTERED_TOOLS under MCP_TOOL_MODE=intent. They are not per-CAPABILITY
        # wrappers (a kg-<verb> skill implies ONE granular tool) — they wrap the
        # WHOLE resolver, and every granular tool they route to already has its
        # own kg-* skill. The dedicated "how to use the intent surface" skill
        # (`kg-intent`, tier: meta) documents the resolver/dispatcher mechanism
        # itself; it is intentionally NOT a `wraps:` entry here — a meta skill
        # never claims verb coverage (see `compute_coverage`'s tier exemption).
        "ask",
        "find",
        "write",
        "act",
        "manage",
        "why",
    }
)

# Known skill-provider packages resolved directly by import path (belt-and-braces
# against stale entry-point metadata). The hub's own provider + the three fleet
# providers that ship kg-* skills.
_PROVIDER_MODULES: tuple[str, ...] = (
    "agent_utilities.skills",
    "universal_skills",
    "epistemic_graph.skills",
    "agent_webui.skills",
)

_VALID_TIERS: frozenset[str] = frozenset({"core", "modality", "meta", "surface"})
_WRAPPER_TIERS: frozenset[str] = frozenset({"core", "modality"})


@dataclass(frozen=True)
class SkillMeta:
    """A discovered ``kg-*`` skill's coverage-relevant frontmatter."""

    name: str
    tier: str  # "" if unset
    wraps: tuple[str, ...]
    path: Path


@dataclass
class CoverageReport:
    uncovered: list[str] = field(default_factory=list)  # verbs with no skill
    orphans: list[tuple[str, str]] = field(default_factory=list)  # (skill, bad_verb)
    bad_tiers: list[tuple[str, str]] = field(default_factory=list)  # (skill, tier)
    covered: dict[str, list[str]] = field(default_factory=dict)  # verb -> [skills]

    @property
    def ok(self) -> bool:
        return not (self.uncovered or self.orphans or self.bad_tiers)


def slug_to_verb(slug: str) -> str:
    """``kg-ontology`` → ``graph_ontology`` (the default 1:1 inference)."""
    cap = slug[len("kg-") :] if slug.startswith("kg-") else slug
    return "graph_" + cap.replace("-", "_")


def verb_universe() -> set[str]:
    """The live graph-os verb surface every skill set must cover."""
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()
    return set(kg_server.REGISTERED_TOOLS)


def _parse_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    try:
        data = yaml.safe_load(text[3:end])
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def _provider_dirs() -> list[Path]:
    """Skill-provider dirs, unioning entry-point discovery with import-path resolve."""
    dirs: dict[str, Path] = {}
    try:
        from agent_utilities.core.providers import (
            SKILL_PROVIDER_GROUP,
            iter_provider_dirs,
        )

        for _name, path in iter_provider_dirs(SKILL_PROVIDER_GROUP):
            dirs[str(path)] = path
    except Exception:  # noqa: BLE001 — discovery must never hard-fail the gate
        pass
    for mod in _PROVIDER_MODULES:
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        locs = list(spec.submodule_search_locations or [])
        if not locs and spec.origin:
            locs = [str(Path(spec.origin).parent)]
        for loc in locs:
            p = Path(loc)
            if p.is_dir():
                dirs[str(p)] = p
    return list(dirs.values())


def discover_skills() -> list[SkillMeta]:
    """Every ``kg-*`` skill discoverable across the installed provider packages."""
    out: dict[str, SkillMeta] = {}  # de-dupe by name (first wins, stable)
    for root in _provider_dirs():
        for md in sorted(root.rglob("SKILL.md")):
            fm = _parse_frontmatter(md.read_text(encoding="utf-8", errors="ignore"))
            name = str(fm.get("name") or md.parent.name)
            if not name.startswith("kg-"):
                continue
            if name in out:
                continue
            wraps_raw = fm.get("wraps") or []
            wraps = (
                tuple(str(w) for w in wraps_raw) if isinstance(wraps_raw, list) else ()
            )
            out[name] = SkillMeta(
                name=name,
                tier=str(fm.get("tier") or ""),
                wraps=wraps,
                path=md,
            )
    return list(out.values())


def compute_coverage() -> CoverageReport:
    """Diff the live verb surface against the discovered kg-* skill set."""
    universe = verb_universe()
    skills = discover_skills()
    report = CoverageReport()

    covered: dict[str, list[str]] = {}
    for s in skills:
        if s.tier and s.tier not in _VALID_TIERS:
            report.bad_tiers.append((s.name, s.tier))
        # meta/surface skills are not verb wrappers — exempt from both checks.
        if s.tier in ("meta", "surface"):
            continue
        # A wrapper skill covers its explicit `wraps:` list, else its slug-inferred verb.
        claimed = list(s.wraps) if s.wraps else [slug_to_verb(s.name)]
        real = [v for v in claimed if v in universe]
        bad = [v for v in claimed if v not in universe]
        for v in bad:
            report.orphans.append((s.name, v))
        for v in real:
            covered.setdefault(v, []).append(s.name)

    report.covered = covered
    required = universe - INTENTIONALLY_UNSKILLED
    report.uncovered = sorted(required - set(covered))
    return report


def main() -> int:
    report = compute_coverage()
    if report.ok:
        n = len(report.covered)
        print(
            f"kg-coverage-doctor OK — {n} graph-os verbs each wrapped by a kg-* skill."
        )
        return 0
    if report.uncovered:
        print(
            "VERBS WITH NO kg-* SKILL (add a skill or add to INTENTIONALLY_UNSKILLED):"
        )
        for v in report.uncovered:
            print(f"  - {v}")
    if report.orphans:
        print("ORPHAN kg-* SKILLS (wrap/slug points at a non-existent verb):")
        for skill, verb in report.orphans:
            print(f"  - {skill} -> {verb}")
    if report.bad_tiers:
        print("INVALID tier: values (use core|modality|meta|surface):")
        for skill, tier in report.bad_tiers:
            print(f"  - {skill}: {tier}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
