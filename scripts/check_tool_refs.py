#!/usr/bin/env python
"""CONCEPT:ECO-4.45 — Reference-integrity gate for agent tool/skill references (Layer 1).

The problem this catches: ``prompts/*.json`` blueprints (and skill docs) reference tools/skills
by *literal display name*. When a tool/skill is renamed, or when the mcp-multiplexer adds a
``<server>__`` prefix at runtime, those references silently dangle — the agent quietly loses a
capability with no error. This gate makes that drift LOUD at build time, the same way
``check_concepts.py`` gates concept markers.

Rules (deliberately unambiguous):
  * A prompt's ``capabilities`` field = bounded INTENT tags (the target model, ECO-4.45). These
    are free-form and EXEMPT — they are resolved to concrete tools by the KG at construction.
  * A prompt's legacy ``tools`` / ``skills`` fields = concrete NAMES. Every entry MUST resolve to
    a known tool function, an installed skill slug, or an MCP catalog tool (prefix-stripped),
    else it is flagged as drift.

This nudges every blueprint toward ``capabilities`` (rename- and prefix-proof) while keeping the
~40 legacy name-list blueprints honest until they migrate.

Default is REPORT mode (exit 0) so the existing drift is visible without breaking CI on day one;
pass ``--strict`` to fail on any unresolved reference (use it for migrated/new blueprints).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO / "agent_utilities" / "tools"
PROMPTS_DIR = REPO / "agent_utilities" / "prompts"
# Best-effort skill catalog locations (skills live in a sibling package).
SKILL_ROOTS = [
    REPO.parents[1] / "skills" / "universal-skills",
    REPO.parents[1] / "universal-skills",
    REPO.parent / "universal-skills",
]

_PREFIX = re.compile(
    r"^[a-z0-9]+__"
)  # mcp-multiplexer per-server prefix, e.g. go__, gitlab__


def known_tool_names() -> set[str]:
    """Tool names = ``@tool_version``-decorated functions + their ``@trace(name=...)`` aliases."""
    names: set[str] = set()
    fn_def = re.compile(r"^async def (\w+)\(", re.M)
    trace_name = re.compile(r'@trace\(\s*name="([^"]+)"')
    versioned = re.compile(r"@tool_version\(")
    for pyfile in TOOLS_DIR.glob("*.py"):
        text = pyfile.read_text(encoding="utf-8", errors="ignore")
        if not versioned.search(text):
            continue
        names.update(fn_def.findall(text))
        names.update(trace_name.findall(text))
    return names


def known_skill_slugs() -> set[str]:
    slugs: set[str] = set()
    for root in SKILL_ROOTS:
        if not root.exists():
            continue
        for skill_md in root.rglob("SKILL.md"):
            slugs.add(skill_md.parent.name)
    return slugs


def _resolves(ref: str, tools: set[str], skills: set[str]) -> bool:
    base = _PREFIX.sub("", ref)  # tolerate an mcp-multiplexer prefix
    candidates = {ref, base, ref.replace("-", "_"), base.replace("-", "_")}
    return bool(candidates & tools) or ref in skills or base in skills


def _known_capabilities() -> set[str]:
    try:
        from agent_utilities.agent.capability_resolver import known_capabilities

        return known_capabilities()
    except Exception:  # noqa: BLE001 - resolver optional at gate time
        return set()


def scan():
    """Return (drift, unknown_caps, n_tools, n_skills).

    ``drift`` = concrete ``tools``/``skills`` refs that resolve to nothing (hard problem).
    ``unknown_caps`` = ``capabilities`` intents the static registry doesn't know (advisory: the
    KG may still resolve them at runtime).
    """
    tools = known_tool_names()
    skills = known_skill_slugs()
    caps = _known_capabilities()
    drift: list[tuple[str, str]] = []
    unknown_caps: list[tuple[str, str]] = []
    for pfile in sorted(PROMPTS_DIR.glob("*.json")):
        try:
            data = json.loads(pfile.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        refs: list[str] = []
        for field in ("tools", "skills"):
            val = data.get(field)
            if isinstance(val, list):
                refs.extend(str(x) for x in val)
        for ref in refs:
            if not _resolves(ref, tools, skills):
                drift.append((pfile.name, ref))
        cap_val = data.get("capabilities")
        if isinstance(cap_val, list) and caps:
            for c in cap_val:
                if str(c) not in caps:
                    unknown_caps.append((pfile.name, str(c)))
    return drift, unknown_caps, len(tools), len(skills)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any drift")
    args = ap.parse_args()

    drift, unknown_caps, n_tools, n_skills = scan()
    print(
        f"check_tool_refs: {n_tools} known tool names, {n_skills} known skill slugs, "
        f"{len(drift)} unresolved concrete references in prompts/*.json"
    )
    if unknown_caps:
        print(
            f"advisory: {len(unknown_caps)} capability intent(s) not in the static registry "
            "(may resolve via the KG at runtime, ECO-4.45):"
        )
        for fname, cap in unknown_caps:
            print(f"  {fname}: {cap}")
    if drift:
        by_file: dict[str, list[str]] = {}
        for fname, ref in drift:
            by_file.setdefault(fname, []).append(ref)
        for fname in sorted(by_file):
            print(f"  {fname}: {', '.join(sorted(by_file[fname]))}")
        print(
            "\nMigrate these blueprints from `tools`/`skills` name-lists to a `capabilities` "
            "intent list (ECO-4.45) so the KG resolves them — rename- and MCP-prefix-proof."
        )
        if not n_skills:
            print(
                "NOTE: no skill catalog found on disk; skill-slug refs could not be checked "
                "(install universal-skills to validate them)."
            )
    if args.strict and drift:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
