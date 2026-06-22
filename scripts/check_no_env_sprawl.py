#!/usr/bin/env python3
"""Config-flag anti-sprawl gate.

Enforces the *Configuration discipline* rule in ``AGENTS.md``: KG/GRAPH/EPISTEMIC
environment variables must be declared as typed fields on ``AgentConfig``
(``agent_utilities/core/config.py``) and read via the ``config`` object — NOT with
bare ``os.environ.get("KG_...")`` / ``os.getenv("GRAPH_...")`` scattered across modules.

Because the codebase already carries ~96 such reads, this is a **ratchet**: the
current set is frozen in ``scripts/env_flag_baseline.txt`` and the gate fails only on
*new* bare reads not in the baseline. Removing a read (routing it through
``AgentConfig``) is always allowed and shrinks the baseline on the next
``--update-baseline``.

Usage:
  python3 scripts/check_no_env_sprawl.py            # check (exit 1 on new sprawl)
  python3 scripts/check_no_env_sprawl.py --update-baseline   # freeze current set

Exit 0 = no new sprawl, 1 = new bare env reads found.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "agent_utilities"
BASELINE = ROOT / "scripts" / "env_flag_baseline.txt"

# Bare env *reads* of ANY variable (not just KG_/GRAPH_/EPISTEMIC_). Modules must
# route every read through ``config.setting(...)`` or a typed ``AgentConfig``
# field — never ``os.environ.get``/``os.getenv``/``os.environ[...]`` directly.
# Two read forms are caught:
#   1. the read APIs: os.environ.get("X") / os.getenv("X")
#   2. subscript reads: os.environ["X"]  (a trailing ``=`` → a write, exempt)
# Writes (os.environ["X"] = ...) are legitimate cross-process signaling and are
# NOT flagged.
PATTERN_GET = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)
PATTERN_SUBSCRIPT = re.compile(
    r"""os\.environ\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\](?!\s*=[^=])"""
)

# Files allowed to read env directly: ``config.py`` maps env → typed fields,
# ``_env.py`` implements the dependency-free ``setting()`` accessor, and
# ``paths.py`` resolves directory overrides before config exists.
ALLOW_FILES = {
    "agent_utilities/core/config.py",
    "agent_utilities/core/_env.py",
    "agent_utilities/core/paths.py",
}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}


def scan() -> set[tuple[str, str]]:
    """Return the set of (relpath, KEY) bare env reads under the package."""
    found: set[tuple[str, str]] = set()
    for py in PKG.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        rel = py.relative_to(ROOT).as_posix()
        # Vendored skill assets (CONCEPT:OS-5.52, ``agent_utilities/skills/``) are
        # shipped skill scripts, not serving-plane code — they follow the skill
        # repos' own convention (standalone CLI tools reading env directly), so the
        # config-discipline gate does not apply to them.
        if rel.startswith("agent_utilities/skills/"):
            continue
        if rel in ALLOW_FILES:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            for m in PATTERN_GET.finditer(line):
                found.add((rel, m.group(1)))
            for m in PATTERN_SUBSCRIPT.finditer(line):
                found.add((rel, m.group(1)))
    return found


def load_baseline() -> set[tuple[str, str]]:
    if not BASELINE.exists():
        return set()
    out: set[tuple[str, str]] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel, _, key = line.partition("\t")
        if rel and key:
            out.add((rel, key))
    return out


def write_baseline(entries: set[tuple[str, str]]) -> None:
    body = "\n".join(f"{rel}\t{key}" for rel, key in sorted(entries))
    BASELINE.write_text(
        "# Frozen baseline of bare os.environ/os.getenv reads across ALL prefixes\n"
        "# (ratchet — burn-down toward zero). The KG_/GRAPH_/EPISTEMIC_ reads were\n"
        "# folded onto config.setting()/AgentConfig fields; what remains here (AGENT_/\n"
        "# VAULT_/OTEL_/connector creds/…) is the tracked burn-down. New entries fail\n"
        "# scripts/check_no_env_sprawl.py — route reads through config.setting(...) or a\n"
        "# typed AgentConfig field instead. Regenerate with --update-baseline after\n"
        "# removing reads. See docs/architecture/configuration.md.\n" + body + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    current = scan()
    if args.update_baseline:
        write_baseline(current)
        print(f"Baseline updated: {len(current)} entries → {BASELINE.name}")
        return 0

    baseline = load_baseline()
    new = sorted(current - baseline)
    if new:
        print("New bare env reads found (add the flag to AgentConfig instead):\n")
        for rel, key in new:
            print(f"  {rel}: {key}")
        print(
            "\nSee AGENTS.md → 'Configuration discipline' and "
            "docs/architecture/configuration.md."
        )
        return 1
    removed = len(baseline) - len(current & baseline)
    msg = f"OK — no new env sprawl ({len(current)} baselined reads"
    print(msg + (f", {removed} removed since baseline)." if removed else ")."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
