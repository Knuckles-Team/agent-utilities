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

# Bare env reads of governed prefixes.
PATTERN = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["'](KG_|GRAPH_|EPISTEMIC_)[A-Z0-9_]+["']"""
)
KEY_RE = re.compile(r"""["']((?:KG_|GRAPH_|EPISTEMIC_)[A-Z0-9_]+)["']""")

# Files allowed to read env directly: the central config object is the ONE place
# that maps env → typed fields; paths.py resolves directory overrides; the gate
# itself names the prefixes.
ALLOW_FILES = {
    "agent_utilities/core/config.py",
    "agent_utilities/core/paths.py",
}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}


def scan() -> set[tuple[str, str]]:
    """Return the set of (relpath, FLAG) bare env reads under the package."""
    found: set[tuple[str, str]] = set()
    for py in PKG.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        rel = py.relative_to(ROOT).as_posix()
        if rel in ALLOW_FILES:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            if PATTERN.search(line):
                key = KEY_RE.search(line)
                if key:
                    found.add((rel, key.group(1)))
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
        "# Frozen baseline of bare KG_/GRAPH_/EPISTEMIC_ env reads (ratchet).\n"
        "# New entries fail scripts/check_no_env_sprawl.py — add flags to AgentConfig\n"
        "# (core/config.py) instead. Regenerate with --update-baseline after removing\n"
        "# reads. See docs/architecture/configuration.md.\n" + body + "\n",
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
