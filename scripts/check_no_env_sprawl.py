#!/usr/bin/env python3
"""Config-flag anti-sprawl gate.

Enforces the *Configuration discipline* rule in ``AGENTS.md``: KG/GRAPH/EPISTEMIC
environment variables must be declared as typed fields on ``AgentConfig``
(``agent_utilities/core/config.py``) and read via the ``config`` object — NOT with
bare ``os.environ.get("KG_...")`` / ``os.getenv("GRAPH_...")`` scattered across modules.

**Absolute gate, no ratchet.** The burn-down finished (the frozen baseline this gate
used to compare against reached zero entries), so per the workspace's no-ratchet policy
the gate now enforces a fixed threshold (``MAX``, a module-level constant — see below)
against the REAL count on every run, pass or fail, instead of comparing against a
baseline file. There is no ``--update-baseline`` escape hatch and no way to freeze a new
violation as "already there": every bare env read outside the allowed files is a failure.

``MAX`` is a module-level constant rather than a ``--max``/CLI flag because both callers
of this script invoke it with **zero arguments**: pre-commit's ``check-no-env-sprawl``
hook (``.config/pre-commit.yaml``, default/blocking stage, ``pass_filenames: false``, no
extra ``args:``) and CI's ``advisory.yml`` (``python3 scripts/check_no_env_sprawl.py``). A
CLI flag would never be passed by either caller, so the threshold has to live in code.

Usage:
  python3 scripts/check_no_env_sprawl.py     # check; prints the real count unconditionally

Exit 0 = count <= MAX, 1 = count > MAX (new/remaining bare env reads found).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

PKG = ROOT / "agent_utilities"

# Absolute threshold — NOT a ratchet baseline. The burn-down already reached zero
# (the last frozen baseline was empty); this constant enforces that permanently. Per
# the workspace's no-ratchet policy, lower this only by actually fixing violations,
# never by re-introducing a baseline file to freeze new ones.
MAX = 0

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


def _candidate_py_files(pkg: Path) -> list[Path]:
    """``.py`` files under ``pkg``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output (e.g. a
    packaging step's ``build/lib/...`` copy of a since-fixed source file),
    which can reintroduce a violation this gate already cleared in real
    source. Falls back to a filtered filesystem walk only when ``pkg`` is not
    inside a git working tree (e.g. a synthetic test fixture).
    """
    return tracked_or_walked(pkg, "*.py", root=ROOT)


def scan() -> set[tuple[str, str]]:
    """Return the set of (relpath, KEY) bare env reads under the package."""
    found: set[tuple[str, str]] = set()
    for py in _candidate_py_files(PKG):
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        rel = py.relative_to(ROOT).as_posix()
        # Vendored skill assets (CONCEPT:AU-OS.deployment.agent-factory-autoload, ``agent_utilities/skills/``) are
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


def main() -> int:
    current = sorted(scan())
    # No-ratchet policy: print the REAL count unconditionally, pass or fail —
    # never silently absorbed into a frozen baseline.
    print(f"env-sprawl bare reads found: {len(current)} (max allowed: {MAX})")
    if len(current) > MAX:
        print("\nBare env reads found (add the flag to AgentConfig instead):\n")
        for rel, key in current:
            print(f"  {rel}: {key}")
        print(
            "\nSee AGENTS.md → 'Configuration discipline' and "
            "docs/architecture/configuration.md."
        )
        return 1
    print("OK — no env sprawl.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
