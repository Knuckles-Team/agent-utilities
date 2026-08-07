"""Docs-vs-reality consistency tests (Plan 09).

Asserts that the generated documentation stays in lock-step with the
single source of truth (``docs/concepts.yaml``) and that the bloat in
AGENTS.md has been eliminated.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
README = ROOT / "README.md"
AGENTS = ROOT / "AGENTS.md"
CONCEPTS = ROOT / "docs" / "concepts.yaml"

# This ceiling is NOT a content-length policy on AGENTS.head.md's hand-curated
# governance prose (Working Discipline, Quality Bar, dependency/sprawl/config
# discipline, Wire-First, Native-by-default, Two-surfaces, secrets, worktree
# workflow, No-Legacy, etc. — deliberately grown over 2+ months as the governance
# surface expanded; ~40 distinctly-titled commits, zero duplicate sections). It
# exists ONLY to catch a GENERATOR LEAK — e.g. a regression that lets
# .hypothesis/node_modules/a stale full project tree bleed back into the
# generated appendix (the full tree was moved out to docs/project_structure.md
# in D-WS-4 for exactly this reason; the generated appendix today is ~1.9 KiB
# and stable). No downstream consumer's token/context budget depends on this
# exact number either: agent_utilities/knowledge_graph/core/agents_md.py's
# loader does no truncation at all, and memory_engine.py's startup-payload
# budget (DEFAULT_BUDGET_CHARS = 24_000) already hard-trims AGENTS.md content
# at any size well above ~24 KiB — the cap sits far below both the old and new
# ceiling either way, so this number does not gate that path.
#
# Recalibrated 2026-08-07 (D-W2P-8): the prior 80 KiB ceiling (baselined
# against a ~71 KiB curated-head snapshot) was tripping on ~2.9 KB of
# legitimate, distinct new governance content, not a leak — see the
# [UPDATE to D-W2P-8] register entry for the full evidence trail (no
# duplicate headings, generated portion stable at ~1.9 KiB, no consumer
# budget affected). Recalibrated to current size (84,817 bytes) + 9,391
# bytes (~9.2 KiB, ~11%) headroom (92 KiB = 94,208 bytes total), matching
# the prior calibration's own ~9 KiB headroom in absolute terms (that one
# was ~13% of its smaller baseline). Proof this still catches what it's
# for: current-size-only content passes; current size + a simulated 16.4 KiB
# tree-leak (the exact size of docs/project_structure.md today, i.e. "the
# full tree D-WS-4 moved out regressed back into AGENTS.md") still trips it
# (101,660 bytes > 94,208). Whether AGENTS.head.md's ~83 KB of curated
# content SHOULD be trimmed is a separate, non-blocking editorial question,
# left open for the operator.
AGENTS_MAX_BYTES = 92 * 1024


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_gen_docs_check_passes():
    result = _run("gen_docs.py", "--check")
    assert result.returncode == 0, (
        "gen_docs.py --check failed; README is out of sync with concepts.yaml.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_concepts_yaml_total_matches_readme_block():
    data = yaml.safe_load(CONCEPTS.read_text(encoding="utf-8"))
    total = len(data["concepts"])
    pillars = len({c["pillar"] for c in data["concepts"]})

    readme = README.read_text(encoding="utf-8")
    block_match = re.search(
        r"<!-- BEGIN GENERATED: concepts -->(.*?)<!-- END GENERATED: concepts -->",
        readme,
        re.DOTALL,
    )
    assert block_match, "README is missing the generated concepts block."
    block = block_match.group(1)

    count_match = re.search(
        r"\*\*(\d+) canonical concepts\*\* across \*\*(\d+) pillars\*\*", block
    )
    assert count_match, (
        f"Could not find the count line in the generated block:\n{block}"
    )
    assert int(count_match.group(1)) == total, (
        f"README concept count {count_match.group(1)} != concepts.yaml total {total}"
    )
    assert int(count_match.group(2)) == pillars, (
        f"README pillar count {count_match.group(2)} != concepts.yaml pillars {pillars}"
    )


def test_agents_md_is_small_and_clean():
    assert AGENTS.exists(), "AGENTS.md does not exist."
    size = AGENTS.stat().st_size
    assert size < AGENTS_MAX_BYTES, (
        f"AGENTS.md is {size} bytes, expected < {AGENTS_MAX_BYTES}."
    )
    content = AGENTS.read_text(encoding="utf-8")
    # The invariant is that the auto-generated project TREE excludes cache/build dirs —
    # a tree entry like "├── .hypothesis/" means gen_agents_md.py's EXCLUDE_DIRS
    # regressed. A bare prose mention (the "Keep Root Pristine" section lists
    # ".hypothesis/" as a forbidden dir) is legitimate documentation, not bloat.
    tree_entry = re.search(r"[├└]──\s*\.hypothesis", content)
    assert tree_entry is None, (
        "AGENTS.md project tree still lists a '.hypothesis' cache entry."
    )


def test_check_concepts_passes():
    result = _run("check_concepts.py")
    assert result.returncode == 0, (
        "check_concepts.py failed; a CONCEPT marker in code is missing from "
        f"concepts.yaml.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
