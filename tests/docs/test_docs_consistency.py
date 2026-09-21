"""Documentation consistency tests.

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

AGENTS_MAX_LINES = 240
AGENTS_HEADINGS = (
    "What this repository owns",
    "Architecture and module map",
    "Commands",
    "Quality gates",
    "Development rules",
    "Documentation",
    "Branching & isolation",
)


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def _h2_headings(content: str) -> tuple[str, ...]:
    return tuple(
        line.removeprefix("## ")
        for line in content.splitlines()
        if line.startswith("## ")
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
    content = AGENTS.read_text(encoding="utf-8")
    assert len(content.splitlines()) <= AGENTS_MAX_LINES
    assert _h2_headings(content) == AGENTS_HEADINGS
    assert content == (ROOT / "AGENTS.head.md").read_text(encoding="utf-8")


def test_check_concepts_passes():
    result = _run("check_concepts.py")
    assert result.returncode == 0, (
        "check_concepts.py failed; a CONCEPT marker in code is missing from "
        f"concepts.yaml.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
