"""AU-QUAL-R005.2.1: tests/unit/patterns/ is carved back into mypy coverage.

The first per-directory grandchild slice of AU-QUAL-R005.2 (itself a child of
AU-QUAL-R005 -- see specs/au-boundary-quality/tasks.md for the recorded
split). This asserts the mypy pre-commit hook's `exclude` regex no longer
matches tests/unit/patterns/, while every other test directory stays
excluded until its own `.2.n` slice lands.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRE_COMMIT_CONFIG = _REPO_ROOT / ".config" / "pre-commit.yaml"


def _mypy_exclude_pattern() -> str:
    config = yaml.safe_load(_PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    for repo in config["repos"]:
        if "mirrors-mypy" not in repo["repo"]:
            continue
        for hook in repo["hooks"]:
            if hook["id"] == "mypy":
                return str(hook["exclude"])
    raise AssertionError("no mypy hook found in .config/pre-commit.yaml")


@pytest.mark.spec("AU-QUAL-R005.2.1")
def test_mypy_config_includes_tests_unit_patterns() -> None:
    pattern = re.compile(_mypy_exclude_pattern())
    assert not pattern.search("tests/unit/patterns/test_prioritization.py"), (
        "AU-QUAL-R005.2.1: tests/unit/patterns/ must be included in mypy coverage"
    )


@pytest.mark.spec("AU-QUAL-R005.2.1")
def test_mypy_config_still_excludes_other_test_directories() -> None:
    """Only the delivered slice is carved back in -- everything else in
    tests/ remains excluded until its own `.2.n` slice lands."""
    pattern = re.compile(_mypy_exclude_pattern())
    assert pattern.search("tests/unit/core/test_something.py")
    assert pattern.search("tests/unit/knowledge_graph/test_something.py")
