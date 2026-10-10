"""Gate: the local analytics-worker entry point is gone.

Binds AU-SEMANTIC-R011.10 -- `agent_utilities/knowledge_graph/analytics_worker.py`
is removed in favor of the EG-served durable-job path. This asserts the
module file is absent, no console-script/entry-point in pyproject.toml still
names it, and no importer under agent_utilities/ or tests/ references its
dotted module path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    _REPO_ROOT / "agent_utilities" / "knowledge_graph" / "analytics_worker.py"
)
_DOTTED_MODULE = "agent_utilities.knowledge_graph.analytics_worker"
_IMPORT_PATTERN = re.compile(
    r"(^|[\s(])"
    r"(import\s+agent_utilities\.knowledge_graph\.analytics_worker"
    r"|from\s+agent_utilities\.knowledge_graph\s+import\s+analytics_worker"
    r"|from\s+agent_utilities\.knowledge_graph\.analytics_worker\s+import)",
)

_SEARCH_DIRS = ("agent_utilities", "tests", "scripts")
_SELF_PATH = Path(__file__).resolve()


@pytest.mark.spec("AU-SEMANTIC-R011.10")
def test_analytics_worker_module_is_absent() -> None:
    assert not _MODULE_PATH.exists(), (
        "AU-SEMANTIC-R011.10 requires "
        "'agent_utilities/knowledge_graph/analytics_worker.py' to be deleted, "
        "but it still exists."
    )


@pytest.mark.spec("AU-SEMANTIC-R011.10")
def test_no_importer_references_analytics_worker() -> None:
    offenders: list[str] = []
    for search_dir in _SEARCH_DIRS:
        base = _REPO_ROOT / search_dir
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.resolve() == _SELF_PATH:
                continue
            text = path.read_text(encoding="utf-8")
            if _IMPORT_PATTERN.search(text):
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], (
        f"No importer may reference {_DOTTED_MODULE!r} once "
        "AU-SEMANTIC-R011.10 deletes it, but found references in: "
        f"{offenders}"
    )


@pytest.mark.spec("AU-SEMANTIC-R011.10")
def test_no_pyproject_entry_point_names_analytics_worker() -> None:
    pyproject_text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert _DOTTED_MODULE not in pyproject_text, (
        "pyproject.toml must not declare a console-script/entry-point that "
        f"names the deleted module {_DOTTED_MODULE!r}."
    )
