"""Regression tests for the repo-node ``owl:sameAs`` crosswalk resolver
(``agent_utilities.observability.repo_crosswalk``) — reconciles the URL-keyed
Portainer ``:Repository`` with the numeric-id code-ingestor repo nodes
(``reports/autonomous-sdlc-loop-design.md`` §4.3). Mirrors
``test_observability_incidents.py``'s fake-KG style.
"""

from __future__ import annotations

from typing import Any

import agent_utilities.knowledge_graph.memory.native_ingest as native_ingest
import agent_utilities.observability.health_ingest as hi
from agent_utilities.observability import repo_crosswalk as rc


class _Capture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, entities, relationships=None, *, source, domain, **kw):
        self.calls.append(
            {
                "entities": entities,
                "relationships": relationships or [],
                "source": source,
            }
        )
        return {"nodes": len(entities), "edges": len(relationships or [])}


class _FakeEngine:
    def __init__(self, by_label: dict[str, list[tuple[str, dict]]]) -> None:
        self._by_label = by_label

    def get_nodes_by_label(self, label: str, limit: int = 0):
        return self._by_label.get(label, [])


# --- normalize_clone_url -------------------------------------------------- #
def test_normalize_collapses_url_variants_to_one_key():
    key = "github.com/knuckles-team/agent-utilities"
    assert (
        rc.normalize_clone_url("https://github.com/Knuckles-Team/agent-utilities.git")
        == key
    )
    assert (
        rc.normalize_clone_url("git@github.com:Knuckles-Team/agent-utilities.git")
        == key
    )
    assert (
        rc.normalize_clone_url("https://www.github.com/Knuckles-Team/agent-utilities/")
        == key
    )
    assert (
        rc.normalize_clone_url("http://github.com/Knuckles-Team/agent-utilities?tab=x")
        == key
    )


def test_normalize_empty_is_blank():
    assert rc.normalize_clone_url("") == ""
    assert rc.normalize_clone_url(None) == ""  # type: ignore[arg-type]


# --- resolve_crosswalk ---------------------------------------------------- #
def test_crosswalk_unifies_url_repo_with_numeric_id_repo():
    """A Portainer URL-keyed :Repository and a GitHub numeric-id :Repository that
    share a normalized clone URL are reconciled — canonical = the numeric-id node
    (where the code graph lives), the URL node an alias."""
    rows = {
        "Repository": [
            (
                "git:repo:github.com/knuckles-team/agent-utilities",
                {"url": "https://github.com/Knuckles-Team/agent-utilities"},
            ),
            (
                "github:repository:42",
                {
                    "htmlUrl": "https://github.com/Knuckles-Team/agent-utilities",
                    "fullName": "Knuckles-Team/agent-utilities",
                },
            ),
        ],
        "Project": [],
    }
    out = rc.resolve_crosswalk(engine=_FakeEngine(rows))

    assert len(out) == 1
    cw = out[0]
    assert cw["url"] == "github.com/knuckles-team/agent-utilities"
    assert cw["canonical"] == "github:repository:42"  # numeric-id node wins
    assert cw["aliases"] == ["git:repo:github.com/knuckles-team/agent-utilities"]


def test_crosswalk_unifies_across_gitlab_project_label():
    rows = {
        "Repository": [
            ("git:repo:gitlab.com/grp/svc", {"url": "https://gitlab.com/grp/svc"}),
        ],
        "Project": [
            (
                "gitlab:project:7",
                {
                    "web_url": "https://gitlab.com/grp/svc",
                    "path_with_namespace": "grp/svc",
                },
            ),
        ],
    }
    out = rc.resolve_crosswalk(engine=_FakeEngine(rows))
    assert len(out) == 1
    assert out[0]["canonical"] == "gitlab:project:7"


def test_crosswalk_ignores_single_or_same_namespace_nodes():
    """One node for a URL, or two nodes from the SAME producer, are not
    cross-linked (nothing to reconcile)."""
    rows = {
        "Repository": [
            ("github:repository:1", {"htmlUrl": "https://github.com/a/only"}),
            ("github:repository:2", {"htmlUrl": "https://github.com/a/dup"}),
            ("git:repo:github.com/a/dup", {"url": "https://github.com/a/dup"}),
        ],
        "Project": [],
    }
    out = rc.resolve_crosswalk(engine=_FakeEngine(rows))
    # only a/dup has two DIFFERENT namespaces -> exactly one crosswalk
    assert [c["url"] for c in out] == ["github.com/a/dup"]


def test_run_repo_crosswalk_writes_samesas_and_aliasof_edges(monkeypatch):
    rows = {
        "Repository": [
            ("git:repo:github.com/o/n", {"url": "https://github.com/o/n"}),
            ("github:repository:9", {"htmlUrl": "https://github.com/o/n"}),
        ],
        "Project": [],
    }
    monkeypatch.setattr(hi, "_engine", lambda: _FakeEngine(rows))
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    out = rc.run_repo_crosswalk(write=True)

    assert out["reconciled"] == 1
    assert out["aliases"] == 1
    assert len(cap.calls) == 1
    rel_types = {r["type"] for r in cap.calls[0]["relationships"]}
    assert rel_types == {"aliasOf", "sameAs"}
    # aliasOf points alias -> canonical (the numeric-id node)
    alias_edge = next(
        r for r in cap.calls[0]["relationships"] if r["type"] == "aliasOf"
    )
    assert alias_edge["source"] == "git:repo:github.com/o/n"
    assert alias_edge["target"] == "github:repository:9"


def test_run_repo_crosswalk_no_engine_is_noop(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    assert rc.resolve_crosswalk() == []
    assert rc.run_repo_crosswalk()["reconciled"] == 0
