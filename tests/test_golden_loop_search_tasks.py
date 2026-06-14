#!/usr/bin/python
"""Live-path test: golden-loop self-play search-task synthesis (KG-2.70/2.71/2.72)."""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.research.golden_loop import GoldenLoopController


class _FakeBackend:
    def execute(self, cypher, params=None):  # noqa: ARG002
        params = params or {}
        if "id" in params:  # neighbor query
            if params["id"] != "Q-target":
                return []
            return [
                {
                    "a": {"id": "Q-target", "name": "Ada Botanist"},
                    "r": {"type": "DESCRIBED", "source": "docA"},
                    "b": {"id": "Q-fern", "name": "Athyrium kenyae", "source": "docA"},
                },
                {
                    "a": {"id": "Q-target", "name": "Ada Botanist"},
                    "r": {"type": "ADVISED_BY", "source": "docB"},
                    "b": {"id": "Q-mentor", "name": "Dr. Mentor", "source": "docB"},
                },
            ]
        # candidate-entity query
        return [{"n": {"id": "Q-target", "name": "Ada Botanist"}}]


class _FakeEngine:
    def __init__(self) -> None:
        self.backend = _FakeBackend()
        self.added: list[tuple[str, dict]] = []

    def add_node(self, node_id, props=None):
        self.added.append((node_id, props or {}))


def test_golden_loop_search_task_stage(tmp_path):
    engine = _FakeEngine()
    controller = GoldenLoopController(
        engine, codebase_root=str(tmp_path), propose_only=True
    )
    report = controller.run_one_cycle(
        synthesize_search=True,
        synthesize=False,
        assimilate=False,
        breadth=False,
        standardize=False,
        distill=False,
    )

    st = report["search_tasks"]
    assert st is not None, report["errors"]
    assert st["tasks"] >= 1
    assert st["persisted_nodes"] >= 1

    # a SearchTask proposal node was persisted, answer never leaked into the question
    assert engine.added and engine.added[0][0].startswith("SearchTask:")
    assert "Ada Botanist" not in engine.added[0][1]["question"]

    # propose-only corpus draft written under .specify/
    corpus = tmp_path / ".specify" / "specs" / "search-tasks" / "tasks.jsonl"
    assert corpus.exists()
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line]
    assert rows and rows[0]["risk_report"]["clear"] is True
