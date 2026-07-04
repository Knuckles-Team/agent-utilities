#!/usr/bin/python
"""Tests for LCM convergence guarantee + hierarchical summary-DAG recovery (b1-05).

CONCEPT:AU-KG.memory.mementified-context
"""

from unittest.mock import MagicMock

import pytest

import agent_utilities.knowledge_graph.memory.memento_compressor as mc
from agent_utilities.knowledge_graph.memory.memento_compressor import (
    _guarantee_shorter,
    compress_to_memento,
    link_parent_memento,
    recover_chain,
)

pytestmark = pytest.mark.concept("AU-KG.memory.mementified-context")


# --- convergence guarantee --------------------------------------------------


def test_guarantee_shorter_truncates_oversized():
    block = "x" * 1000
    memento = "y" * 2000  # compression "failed" — bigger than the block
    out = _guarantee_shorter(memento, block, max_ratio=0.9)
    assert len(out) < len(block)
    assert out.endswith("…[truncated:recoverable]")


def test_guarantee_shorter_leaves_small_memento_untouched():
    block = "x" * 1000
    memento = "small summary"
    assert _guarantee_shorter(memento, block) == memento


def test_compress_to_memento_guarantees_reduction(monkeypatch):
    # LLM returns a memento LONGER than the block; judge accepts it.
    monkeypatch.setattr(mc, "_memento_llm", lambda *a, **k: "y" * 5000)
    monkeypatch.setattr(mc, "judge_memento", lambda *a, **k: (9, ""))
    block_content = "x" * 1000
    out = compress_to_memento(
        MagicMock(), [{"role": "user", "content": block_content}], dry_run=True
    )
    # output is guaranteed smaller than the rendered block despite the LLM's bloat
    assert out is not None
    assert len(out) < len(block_content)


# --- hierarchical summary-DAG recovery -------------------------------------


def test_recover_chain_walks_multi_level_dag():
    engine = MagicMock()
    engine.backend = MagicMock()

    def _execute(cypher, params=None):
        pid = (params or {}).get("id")
        return {
            "m1": [{"id": "m2", "content": "mid-level summary"}],
            "m2": [{"id": "b1", "content": "RAW BLOCK"}],
        }.get(pid, [])

    engine.backend.execute.side_effect = _execute
    assert recover_chain(engine, "m1") == "RAW BLOCK"  # deepest leaf content


def test_recover_chain_no_backend_returns_none():
    engine = MagicMock()
    engine.backend = None
    assert recover_chain(engine, "m1") is None


# --- DAG level builder ------------------------------------------------------


def test_link_parent_memento_creates_summarizes_edges():
    engine = MagicMock()
    n = link_parent_memento(engine, "parent", ["c1", "c2"])
    assert n == 2
    calls = [c.args for c in engine.link_nodes.call_args_list]
    assert ("parent", "c1", "SUMMARIZES") in calls
    assert ("parent", "c2", "SUMMARIZES") in calls
