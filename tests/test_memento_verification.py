#!/usr/bin/python
"""Tests for the external-verification + provenance stamp on memento compaction.

CONCEPT:KG-2.20
"""

from unittest.mock import MagicMock

import pytest

import agent_utilities.knowledge_graph.memory.memento_compressor as mc
from agent_utilities.knowledge_graph.memory.memento_compressor import (
    _persist_memento,
    compress_to_memento,
    verify_memento,
)

pytestmark = pytest.mark.concept("KG-2.20")

_BLOCK = "The capital of France is Paris. The Eiffel Tower is in Paris."


# --- verify_memento (deterministic, external) ------------------------------


def test_verify_memento_faithful():
    v = verify_memento(_BLOCK, "The capital of France is Paris.")
    assert v["verified"] is True
    assert v["faithful_ratio"] == 1.0
    assert v["ungrounded"] == 0
    assert v["verifier"] == "faithfulness"


def test_verify_memento_hallucinated():
    v = verify_memento(
        _BLOCK, "The moon is made entirely of green cheese and stocks crashed."
    )
    assert v["verified"] is False
    assert v["faithful_ratio"] < 1.0
    assert v["ungrounded"] >= 1


# --- provenance stamp on persistence ---------------------------------------


def _capture_engine():
    engine = MagicMock()
    engine.backend = object()  # truthy → _persist_memento proceeds
    return engine


def test_persist_memento_stamps_provenance():
    engine = _capture_engine()
    verdict = {
        "verified": True,
        "faithful_ratio": 0.95,
        "ungrounded": 0,
        "verifier": "faithfulness",
    }
    mem_id = _persist_memento(engine, "dense memento", verification=verdict)
    assert mem_id
    # First add_node call is the Memento node; inspect its properties.
    memento_call = next(
        c for c in engine.add_node.call_args_list if c.args[1] == "Memento"
    )
    props = memento_call.kwargs["properties"]
    assert props["provenance_verified"] is True
    assert props["provenance_faithfulness"] == 0.95
    assert props["provenance_verifier"] == "faithfulness"


def test_persist_memento_without_verification_has_no_stamp():
    engine = _capture_engine()
    _persist_memento(engine, "dense memento")
    memento_call = next(
        c for c in engine.add_node.call_args_list if c.args[1] == "Memento"
    )
    props = memento_call.kwargs["properties"]
    assert "provenance_verified" not in props


# --- live path: compress_to_memento runs the gate --------------------------


def test_compress_to_memento_runs_external_gate(monkeypatch):
    # Drive the compressor deterministically without any LLM.
    monkeypatch.setattr(
        mc, "_memento_llm", lambda *a, **k: "The capital of France is Paris."
    )
    monkeypatch.setattr(mc, "judge_memento", lambda *a, **k: (9, ""))

    captured = {}

    def fake_persist(
        engine, memento_text, *, source="", raw_block=None, verification=None
    ):
        captured["verification"] = verification
        return "mem_x"

    monkeypatch.setattr(mc, "_persist_memento", fake_persist)

    out = compress_to_memento(
        MagicMock(), [{"role": "user", "content": _BLOCK}], refine=True
    )
    assert out == "The capital of France is Paris."
    # The external verification verdict was computed and handed to persistence.
    assert captured["verification"] is not None
    assert captured["verification"]["verified"] is True
    assert captured["verification"]["verifier"] == "faithfulness"
