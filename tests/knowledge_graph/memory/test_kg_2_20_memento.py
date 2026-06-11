"""CONCEPT:KG-2.20 — Mementified Context Management (MEM-0..4).

Covers the judge-refine loop, semantic-boundary segmentation, lossless recoverability, and the
live block-compress-evict capability (the Wire-First *_live_path test exercises the capability's
transform on real pydantic-ai ModelMessage objects, not just the helpers in isolation).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.memory import (
    boundary_score,
    compress_to_memento,
    judge_memento,
    plan_block_eviction,
    segment_into_blocks,
)
from agent_utilities.knowledge_graph.memory import memento_compressor as mc

# ── MEM-2: judge-refine loop ─────────────────────────────────────────────────────


def test_judge_parses_score_and_feedback(monkeypatch):
    monkeypatch.setattr(
        mc, "_memento_llm", lambda s, u: "SCORE: 6\nFEEDBACK: missing formula b+7 | 56"
    )
    score, fb = judge_memento("block", "weak memento")
    assert score == 6 and "b+7" in fb


def test_judge_accepts_when_no_llm(monkeypatch):
    # No model available -> degrade to accept (single-shot), never block compression.
    monkeypatch.setattr(mc, "_memento_llm", lambda s, u: None)
    score, fb = judge_memento("block", "memento")
    assert score >= mc.MEMENTO_ACCEPT_THRESHOLD


def test_compress_refines_until_threshold(monkeypatch):
    calls = {"n": 0}

    def fake(system, user):
        calls["n"] += 1
        if "judge" in system.lower() or "Memento judge" in system:
            # fail the first judge, then pass -> forces exactly one refine pass
            return (
                "SCORE: 5\nFEEDBACK: missing value x=5"
                if calls["n"] < 4
                else "SCORE: 9\nFEEDBACK: OK"
            )
        return "STATE: x=5; next=verify"

    monkeypatch.setattr(mc, "_memento_llm", fake)
    # Long enough that the F4 shrink-guarantee never truncates the memento.
    out = compress_to_memento(
        None,  # type: ignore[arg-type]
        [{"role": "user", "content": "do x, then verify it and record state " * 2}],
        dry_run=True,
    )
    assert out == "STATE: x=5; next=verify"
    # 1 compress + (judge-fail + recompress) + judge-pass = 4 calls
    assert calls["n"] == 4


# ── MEM-3: semantic-boundary segmentation ────────────────────────────────────────


def test_boundary_score_never_cuts_mid_calculation():
    prev = {"role": "assistant", "content": "compute b+7 ="}
    nxt = {"role": "assistant", "content": "56, so b=49"}
    assert boundary_score(prev, nxt) == 0.0  # ends with '=', continuation 'so'


def test_boundary_score_high_at_action_observation_cycle():
    tool = {"role": "tool", "content": "result=ok", "tool_call_id": "1"}
    asst = {"role": "assistant", "content": "Now the next step"}
    assert boundary_score(tool, asst) >= 2.0


def test_segment_respects_min_block_and_no_tiny_dangler():
    msgs = [
        {"role": "assistant", "content": "x " * 60},
        {"role": "tool", "content": "y " * 40},
    ]
    msgs = [{"role": "system", "content": "s " * 5}] + msgs * 6
    blocks = segment_into_blocks(msgs, min_block_tokens=200)
    assert len(blocks) >= 2
    # every index assigned exactly once, contiguous, no empty blocks
    flat = [i for b in blocks for i in b]
    assert flat == list(range(len(msgs)))


# ── MEM-4: lossless eviction planning + recoverability ───────────────────────────


def test_plan_eviction_preserves_head_and_recent_and_is_minimal():
    msgs = [{"role": "system", "content": "sys " * 5}]
    for k in range(8):
        msgs.append({"role": "assistant", "content": f"reason {k} " * 60})
        msgs.append(
            {"role": "tool", "content": f"obs {k} " * 40, "tool_call_id": str(k)}
        )
    ev, kept = plan_block_eviction(
        msgs, budget_tokens=800, keep_recent_blocks=1, keep_head=1
    )
    assert ev, "should evict to fit budget"
    assert 0 in kept, "system-prompt head must be preserved"
    assert any(i >= len(msgs) - 2 for i in kept), "recent block must be preserved"


def test_persist_memento_is_lossless_and_recoverable():
    engine = MagicMock()
    engine.backend = MagicMock()
    mid = mc._persist_memento(
        engine, "MEMENTO: state", source="t", raw_block="the full raw block text"
    )
    assert mid is not None
    # a Memento node + an EvictedBlock node + a SUMMARIZES edge were written
    node_labels = [c.args[1] for c in engine.add_node.call_args_list]
    assert "Memento" in node_labels and "EvictedBlock" in node_labels
    engine.link_nodes.assert_called_once()
    assert engine.link_nodes.call_args.args[2] == "SUMMARIZES"

    # recovery follows the pointer back to the raw block
    engine.backend.execute.return_value = [{"content": "the full raw block text"}]
    assert mc.recover_evicted_block(engine, mid) == "the full raw block text"


# ── MEM-1: live capability transform (Wire-First *_live_path) ─────────────────────


def test_memento_capability_live_path_evicts_and_inserts_memento(monkeypatch):
    """Exercise the capability the way the agent loop does: hand it a real ModelMessage list and
    assert it evicts old blocks, inserts a memento, preserves the head, and cuts tokens."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ToolReturnPart,
    )

    from agent_utilities.capabilities import MementoCompaction
    from agent_utilities.capabilities.memento import _message_to_dict
    from agent_utilities.knowledge_graph.memory.agent_context import (
        estimate_message_tokens,
    )

    monkeypatch.setattr(mc, "_memento_llm", lambda s, u: "MEMENTO: compressed state")

    msgs = [ModelRequest(parts=[SystemPromptPart(content="system prompt " * 5)])]
    for k in range(8):
        msgs.append(ModelResponse(parts=[TextPart(content=f"reasoning {k} " * 60)]))
        msgs.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="t", content=f"obs {k} " * 40, tool_call_id=str(k)
                    )
                ]
            )
        )

    cap = MementoCompaction(
        max_tokens=1000, keep_recent_blocks=1, keep_head=1, source="t"
    )
    new_msgs, n_evicted = cap.mementoize_messages(msgs, budget_tokens=800, engine=None)

    before = estimate_message_tokens([_message_to_dict(m) for m in msgs])
    after = estimate_message_tokens([_message_to_dict(m) for m in new_msgs])
    assert n_evicted > 0
    assert after < before * 0.6  # >40% reduction (success metric)
    # the head (system prompt) survives, and a memento was injected
    assert any(
        getattr(p, "content", "").startswith("system prompt") for p in new_msgs[0].parts
    )
    assert any(
        isinstance(m, ModelRequest)
        and any(
            getattr(p, "content", "").startswith("PRIOR CONTEXT MEMENTO")
            for p in m.parts
        )
        for m in new_msgs
    )


def test_memento_capability_noop_under_budget():
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from agent_utilities.capabilities import MementoCompaction

    cap = MementoCompaction(max_tokens=100_000)
    msgs = [ModelRequest(parts=[UserPromptPart(content="hi")])] * 5
    out, n = cap.mementoize_messages(msgs, budget_tokens=100_000, engine=None)
    assert n == 0 and out is msgs


def test_memento_capability_registered_in_factory_default_on():
    """Wire-First: the capability must be wired into the factory with integration ON by default."""
    import inspect

    from agent_utilities.agent import factory

    sig = inspect.signature(factory.create_agent)
    assert sig.parameters["memento_compaction"].default is True
    src = inspect.getsource(factory)
    assert "MementoCompaction(" in src  # actually appended, not just imported
