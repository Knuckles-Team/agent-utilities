"""R020: a run envelope never reaches the user as raw JSON."""

from __future__ import annotations

import json

from agent_utilities.messaging.router import _unwrap_agent_envelope
from agent_utilities.orchestration.agent_runner import _render_agent_result
from agent_utilities.orchestration.run_envelope import unwrap_run_envelope

_LEAKED = {
    "output": "Based on the provided evidence, the answer is 42.",
    "run_id": "run:ce25",
    "channel_id": "orch:messaging:telegram:123",
    "run_summary": {
        "route": {"agent": "messaging-assistant"},
        "outcome": "ok",
        "stage_reached": "multi-agent-graph",
        "trace_ref": "trace:pref_run_ce25",
        "execution_mode": "direct_completion",
        "provenance_recorded": True,
    },
    "provenance_recorded": True,
}


def test_exact_leaked_envelope_string_sends_only_output() -> None:
    text, summary = _unwrap_agent_envelope(json.dumps(_LEAKED))
    assert text == "Based on the provided evidence, the answer is 42."
    assert summary is not None and summary["trace_ref"] == "trace:pref_run_ce25"


def test_envelope_dict_sends_only_output() -> None:
    text, summary = unwrap_run_envelope(_LEAKED)
    assert text == _LEAKED["output"]
    assert summary == _LEAKED["run_summary"]


def test_every_renderer_key_unwraps() -> None:
    rendered = _render_agent_result(
        "hello",
        run_id="run:1",
        return_mermaid=True,
        mermaid="graph TD",
        channel_id="c",
        run_summary={"outcome": "ok"},
        execution_evidence={"model_ref": "m"},
        provenance_recorded=False,
    )
    assert unwrap_run_envelope(rendered) == ("hello", {"outcome": "ok"})


def test_genuine_json_reply_passes_through() -> None:
    reply = json.dumps({"output": "x", "extra": 1})
    assert unwrap_run_envelope(reply) == (reply, None)
    assert unwrap_run_envelope("plain text") == ("plain text", None)
    assert unwrap_run_envelope(None) == ("", None)
