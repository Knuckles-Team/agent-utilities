"""Tests for the AG-UI wire format emitter.

Validates that :class:`AGUIGraphEmitter` correctly translates graph
execution events from :func:`run_graph_iter` into the AG-UI wire
protocol format (``0:``, ``2:``, ``8:``, ``9:`` line prefixes).

CONCEPT:AU-002 Graph Orchestration
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_utilities.agui_emitter import AGUIGraphEmitter


@pytest.fixture
def emitter():
    return AGUIGraphEmitter()


class TestTextDelta:
    """Validate ``2:`` prefix text streaming."""

    def test_simple_text(self, emitter):
        chunks = emitter.format_text_delta("Hello world")
        assert len(chunks) == 2
        # First chunk is the text delta
        assert chunks[0].startswith(b"2:")
        decoded = json.loads(chunks[0].decode("utf-8")[2:].strip())
        assert decoded == "Hello world"
        # Second chunk is a heartbeat
        assert chunks[1] == b'0 " "\n'

    def test_text_with_special_chars(self, emitter):
        chunks = emitter.format_text_delta('He said "hello" & <goodbye>')
        text_line = chunks[0].decode("utf-8")
        assert text_line.startswith("2:")
        decoded = json.loads(text_line[2:].strip())
        assert decoded == 'He said "hello" & <goodbye>'


class TestToolCall:
    """Validate ``9:`` prefix tool call formatting."""

    def test_basic_tool_call(self, emitter):
        chunks = emitter.format_tool_call("router", {"query": "test"})
        assert len(chunks) == 2
        assert chunks[0].startswith(b"9:")
        payload = json.loads(chunks[0].decode("utf-8")[2:].strip())
        assert payload["node_id"] == "router"
        assert payload["inputs"]["query"] == "test"


class TestHeartbeat:
    """Validate ``0:`` heartbeat format."""

    def test_heartbeat_format(self, emitter):
        hb = emitter.format_heartbeat()
        assert hb == b'0 " "\n'


class TestTranslate:
    """Validate event translation dispatch."""

    def test_node_transition_event(self, emitter):
        event = {
            "type": "node_transition",
            "step": 1,
            "active_nodes": [{"node_id": "router", "task_id": "t:0"}],
            "state_snapshot": {"routed_domain": "git"},
        }
        chunks = emitter.translate(event)
        assert len(chunks) >= 1
        first_line = chunks[0].decode("utf-8")
        assert first_line.startswith("8:")
        payload = json.loads(first_line[2:].strip())
        assert payload["type"] == "graph_node_transition"
        assert payload["step"] == 1
        assert payload["active_nodes"][0]["node_id"] == "router"

    def test_sideband_event(self, emitter):
        event = {
            "type": "sideband",
            "event": {"type": "graph_start", "run_id": "abc"},
        }
        chunks = emitter.translate(event)
        first_line = chunks[0].decode("utf-8")
        assert first_line.startswith("8:")
        payload = json.loads(first_line[2:].strip())
        assert payload["type"] == "graph_start"
        assert payload["run_id"] == "abc"

    def test_graph_complete_event(self, emitter):
        event = {
            "type": "graph_complete",
            "run_id": "xyz",
            "output": "Final answer",
            "state_snapshot": {"routed_domain": "code"},
        }
        chunks = emitter.translate(event)
        # Should have text delta + heartbeat + sideband annotation + heartbeat
        assert len(chunks) >= 3
        # First chunk should be a text delta
        text_line = chunks[0].decode("utf-8")
        assert text_line.startswith("2:")
        decoded = json.loads(text_line[2:].strip())
        assert decoded == "Final answer"
        # Should also contain a sideband with graph_complete
        sideband_chunks = [
            c for c in chunks if c.decode("utf-8").startswith("8:")
        ]
        assert len(sideband_chunks) >= 1
        sb_payload = json.loads(sideband_chunks[0].decode("utf-8")[2:].strip())
        assert sb_payload["type"] == "graph_complete"

    def test_error_event(self, emitter):
        event = {
            "type": "error",
            "run_id": "err-1",
            "error": "Something went wrong",
        }
        chunks = emitter.translate(event)
        first_line = chunks[0].decode("utf-8")
        assert first_line.startswith("8:")
        payload = json.loads(first_line[2:].strip())
        assert payload["type"] == "error"
        assert payload["error"] == "Something went wrong"

    def test_elicitation_event(self, emitter):
        event = {
            "type": "elicitation",
            "reason": "human_approval_required",
            "state_snapshot": {"mode": "plan"},
        }
        chunks = emitter.translate(event)
        first_line = chunks[0].decode("utf-8")
        assert first_line.startswith("8:")
        payload = json.loads(first_line[2:].strip())
        assert payload["type"] == "elicitation_request"
        assert payload["reason"] == "human_approval_required"

    def test_unknown_event_emitted_as_sideband(self, emitter):
        event = {"type": "custom_event", "data": "anything"}
        chunks = emitter.translate(event)
        first_line = chunks[0].decode("utf-8")
        assert first_line.startswith("8:")

    def test_dict_output_extraction(self, emitter):
        event = {
            "type": "graph_complete",
            "run_id": "x",
            "output": {"results": {"output": "nested result"}},
            "state_snapshot": {},
        }
        chunks = emitter.translate(event)
        text_line = chunks[0].decode("utf-8")
        decoded = json.loads(text_line[2:].strip())
        assert decoded == "nested result"

    def test_none_output_produces_no_text(self, emitter):
        event: dict[str, Any] = {
            "type": "graph_complete",
            "run_id": "x",
            "output": None,
            "state_snapshot": {},
        }
        chunks = emitter.translate(event)
        # Should only have sideband, no text delta
        text_chunks = [
            c for c in chunks if c.decode("utf-8").startswith("2:")
        ]
        assert len(text_chunks) == 0
