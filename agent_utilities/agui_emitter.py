#!/usr/bin/python
"""AG-UI Wire Format Emitter Module.

This module translates graph execution events from :func:`run_graph_iter`
into the AG-UI wire protocol format used by the Agent UI frontend. The
wire protocol uses numbered line prefixes:

* ``0:``  — Text content / heartbeat
* ``2:``  — Text delta streaming chunks
* ``8:``  — Sideband annotation data (graph events, tool calls)
* ``9:``  — Tool call start/progress information

CONCEPT:AU-002 Graph Orchestration
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AGUIGraphEmitter:
    """Translates graph execution events to AG-UI wire format.

    This emitter converts the structured event dictionaries yielded by
    :func:`~agent_utilities.graph.runner.run_graph_iter` into byte-encoded
    AG-UI wire format lines suitable for ``StreamingResponse``.

    The AG-UI protocol uses specific line prefixes:

    - ``0:`` for text content and heartbeats
    - ``2:`` for streaming text deltas (partial tokens)
    - ``8:`` for sideband annotations (graph events, metadata)
    - ``9:`` for tool call information

    Usage::

        emitter = AGUIGraphEmitter()
        async for event in run_graph_iter(graph, config, query):
            for chunk in emitter.translate(event):
                yield chunk

    """

    def translate(self, event: dict[str, Any]) -> list[bytes]:
        """Translate a single graph event into AG-UI wire format chunks.

        Args:
            event: A structured event dictionary from :func:`run_graph_iter`.

        Returns:
            A list of byte-encoded wire format lines to send to the client.
            May return multiple chunks (e.g., a sideband annotation followed
            by a heartbeat for flush).

        """
        event_type = event.get("type", "")
        handler = getattr(self, f"_translate_{event_type}", None)
        if handler:
            return handler(event)
        # Unknown event type — emit as generic sideband
        return self._format_sideband(event)

    def _translate_node_transition(self, event: dict[str, Any]) -> list[bytes]:
        """Emit a sideband annotation for a node transition."""
        annotation = {
            "type": "graph_node_transition",
            "step": event.get("step"),
            "active_nodes": event.get("active_nodes", []),
            "state": event.get("state_snapshot", {}),
        }
        return self._format_sideband(annotation)

    def _translate_sideband(self, event: dict[str, Any]) -> list[bytes]:
        """Forward a raw sideband event from the graph event queue."""
        inner = event.get("event", event)
        return self._format_sideband(inner)

    def _translate_elicitation(self, event: dict[str, Any]) -> list[bytes]:
        """Emit an elicitation request as a sideband annotation."""
        annotation = {
            "type": "elicitation_request",
            "reason": event.get("reason"),
            "state": event.get("state_snapshot", {}),
        }
        return self._format_sideband(annotation)

    def _translate_graph_complete(self, event: dict[str, Any]) -> list[bytes]:
        """Emit the final graph output as a text content chunk."""
        output = event.get("output")
        chunks: list[bytes] = []

        # Extract text from GraphResponse-like outputs
        text = self._extract_output_text(output)
        if text:
            chunks.extend(self.format_text_delta(text))

        # Also emit a completion sideband annotation
        annotation = {
            "type": "graph_complete",
            "run_id": event.get("run_id"),
            "state": event.get("state_snapshot", {}),
        }
        chunks.extend(self._format_sideband(annotation))
        return chunks

    def _translate_error(self, event: dict[str, Any]) -> list[bytes]:
        """Emit an error as a sideband annotation."""
        annotation = {
            "type": "error",
            "error": event.get("error", "Unknown error"),
            "run_id": event.get("run_id"),
        }
        return self._format_sideband(annotation)

    # -- Formatting Primitives --

    def format_text_delta(self, text: str) -> list[bytes]:
        """Format text as AG-UI streaming text delta chunks.

        Args:
            text: The text content to stream.

        Returns:
            A list of byte-encoded ``2:`` prefixed lines.

        """
        # AG-UI text deltas use the ``2:`` prefix with JSON-encoded content
        encoded = json.dumps(text)
        return [
            f"2:{encoded}\n".encode(),
            self.format_heartbeat(),
        ]

    def format_tool_call(self, node_id: str, inputs: dict[str, Any]) -> list[bytes]:
        """Format a tool/node call as AG-UI tool call information.

        Args:
            node_id: The ID of the graph node being executed.
            inputs: The input data for the node.

        Returns:
            A list of byte-encoded ``9:`` prefixed lines.

        """
        payload = json.dumps({"node_id": node_id, "inputs": inputs})
        return [
            f"9:{payload}\n".encode(),
            self.format_heartbeat(),
        ]

    def format_heartbeat(self) -> bytes:
        """Format a heartbeat/flush marker.

        Returns:
            A byte-encoded ``0:`` heartbeat line.

        """
        return b'0 " "\n'

    # -- Internal Helpers --

    def _format_sideband(self, annotation: dict[str, Any]) -> list[bytes]:
        """Format a sideband annotation as an ``8:`` prefixed line.

        Args:
            annotation: The annotation data to encode.

        Returns:
            A list containing the annotation line and a heartbeat for flush.

        """
        payload = json.dumps(annotation)
        return [
            f"8:{payload}\n".encode(),
            self.format_heartbeat(),
        ]

    @staticmethod
    def _extract_output_text(output: Any) -> str:
        """Extract display text from various graph output types.

        Handles ``GraphResponse`` objects, plain dicts, and string results.

        Args:
            output: The raw output from the graph execution.

        Returns:
            A string representation of the output.

        """
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        # GraphResponse-like object with .results dict
        if hasattr(output, "results"):
            results = output.results
            if isinstance(results, dict):
                return results.get("output", str(results))
            return str(results)
        # Plain dict (from .model_dump())
        if isinstance(output, dict):
            results = output.get("results", {})
            if isinstance(results, dict):
                return results.get("output", str(results))
            return str(output)
        return str(output)
