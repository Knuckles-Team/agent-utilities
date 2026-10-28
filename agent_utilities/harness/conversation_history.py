"""Validate host-authorized completed text messages for a fresh harness run."""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


def child_history_envelope(content: str | None, binding: Any, *, run_id: str) -> str:
    """Validate the host's explicit child selection before ordinary hydration.

    The host owns authorization; this checks that the selected payload matches
    its independently supplied binding and contains only completed text turns.
    """
    from agent_utilities.host_ports import CompletedChildBinding

    try:
        if not isinstance(binding, CompletedChildBinding):
            raise ValueError
        fields = (
            "child_id",
            "conversation_id",
            "predecessor_run_id",
            "history_revision",
        )
        if any(
            type(getattr(binding, key)) is not str
            or not getattr(binding, key).strip()
            or len(getattr(binding, key).encode()) > 512
            for key in fields
        ):
            raise ValueError
        if not isinstance(content, str) or len(content.encode()) > 1_048_576:
            raise ValueError
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {
            "version",
            "phase",
            "messages",
            *fields,
        }:
            raise ValueError
        if (
            type(value["version"]) is not int
            or value["version"] != 1
            or value["phase"] != "completed"
        ):
            raise ValueError
        if any(value[key] != getattr(binding, key) for key in fields):
            raise ValueError
        normalized = json.dumps(
            {
                "version": 1,
                "phase": "completed",
                "run_id": binding.predecessor_run_id,
                "messages": value["messages"],
            }
        )
        decode_completed_history(normalized, run_id=run_id)
        return normalized
    except Exception:
        raise ValueError("completed child history is invalid") from None


def decode_completed_history(content: str, *, run_id: str) -> list[ModelMessage]:
    """Decode only completed text turns, never budgets or execution authority."""
    try:
        if not isinstance(content, str) or len(content.encode()) > 1_048_576:
            raise ValueError
        value = json.loads(content)
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "phase", "run_id", "messages"}
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["phase"] != "completed"
            or not isinstance(value["run_id"], str)
            or not value["run_id"].strip()
            or value["run_id"] == run_id
            or not isinstance(value["messages"], list)
            or not 2 <= len(value["messages"]) <= 1024
            or len(value["messages"]) % 2
        ):
            raise ValueError
        messages: list[ModelMessage] = []
        for index, message in enumerate(value["messages"]):
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "text"}
                or message["role"] != ("assistant" if index % 2 else "user")
                or not isinstance(message["text"], str)
            ):
                raise ValueError
            if index % 2:
                messages.append(ModelResponse(parts=[TextPart(message["text"])]))
            else:
                messages.append(ModelRequest(parts=[UserPromptPart(message["text"])]))
        return messages
    except Exception:
        raise ValueError("completed conversation history is invalid") from None
