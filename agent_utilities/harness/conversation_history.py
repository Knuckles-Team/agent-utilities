"""Validate host-authorized completed text messages for a fresh harness run."""

from __future__ import annotations

import json

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


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
