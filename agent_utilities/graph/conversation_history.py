"""Decode host-authorized prior model messages for a fresh execution turn."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessagesTypeAdapter

from agent_utilities import host_ports

from .checkpoint import CheckpointUnavailable, _Snapshot
from .state import RunRequest


async def read_conversation_history(request: RunRequest) -> list[Any]:
    """Import messages only, never old turn counters, instructions or effects."""
    value = await host_ports.read_run_history(request)
    if value is None:
        return []
    if not isinstance(value, Mapping) or set(value) != {"run_id", "content"}:
        raise CheckpointUnavailable("conversation history projection is invalid")
    run_id, content = value["run_id"], value["content"]
    if not isinstance(run_id, str) or not run_id or run_id == request.run_id:
        raise CheckpointUnavailable("conversation history run binding is invalid")
    if not isinstance(content, str) or len(content.encode()) > 1_048_576:
        raise CheckpointUnavailable("conversation history content is invalid")
    try:
        saved = _Snapshot.model_validate_json(content)
        messages = ModelMessagesTypeAdapter.validate_python(saved.messages)
    except ValidationError as exc:
        raise CheckpointUnavailable("conversation history content is invalid") from exc
    if (
        saved.run_id != run_id
        or saved.phase != "completed"
        or saved.status != "completed"
        or saved.next_step != len(saved.outputs)
        or not saved.next_step
        or not messages
    ):
        raise CheckpointUnavailable("conversation history is not a completed turn")
    return list(messages)
