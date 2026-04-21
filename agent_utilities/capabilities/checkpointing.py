#!/usr/bin/python
"""Conversation checkpointing capability with graph persistence.

Allows snapshotting conversation state, persisting it to the knowledge graph,
and rewinding or forking from specific points.
"""

from __future__ import annotations

import abc
import builtins
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage

from ..models.knowledge_graph import CheckpointNode, RegistryNodeType

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    id: str
    label: str
    turn: int
    messages: list[ModelMessage]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        # Simplistic serialization for messages
        # In a real impl, we'd use pydantic-ai's message serialization
        data = {
            "id": self.id,
            "label": self.label,
            "turn": self.turn,
            "messages": [
                m.model_dump() if hasattr(m, "model_dump") else str(m)
                for m in self.messages
            ],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data_str: str) -> Checkpoint:
        data = json.loads(data_str)
        # Note: Deserializing ModelMessages requires care (using pydantic-ai types)
        # This is a stub for the storage layer
        return cls(**data)


class CheckpointStore(abc.ABC):
    @abc.abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None:
        pass

    @abc.abstractmethod
    async def get(self, checkpoint_id: str) -> Checkpoint | None:
        pass

    @abc.abstractmethod
    async def list(self, limit: int = 10) -> builtins.list[Checkpoint]:
        pass


class InMemoryCheckpointStore(CheckpointStore):
    def __init__(self):
        self._checkpoints: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        self._checkpoints[checkpoint.id] = checkpoint

    async def get(self, checkpoint_id: str) -> Checkpoint | None:
        return self._checkpoints.get(checkpoint_id)

    async def list(self, limit: int = 10) -> builtins.list[Checkpoint]:
        return sorted(
            self._checkpoints.values(), key=lambda x: x.timestamp, reverse=True
        )[:limit]


class GraphCheckpointStore(CheckpointStore):
    """Persists checkpoints to the Knowledge Graph as CheckpointNodes."""

    def __init__(self, engine: Any):
        self.engine = engine

    async def save(self, checkpoint: Checkpoint) -> None:
        node = CheckpointNode(
            id=checkpoint.id,
            type=RegistryNodeType.CHECKPOINT,
            name=f"Checkpoint: {checkpoint.label}",
            label=checkpoint.label,
            turn=checkpoint.turn,
            message_count=len(checkpoint.messages),
            message_data=checkpoint.to_json(),
            importance_score=0.7,
            timestamp=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(checkpoint.timestamp)
            ),
            metadata=checkpoint.metadata,
        )
        try:
            self.engine.graph.add_node(node.id, **node.model_dump())
            if self.engine.backend:
                await self.engine.backend.upsert_node(
                    RegistryNodeType.CHECKPOINT, node.id, node.model_dump()
                )

            # Link to episode
            episode_id = checkpoint.metadata.get("episode_id")
            if episode_id:
                self.engine.graph.add_edge(node.id, episode_id, type="SNAPSHOT_OF")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to graph: {e}")

    async def get(self, checkpoint_id: str) -> Checkpoint | None:
        try:
            if self.engine.backend:
                data = await self.engine.backend.get_node(
                    RegistryNodeType.CHECKPOINT, checkpoint_id
                )
                if data:
                    return Checkpoint.from_json(data["message_data"])

            if checkpoint_id in self.engine.graph:
                data = self.engine.graph.nodes[checkpoint_id]
                return Checkpoint.from_json(data["message_data"])
        except Exception as e:
            logger.error(f"Failed to get checkpoint from graph: {e}")
        return None

    async def list(self, limit: int = 10) -> builtins.list[Checkpoint]:
        # Implementation would use Cypher or graph iteration
        return []


class FileCheckpointStore(CheckpointStore):
    """Persists checkpoints to local files."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    async def save(self, checkpoint: Checkpoint) -> None:
        path = self.directory / f"{checkpoint.id}.json"
        path.write_text(checkpoint.to_json(), encoding="utf-8")

    async def get(self, checkpoint_id: str) -> Checkpoint | None:
        path = self.directory / f"{checkpoint_id}.json"
        if path.exists():
            return Checkpoint.from_json(path.read_text(encoding="utf-8"))
        return None

    async def list(self, limit: int = 10) -> builtins.list[Checkpoint]:
        files = sorted(
            self.directory.glob("*.json"), key=os.path.getmtime, reverse=True
        )[:limit]
        return [Checkpoint.from_json(f.read_text(encoding="utf-8")) for f in files]


class CheckpointToolset:
    """A toolset that exposes checkpointing operations to the agent."""

    def __init__(self, store: CheckpointStore):
        self.store = store

    async def create_checkpoint(self, ctx: RunContext[Any], label: str) -> str:
        """Manually create a checkpoint of the current state."""
        # This would be integrated with the middleware but for now it's a stub
        return "Checkpoint created."

    async def list_checkpoints(self, ctx: RunContext[Any]) -> str:
        """List available checkpoints for rewinding."""
        ckpts = await self.store.list()
        return "\n".join([f"- {c.id}: {c.label} ({c.turn} turns)" for c in ckpts])

    async def rewind(self, ctx: RunContext[Any], checkpoint_id: str) -> None:
        """Rewind the conversation to a specific checkpoint."""
        raise RewindRequested(checkpoint_id)


class RewindRequested(Exception):
    """Raised to trigger a conversation rewind."""

    def __init__(self, checkpoint_id: str):
        self.checkpoint_id = checkpoint_id


@dataclass
class CheckpointMiddleware(AbstractCapability[Any]):
    """Capability that automatically saves checkpoints during a run."""

    store: CheckpointStore
    frequency: Literal["every_tool", "every_turn", "manual_only"] = "every_tool"

    _current_turn: int = 0

    async def after_tool_execute(self, ctx: RunContext[Any], **kwargs) -> Any:
        call = kwargs.get("call")
        tool_name = call.tool_name if call else "unknown"
        if self.frequency == "every_tool":
            await self._checkpoint(ctx, label=f"After tool: {tool_name}")
        return kwargs.get("result")

    async def _checkpoint(self, ctx: RunContext[Any], label: str) -> str:
        checkpoint_id = f"ckpt_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        messages = ctx.messages

        cp = Checkpoint(
            id=checkpoint_id,
            label=label,
            turn=self._current_turn,
            messages=messages,
            metadata={"episode_id": getattr(ctx.deps, "episode_id", None)},
        )
        await self.store.save(cp)
        return checkpoint_id


async def fork_from_checkpoint(
    agent: Agent, checkpoint: Checkpoint, user_input: str
) -> Any:
    """Start a new run from a specific checkpoint."""
    return await agent.run(user_input, message_history=checkpoint.messages)
