#!/usr/bin/python

from __future__ import annotations

import sys
import json
import logging
import asyncio


from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    pass


from pydantic_ai import Agent



from .config import *
from .workspace import *


from .models import PeriodicTask

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()




logger = logging.getLogger(__name__)


def save_chat_to_disk(chat_id: str, messages: List[Dict[str, Any]]):
    """Save a chat conversation to a JSON file in the chats directory."""
    chats_dir = get_workspace_path(CORE_FILES["CHATS"])
    chats_dir.mkdir(parents=True, exist_ok=True)
    path = chats_dir / f"{chat_id}.json"

    chat_data = {
        "id": chat_id,
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
    }

    path.write_text(json.dumps(chat_data, indent=2), encoding="utf-8")
    logger.debug(f"Saved chat {chat_id} to disk")


def list_chats_from_disk() -> List[Dict[str, Any]]:
    """List all chats stored in the workspace."""
    chats_dir = get_workspace_path(CORE_FILES["CHATS"])
    if not chats_dir.exists():
        return []

    chats = []
    for f in chats_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            message_text = ""
            if data.get("messages") and len(data["messages"]) > 0:
                first_msg = data["messages"][0]
                if isinstance(first_msg.get("content"), str):
                    message_text = first_msg["content"]
                elif isinstance(first_msg.get("content"), list):

                    message_text = str(first_msg["content"][0])

            chats.append(
                {
                    "id": data.get("id", f.stem),
                    "firstMessage": message_text[:100],
                    "timestamp": data.get(
                        "timestamp",
                        datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    ),
                }
            )
        except Exception as e:
            logger.debug(f"Error loading chat file {f}: {e}")

    return sorted(chats, key=lambda x: x["timestamp"], reverse=True)


def get_chat_from_disk(chat_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific chat from disk."""
    path = get_workspace_path(CORE_FILES["CHATS"]) / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error reading chat {chat_id}: {e}")
    return None


def delete_chat_from_disk(chat_id: str) -> bool:
    """Delete a chat file from workspace."""
    path = get_workspace_path(CORE_FILES["CHATS"]) / f"{chat_id}.json"
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {e}")
    return False


def prune_large_messages(messages: list[Any], max_length: int = 5000) -> list[Any]:
    """
    Summarize large tool outputs in the message history to save context window.
    Keeps the most recent tool outputs intact if they are the very last message,
    but generally we want to prune history.
    """
    pruned_messages = []
    for i, msg in enumerate(messages):
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if isinstance(content, str) and len(content) > max_length:
            summary = (
                f"{content[:200]} ... "
                f"[Output truncated, original length {len(content)} characters] "
                f"... {content[-200:]}"
            )

            if isinstance(msg, dict):
                msg["content"] = summary
                pruned_messages.append(msg)
            elif hasattr(msg, "content"):
                try:
                    from copy import copy

                    new_msg = copy(msg)
                    new_msg.content = summary
                    pruned_messages.append(new_msg)
                except Exception:
                    pruned_messages.append(msg)
            else:
                pruned_messages.append(msg)
        else:
            pruned_messages.append(msg)

    return pruned_messages


async def chat(agent: Agent, prompt: str):
    result = await agent.run(prompt)
    print(f"Response:\n\n{result.output}", file=sys.stderr)


async def node_chat(agent: Agent, prompt: str) -> List:
    nodes = []
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            nodes.append(node)
            print(node, file=sys.stderr)
    return nodes


async def stream_chat(agent: Agent, prompt: str) -> None:
    async with agent.run_stream(prompt) as result:
        async for text_chunk in result.stream_text(delta=True):
            print(text_chunk, end="", flush=True, file=sys.stderr)
        print("\nDone!", file=sys.stderr)
