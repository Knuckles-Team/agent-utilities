"""Shared helpers for JSONL-family session parsers (CONCEPT:ECO-4.38)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, skipping malformed lines."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def text_from_content(content: Any) -> tuple[str, str, bool]:
    """Flatten an Anthropic-style content list to ``(text, thinking, has_tool_use)``.

    Handles plain strings and lists of typed blocks (text/thinking/tool_use).
    """
    if isinstance(content, str):
        return content, "", False
    if not isinstance(content, list):
        return "", "", False
    parts: list[str] = []
    thinking: list[str] = []
    has_tool = False
    for block in content:
        if not isinstance(block, dict):
            if isinstance(block, str):
                parts.append(block)
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "thinking":
            thinking.append(str(block.get("thinking", "")))
        elif btype == "tool_use":
            has_tool = True
        elif btype == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                parts.append(inner)
            elif isinstance(inner, list):
                for ib in inner:
                    if isinstance(ib, dict) and ib.get("type") == "text":
                        parts.append(str(ib.get("text", "")))
    return "\n".join(parts), "\n".join(thinking), has_tool


def tool_uses_from_content(content: Any) -> list[dict[str, Any]]:
    """Extract tool_use blocks (name, id, input) from a content list."""
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            out.append(
                {
                    "name": block.get("name", ""),
                    "id": block.get("id", ""),
                    "input": block.get("input", {}),
                }
            )
    return out


def categorize_tool(name: str) -> str:
    """Map a tool name to a coarse category (read|edit|bash|search|other)."""
    n = (name or "").lower()
    if n in {"read", "notebookread"}:
        return "read"
    if n in {"edit", "write", "multiedit", "notebookedit"}:
        return "edit"
    if n in {"bash", "shell", "run"}:
        return "bash"
    if n in {"grep", "glob", "search", "websearch", "webfetch"}:
        return "search"
    if "skill" in n or n.startswith("/"):
        return "skill"
    return "other"
