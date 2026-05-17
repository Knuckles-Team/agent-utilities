#!/usr/bin/python
"""Multi-IDE Conversation Log Ingestion.

CONCEPT:KG-2.1 — Cross-IDE Chat Memory

Parses conversation logs from external IDE/agent platforms and ingests
them into the Knowledge Graph as Thread/Message nodes. Supports:
- Antigravity: ~/.gemini/antigravity/brain/*/overview.txt
- Windsurf:    ~/.codeium/windsurf/memories/ or ~/.windsurf/memories/
- Claude Code: ~/.claude/projects/ or ~/.config/claude/
- Codex:       ~/.codex/sessions/
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# IDE-specific log directory patterns
IDE_LOG_PATHS: dict[str, list[str]] = {
    "antigravity": [
        "~/.gemini/antigravity/brain",
    ],
    "windsurf": [
        "~/.codeium/windsurf/memories",
        "~/.windsurf/memories",
    ],
    "claude": [
        "~/.claude/projects",
        "~/.config/claude",
    ],
    "codex": [
        "~/.codex/sessions",
    ],
}


def _resolve_paths(ide: str) -> list[Path]:
    """Resolve IDE log paths, returning only those that exist."""
    patterns = IDE_LOG_PATHS.get(ide, [])
    resolved = []
    for p in patterns:
        expanded = Path(os.path.expanduser(p))
        if expanded.exists():
            resolved.append(expanded)
    return resolved


def parse_antigravity_logs(brain_dir: Path) -> list[dict[str, Any]]:
    """Parse Antigravity conversation logs from brain directory.

    Each conversation is in a subdirectory with an overview.txt file
    containing the full transcript.
    """
    conversations: list[dict[str, Any]] = []
    if not brain_dir.exists():
        return conversations

    # Each subdirectory is a conversation
    for conv_dir in sorted(brain_dir.iterdir()):
        if not conv_dir.is_dir():
            continue

        overview_path = conv_dir / ".system_generated" / "logs" / "overview.txt"
        if not overview_path.exists():
            # Try direct overview.txt
            overview_path = conv_dir / "overview.txt"
        if not overview_path.exists():
            continue

        try:
            content = overview_path.read_text(errors="replace")
            if not content.strip():
                continue

            conv_id = conv_dir.name
            # Extract timestamp from directory name or file mtime
            try:
                mtime = overview_path.stat().st_mtime
                timestamp = datetime.fromtimestamp(mtime).isoformat()
            except Exception:
                timestamp = datetime.now().isoformat()

            # Parse messages from overview.txt
            messages = _parse_overview_messages(content)

            # Extract title from first user message, fallback to first assistant content
            title = "Untitled"
            for msg in messages:
                if msg["role"] == "user" and len(msg["content"]) > 5:
                    title = msg["content"][:100]
                    break
            if title == "Untitled":
                for msg in messages:
                    if msg["role"] == "assistant" and not msg["content"].startswith(
                        "[Tool"
                    ):
                        title = msg["content"][:100]
                        break

            conversations.append(
                {
                    "id": f"antigravity:{conv_id}",
                    "source": "antigravity",
                    "title": title,
                    "timestamp": timestamp,
                    "messages": messages,
                    "path": str(overview_path),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to parse Antigravity log {conv_dir}: {e}")

    return conversations


def _parse_overview_messages(content: str) -> list[dict[str, Any]]:
    """Parse messages from an Antigravity overview.txt file.

    Format is JSONL — each line is a JSON object with fields:
    - step_index: int
    - source: "MODEL" | "USER" | "TOOL"
    - type: e.g. "PLANNER_RESPONSE"
    - content: str (may be null)
    - tool_calls: list (may be null)
    """
    messages = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            source = obj.get("source", "")
            msg_content = obj.get("content", "")

            # Map source to chat role
            if source in ("USER", "USER_EXPLICIT", "USER_IMPLICIT"):
                role = "user"
            elif source == "MODEL":
                role = "assistant"
            elif source in ("TOOL", "TOOL_RESULT"):
                role = "tool"
            else:
                continue

            # Skip entries with no content
            if not msg_content:
                # If there are tool calls, summarize them
                tool_calls = obj.get("tool_calls", [])
                if tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                    msg_content = f"[Tool calls: {', '.join(tool_names)}]"
                else:
                    continue

            messages.append(
                {
                    "role": role,
                    "content": str(msg_content)[:5000],  # Cap message size
                }
            )
        except json.JSONDecodeError:
            # Fall back to plain text role detection
            stripped = line.strip()
            if stripped.startswith("USER:") or stripped.startswith("Human:"):
                remainder = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                if remainder:
                    messages.append({"role": "user", "content": remainder})
            elif stripped.startswith("ASSISTANT:") or stripped.startswith("Model:"):
                remainder = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                if remainder:
                    messages.append({"role": "assistant", "content": remainder})

    return messages


def parse_windsurf_logs(memories_dir: Path) -> list[dict[str, Any]]:
    """Parse Windsurf conversation logs from memories directory."""
    conversations: list[dict[str, Any]] = []
    if not memories_dir.exists():
        return conversations

    for f in sorted(memories_dir.glob("**/*.json")):
        try:
            data = json.loads(f.read_text(errors="replace"))
            if isinstance(data, dict):
                conv_id = f.stem
                messages = []
                for msg in data.get("messages", []):
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": str(msg.get("content", "")),
                        }
                    )
                if messages:
                    conversations.append(
                        {
                            "id": f"windsurf:{conv_id}",
                            "source": "windsurf",
                            "title": str(
                                data.get("title", messages[0]["content"][:100])
                            ),
                            "timestamp": str(
                                data.get("timestamp", datetime.now().isoformat())
                            ),
                            "messages": messages,
                            "path": str(f),
                        }
                    )
        except Exception as e:
            logger.debug(f"Failed to parse Windsurf log {f}: {e}")

    return conversations


def parse_claude_logs(projects_dir: Path) -> list[dict[str, Any]]:
    """Parse Claude Code conversation logs."""
    conversations: list[dict[str, Any]] = []
    if not projects_dir.exists():
        return conversations

    # Claude stores conversations as JSONL files
    for f in sorted(projects_dir.glob("**/*.jsonl")):
        try:
            messages = []
            for line in f.read_text(errors="replace").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": str(msg.get("content", "")),
                        }
                    )
                except json.JSONDecodeError:
                    continue

            if messages:
                conv_id = f.stem
                conversations.append(
                    {
                        "id": f"claude:{conv_id}",
                        "source": "claude",
                        "title": messages[0]["content"][:100]
                        if messages
                        else "Untitled",
                        "timestamp": datetime.fromtimestamp(
                            f.stat().st_mtime
                        ).isoformat(),
                        "messages": messages,
                        "path": str(f),
                    }
                )
        except Exception as e:
            logger.debug(f"Failed to parse Claude log {f}: {e}")

    return conversations


def parse_codex_logs(sessions_dir: Path) -> list[dict[str, Any]]:
    """Parse Codex session logs."""
    conversations: list[dict[str, Any]] = []
    if not sessions_dir.exists():
        return conversations

    for f in sorted(sessions_dir.glob("**/*.json")):
        try:
            data = json.loads(f.read_text(errors="replace"))
            if isinstance(data, dict):
                messages = []
                for msg in (
                    data.get("messages", None) or data.get("history", None) or []
                ):
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": str(msg.get("content", msg.get("text", ""))),
                        }
                    )
                if messages:
                    conv_id = f.stem
                    conversations.append(
                        {
                            "id": f"codex:{conv_id}",
                            "source": "codex",
                            "title": messages[0]["content"][:100]
                            if messages
                            else "Untitled",
                            "timestamp": datetime.fromtimestamp(
                                f.stat().st_mtime
                            ).isoformat(),
                            "messages": messages,
                            "path": str(f),
                        }
                    )
        except Exception as e:
            logger.debug(f"Failed to parse Codex log {f}: {e}")

    return conversations


# Parser registry
_PARSERS = {
    "antigravity": parse_antigravity_logs,
    "windsurf": parse_windsurf_logs,
    "claude": parse_claude_logs,
    "codex": parse_codex_logs,
}


def discover_all_conversations(
    ides: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Discover conversations from all supported IDEs.

    Args:
        ides: Optional list of IDEs to scan. Defaults to all.

    Returns:
        List of conversation dicts ready for KG ingestion.
    """
    target_ides = ides or list(IDE_LOG_PATHS.keys())
    all_conversations = []

    for ide in target_ides:
        parser = _PARSERS.get(ide)
        if not parser:
            logger.warning(f"No parser for IDE: {ide}")
            continue

        paths = _resolve_paths(ide)
        for path in paths:
            logger.info(f"Scanning {ide} logs at {path}")
            conversations = parser(path)
            all_conversations.extend(conversations)
            logger.info(f"  Found {len(conversations)} conversations from {ide}")

    return all_conversations


def ingest_conversations_to_kg(
    conversations: list[dict[str, Any]] | None = None,
    ides: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest external IDE conversations into the Knowledge Graph.

    Creates Thread and Message nodes with proper provenance linking.

    Args:
        conversations: Pre-parsed conversations. If None, auto-discovers.
        ides: List of IDEs to scan (only used if conversations is None).

    Returns:
        Summary dict with counts per IDE.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return {"error": "KG engine not available"}

    if conversations is None:
        conversations = discover_all_conversations(ides)

    summary: dict[str, int] = {}
    ingested = 0

    for conv in conversations:
        source = conv.get("source", "unknown")
        conv_id = conv["id"]
        messages = conv.get("messages", [])

        if not messages:
            continue

        try:
            partition = f"partition:{source}"

            # Create Thread node via engine API (backend-safe)
            engine.add_node(
                node_id=conv_id,
                node_type="Thread",
                properties={
                    "title": conv.get("title", "Untitled"),
                    "source": source,
                    "partition": partition,
                    "timestamp": conv.get("timestamp", datetime.now().isoformat()),
                    "valid_from": conv.get("timestamp", datetime.now().isoformat()),
                    "path": conv.get("path", ""),
                    "message_count": len(messages),
                },
            )

            # Create Message nodes (cap at 50 per thread to manage DB size)
            max_msgs = min(len(messages), 50)
            for i, msg in enumerate(messages[:max_msgs]):
                msg_id = f"msg:{conv_id}:{i}"
                content = msg.get("content", "")
                # Truncate very long messages
                if len(content) > 5000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-2000:]

                engine.add_node(
                    node_id=msg_id,
                    node_type="Message",
                    properties={
                        "role": msg.get("role", "user"),
                        "content": content,
                        "timestamp": conv.get("timestamp", ""),
                        "source": source,
                        "partition": partition,
                    },
                )

                # Link Message to Thread
                engine.link_nodes(
                    source_id=conv_id,
                    target_id=msg_id,
                    rel_type="CONTAINS",
                    properties={"source": source},
                )

            summary[source] = summary.get(source, 0) + 1
            ingested += 1

        except Exception as e:
            logger.warning(f"Failed to ingest conversation {conv_id}: {e}")

    return {
        "total_ingested": ingested,
        "per_ide": summary,
        "total_messages": sum(len(c.get("messages", [])) for c in conversations),
    }
