"""Antigravity IDE session parser (CONCEPT:ECO-4.38).

Google's Antigravity IDE keeps each session under ``~/.gemini/antigravity/``::

    conversations/<uuid>.pb        per-session transcript (AES-encrypted protobuf)
    brain/<uuid>/<artifact>.md     plaintext task / plan / walkthrough artifacts
    brain/<uuid>/<artifact>.md.metadata.json   {artifactType, summary, updatedAt}
    annotations/<uuid>.pbtxt       last_user_view_time + archived flag

The canonical transcript (``conversations/<uuid>.pb``) is **AES-encrypted on
disk** (CTR/CBC/GCM with a key Antigravity never writes to disk — see
agentsview ``internal/parser/antigravity_crypto.go``); decrypting it needs an
``ANTIGRAVITY_KEY`` and an undocumented, version-fragile protobuf schema, so the
full turn-by-turn transcript stays a documented follow-up (REMAINING).

What we ingest **today, with zero key/crypto dependency**, is the per-session
**brain artifacts** — the plaintext markdown documents Antigravity writes for
every session (the task checklist, the implementation plan, the code
walkthrough, analyses, …). They are the high-signal, already-decrypted record of
what each session was about, so each artifact becomes a :class:`UsageMessage` and
the whole session is tagged ``agent="antigravity"``. The ``conversations/<uuid>.pb``
file is the discovery anchor (one per session) and drives the mtime/size
skip-cache; the brain dir beside it supplies the content.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent_utilities.usage.models import (
    ParsedSessionBundle,
    UsageMessage,
    UsageSession,
)

# Brain artifacts whose content represents the human's intent (rendered as the
# user turn); every other artifact is an agent-produced document.
_USER_ARTIFACTS = {"task.md", "ARTIFACT_TYPE_TASK"}


def _session_root(path: Path) -> Path:
    """The Antigravity base dir given a ``conversations/<uuid>.pb`` path."""
    # path = <base>/conversations/<uuid>.pb  ->  <base>
    return path.parent.parent


def _read_metadata(md_path: Path) -> dict[str, Any]:
    """Best-effort read of the ``<artifact>.md.metadata.json`` sidecar."""
    sidecar = md_path.with_name(md_path.name + ".metadata.json")
    try:
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _iter_brain_artifacts(brain_dir: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(md_path, metadata)`` for each top-level brain artifact.

    Only the canonical ``*.md`` artifacts are read — the ``.resolved`` /
    ``.resolved.N`` snapshots and ``.metadata.json`` sidecars beside them are
    skipped (``*.md`` matches none of those suffixes).
    """
    if not brain_dir.is_dir():
        return
    for md_path in sorted(brain_dir.glob("*.md")):
        if not md_path.is_file():
            continue
        yield md_path, _read_metadata(md_path)


def parse(path: Path, source) -> Iterator[ParsedSessionBundle]:
    session_id = path.stem
    base = _session_root(path)
    brain_dir = base / "brain" / session_id

    artifacts = list(_iter_brain_artifacts(brain_dir))
    if not artifacts:
        # No plaintext content for this session (encrypted .pb only) — yield
        # nothing rather than a contentless session, matching the framework's
        # "detected but not yet parseable" behaviour for opaque stores.
        return

    # Order artifacts by their metadata ``updatedAt`` so the transcript reads
    # chronologically; missing timestamps sort last but keep a stable order.
    def _sort_key(item: tuple[Path, dict[str, Any]]) -> tuple[str, str]:
        md_path, meta = item
        return (str(meta.get("updatedAt") or "~"), md_path.name)

    artifacts.sort(key=_sort_key)

    messages: list[UsageMessage] = []
    started_at: str | None = None
    ended_at: str | None = None
    first_message = ""
    project = ""

    for ordinal, (md_path, meta) in enumerate(artifacts):
        try:
            content = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not content.strip():
            continue
        ts = meta.get("updatedAt")
        artifact_type = str(meta.get("artifactType") or "")
        is_user = md_path.name in _USER_ARTIFACTS or artifact_type in _USER_ARTIFACTS
        role = "user" if is_user else "assistant"
        if ts:
            started_at = started_at or str(ts)
            ended_at = str(ts)
        if role == "user" and not first_message:
            first_message = (meta.get("summary") or content)[:300]
        messages.append(
            UsageMessage(
                session_id=session_id,
                ordinal=ordinal,
                role=role,
                content=content,
                timestamp=(str(ts) if ts else None),
                content_length=len(content),
            )
        )

    if not messages:
        return

    if not first_message:
        # No explicit task artifact — seed from the first artifact's summary/body.
        first_message = messages[0].content[:300]

    user_count = sum(1 for m in messages if m.role == "user")
    yield ParsedSessionBundle(
        session=UsageSession(
            id=session_id,
            project=project or source.agent_type,
            agent=source.agent_type,
            first_message=first_message,
            started_at=started_at,
            ended_at=ended_at,
            message_count=len(messages),
            user_message_count=user_count,
            file_path=str(path),
        ),
        messages=messages,
        tool_calls=[],
        usage_events=[],
    )
