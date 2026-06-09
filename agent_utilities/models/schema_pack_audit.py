#!/usr/bin/python
from __future__ import annotations

"""Schema-Pack candidate-type audit (CONCEPT:KG-2.35).

When an EXCLUSIVE pack is active and a write introduces a node/edge type *outside*
the active set, that type is recorded as a *candidate* — a type the brain is
encountering but the active pack does not (yet) model. This mirrors gbrain's
``candidate-audit.ts`` and feeds a ``gbrain schema review-candidates``-style flow
(exposed here via ``graph_configure(action="schema_candidates")``).

The auditor is **observe-only**: it must never reject or block a write. It is also
**privacy-first** — by default raw type names are stored as a salted-free SHA-256
prefix plus a 4-character slug so a leaked audit log does not reveal a deployment's
private domain vocabulary. Set ``GRAPH_SCHEMA_AUDIT_VERBOSE=1`` to store raw names.
"""


import hashlib
import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

CandidateKind = Literal["node", "edge"]


def _redact(type_name: str) -> str:
    """Return a privacy-preserving token for a type name (hash + short prefix)."""
    digest = hashlib.sha256(type_name.encode("utf-8")).hexdigest()[:12]
    prefix = type_name[:4]
    return f"{prefix}…:{digest}"


def _audit_path() -> Path:
    """Resolve the JSONL audit file under the XDG state dir (created on demand)."""
    override = os.environ.get("GRAPH_SCHEMA_AUDIT_DIR")
    if override:
        base = Path(override).expanduser()
    else:
        try:
            import platformdirs

            base = Path(
                platformdirs.user_state_path("agent-utilities", "knuckles-team")
            )
        except Exception:  # pragma: no cover - platformdirs always present in deps
            base = Path.home() / ".local" / "state" / "agent-utilities"
    base.mkdir(parents=True, exist_ok=True)
    return base / "schema_candidates.jsonl"


class SchemaCandidateAuditor:
    """Process-singleton recorder of out-of-pack candidate types (KG-2.35).

    In-memory de-duplication bounds write volume: a given
    ``(kind, type_name, pack)`` triple is appended at most once per process.
    """

    _instance: SchemaCandidateAuditor | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._seen: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> SchemaCandidateAuditor:
        """Return the shared process-wide auditor."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def _verbose() -> bool:
        return os.environ.get("GRAPH_SCHEMA_AUDIT_VERBOSE", "") not in (
            "",
            "0",
            "false",
        )

    def record(self, kind: CandidateKind, type_name: str, pack_name: str) -> bool:
        """Record an out-of-pack candidate type. Returns True if newly written.

        Best-effort and exception-safe: any I/O failure is swallowed (the write
        path must never be impacted by auditing).
        """
        key = (kind, type_name, pack_name)
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "kind": kind,
            "pack": pack_name,
            "type": type_name if self._verbose() else _redact(type_name),
            "redacted": not self._verbose(),
        }
        try:
            with open(_audit_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as e:  # pragma: no cover - never block the write path
            logger.debug("schema candidate audit write failed: %s", e)
            return False
        return True

    def review(self, limit: int = 100) -> list[dict]:
        """Return the most recent candidate records (newest last), up to ``limit``."""
        path = _audit_path()
        if not path.exists():
            return []
        out: list[dict] = []
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        except Exception as e:  # pragma: no cover - defensive read
            logger.debug("schema candidate audit read failed: %s", e)
        return out

    def reset(self) -> None:
        """Clear the in-memory de-dup set (test hook)."""
        with self._lock:
            self._seen.clear()


__all__ = ["SchemaCandidateAuditor", "CandidateKind"]
