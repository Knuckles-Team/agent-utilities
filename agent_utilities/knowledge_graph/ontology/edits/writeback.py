#!/usr/bin/python
from __future__ import annotations

"""Edit write-back — pushing committed edits to a source datasource.

Provenance (Palantir Foundry doc: *object-edits/overview*): edits captured on an
object can be **written back to the source datasource** the object was built
from, so the system of record reflects user changes. This module provides the
write-back leg: a registerable :class:`EditSink` interface plus a concrete
durable, append-only **JSONL sink keyed by object type** (one ``<type>.jsonl``
file per object type), and a :class:`WriteBackRouter` that fans committed edits
out to the registered sinks.

CONCEPT:KG-2.43
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .ledger import Edit

logger = logging.getLogger(__name__)

__all__ = [
    "EditSink",
    "JsonlEditSink",
    "WriteBackRouter",
    "object_type_of",
]


def object_type_of(edit: Edit) -> str:
    """Derive the source object-type key an edit writes back under.

    Prefers an explicit ``type`` on the edit's after/before snapshot; falls back
    to the id namespace (``paper:001`` → ``paper``); else ``object``.
    """
    for snap in (edit.after, edit.before):
        t = snap.get("type") if isinstance(snap, dict) else None
        if isinstance(t, str) and t:
            return t
    oid = edit.object_id or ""
    if ":" in oid:
        return oid.split(":", 1)[0]
    return "object"


@runtime_checkable
class EditSink(Protocol):
    """A registerable write-back target for committed edits.

    Implementations push an :class:`Edit` to a source datasource (a file, a REST
    API, a relational table, a message bus). ``write`` must be idempotent-safe
    for replay where the underlying datasource allows it.
    """

    def write(self, edit: Edit) -> bool:
        """Persist one edit to the sink. Returns True on success."""
        ...

    def flush(self) -> None:
        """Flush any buffered writes to durable storage."""
        ...


class JsonlEditSink:
    """Durable append-only JSONL sink, one file per object type. CONCEPT:KG-2.43.

    Each committed edit is serialized as a single JSON line appended to
    ``<root>/<object_type>.jsonl``. Append-only + per-type partitioning mirrors
    Foundry's "edits datasource": an immutable change log per object type that a
    downstream sync can tail back into the system of record.

    Args:
        root: Directory under which per-type ``.jsonl`` files are written.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, object_type: str) -> Path:
        """The JSONL file path for a given object type."""
        safe = object_type.replace("/", "_").replace(os.sep, "_")
        return self.root / f"{safe}.jsonl"

    def write(self, edit: Edit) -> bool:
        """Append one edit as a JSON line under its object-type file."""
        try:
            path = self.path_for(object_type_of(edit))
            line = json.dumps(edit.model_dump(), default=str, sort_keys=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return True
        except Exception as exc:  # noqa: BLE001 — write-back never blocks the ledger
            logger.warning("JsonlEditSink: failed to write edit %s: %s", edit.id, exc)
            return False

    def flush(self) -> None:
        """No-op: each write is flushed by the context-managed append."""
        return None

    def read_all(self, object_type: str) -> list[dict[str, Any]]:
        """Read back every edit record written for an object type (replay/audit)."""
        path = self.path_for(object_type)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                records.append(json.loads(raw))
        return records


class WriteBackRouter:
    """Fans committed edits out to one or more registered :class:`EditSink`.

    CONCEPT:KG-2.43 — the write-back leg of the edit ledger. Sinks are registered
    by name; :meth:`write_back` pushes an edit (or a batch) to every registered
    sink and returns a per-sink success map.
    """

    def __init__(self) -> None:
        self._sinks: dict[str, EditSink] = {}

    def register_sink(self, name: str, sink: EditSink) -> None:
        """Register a write-back sink under a name (overwrites an existing one)."""
        if not isinstance(sink, EditSink):
            raise TypeError(
                f"sink {name!r} does not satisfy the EditSink protocol "
                "(needs write() and flush())"
            )
        self._sinks[name] = sink

    def unregister_sink(self, name: str) -> None:
        """Remove a previously registered sink (no-op if absent)."""
        self._sinks.pop(name, None)

    @property
    def sink_names(self) -> list[str]:
        """Names of the currently registered sinks."""
        return list(self._sinks)

    def write_back(self, edit: Edit) -> dict[str, bool]:
        """Push one edit to every registered sink; returns per-sink success."""
        results: dict[str, bool] = {}
        for name, sink in self._sinks.items():
            try:
                results[name] = bool(sink.write(edit))
            except Exception as exc:  # noqa: BLE001
                logger.warning("WriteBackRouter: sink %s raised: %s", name, exc)
                results[name] = False
        return results

    def write_back_many(self, edits: list[Edit]) -> dict[str, int]:
        """Push a batch of edits; returns per-sink count of successful writes."""
        counts: dict[str, int] = {name: 0 for name in self._sinks}
        for edit in edits:
            for name, ok in self.write_back(edit).items():
                if ok:
                    counts[name] += 1
        self.flush()
        return counts

    def flush(self) -> None:
        """Flush every registered sink."""
        for sink in self._sinks.values():
            try:
                sink.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("WriteBackRouter: flush failed: %s", exc)
