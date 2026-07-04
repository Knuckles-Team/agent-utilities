"""CONCEPT:AU-KG.ingest.wiki-delta-ingest — Self-Curating Wiki (delta-skip continuous ingest).

Assimilated from memory-os (ClaudioDrews/memory-os@a4ca094, scripts/wiki_continuous_ingest.py): a
self-curating knowledge vault whose markdown pages are continuously ingested into the graph, but
**only when they change** — each file's SHA-256 is tracked so unchanged pages are skipped. State is
written atomically (tempfile → fsync → rename) so a crash never corrupts it.

This is the wiki layer on top of agent-utilities' existing ingestion: changed pages are handed to
``IngestionEngine.ingest`` (which already does concept/entity extraction, KG-2.8 delta hashing, and
linking), and the existing ``SynthesisEngine`` auto-curates promotions. The curator itself is pure
I/O over hashes and is fully unit-testable via an injected ``ingest_fn``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes (CONCEPT:AU-KG.ingest.wiki-delta-ingest delta detection)."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


class WikiCurator:
    """Tracks wiki-page content hashes so only new/changed pages are re-ingested."""

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)

    def _load_state(self) -> dict[str, str]:
        if not self.state_path.is_file():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except (ValueError, OSError):
            return {}

    def _save_state_atomic(self, state: dict[str, str]) -> None:
        """Write state via tempfile + fsync + atomic rename (crash-safe)."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def changed_files(self, wiki_dir: str | Path) -> list[Path]:
        """Return wiki .md files that are new or whose content hash changed since last ingest."""
        state = self._load_state()
        changed: list[Path] = []
        for path in sorted(Path(wiki_dir).rglob("*.md")):
            if state.get(str(path)) != file_hash(path):
                changed.append(path)
        return changed

    def curate(
        self,
        wiki_dir: str | Path,
        ingest_fn: Callable[[Path], Any],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Ingest only changed wiki pages via ``ingest_fn``, then commit the new hash state.

        ``ingest_fn(path)`` does the actual ingestion (e.g. ``IngestionEngine.ingest``). Returns a
        summary ``{scanned, ingested, skipped, errors}``. On ``dry_run`` nothing is ingested or
        committed — it just reports what would change.
        """
        all_files = sorted(Path(wiki_dir).rglob("*.md"))
        state = self._load_state()
        changed = [p for p in all_files if state.get(str(p)) != file_hash(p)]
        summary: dict[str, Any] = {
            "scanned": len(all_files),
            "ingested": 0,
            "skipped": len(all_files) - len(changed),
            "errors": 0,
            "dry_run": dry_run,
        }
        if dry_run:
            summary["would_ingest"] = [str(p) for p in changed]
            return summary
        for path in changed:
            try:
                ingest_fn(path)
                state[str(path)] = file_hash(path)
                summary["ingested"] += 1
            except Exception as e:  # noqa: BLE001 - one bad page must not abort the run
                logger.warning("Wiki ingest failed for %s: %s", path, e)
                summary["errors"] += 1
        self._save_state_atomic(state)
        return summary


def curate_wiki(
    engine: Any,
    wiki_dir: str | Path,
    *,
    state_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """CLI/graph_ingest entry point: continuously curate a wiki dir into the graph (CONCEPT:AU-KG.ingest.wiki-delta-ingest).

    Changed pages are routed through ``IngestionEngine.ingest`` (reusing concept/entity extraction
    + KG-2.8 delta hashing); ``SynthesisEngine`` then auto-curates promotions.
    """
    sp = state_path or (Path(wiki_dir) / ".wiki_ingest_state.json")
    curator = WikiCurator(sp)

    def _ingest(path: Path) -> Any:
        import asyncio

        from .engine import ContentType, IngestionEngine, IngestionManifest

        ing = IngestionEngine(kg_engine=engine)
        manifest = IngestionManifest(
            content_type=ContentType.DOCUMENT, source_uri=str(path)
        )
        return asyncio.run(ing.ingest(manifest))

    return curator.curate(wiki_dir, _ingest, dry_run=dry_run)
