from __future__ import annotations

"""Filesystem document-source connector.

CONCEPT:ECO-4.25 — reference ``Load`` + ``Poll`` + permission-sync connector.
CONCEPT:ECO-4.26 — incremental poll keyed on a max-mtime watermark.
CONCEPT:ECO-4.28 — POSIX owner/group → :class:`ExternalAccess`.

Walks a directory, reading document files into :class:`SourceDocument`s. Reuses
the ingestion engine's document-extension allow-list and skip-dirs, and the
enrichment ``read_document_text`` reader — so it never reinvents extraction. Fully
offline + deterministic (no network), which makes it the primary connector the
live-path ingestion test exercises end-to-end.
"""

from collections.abc import Iterator
from pathlib import Path

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    ExternalAccess,
    LoadConnector,
    PermSyncConnector,
    PollConnector,
    SlimDocument,
    SourceDocument,
)
from ..registry import register_source


def _doc_extensions() -> set[str]:
    """Reuse the ingestion engine's document extension allow-list."""
    from ....knowledge_graph.ingestion.engine import IngestionEngine

    return set(IngestionEngine._DOC_EXTENSIONS)


def _skip_dirs() -> set[str]:
    from ....knowledge_graph.ingestion.engine import _SKIP_DIRS

    return set(_SKIP_DIRS)


@register_source("filesystem")
class FilesystemConnector(LoadConnector, PollConnector, PermSyncConnector):
    """Ingest document files under a root directory.

    CONCEPT:ECO-4.25.

    Config:
        root: Directory to walk (required).
        extensions: Optional override of the document extension allow-list.
        recursive: Walk subdirectories (default True).
        public: Mark every document world-readable (default True). When False,
            POSIX owner/group are mapped to :class:`ExternalAccess` groups.
    """

    provider = "Local Filesystem"
    priority = 40

    def configure(
        self,
        *,
        root: str = "",
        extensions: list[str] | None = None,
        recursive: bool = True,
        public: bool = True,
        **_: object,
    ) -> None:
        if not root:
            raise ValueError("FilesystemConnector requires a 'root' directory")
        self.root = Path(root).expanduser()
        self.extensions = (
            {e.lower() for e in extensions} if extensions else _doc_extensions()
        )
        self.recursive = recursive
        self.public = public
        self._skip = _skip_dirs()

    def health_check(self) -> bool:
        return self.root.exists() and self.root.is_dir()

    # -- enumeration -------------------------------------------------------

    def _iter_files(self) -> Iterator[Path]:
        if not self.health_check():
            return
        walker = self.root.rglob("*") if self.recursive else self.root.glob("*")
        for p in sorted(walker):
            if not p.is_file():
                continue
            if p.suffix.lower() not in self.extensions:
                continue
            if any(part in self._skip for part in p.parts):
                continue
            yield p

    def _read(self, path: Path) -> str:
        from ....knowledge_graph.enrichment.extractors.document import (
            read_document_text,
        )

        try:
            return read_document_text(str(path))
        except Exception:  # noqa: BLE001 — unreadable file → empty (skipped upstream)
            return ""

    def _access(self, path: Path) -> ExternalAccess:
        if self.public:
            return ExternalAccess.public()
        groups: list[str] = []
        try:
            import grp

            gid = path.stat().st_gid
            groups.append(grp.getgrgid(gid).gr_name)
        except Exception:  # noqa: BLE001 — group lookup is best-effort on the platform
            pass
        return ExternalAccess(is_public=False, group_ids=groups)

    def _to_document(self, path: Path) -> SourceDocument | None:
        text = self._read(path)
        if not text.strip():
            return None
        st = path.stat()
        return SourceDocument(
            id=str(path.resolve()),
            source_uri=str(path.resolve()),
            title=path.name,
            text=text,
            doc_type=path.suffix.lstrip(".").lower() or "document",
            metadata={"size": st.st_size, "mtime_ns": st.st_mtime_ns},
            external_access=self._access(path),
            updated_at=str(st.st_mtime_ns),
        )

    # -- LoadConnector -----------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        for path in self._iter_files():
            doc = self._to_document(path)
            if doc is not None:
                yield doc

    # -- PollConnector -----------------------------------------------------

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Yield only files modified since the watermark; advance it to the new max.

        CONCEPT:ECO-4.26 — the watermark is the max ``st_mtime_ns`` seen so far,
        so a re-poll over an unchanged tree returns zero documents.
        """
        prior = int(checkpoint.watermark) if checkpoint and checkpoint.watermark else -1
        docs: list[SourceDocument] = []
        max_mtime = prior
        for path in self._iter_files():
            mtime = path.stat().st_mtime_ns
            if mtime <= prior:
                continue
            doc = self._to_document(path)
            if doc is None:
                continue
            docs.append(doc)
            max_mtime = max(max_mtime, mtime)
        cp = ConnectorCheckpoint(
            has_more=False,
            watermark=str(max_mtime if max_mtime >= 0 else prior),
        )
        return CheckpointedBatch(documents=docs, checkpoint=cp)

    # -- SlimConnector / PermSyncConnector ---------------------------------

    def slim(self) -> Iterator[SlimDocument]:
        for path in self._iter_files():
            yield SlimDocument(
                id=str(path.resolve()),
                source_uri=str(path.resolve()),
                external_access=self._access(path),
            )

    def fetch_access(self) -> Iterator[tuple[str, ExternalAccess]]:
        for path in self._iter_files():
            yield str(path.resolve()), self._access(path)
