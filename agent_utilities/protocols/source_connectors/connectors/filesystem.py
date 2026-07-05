from __future__ import annotations

"""Filesystem document-source connector.

CONCEPT:AU-ECO.connector.document-source-framework — reference ``Load`` + ``Poll`` + permission-sync connector.
CONCEPT:AU-ECO.connector.incremental-poll-watermark — incremental poll keyed on a max-mtime watermark.
CONCEPT:AU-ECO.connector.posix-owner-mapping — POSIX owner/group → :class:`ExternalAccess`.

Walks a directory, reading document files into :class:`SourceDocument`s. Reuses
the ingestion engine's document-extension allow-list and skip-dirs, and the
enrichment ``read_document_text`` reader — so it never reinvents extraction. Fully
offline + deterministic (no network), which makes it the primary connector the
live-path ingestion test exercises end-to-end.
"""

import hashlib
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

# ── Filesystem-connector presets (CONCEPT:AU-ECO.connector.openwiki-preset) ──────
#
# CLAUDE.md rule: a NEW external source is a *preset*, not a new connector module.
# openwiki is a pure filesystem corpus (a repo's ``openwiki/`` dir of markdown, no
# server/protocol/auth), so it rides the existing FilesystemConnector via a data-only
# preset. The preset points at ``openwiki/``, restricts to markdown, uses the
# ``.last-update.json`` SHA-256 snapshot as a content watermark (better than mtime),
# and stamps OKF frontmatter on ingest (openwiki md carries none) so every page
# normalizes into the skill-graph / OKF-CIS standard. Per-repo provenance = SLUG.
FILESYSTEM_PRESETS: dict[str, dict[str, object]] = {
    "openwiki": {
        "subdir": "openwiki",
        "extensions": [".md", ".markdown"],
        "watermark_file": "openwiki/.last-update.json",
        "stamp_okf": True,
        "okf_type": "wiki",
        "doc_type": "wiki",
    },
}


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

    CONCEPT:AU-ECO.connector.document-source-framework.

    Config:
        root: Directory to walk (required).
        extensions: Optional override of the document extension allow-list.
        recursive: Walk subdirectories (default True).
        public: Mark every document world-readable (default True). When False,
            POSIX owner/group are mapped to :class:`ExternalAccess` groups.
    """

    provider = "Local Filesystem"

    def configure(
        self,
        *,
        root: str = "",
        preset: str = "",
        subdir: str = "",
        extensions: list[str] | None = None,
        recursive: bool = True,
        public: bool = True,
        watermark_file: str = "",
        stamp_okf: bool = False,
        okf_type: str = "Reference",
        doc_type: str = "",
        slug: str = "",
        **_: object,
    ) -> None:
        # A preset supplies data-only defaults (openwiki, …); explicit kwargs win.
        if preset:
            base = FILESYSTEM_PRESETS.get(preset)
            if base is None:
                raise ValueError(
                    f"Unknown filesystem preset {preset!r}. "
                    f"Available: {', '.join(sorted(FILESYSTEM_PRESETS)) or '(none)'}"
                )
            merged = {
                **base,
                **{
                    k: v
                    for k, v in self._config.items()
                    if k != "preset" and v not in ("", None)
                },
            }
            merged.pop("preset", None)
            self._config = merged
            self.configure(**merged)  # type: ignore[arg-type]
            return
        if not root:
            raise ValueError("FilesystemConnector requires a 'root' directory")
        base_root = Path(root).expanduser()
        # ``slug`` provenance defaults to the repo-dir name a ``subdir`` hangs off.
        self.slug = slug or (base_root.name if subdir else "")
        self.root = base_root / subdir if subdir else base_root
        self.extensions = (
            {e.lower() for e in extensions} if extensions else _doc_extensions()
        )
        self.recursive = recursive
        self.public = public
        self.watermark_file = watermark_file
        self.stamp_okf = bool(stamp_okf)
        self.okf_type = okf_type
        self.doc_type_override = doc_type
        self._skip = _skip_dirs()

    def _snapshot_watermark(self) -> str | None:
        """SHA-256 of the ``watermark_file`` snapshot, or ``None`` if not configured.

        CONCEPT:AU-ECO.connector.openwiki-preset — openwiki writes a
        ``.last-update.json`` on every corpus refresh; its content hash is a
        far tighter delta signal than per-file mtime (unchanged corpus ⇒ identical
        hash ⇒ zero re-ingest), and survives a git checkout that rewrites mtimes.
        """
        if not self.watermark_file:
            return None
        wf = Path(self.watermark_file)
        if not wf.is_absolute():
            # Resolve relative to the repo root (parent of ``subdir``) then root.
            for cand in (self.root.parent / wf.name, self.root / wf.name, wf):
                if cand.exists():
                    wf = cand
                    break
        if not wf.exists():
            return None
        return hashlib.sha256(wf.read_bytes()).hexdigest()

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
        metadata: dict[str, object] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        doc_type = (
            self.doc_type_override or path.suffix.lstrip(".").lower() or "document"
        )
        # OKF normalization on ingest (CONCEPT:AU-ECO.connector.openwiki-preset):
        # stamp minimal OKF frontmatter on bodies that carry none, and record the
        # per-repo SLUG so every ingested page traces back to its openwiki instance.
        if self.stamp_okf:
            from agent_utilities.knowledge_graph.distillation.okf_bundle import (
                frontmatter_text,
                resolve_type_domain,
            )

            provenance = self.slug or self.root.name
            _pillar, domain = resolve_type_domain(self.okf_type, provenance=provenance)
            metadata.update(
                {
                    "slug": provenance,
                    "okf_type": self.okf_type,
                    "okf_domain": domain,
                    "overlay_source": str(path.resolve()),
                }
            )
            text = frontmatter_text(
                text,
                ftype=self.okf_type,
                resource=str(path.resolve()),
                title=path.stem.replace("-", " ").replace("_", " ").title(),
                extra={"slug": provenance, "domain": domain},
            )
        return SourceDocument(
            id=str(path.resolve()),
            source_uri=str(path.resolve()),
            title=path.name,
            text=text,
            doc_type=doc_type,
            metadata=metadata,
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

        CONCEPT:AU-ECO.connector.incremental-poll-watermark — the watermark is the max ``st_mtime_ns`` seen so far,
        so a re-poll over an unchanged tree returns zero documents.

        When a ``watermark_file`` is configured (the openwiki preset's
        ``.last-update.json``) its SHA-256 snapshot is the authoritative watermark:
        an unchanged snapshot short-circuits the whole poll to zero documents
        (CONCEPT:AU-ECO.connector.openwiki-preset), independent of file mtimes.
        """
        snapshot = self._snapshot_watermark()
        prior_snapshot = (
            checkpoint.watermark if checkpoint and checkpoint.watermark else ""
        )
        if snapshot is not None:
            if snapshot == prior_snapshot:
                return CheckpointedBatch(
                    documents=[],
                    checkpoint=ConnectorCheckpoint(has_more=False, watermark=snapshot),
                )
            return CheckpointedBatch(
                documents=list(self.load()),
                checkpoint=ConnectorCheckpoint(has_more=False, watermark=snapshot),
            )
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
