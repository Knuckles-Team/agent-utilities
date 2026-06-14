#!/usr/bin/env python3
"""Profile document ingestion over a corpus of (large) PDFs — books by default.

Times each stage so the bottleneck is obvious:
  * read   — PDF → text (PyMuPDF fast path in read_document_text)
  * chunk  — text → chunks (the retrieval/enrichment substrate)
  * ingest — full IngestionEngine.ingest (structural + enrichment), if --ingest
             and a live engine/backend are reachable.

Stages degrade gracefully: if PyMuPDF/pypdf or a live engine is absent (e.g. a
bare sandbox), that stage reports "unavailable" rather than failing. Run it on
the deployed stack (engine + vLLM + pymupdf) for the full end-to-end picture.

Usage::

    python scripts/profile_book_ingestion.py                       # read+chunk only
    python scripts/profile_book_ingestion.py --dir /path/to/pdfs
    python scripts/profile_book_ingestion.py --ingest              # full e2e (needs engine)
    python scripts/profile_book_ingestion.py --ingest --no-enrich  # structural only
    python scripts/profile_book_ingestion.py --limit 3 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

DEFAULT_DIR = "/home/apps/workspace/prompts/books"


def _read(path: str) -> tuple[str, float]:
    from agent_utilities.knowledge_graph.enrichment.extractors.document import (
        read_document_text,
    )

    t = time.monotonic()
    text = read_document_text(path)
    return text, time.monotonic() - t


def _chunk(text: str) -> tuple[int, float]:
    from agent_utilities.knowledge_graph.distillation.distillation_engine import (
        chunk_text,
    )

    t = time.monotonic()
    chunks = chunk_text(text)
    return len(chunks), time.monotonic() - t


async def _ingest(path: str, enrich: bool) -> tuple[dict, float]:
    from agent_utilities.knowledge_graph.ingestion.engine import (
        ContentType,
        IngestionEngine,
        IngestionManifest,
    )

    eng = IngestionEngine()  # active singleton engine + backend
    manifest = IngestionManifest(
        content_type=ContentType.DOCUMENT,
        source_uri=path,
        metadata={} if enrich else {"enrich": False},
        force=True,
    )
    t = time.monotonic()
    res = await eng.ingest(manifest)
    dt = time.monotonic() - t
    return {
        "status": res.status,
        "nodes": res.nodes_created,
        "edges": res.edges_created,
        "enrichment": res.details.get("enrichment"),
    }, dt


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile book/PDF ingestion stages.")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="Only the N smallest files.")
    ap.add_argument(
        "--ingest", action="store_true", help="Run full ingest (needs engine)."
    )
    ap.add_argument("--no-enrich", action="store_true", help="Structural ingest only.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    pdfs = sorted(root.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"no PDFs under {root}")
        return 1

    rows = []
    for p in pdfs:
        mb = p.stat().st_size / 1e6
        row = {"file": p.name, "size_mb": round(mb, 1)}
        try:
            text, t_read = _read(str(p))
            row["read_s"] = round(t_read, 2)
            row["chars"] = len(text)
            row["read_mb_s"] = round(mb / t_read, 2) if t_read else None
            if text:
                n_chunks, t_chunk = _chunk(text)
                row["chunks"] = n_chunks
                row["chunk_s"] = round(t_chunk, 2)
            else:
                row["note"] = "0 chars (PDF reader unavailable?)"
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        if args.ingest and row.get("chars"):
            try:
                ing, t_ing = asyncio.run(_ingest(str(p), enrich=not args.no_enrich))
                row["ingest_s"] = round(t_ing, 2)
                row["ingest"] = ing
            except Exception as e:  # noqa: BLE001
                row["ingest_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            extra = (
                f" chunks={r.get('chunks', '-')}"
                f" ingest={r.get('ingest_s', '-')}s {r.get('ingest', '')}"
            )
            print(
                f"{r['file']:42} {r['size_mb']:6.1f}MB "
                f"read={r.get('read_s', '-')}s ({r.get('read_mb_s', '-')}MB/s) "
                f"chars={r.get('chars', '-')}{extra}"
                + (
                    f"  ERR {r.get('error') or r.get('ingest_error')}"
                    if r.get("error") or r.get("ingest_error")
                    else ""
                )
            )
        reads = [r["read_s"] for r in rows if r.get("read_s")]
        if reads:
            print(f"\ntotal read: {sum(reads):.1f}s over {len(reads)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
