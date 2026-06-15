#!/usr/bin/python
from __future__ import annotations

"""The unified skill-graph distillation pipeline — one contract, every source kind.

CONCEPT:KG-2.7 — A skill-graph captures an externally-consumable corpus of knowledge
as an agent skill. Previously each source type (web crawl / PDF / KG distillation /
hand-authored) was built a different way and only the KG path recorded provenance.
This module is the **single spine**: any source — a website, a PDF/Office file, a
local directory, a single URL, a REST/DB/MCP connector, freshly LLM-generated text,
or a coherent Knowledge-Graph subgraph — is normalized to a ``reference/`` markdown
tree and rendered into a *standardized* skill-graph with a uniform
``sources.json`` provenance/freshness manifest (see :mod:`.skill_graph_schema`).

Design — reuse, don't reinvent:

* Each acquisition routes to machinery that already exists: the source-connector
  registry (``protocols.source_connectors`` — web/reader/rest/database/mcp_tool),
  ``markitdown``/``pymupdf4llm`` for documents, and :class:`SkillGraphDistiller`
  for KG subgraphs.
* KG enrichment is **hybrid-auto**: the offline corpus is always produced; when the
  graph daemon is reachable the tree is *also* ingested (so the KG can reason over
  it) — guarded and bounded, degrading cleanly to ``kg_ingested: false`` when down.
* The web crawler and the text generator are **injectable** so the thin
  ``skill-graph-builder`` CLI can supply its richer crawl4ai path while the core
  stays self-contained and offline-testable with the built-in web connector.

CLI::

    python -m agent_utilities.knowledge_graph.distillation.skill_graph_pipeline \
        build --name servicenow-docs --out /tmp/sg \
        --source web=https://docs.site --source pdf=/tmp/manual.pdf
    python -m ...skill_graph_pipeline status --dir /path/to/skill-graph
    python -m ...skill_graph_pipeline rebuild --dir /path/to/skill-graph
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .skill_graph_schema import (
    SOURCE_KINDS,
    SOURCES_SCHEMA,
    parse_frontmatter,
    sha256_bytes,
    sha256_text,
    validate_skill_graph,
)

logger = logging.getLogger(__name__)

# Source kinds whose freshness can be re-checked without network I/O.
_LOCAL_KINDS = frozenset({"dir", "pdf", "office", "generated"})
# Document file extensions converted to markdown via markitdown/pymupdf4llm.
_DOC_EXTS = (".pdf", ".docx", ".pptx", ".xlsx", ".csv")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("agent-utilities")
    except Exception:  # noqa: BLE001 — editable/uninstalled checkout
        return "0+unknown"


def _slug(text: str, fallback: str = "doc") -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", (text or "").strip().lower()).strip("-")
    return (s or fallback)[:80]


def _bump_patch(version: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", version or "")
    if not m:
        return "0.1.0"
    return f"{m.group(1)}.{m.group(2)}.{int(m.group(3)) + 1}"


# ── data model ────────────────────────────────────────────────────────────────


@dataclass
class SourceSpec:
    """One declared input to a skill-graph build.

    ``kind`` selects the acquisition route (see :data:`.skill_graph_schema.SOURCE_KINDS`);
    ``uri`` is the url/path/query/seed; ``options`` carries per-kind knobs
    (``max_depth``/``max_pages`` for web, connector ``config``, generation ``prompt`` …).
    """

    kind: str
    uri: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError(
                f"Unknown source kind {self.kind!r}; valid: {', '.join(SOURCE_KINDS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "uri": self.uri, "options": dict(self.options)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceSpec:
        return cls(
            kind=d["kind"], uri=d.get("uri", ""), options=dict(d.get("options") or {})
        )

    @classmethod
    def parse(cls, token: str) -> SourceSpec:
        """Parse a ``kind=uri`` CLI token (e.g. ``web=https://x`` or ``pdf=/a.pdf``)."""
        kind, _, uri = token.partition("=")
        return cls(kind=kind.strip(), uri=uri.strip())


@dataclass
class AcquiredDoc:
    rel_path: str  # path under reference/ (POSIX)
    text: str  # markdown body
    title: str = ""
    source_uri: str = ""


@dataclass
class AcquiredBundle:
    spec: SourceSpec
    docs: list[AcquiredDoc]
    extractor: str
    fetched_at: str = field(default_factory=_now_iso)
    kg_manifest: dict[str, Any] | None = None  # set only by the kg_query kind

    @property
    def content_hash(self) -> str:
        """Order-independent digest of what this source yielded (drives staleness)."""
        joined = "\n".join(sorted(f"{d.source_uri}\n{d.text}" for d in self.docs))
        return sha256_text(joined)


CrawlerFn = Callable[[SourceSpec], list[AcquiredDoc]]
GeneratorFn = Callable[[SourceSpec], list[AcquiredDoc]]


# ── pipeline ────────────────────────────────────────────────────────────────


class SkillGraphPipeline:
    """Acquire from any source kind → standardized skill-graph (+ provenance manifest)."""

    def __init__(
        self,
        *,
        crawler_fn: CrawlerFn | None = None,
        generator_fn: GeneratorFn | None = None,
        kg_enrich: bool = True,
        kg_timeout: float = 1800.0,
    ) -> None:
        self.crawler_fn = crawler_fn
        self.generator_fn = generator_fn
        self.kg_enrich = kg_enrich
        self.kg_timeout = kg_timeout

    # ── acquisition (route per kind to existing extractors) ──────────────────

    def acquire(self, spec: SourceSpec) -> AcquiredBundle:
        route = {
            "web": self._acquire_web,
            "pdf": self._acquire_document,
            "office": self._acquire_document,
            "dir": self._acquire_dir,
            "url_reader": self._acquire_reader,
            "rest": self._acquire_connector,
            "database": self._acquire_connector,
            "mcp_tool": self._acquire_connector,
            "generated": self._acquire_generated,
            "kg_query": self._acquire_kg,
        }[spec.kind]
        return route(spec)

    def _acquire_web(self, spec: SourceSpec) -> AcquiredBundle:
        if self.crawler_fn is not None:
            docs = self.crawler_fn(spec)
            return AcquiredBundle(spec, docs, extractor="crawler_fn")
        # Self-contained fallback: the in-tree recursive web connector (ECO-4.25).
        from agent_utilities.protocols.source_connectors.registry import (
            build_connector,
        )

        conn = build_connector(
            "web",
            {
                "base_url": spec.uri,
                "max_depth": int(spec.options.get("max_depth", 2)),
                "max_pages": int(spec.options.get("max_pages", 1000)),
            },
        )
        docs = [self._doc_from_source(d) for d in conn.load()]  # type: ignore[attr-defined]
        return AcquiredBundle(spec, docs, extractor="web-connector")

    def _acquire_reader(self, spec: SourceSpec) -> AcquiredBundle:
        from agent_utilities.protocols.source_connectors.registry import (
            build_connector,
        )

        conn = build_connector("reader", {"url": spec.uri, **spec.options})
        docs = [self._doc_from_source(d) for d in conn.load()]  # type: ignore[attr-defined]
        return AcquiredBundle(spec, docs, extractor="reader")

    def _acquire_connector(self, spec: SourceSpec) -> AcquiredBundle:
        from agent_utilities.protocols.source_connectors.base import PollConnector
        from agent_utilities.protocols.source_connectors.registry import (
            build_connector,
        )

        config = dict(spec.options.get("config") or spec.options)
        if spec.uri and "url" not in config and "base_url" not in config:
            config.setdefault("base_url", spec.uri)
        conn = build_connector(spec.kind, config)
        if hasattr(conn, "load"):
            raw = list(conn.load())  # type: ignore[attr-defined]
        elif isinstance(conn, PollConnector):
            raw = list(conn.poll_all())
        else:  # pragma: no cover — registry guarantees one of the two
            raw = []
        docs = [self._doc_from_source(d, subdir=spec.kind) for d in raw]
        return AcquiredBundle(spec, docs, extractor=f"{spec.kind}-connector")

    def _acquire_document(self, spec: SourceSpec) -> AcquiredBundle:
        """Convert a single PDF/Office file (local path or http URL) to markdown."""
        uri = spec.uri
        if uri.startswith("http"):
            import requests

            resp = requests.get(uri, timeout=60)
            resp.raise_for_status()
            ext = Path(uri.split("?")[0]).suffix.lower() or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(resp.content)
                local = tmp.name
            stem = Path(uri.split("?")[0]).stem or "document"
        else:
            local = uri
            stem = Path(uri).stem
        text = _convert_document(local)
        if uri.startswith("http"):
            Path(local).unlink(missing_ok=True)
        doc = AcquiredDoc(
            rel_path=f"{_slug(stem)}.md", text=text, title=stem, source_uri=uri
        )
        return AcquiredBundle(spec, [doc], extractor="markitdown")

    def _acquire_dir(self, spec: SourceSpec) -> AcquiredBundle:
        """Read a local directory: .md/.txt verbatim, documents converted to markdown.

        An existing skill directory (``SKILL.md`` + ``reference/``) contributes its
        ``reference/`` subtree, mirroring the builder's prior behavior.
        """
        root = Path(spec.uri).expanduser()
        if (root / "SKILL.md").exists() and (root / "reference").is_dir():
            root = root / "reference"
        docs: list[AcquiredDoc] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            rel = p.relative_to(root).as_posix()
            if ext in (".md", ".markdown", ".txt"):
                text = p.read_text(encoding="utf-8", errors="replace")
                rel = re.sub(r"\.(markdown|txt)$", ".md", rel)
            elif ext in _DOC_EXTS:
                text = _convert_document(str(p))
                rel = rel.rsplit(".", 1)[0] + ".md"
            else:
                continue
            docs.append(AcquiredDoc(rel_path=rel, text=text, source_uri=str(p)))
        return AcquiredBundle(spec, docs, extractor="filesystem")

    def _acquire_generated(self, spec: SourceSpec) -> AcquiredBundle:
        """Author a fresh markdown corpus with an LLM ("generate our own")."""
        if self.generator_fn is not None:
            docs = self.generator_fn(spec)
            return AcquiredBundle(spec, docs, extractor="generator_fn")
        docs = _default_generate(spec)
        return AcquiredBundle(spec, docs, extractor="llm-generated")

    def _acquire_kg(self, spec: SourceSpec) -> AcquiredBundle:
        """Distill a coherent KG subgraph into a reference tree (+ kg_manifest)."""
        import asyncio

        from .skill_graph_distiller import SkillGraphDistiller

        selector = spec.uri.strip()
        is_seed = bool(selector) and not any(ch.isspace() for ch in selector)
        depth = int(spec.options.get("max_depth", 2))
        tmp = Path(tempfile.mkdtemp(prefix="sg_kg_"))

        async def _run() -> dict[str, Any]:
            dist = await SkillGraphDistiller.connect()
            try:
                return await dist.distill(
                    seed=selector if is_seed else None,
                    query=None if is_seed else selector,
                    depth=depth,
                    out_dir=tmp,
                )
            finally:
                await dist.close()

        manifest = asyncio.run(_run())
        ref = tmp / "reference"
        docs: list[AcquiredDoc] = []
        if ref.exists():
            for p in sorted(ref.rglob("*.md")):
                docs.append(
                    AcquiredDoc(
                        rel_path=p.relative_to(ref).as_posix(),
                        text=p.read_text(encoding="utf-8", errors="replace"),
                        source_uri=f"kg://{selector}",
                    )
                )
        bundle = AcquiredBundle(
            spec, docs, extractor="kg-distiller", kg_manifest=manifest
        )
        shutil.rmtree(tmp, ignore_errors=True)
        return bundle

    @staticmethod
    def _doc_from_source(sd: Any, subdir: str = "") -> AcquiredDoc:
        title = (getattr(sd, "title", "") or "").strip()
        uri = getattr(sd, "source_uri", "") or getattr(sd, "id", "")
        name = _slug(title or Path(uri).name or getattr(sd, "id", "doc"))
        rel = f"{subdir}/{name}.md" if subdir else f"{name}.md"
        return AcquiredDoc(
            rel_path=rel,
            text=getattr(sd, "text", "") or "",
            title=title,
            source_uri=uri,
        )

    # ── build ────────────────────────────────────────────────────────────────

    def build(
        self,
        *,
        name: str,
        specs: list[SourceSpec],
        out_dir: str | Path,
        description: str | None = None,
        max_file_kb: int = 50,
        kg_enrich: bool | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Acquire all sources → write a standardized skill-graph at ``out_dir/name``."""
        skill_dir = Path(out_dir) / name
        ref = skill_dir / "reference"
        if ref.exists():
            shutil.rmtree(ref)
        ref.mkdir(parents=True, exist_ok=True)

        bundles = [self.acquire(s) for s in specs]

        # Write the merged, de-duplicated reference tree.
        used: set[str] = set()
        for bundle in bundles:
            for doc in bundle.docs:
                rel = _unique_rel(doc.rel_path, used)
                target = ref / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(doc.text, encoding="utf-8")

        if max_file_kb > 0:
            _split_oversized(ref, max_file_kb)

        # Ship the KG-distiller provenance manifest (round-trip) if a kg_query ran.
        kg_query_manifest = next(
            (b.kg_manifest for b in bundles if b.kg_manifest), None
        )
        if kg_query_manifest is not None:
            (skill_dir / "kg_manifest.json").write_text(
                json.dumps(kg_query_manifest, indent=2), encoding="utf-8"
            )

        # Hybrid-auto KG enrichment (graceful — never blocks the offline graph).
        do_kg = self.kg_enrich if kg_enrich is None else kg_enrich
        kg_result: dict[str, Any] = (
            self._maybe_ingest_kg(ref, name) if do_kg else {"kg_ingested": False}
        )
        if kg_query_manifest is not None:
            kg_result.setdefault("kg_manifest", "kg_manifest.json")
            kg_result.setdefault("kg_ontology", kg_query_manifest.get("ontology"))
            kg_result.setdefault("concepts", _manifest_concepts(kg_query_manifest))

        md_files = sorted(ref.rglob("*.md"))
        version = version or "0.1.0"
        description = description or _default_description(name)

        self._write_sources_manifest(
            skill_dir, name, version, bundles, md_files, ref, kg_result
        )
        self._render_skill_md(
            skill_dir, name, description, version, bundles, md_files, ref, kg_result
        )

        errors = validate_skill_graph(skill_dir)
        if errors:
            logger.warning("skill-graph %s validation: %s", name, errors)
        return {
            "skill_dir": str(skill_dir),
            "name": name,
            "version": version,
            "file_count": len(md_files),
            "source_count": len(specs),
            "kg_ingested": bool(kg_result.get("kg_ingested")),
            "validation_errors": errors,
        }

    def _maybe_ingest_kg(self, ref: Path, name: str) -> dict[str, Any]:
        """Best-effort, bounded ingest of the reference tree into the live KG.

        Decoupled process-boundary shell-out (mirrors the builder's prior pattern):
        sidesteps event-loop reentrancy and naturally bounds via the subprocess
        timeout. Any failure (no daemon, embedder 502, timeout) degrades to
        ``kg_ingested: false`` — the offline graph is already on disk.
        """
        cmd = [
            sys.executable,
            "-m",
            "agent_utilities.knowledge_graph.ingestion",
            str(ref),
            "--content-type",
            "document",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.kg_timeout
            )
        except Exception as exc:  # noqa: BLE001 — FileNotFound, timeout, etc.
            logger.info("KG enrichment skipped for %s: %s", name, exc)
            return {"kg_ingested": False, "reason": str(exc)}
        if proc.returncode != 0:
            logger.info(
                "KG enrichment failed for %s (daemon down?): %s",
                name,
                (proc.stderr or proc.stdout or "").strip()[:300],
            )
            return {"kg_ingested": False, "reason": "ingest_failed"}
        nodes = edges = 0
        try:
            payload = json.loads(proc.stdout or "{}")
            for r in payload.get("results", []):
                nodes += int(r.get("nodes_created") or 0)
                edges += int(r.get("edges_created") or 0)
        except ValueError:
            pass
        return {
            "kg_ingested": True,
            "nodes": nodes,
            "edges": edges,
            "domain": f"skillgraph:{name}",
            "kg_ontology": "agent-utilities",
        }

    # ── manifest + SKILL.md ──────────────────────────────────────────────────

    def _write_sources_manifest(
        self,
        skill_dir: Path,
        name: str,
        version: str,
        bundles: list[AcquiredBundle],
        md_files: list[Path],
        ref: Path,
        kg_result: dict[str, Any],
    ) -> None:
        sources = [
            {
                "kind": b.spec.kind,
                "uri": b.spec.uri,
                "options": dict(b.spec.options),
                "extractor": b.extractor,
                "fetched_at": b.fetched_at,
                "content_hash": b.content_hash,
                "doc_count": len(b.docs),
            }
            for b in bundles
        ]
        files = [
            {
                "path": p.relative_to(skill_dir).as_posix(),
                "sha256": sha256_bytes(p.read_bytes()),
                "bytes": p.stat().st_size,
            }
            for p in md_files
        ]
        manifest = {
            "schema": SOURCES_SCHEMA,
            "name": name,
            "skill_graph_version": version,
            "built_at": _now_iso(),
            "builder_version": _pkg_version(),
            "kg_ingested": bool(kg_result.get("kg_ingested")),
            "kg_manifest": kg_result.get("kg_manifest"),
            "kg_ontology": kg_result.get("kg_ontology"),
            "concepts": kg_result.get("concepts") or [],
            "sources": sources,
            "files": files,
            "stats": {"file_count": len(md_files), "source_count": len(bundles)},
        }
        (skill_dir / "sources.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _render_skill_md(
        self,
        skill_dir: Path,
        name: str,
        description: str,
        version: str,
        bundles: list[AcquiredBundle],
        md_files: list[Path],
        ref: Path,
        kg_result: dict[str, Any],
    ) -> None:
        source_types = sorted({b.spec.kind for b in bundles})
        concepts = kg_result.get("concepts") or []
        toc = _render_toc(_build_doc_tree(ref))
        title = name.replace("-", " ").replace("docs", "").strip().title()

        lines = ["---", f"name: {name}", f"description: {description}"]
        lines.append(f"skill_graph_version: {version}")
        lines.append(f"source_types: [{', '.join(source_types)}]")
        lines.append(f"built_at: {_now_iso()}")
        lines.append(f"builder_version: {_pkg_version()}")
        lines.append(f"file_count: {len(md_files)}")
        lines.append(f"kg_ingested: {str(bool(kg_result.get('kg_ingested'))).lower()}")
        if kg_result.get("kg_manifest"):
            lines.append(f"kg_manifest: {kg_result['kg_manifest']}")
        if kg_result.get("kg_ontology"):
            lines.append(f"kg_ontology: {kg_result['kg_ontology']}")
        if concepts:
            lines.append(f"concepts: [{', '.join(repr(c) for c in concepts)}]")
        lines.append("categories: [Documentation, Knowledge Base, Reference]")
        lines.append(f"tags: [docs, reference, {name}, knowledge-base]")
        lines.append("---")
        lines.append("")
        lines.append(f"# {title} Documentation")
        lines.append("")
        lines.append(description)
        lines.append("")
        lines.append(f"**Sources** ({len(bundles)}):")
        for b in bundles:
            label = b.spec.uri or b.spec.kind
            lines.append(f"- `{b.spec.kind}` — {label}")
        lines.append("")
        lines.append(
            f"**Contains**: {len(md_files)} markdown files. "
            f"*Built {time.strftime('%B %d, %Y', time.gmtime())}.*"
        )
        if kg_result.get("kg_ingested"):
            lines.append(
                f"\n*Ingested into the Knowledge Graph "
                f"({kg_result.get('nodes', 0)} nodes) — query it via `graph_search`.*"
            )
        lines.append("")
        lines.append("## 📚 Table of Contents")
        lines.append("")
        lines.append("\n".join(toc) if toc else "*No markdown files found.*")
        lines.append("")
        lines.append("## 🤖 Agent Usage Guide")
        lines.append("")
        lines.append(
            f"- When asked about **{title}**, consult the reference files above."
        )
        lines.append(
            "- Prefer exact quotes and direct links to the relevant file/section."
        )
        lines.append(
            "- `sources.json` records provenance + freshness; rebuild when stale."
        )
        lines.append("")
        (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")

    # ── freshness + rebuild ───────────────────────────────────────────────────

    def status(self, skill_dir: str | Path, *, quick: bool = False) -> dict[str, Any]:
        """Re-check each source against its recorded content hash.

        ``quick`` skips network-backed sources (web/url_reader/connectors), reporting
        them ``unknown`` rather than re-fetching — useful for a fast local gate.
        """
        d = Path(skill_dir)
        manifest_path = d / "sources.json"
        if not manifest_path.exists():
            return {"name": d.name, "status": "unknown", "reason": "no sources.json"}
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        results: list[dict[str, Any]] = []
        any_stale = False
        for src in data.get("sources", []):
            spec = SourceSpec.from_dict(src)
            if quick and spec.kind not in _LOCAL_KINDS:
                results.append(
                    {"kind": spec.kind, "uri": spec.uri, "status": "unknown"}
                )
                continue
            try:
                current = self.acquire(spec).content_hash
            except Exception as exc:  # noqa: BLE001 — source unreachable now
                results.append(
                    {
                        "kind": spec.kind,
                        "uri": spec.uri,
                        "status": "unknown",
                        "reason": str(exc)[:120],
                    }
                )
                continue
            fresh = current == src.get("content_hash")
            any_stale = any_stale or not fresh
            results.append(
                {
                    "kind": spec.kind,
                    "uri": spec.uri,
                    "status": "fresh" if fresh else "stale",
                }
            )
        return {
            "name": data.get("name", d.name),
            "skill_graph_version": data.get("skill_graph_version"),
            "status": "stale" if any_stale else "fresh",
            "sources": results,
        }

    def rebuild(
        self, skill_dir: str | Path, *, kg_enrich: bool | None = None
    ) -> dict[str, Any]:
        """Re-acquire from the recorded specs and bump the patch version."""
        d = Path(skill_dir)
        data = json.loads((d / "sources.json").read_text(encoding="utf-8"))
        specs = [SourceSpec.from_dict(s) for s in data.get("sources", [])]
        fm = parse_frontmatter((d / "SKILL.md").read_text(encoding="utf-8"))
        version = _bump_patch(data.get("skill_graph_version") or "0.1.0")
        return self.build(
            name=data.get("name", d.name),
            specs=specs,
            out_dir=d.parent,
            description=fm.get("description"),
            kg_enrich=kg_enrich,
            version=version,
        )

    # ── legacy migration ──────────────────────────────────────────────────────

    def classify_legacy(self, skill_dir: str | Path) -> dict[str, Any]:
        """Classify a (possibly legacy) skill-graph for migration to the contract.

        Modes: ``managed`` (already has sources.json — nothing to do),
        ``reacquire`` (legacy ``source_url`` in SKILL.md → rebuild from those URLs),
        ``wrap`` (no source_url but has a ``reference/`` corpus → re-package the
        existing markdown as a ``dir`` source, preserving content), or ``native``
        (hand-authored / nested, no corpus → leave alone).
        """
        d = Path(skill_dir)
        skill_md = d / "SKILL.md"
        fm = (
            parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            if skill_md.exists()
            else {}
        )
        ref = d / "reference"
        md_count = len(list(ref.rglob("*.md"))) if ref.is_dir() else 0
        source_url = fm.get("source_url")
        if isinstance(source_url, list):
            source_url = ", ".join(source_url)
        if (d / "sources.json").exists():
            mode = "managed"
        elif source_url:
            mode = "reacquire"
        elif md_count:
            mode = "wrap"
        else:
            mode = "native"
        return {
            "name": d.name,
            "mode": mode,
            "source_url": source_url or "",
            "crawl_depth": fm.get("crawl_depth") or "",
            "file_count": md_count,
            "description": fm.get("description") or "",
        }

    @staticmethod
    def _specs_from_source_url(source_url: str, depth: int) -> list[SourceSpec]:
        specs: list[SourceSpec] = []
        for raw in (u.strip() for u in source_url.split(",")):
            if not raw:
                continue
            low = raw.lower().split("?")[0]
            if raw.startswith("http") and low.endswith(_DOC_EXTS):
                specs.append(
                    SourceSpec("pdf" if low.endswith(".pdf") else "office", raw)
                )
            elif raw.startswith("http"):
                specs.append(SourceSpec("web", raw, {"max_depth": depth}))
            else:
                specs.append(SourceSpec("dir", raw))
        return specs

    def migrate_legacy(
        self,
        skill_dir: str | Path,
        *,
        mode: str = "auto",
        kg_enrich: bool | None = None,
    ) -> dict[str, Any]:
        """Migrate one legacy skill-graph in place to the standardized contract.

        ``mode='auto'`` follows :meth:`classify_legacy`; ``reacquire`` re-crawls the
        legacy ``source_url``; ``wrap`` re-packages the existing ``reference/`` tree
        (offline, content-preserving). The first standardized build is versioned
        ``1.0.0``. Returns the build result (or a ``skipped`` record for native graphs).
        """
        d = Path(skill_dir)
        info = self.classify_legacy(d)
        chosen = info["mode"] if mode == "auto" else mode
        if chosen in ("managed", "native"):
            return {"name": d.name, "skipped": True, "reason": chosen}

        depth = int(info["crawl_depth"] or 2)
        tmp: Path | None = None
        if chosen == "reacquire":
            specs = self._specs_from_source_url(info["source_url"], depth)
            if not specs:
                return {
                    "name": d.name,
                    "skipped": True,
                    "reason": "no usable source_url",
                }
        else:  # wrap — copy the existing corpus aside so build() can wipe reference/
            tmp = Path(tempfile.mkdtemp(prefix="sg_wrap_"))
            shutil.copytree(d / "reference", tmp, dirs_exist_ok=True)
            specs = [SourceSpec("dir", str(tmp))]
        try:
            result = self.build(
                name=d.name,
                specs=specs,
                out_dir=d.parent,
                description=info["description"] or None,
                kg_enrich=kg_enrich,
                version="1.0.0",
            )
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        result["migrated_mode"] = chosen
        return result


# ── module helpers (pure / optional-dep guarded) ───────────────────────────────


def _unique_rel(rel: str, used: set[str]) -> str:
    if rel not in used:
        used.add(rel)
        return rel
    stem, _, ext = rel.rpartition(".")
    i = 2
    while True:
        cand = f"{stem}-{i}.{ext}" if stem else f"{rel}-{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


def _convert_document(path: str) -> str:
    """PDF/Office → markdown via markitdown, falling back to pymupdf4llm."""
    try:
        from markitdown import MarkItDown

        return MarkItDown().convert(path).text_content
    except ImportError:
        pass
    try:
        import pymupdf4llm

        return pymupdf4llm.to_markdown(path)
    except ImportError as exc:
        raise RuntimeError(
            "Document conversion needs 'markitdown' or 'pymupdf4llm'. "
            "Install with: pip install 'universal-skills[skill-graph-builder]'"
        ) from exc


def _default_generate(spec: SourceSpec) -> list[AcquiredDoc]:
    """Generate a markdown corpus from a prompt via the role-resolved chat model."""
    prompt = spec.options.get("prompt") or spec.uri
    if not prompt:
        raise ValueError("generated source needs options.prompt or uri (the topic)")
    from pydantic_ai import Agent

    from agent_utilities.core.model_factory import create_model

    agent = Agent(
        create_model(role="generator"),
        system_prompt=(
            "You are a technical writer. Produce a thorough, well-structured Markdown "
            "reference document on the requested topic, using clear headings."
        ),
    )
    result = agent.run_sync(prompt)
    text = getattr(result, "output", None) or getattr(result, "data", "")
    title = (spec.options.get("title") or prompt)[:80]
    return [
        AcquiredDoc(
            rel_path=f"{_slug(title)}.md",
            text=str(text),
            title=title,
            source_uri=f"generated://{_slug(title)}",
        )
    ]


def _manifest_concepts(manifest: dict[str, Any]) -> list[str]:
    """Surface CONCEPT:* ids from a KG-distiller manifest's node set."""
    out: list[str] = []
    for node in manifest.get("nodes", []):
        nid = str(node.get("id", ""))
        if nid.upper().startswith("CONCEPT:"):
            out.append(nid)
    return sorted(set(out))


def _default_description(name: str) -> str:
    return (
        f"Comprehensive reference documentation for {name.replace('-', ' ').title()}."
    )


# --- TOC builders (markdown index over the reference tree) ---------------------


def _extract_title(file_path: Path) -> str:
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")[:4000]
        fm = parse_frontmatter(content)
        if fm.get("title"):
            return str(fm["title"])
        h1 = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        if h1:
            return h1.group(1).strip()
    except OSError:
        pass
    return file_path.stem.replace("-", " ").replace("_", " ").title()


def _build_doc_tree(reference_dir: Path) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for md in sorted(reference_dir.rglob("*.md")):
        rel = md.relative_to(reference_dir)
        cur = tree
        for part in rel.parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[rel.name] = (_extract_title(md), f"reference/{rel.as_posix()}")
    return tree


def _render_toc(tree: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    dirs = sorted(k for k in tree if isinstance(tree[k], dict))
    files = sorted(k for k in tree if not isinstance(tree[k], dict))
    for key in dirs:
        lines.append("  " * indent + f"- 📁 **{key}/**")
        lines.extend(_render_toc(tree[key], indent + 1))
    for key in files:
        title, link = tree[key]
        lines.append("  " * indent + f"- [{title}]({link})")
    return lines


def _split_oversized(reference_dir: Path, max_file_kb: int) -> None:
    """Split markdown files above ``max_file_kb`` (mdsplit if present; line fallback)."""
    max_bytes = max_file_kb * 1024
    for md in list(reference_dir.rglob("*.md")):
        try:
            if md.stat().st_size <= max_bytes:
                continue
        except OSError:
            continue
        out = md.parent / md.stem
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        split_ok = _try_mdsplit(md, out, max_bytes)
        if not split_ok:
            _line_split(md, out, max_bytes)
        for big in [
            p
            for p in out.rglob("*.md")
            if p.stat().st_size > max_bytes and p.name != "toc.md"
        ]:
            _line_split(big, out, max_bytes)
            big.unlink(missing_ok=True)
        md.unlink(missing_ok=True)


def _try_mdsplit(md: Path, out: Path, max_bytes: int) -> bool:
    for level in ("1", "2"):
        try:
            subprocess.run(
                [
                    "mdsplit",
                    str(md),
                    "--max-level",
                    level,
                    "--table-of-contents",
                    "--output",
                    str(out),
                    "--force",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
        if not any(p.stat().st_size > max_bytes for p in out.rglob("*.md")):
            return True
    return True  # mdsplit ran; oversized leftovers handled by caller's line-split


def _line_split(src: Path, out: Path, max_bytes: int) -> None:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    target = max_bytes - (50 * 1024) if max_bytes > (50 * 1024) else max_bytes // 2
    target = target or max_bytes
    chunk: list[str] = []
    size = idx = 0
    base = src.stem
    for line in lines:
        chunk.append(line)
        size += len(line.encode("utf-8"))
        if size >= target:
            idx += 1
            (out / f"{base}_pt{idx}.md").write_text("".join(chunk), encoding="utf-8")
            chunk, size = [], 0
    if chunk:
        idx += 1
        (out / f"{base}_pt{idx}.md").write_text("".join(chunk), encoding="utf-8")


# ── CLI ─────────────────────────────────────────────────────────────────────


def _cmd_build(args: argparse.Namespace) -> int:
    specs = [SourceSpec.parse(t) for t in (args.source or [])]
    if args.depth is not None:
        for s in specs:
            if s.kind in ("web", "kg_query"):
                s.options.setdefault("max_depth", args.depth)
    pipe = SkillGraphPipeline(kg_enrich=not args.no_kg)
    result = pipe.build(
        name=args.name,
        specs=specs,
        out_dir=args.out,
        description=args.description,
        max_file_kb=args.max_file_kb,
    )
    print(json.dumps(result, indent=2))
    return 0 if not result["validation_errors"] else 1


def _cmd_status(args: argparse.Namespace) -> int:
    print(json.dumps(SkillGraphPipeline().status(args.dir, quick=args.quick), indent=2))
    return 0


def _cmd_rebuild(args: argparse.Namespace) -> int:
    pipe = SkillGraphPipeline(kg_enrich=not args.no_kg)
    print(json.dumps(pipe.rebuild(args.dir), indent=2))
    return 0


def _iter_graph_dirs(root: Path):
    """Yield each skill-graph directory (one per SKILL.md) under ``root``."""
    for skill_md in sorted(root.rglob("SKILL.md")):
        yield skill_md.parent


def _cmd_plan(args: argparse.Namespace) -> int:
    """Scan a skill_graphs root and print a migration plan (classification table)."""
    pipe = SkillGraphPipeline(kg_enrich=False)
    rows = [pipe.classify_legacy(d) for d in _iter_graph_dirs(Path(args.root))]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["mode"]] = counts.get(r["mode"], 0) + 1
    if args.json:
        print(json.dumps({"counts": counts, "graphs": rows}, indent=2))
        return 0
    print(f"Migration plan for {args.root}\n")
    print(f"{'MODE':<10} {'FILES':>6}  NAME")
    for r in sorted(rows, key=lambda x: (x["mode"], x["name"])):
        print(f"{r['mode']:<10} {r['file_count']:>6}  {r['name']}")
    print("\nTotals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(
        "\nLegend: reacquire=re-crawl source_url · wrap=re-package existing reference/ "
        "· managed=already on contract · native=hand-authored, left alone"
    )
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    pipe = SkillGraphPipeline(kg_enrich=not args.no_kg)
    if args.dir:
        result = pipe.migrate_legacy(args.dir, mode=args.mode)
        print(json.dumps(result, indent=2))
        return 0 if not result.get("validation_errors") else 1
    # Batch over a root.
    root = Path(args.root)
    only = {n.strip() for n in (args.only or "").split(",") if n.strip()}
    targets = [d for d in _iter_graph_dirs(root) if not only or d.name in only]
    results = []
    migrated = 0
    for d in targets:
        info = pipe.classify_legacy(d)
        if info["mode"] in ("managed", "native") and args.mode == "auto":
            continue
        if not args.apply:
            results.append({"name": d.name, "would_migrate": info["mode"]})
            continue
        res = pipe.migrate_legacy(d, mode=args.mode)
        results.append(res)
        migrated += 0 if res.get("skipped") else 1
        if args.limit and migrated >= args.limit:
            break
    print(json.dumps({"applied": args.apply, "results": results}, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified skill-graph distillation pipeline."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build a skill-graph from one or more sources.")
    b.add_argument("--name", required=True)
    b.add_argument(
        "--out", required=True, help="Parent directory; graph written to <out>/<name>."
    )
    b.add_argument(
        "--source",
        action="append",
        metavar="KIND=URI",
        help="Repeatable. e.g. web=https://x, pdf=/a.pdf, kg_query='servicenow'.",
    )
    b.add_argument("--description", default=None)
    b.add_argument(
        "--depth", type=int, default=None, help="max_depth for web/kg sources."
    )
    b.add_argument("--max-file-kb", type=int, default=50)
    b.add_argument("--no-kg", action="store_true", help="Skip KG enrichment entirely.")
    b.set_defaults(func=_cmd_build)

    s = sub.add_parser("status", help="Report freshness of an existing skill-graph.")
    s.add_argument("--dir", required=True)
    s.add_argument("--quick", action="store_true", help="Skip network sources.")
    s.set_defaults(func=_cmd_status)

    r = sub.add_parser("rebuild", help="Re-acquire from recorded specs; bump version.")
    r.add_argument("--dir", required=True)
    r.add_argument("--no-kg", action="store_true")
    r.set_defaults(func=_cmd_rebuild)

    p = sub.add_parser(
        "plan", help="Classify every skill-graph under a root for migration."
    )
    p.add_argument("--root", required=True, help="A skill_graphs/ directory.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_plan)

    m = sub.add_parser(
        "migrate",
        help="Migrate legacy skill-graph(s) to the standardized contract.",
    )
    mg = m.add_mutually_exclusive_group(required=True)
    mg.add_argument("--dir", help="Migrate a single skill-graph directory.")
    mg.add_argument("--root", help="Batch over every skill-graph under this root.")
    m.add_argument(
        "--mode",
        choices=["auto", "reacquire", "wrap"],
        default="auto",
        help="auto follows classification; reacquire re-crawls; wrap re-packages.",
    )
    m.add_argument(
        "--apply",
        action="store_true",
        help="With --root: actually migrate (default is a dry-run preview).",
    )
    m.add_argument("--only", default="", help="Comma-separated graph names to migrate.")
    m.add_argument("--limit", type=int, default=0, help="Max graphs to migrate.")
    m.add_argument("--no-kg", action="store_true")
    m.set_defaults(func=_cmd_migrate)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
