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
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
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

# Shrink guard (re-fetch): a re-crawl that returns far less content than the graph
# already has usually means the source_url moved (a docs site restructured to a landing
# page). Rather than overwrite rich content with a sparse crawl, the guard keeps the
# existing graph and flags it for a URL fix. Size-based so it is robust to file-splitting.
_SHRINK_RATIO = 0.5
_SHRINK_MIN_BYTES = 10_000


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


def _run_bounded(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    """Run a subprocess in its own process group, killing the WHOLE group on timeout.

    Plain ``subprocess.run(timeout=…)`` kills only the direct child, orphaning any
    grandchildren (crawl.py spawns a worker + Chromium; the ingest CLI spawns helpers)
    — they linger and pile up. Here a timeout SIGTERMs then SIGKILLs the group so a
    stalled crawl/ingest leaves nothing behind. Returns ``(returncode, stdout, stderr)``.
    """
    import signal

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", err or ""
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
                proc.wait(timeout=5)
                break
            except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                continue
        raise


@lru_cache(maxsize=1)
def _embed_probe_model() -> Any:
    from agent_utilities.core.embedding_utilities import create_embedding_model

    return create_embedding_model()


def _embedder_responsive(timeout: float = 6.0) -> bool | None:
    """Real bounded probe of the embedder. True=up, False=definitively down, None=unknown.

    A ``/health`` 200 from the serving proxy does NOT mean the GPU behind it can answer
    (it power-cycles); only a real, time-bounded embed call tells the truth. ``None``
    (can't construct a probe) means "don't block" — let the bounded ingest try.
    """
    try:
        model = _embed_probe_model()
    except Exception:  # noqa: BLE001 — no embedder configured → don't block
        return None
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        ex.submit(lambda: model.get_text_embedding("ping")).result(timeout=timeout)
        return True
    except concurrent.futures.TimeoutError:
        return False
    except Exception:  # noqa: BLE001 — transient error → unknown, let ingest try
        return None
    finally:
        ex.shutdown(wait=False)


def _ref_bytes(reference_dir: Path) -> int:
    """Total markdown bytes currently in a graph's ``reference/`` tree."""
    if not reference_dir.is_dir():
        return 0
    return sum(p.stat().st_size for p in reference_dir.rglob("*.md"))


def _bundle_bytes(bundles: list[AcquiredBundle]) -> int:
    """Total markdown bytes a fresh acquisition produced (pre-split)."""
    return sum(len(d.text.encode("utf-8")) for b in bundles for d in b.docs)


def _is_shrink(existing_bytes: int, new_bytes: int) -> bool:
    """True when a re-fetch is suspiciously smaller than the existing corpus."""
    return (
        existing_bytes >= _SHRINK_MIN_BYTES
        and new_bytes < _SHRINK_RATIO * existing_bytes
    )


# ── crawl4ai web crawler discovery ─────────────────────────────────────────────
# Real (JS-rendering) web acquisition reuses the universal-skills ``web-crawler``
# (crawl4ai) instead of duplicating it. The crawler runs in its own interpreter
# (``SKILL_GRAPH_CRAWLER_PYTHON``) so crawl4ai/Playwright can live in a dedicated
# venv; ``SKILL_GRAPH_CRAWLER`` overrides the script path. When unavailable, the
# pipeline falls back to the in-tree basic web connector.


def _discover_crawl_script() -> Path | None:
    env = os.environ.get("SKILL_GRAPH_CRAWLER", "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None
    try:
        import importlib.util

        spec = importlib.util.find_spec("universal_skills")
    except Exception:  # noqa: BLE001 — package layout probing is best-effort
        spec = None
    for base in list(getattr(spec, "submodule_search_locations", []) or []):
        cand = Path(base) / "research" / "web-crawler" / "scripts" / "crawl.py"
        if cand.exists():
            return cand
    return None


@lru_cache(maxsize=1)
def _resolve_crawler() -> tuple[str, str] | None:
    """Return ``(crawler_python, crawl_script)`` if a crawl4ai crawler is usable."""
    script = _discover_crawl_script()
    if script is None:
        return None
    crawler_py = (
        os.environ.get("SKILL_GRAPH_CRAWLER_PYTHON", "").strip() or sys.executable
    )
    try:
        probe = subprocess.run(
            [crawler_py, "-c", "import crawl4ai"], capture_output=True, timeout=60
        )
    except Exception:  # noqa: BLE001 — interpreter missing / probe failed
        return None
    if probe.returncode != 0:
        return None
    return crawler_py, str(script)


def _crawl_via_script(
    spec: SourceSpec, crawler_py: str, script: str
) -> list[AcquiredDoc]:
    """Crawl ``spec.uri`` with the crawl4ai web-crawler subprocess → markdown docs.

    Raises ``RuntimeError`` on crawl failure so callers preserve existing content
    (``build`` acquires before wiping) rather than silently degrading to a sparse crawl.
    """
    tmp = Path(tempfile.mkdtemp(prefix="sg_crawl_"))
    try:
        cmd = [
            crawler_py,
            script,
            "--urls",
            spec.uri,
            "--strategy",
            "recursive",
            "--output-dir",
            str(tmp),
            "--max-depth",
            str(int(spec.options.get("max_depth", 2))),
            "--max-pages",
            str(
                int(
                    spec.options.get("max_pages")
                    or os.environ.get("SKILL_GRAPH_MAX_PAGES")
                    or 1000
                )
            ),
        ]
        if spec.options.get("disable_magic_js"):
            cmd.append("--disable-magic-js")
        if spec.options.get("no_sitemap"):
            cmd.append("--no-sitemap")
        if spec.options.get("wait_for"):
            cmd.extend(["--wait-for", str(spec.options["wait_for"])])
        # Per-crawl wall-clock bound so one slow/looping site can't stall a batch
        # (env-overridable; option wins). Timeout → RuntimeError → existing content kept.
        timeout = float(
            spec.options.get("crawl_timeout")
            or os.environ.get("SKILL_GRAPH_CRAWL_TIMEOUT")
            or 900
        )
        # Bounded + process-group kill: a stalled crawl (e.g. a JS-heavy site) must not
        # orphan its Chromium/worker children when it times out.
        returncode, _out, err = _run_bounded(cmd, timeout=timeout)
        if returncode != 0:
            raise RuntimeError(f"crawl failed for {spec.uri}: {err[-400:]}")
        return [
            AcquiredDoc(
                rel_path=p.relative_to(tmp).as_posix(),
                text=p.read_text(encoding="utf-8", errors="replace"),
                source_uri=spec.uri,
            )
            for p in sorted(tmp.rglob("*.md"))
        ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── llms.txt / llms-full.txt acquisition + per-site strategy detection ──────────
# The robustness layer: modern doc platforms publish the llms.txt standard — a clean,
# complete, LLM-optimized rendering of their docs. Preferring it sidesteps JS rendering,
# bot-blocking, sparse SPAs, and incomplete recursive crawls entirely.


def _http_get(
    url: str, *, timeout: float = 25.0, max_bytes: int = 25_000_000
) -> str | None:
    """Plain bounded HTTP GET → text, or None on any non-200/error. No deps."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "skill-graph-builder"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — http(s) only
            if getattr(r, "status", 200) != 200:
                return None
            return r.read(max_bytes).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — unreachable/404/timeout → caller falls back
        return None


def _site_root(url: str) -> str:
    from urllib.parse import urlparse

    p = urlparse(url if "://" in url else f"https://{url}")
    return f"{p.scheme or 'https'}://{p.netloc}"


# Above this many heading sections an llms-full split is *over-fragmented*
# (e.g. langchain's 2548 per-API-method H1s). Past the cap we coalesce adjacent
# sections into size-bounded files so the reference/ tree stays navigable.
_MAX_LLMS_SECTIONS = 400
_LLMS_COALESCE_BYTES = 48_000


def _split_llms_full(text: str, base: str) -> list[AcquiredDoc]:
    """Split an llms-full.txt corpus into per-section reference docs (by top headings).

    Splits on H1 (then H2). If that yields more sections than
    :data:`_MAX_LLMS_SECTIONS`, adjacent sections are packed into ~48 KB files —
    preserving heading boundaries while keeping the file count sane. A corpus with
    no usable headings becomes a single doc (``split_oversized`` chunks it by size).
    """
    text = text.replace("\r\n", "\n")
    src = f"{base}/llms-full.txt"
    for level in ("# ", "## "):
        parts = re.split(rf"(?m)^(?={re.escape(level)})", text)
        sections = [p.strip() for p in parts if p.strip()]
        if len(sections) < 2:
            continue
        if len(sections) <= _MAX_LLMS_SECTIONS:
            return [_section_doc(sec, src) for sec in sections]
        # Over-fragmented: pack adjacent sections into size-bounded files.
        docs: list[AcquiredDoc] = []
        buf: list[str] = []
        size = 0
        for sec in sections:
            if buf and size + len(sec) > _LLMS_COALESCE_BYTES:
                docs.append(_section_doc("\n\n".join(buf), src, idx=len(docs)))
                buf, size = [], 0
            buf.append(sec)
            size += len(sec)
        if buf:
            docs.append(_section_doc("\n\n".join(buf), src, idx=len(docs)))
        return docs
    return [
        AcquiredDoc(
            rel_path="llms-full.md",
            text=text.strip() + "\n",
            title="Full documentation",
            source_uri=src,
        )
    ]


def _section_doc(sec: str, src: str, idx: int | None = None) -> AcquiredDoc:
    """Build one reference doc from a markdown section (title from its first heading)."""
    m = re.match(r"^#{1,6}\s+(.+)", sec)
    title = (m.group(1).strip() if m else "section")[:80]
    slug = _slug(title)
    rel = f"{idx:04d}-{slug}.md" if idx is not None else f"{slug}.md"
    return AcquiredDoc(rel_path=rel, text=sec.strip() + "\n", title=title, source_uri=src)


def _fetch_llms_index(idx: str, base: str, max_pages: int) -> list[AcquiredDoc]:
    """Fetch each page linked from an llms.txt index (.md raw, else strip-to-text)."""
    from urllib.parse import urljoin

    seen: set[str] = set()
    docs: list[AcquiredDoc] = []
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", idx):
        if len(docs) >= max_pages:
            break
        name, url = m.group(1).strip(), urljoin(base + "/", m.group(2).strip())
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        body = _http_get(url)
        if not body or not body.strip():
            continue
        rel = re.sub(r"[^a-zA-Z0-9._/-]+", "-", url.split("://", 1)[-1]).strip("-/")
        rel = (rel or _slug(name)) + ("" if rel.endswith(".md") else ".md")
        docs.append(
            AcquiredDoc(rel_path=rel, text=body, title=name[:80], source_uri=url)
        )
    return docs


def _fetch_llms_docs(url: str, *, max_pages: int = 1000) -> list[AcquiredDoc]:
    """Acquire a site's docs via the llms.txt standard; [] if not published."""
    base = _site_root(url)
    full = _http_get(f"{base}/llms-full.txt")
    if full and len(full) > 2000:
        return _split_llms_full(full, base)
    idx = _http_get(f"{base}/llms.txt")
    if idx and "](" in idx:
        return _fetch_llms_index(idx, base, max_pages)
    return []


def _has_sitemap(base: str) -> bool:
    if _http_get(f"{base}/sitemap.xml") is not None:
        return True
    robots = _http_get(f"{base}/robots.txt") or ""
    return "sitemap:" in robots.lower()


def detect_scrape_strategy(url: str) -> tuple[SourceSpec, dict[str, Any]]:
    """Probe a site and pick the best acquisition strategy (the SiteProfiler).

    Priority: llms.txt/llms-full.txt → sitemap crawl → recursive render. Returns the
    chosen ``SourceSpec`` plus a ``scrape_profile`` dict (what was detected + why) for
    the robustness ledger recorded in sources.json.
    """
    base = _site_root(url)
    profile: dict[str, Any] = {"url": url, "root": base}
    if _http_get(f"{base}/llms-full.txt", max_bytes=4096) is not None:
        profile["strategy"] = "llms"
        profile["signal"] = "llms-full.txt"
        return SourceSpec("llms", base), profile
    idx = _http_get(f"{base}/llms.txt")
    if idx and "](" in idx:
        profile["strategy"] = "llms"
        profile["signal"] = "llms.txt"
        return SourceSpec("llms", base), profile
    if _has_sitemap(base):
        profile["strategy"] = "web+sitemap"
        profile["signal"] = "sitemap.xml"
        return SourceSpec("web", url, {"no_sitemap": False}), profile
    profile["strategy"] = "web+render"
    profile["signal"] = "recursive crawl4ai (no llms.txt/sitemap)"
    return SourceSpec("web", url), profile


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
# (graph_name, corpus_digest) -> distilled OVERVIEW.md markdown
DistillerFn = Callable[[str, str], str]


# ── pipeline ────────────────────────────────────────────────────────────────


class SkillGraphPipeline:
    """Acquire from any source kind → standardized skill-graph (+ provenance manifest)."""

    def __init__(
        self,
        *,
        crawler_fn: CrawlerFn | None = None,
        generator_fn: GeneratorFn | None = None,
        distiller_fn: DistillerFn | None = None,
        kg_enrich: bool = True,
        kg_timeout: float = 300.0,
    ) -> None:
        self.crawler_fn = crawler_fn
        self.generator_fn = generator_fn
        self.distiller_fn = distiller_fn
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
            "llms": self._acquire_llms,
        }[spec.kind]
        return route(spec)

    def _acquire_llms(self, spec: SourceSpec) -> AcquiredBundle:
        """Acquire from the llms.txt / llms-full.txt standard — clean, complete docs.

        ``uri`` is the site root (or any page on it). Prefers ``llms-full.txt`` (the
        whole corpus in one fetch — no JS, no bot-blocking, no shrink), else parses
        ``llms.txt`` (a curated index of ``[name](url.md)`` links) and fetches each.
        Returns an empty bundle if neither is present so the caller can fall back.
        """
        docs = _fetch_llms_docs(
            spec.uri, max_pages=int(spec.options.get("max_pages", 1000))
        )
        return AcquiredBundle(spec, docs, extractor="llms.txt")

    def _acquire_web(self, spec: SourceSpec) -> AcquiredBundle:
        if self.crawler_fn is not None:
            docs = self.crawler_fn(spec)
            return AcquiredBundle(spec, docs, extractor="crawler_fn")
        # Prefer the real (JS-rendering) crawl4ai web-crawler when available so a
        # re-fetch reproduces the original crawl fidelity.
        crawler = _resolve_crawler()
        if crawler is not None:
            crawler_py, script = crawler
            docs = _crawl_via_script(spec, crawler_py, script)
            return AcquiredBundle(spec, docs, extractor="crawl4ai")
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
        record_specs: list[SourceSpec] | None = None,
    ) -> dict[str, Any]:
        """Acquire all sources → write a standardized skill-graph at ``out_dir/name``.

        ``record_specs`` overrides the *logical* sources recorded in ``sources.json``
        (and the SKILL.md ``source_url``) while ``specs`` still drives acquisition —
        used by wrap-migration to adopt existing content (``specs=[dir]``) yet record
        the durable upstream web sources so the graph stays re-crawlable.
        """
        # Acquire FIRST (into memory), THEN wipe + rewrite — so a slow/failed crawl
        # never destroys the existing corpus (acquisition touches its own temp dirs
        # only, never ``ref``). Critical for safe in-place migration/rebuild/refresh.
        bundles = [self.acquire(s) for s in specs]
        return self._build_from_bundles(
            name=name,
            bundles=bundles,
            out_dir=out_dir,
            description=description,
            max_file_kb=max_file_kb,
            kg_enrich=kg_enrich,
            version=version,
            record_specs=record_specs,
        )

    def _build_from_bundles(
        self,
        *,
        name: str,
        bundles: list[AcquiredBundle],
        out_dir: str | Path,
        description: str | None,
        max_file_kb: int,
        kg_enrich: bool | None,
        version: str | None,
        record_specs: list[SourceSpec] | None,
    ) -> dict[str, Any]:
        """Finalize from already-acquired bundles: wipe → write → split → KG → render.

        Split out of :meth:`build` so callers that need the acquired content *before*
        committing (e.g. :meth:`refresh_one`, to compare hashes and skip unchanged
        graphs) can acquire once and reuse it.
        """
        skill_dir = Path(out_dir) / name
        ref = skill_dir / "reference"

        if ref.exists():
            shutil.rmtree(ref)
        _write_reference_tree(ref, bundles, max_file_kb)

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
            self._maybe_ingest_kg([str(ref)], name) if do_kg else {"kg_ingested": False}
        )
        if kg_query_manifest is not None:
            kg_result.setdefault("kg_manifest", "kg_manifest.json")
            kg_result.setdefault("kg_ontology", kg_query_manifest.get("ontology"))
            kg_result.setdefault("concepts", _manifest_concepts(kg_query_manifest))

        md_files = sorted(ref.rglob("*.md"))
        version = version or "0.1.0"
        description = description or _default_description(name)

        # Logical sources recorded in the manifest: the override (durable upstream)
        # when given, else the acquisition bundles' own specs.
        if record_specs is not None:
            corpus_hash = sha256_text(
                "\n".join(
                    sorted(
                        f"{doc.source_uri}\n{doc.text}"
                        for b in bundles
                        for doc in b.docs
                    )
                )
            )
            recorded = [
                {
                    "kind": s.kind,
                    "uri": s.uri,
                    "options": dict(s.options),
                    "extractor": "wrap-existing",
                    "fetched_at": _now_iso(),
                    "content_hash": corpus_hash,
                    "doc_count": len(md_files),
                }
                for s in record_specs
            ]
            record_kinds = sorted({s.kind for s in record_specs})
            source_url = ", ".join(
                s.uri for s in record_specs if s.uri.startswith("http")
            )
        else:
            recorded = None
            record_kinds = None
            source_url = ", ".join(
                b.spec.uri for b in bundles if b.spec.uri.startswith("http")
            )

        self._write_sources_manifest(
            skill_dir,
            name,
            version,
            bundles,
            md_files,
            ref,
            kg_result,
            recorded=recorded,
        )
        _write_index_json(skill_dir, ref, name, version, source_url, kg_result)
        self._render_skill_md(
            skill_dir,
            name,
            description,
            version,
            bundles,
            md_files,
            ref,
            kg_result,
            source_types_override=record_kinds,
            source_url=source_url,
        )

        errors = validate_skill_graph(skill_dir)
        if errors:
            logger.warning("skill-graph %s validation: %s", name, errors)
        return {
            "skill_dir": str(skill_dir),
            "name": name,
            "version": version,
            "file_count": len(md_files),
            "source_count": len(bundles),
            "kg_ingested": bool(kg_result.get("kg_ingested")),
            "validation_errors": errors,
        }

    def _maybe_ingest_kg(self, targets: list[str], name: str) -> dict[str, Any]:
        """Best-effort, bounded ingest of ``targets`` (the reference dir, or — for a
        delta refresh — just the changed/added files) into the live KG.

        Decoupled process-boundary shell-out (mirrors the builder's prior pattern):
        sidesteps event-loop reentrancy and naturally bounds via the subprocess
        timeout. Any failure (no daemon, embedder 502, timeout) degrades to
        ``kg_ingested: false`` — the offline graph is already on disk.
        """
        # Fast health-gate: a real bounded embed probe. If the embedder is
        # definitively down (the GPU power-cycled), skip in ~6s instead of waiting out
        # the full ingest timeout — the offline graph is already on disk.
        if _embedder_responsive() is False:
            logger.info("KG enrichment skipped for %s: embedder unresponsive", name)
            return {"kg_ingested": False, "reason": "embedder_unhealthy"}
        if not targets:
            return {"kg_ingested": False, "reason": "no_targets"}
        cmd = [
            sys.executable,
            "-m",
            "agent_utilities.knowledge_graph.ingestion",
            *targets,
            "--content-type",
            "document",
        ]
        try:
            returncode, stdout, stderr = _run_bounded(cmd, timeout=self.kg_timeout)
        except Exception as exc:  # noqa: BLE001 — FileNotFound, timeout, etc.
            logger.info("KG enrichment skipped for %s: %s", name, exc)
            return {"kg_ingested": False, "reason": str(exc)}
        if returncode != 0:
            logger.info(
                "KG enrichment failed for %s (daemon down?): %s",
                name,
                (stderr or stdout or "").strip()[:300],
            )
            return {"kg_ingested": False, "reason": "ingest_failed"}
        nodes = edges = 0
        try:
            payload = json.loads(stdout or "{}")
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
        recorded: list[dict[str, Any]] | None = None,
    ) -> None:
        sources = (
            recorded
            if recorded is not None
            else [
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
        )
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
            "stats": {"file_count": len(md_files), "source_count": len(sources)},
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
        source_types_override: list[str] | None = None,
        source_url: str = "",
    ) -> None:
        source_types = source_types_override or sorted({b.spec.kind for b in bundles})
        concepts = kg_result.get("concepts") or []
        toc = _render_toc(_build_doc_tree(ref))
        title = name.replace("-", " ").replace("docs", "").strip().title()
        kg_on = bool(kg_result.get("kg_ingested"))
        domain = kg_result.get("domain") or f"skillgraph:{name}"
        urls = [u.strip() for u in source_url.split(",") if u.strip()]
        total_kb = sum(p.stat().st_size for p in md_files) // 1024

        # ── frontmatter ──
        lines = ["---", f"name: {name}", f"description: {description}"]
        lines.append(f"skill_graph_version: {version}")
        lines.append(f"source_types: [{', '.join(source_types)}]")
        if source_url:
            lines.append(f"source_url: {source_url}")
        lines.append(f"built_at: {_now_iso()}")
        lines.append(f"builder_version: {_pkg_version()}")
        lines.append(f"file_count: {len(md_files)}")
        lines.append(f"kg_ingested: {str(kg_on).lower()}")
        lines.append("index: index.json")
        if (skill_dir / "OVERVIEW.md").exists():
            lines.append("overview: OVERVIEW.md")
        if kg_result.get("kg_manifest"):
            lines.append(f"kg_manifest: {kg_result['kg_manifest']}")
        if kg_result.get("kg_ontology"):
            lines.append(f"kg_ontology: {kg_result['kg_ontology']}")
        if concepts:
            lines.append(f"concepts: [{', '.join(repr(c) for c in concepts)}]")
        lines.append("categories: [Documentation, Knowledge Base, Reference]")
        lines.append(f"tags: [docs, reference, {name}, knowledge-base]")
        lines.append("---")

        # ── header + badge table ──
        _nodes = int(kg_result.get("nodes") or 0)
        kg_cell = (
            f"✅ ingested ({f'{_nodes} nodes, ' if _nodes else ''}domain `{domain}`)"
            if kg_on
            else "— (offline corpus)"
        )
        lines += ["", f"# {title} — Reference Skill-Graph", "", f"> {description}", ""]
        lines += [
            "| | |",
            "|---|---|",
            f"| **Version** | {version} |",
            f"| **Files** | {len(md_files)} ({total_kb} KB) |",
            f"| **Source types** | {', '.join(source_types)} |",
            f"| **Knowledge Graph** | {kg_cell} |",
            f"| **Built** | {time.strftime('%B %d, %Y', time.gmtime())} |",
        ]
        if urls:
            lines.append("")
            lines.append("**Sources:** " + ", ".join(f"[{u}]({u})" for u in urls))
        lines.append("")

        # ── agent usage guidance (the leverage layer) ──
        lines += ["## 🧭 How to use this skill-graph", ""]
        lines.append(
            f"This is a **full reference corpus for {title}** — a manual at your "
            "disposal. Treat it as ground truth: quote it, don't paraphrase from memory."
        )
        lines.append("")
        if (skill_dir / "OVERVIEW.md").exists():
            lines.append(
                "- **Start here:** read **[OVERVIEW.md](OVERVIEW.md)** — the distilled "
                "essence + cheatsheet of this corpus — then drill into `reference/` for detail."
            )
        lines.append(
            "- **Look something up:** scan the Table of Contents (or `index.json` for a "
            "machine-readable map), open the specific `reference/…` file, quote it + link it."
        )
        if kg_on:
            lines.append(
                "- **Cross-cutting question:** this corpus is in the Knowledge Graph — "
                f'`graph_search(query="…", mode="hybrid")` retrieves the right passages '
                f"across all files at once (domain `{domain}`). Prefer it for synthesis."
            )
        lines.append(
            "- **Stay grounded:** never invent APIs/flags — verify against the reference "
            "and cite the file. `sources.json` tracks provenance + freshness."
        )
        lines.append("")

        # ── table of contents ──
        lines += ["## 📚 Table of Contents", ""]
        lines.append("\n".join(toc) if toc else "*No markdown files found.*")
        lines.append("")

        # ── knowledge-graph / ontology cross-links ──
        if kg_on:
            lines += ["## 🔗 Knowledge Graph & Ontology", ""]
            lines.append(
                f"Ingested as a `SkillGraph` ontology object over domain `{domain}`: it "
                "`CONTAINS` its Documents, `RELATES_TO` the Concepts it covers, and is "
                "`DERIVED_FROM` its sources. Discover overlap/related graphs via "
                "`ontology_interface(action='implementers', name='SkillGraph')` or "
                "`graph_search`."
            )
            if concepts:
                lines.append("")
                lines.append(
                    "**Covers concepts:** " + ", ".join(f"`{c}`" for c in concepts)
                )
            lines.append("")

        (skill_dir / "SKILL.md").write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )

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
        shrink_guard: bool = True,
        detect: bool = True,
    ) -> dict[str, Any]:
        """Migrate one legacy skill-graph in place to the standardized contract.

        ``mode='auto'`` follows :meth:`classify_legacy`; ``reacquire`` re-crawls the
        legacy ``source_url``; ``wrap`` re-packages the existing ``reference/`` tree
        (offline, content-preserving). The first standardized build is versioned
        ``1.0.0``. With ``shrink_guard`` a reacquire that crawls far less than the
        existing corpus (a moved source_url) keeps the old content and reports
        ``stale_url``. Returns the build result (or a ``skipped`` record for natives).
        """
        d = Path(skill_dir)
        info = self.classify_legacy(d)
        chosen = info["mode"] if mode == "auto" else mode
        if chosen in ("managed", "native"):
            return {"name": d.name, "skipped": True, "reason": chosen}

        depth = int(info["crawl_depth"] or 2)
        scrape_profile: dict[str, Any] = {}
        if chosen == "reacquire":
            specs = self._specs_from_source_url(info["source_url"], depth)
            if not specs:
                return {
                    "name": d.name,
                    "skipped": True,
                    "reason": "no usable source_url",
                }
            # SiteProfiler: probe the primary URL and prefer the best strategy
            # (llms.txt/llms-full.txt → sitemap → recursive render) over a blind crawl.
            if detect:
                primary = next(
                    (s.uri for s in specs if s.uri.startswith("http")), specs[0].uri
                )
                best, scrape_profile = detect_scrape_strategy(primary)
                if best.kind == "web":
                    best.options.setdefault("max_depth", depth)
                specs = [best]
            try:
                bundles = [self.acquire(s) for s in specs]
            except Exception as exc:  # noqa: BLE001 — crawl failed → keep content
                return {
                    "name": d.name,
                    "status": "failed",
                    "reason": str(exc)[:200],
                    "scrape_profile": scrape_profile,
                }
            if shrink_guard and _is_shrink(
                _ref_bytes(d / "reference"), _bundle_bytes(bundles)
            ):
                return {
                    "name": d.name,
                    "status": "stale_url",
                    "existing_bytes": _ref_bytes(d / "reference"),
                    "new_bytes": _bundle_bytes(bundles),
                    "reason": "re-crawl far smaller than existing — source_url likely moved",
                    "scrape_profile": scrape_profile,
                }
            result = self._build_from_bundles(
                name=d.name,
                bundles=bundles,
                out_dir=d.parent,
                description=info["description"] or None,
                max_file_kb=50,
                kg_enrich=kg_enrich,
                version="1.0.0",
                record_specs=None,
            )
            # Stamp the chosen strategy into the manifest (the per-graph ledger entry).
            if scrape_profile:
                mpath = d / "sources.json"
                if mpath.exists():
                    data = json.loads(mpath.read_text(encoding="utf-8"))
                    data["scrape_profile"] = scrape_profile
                    mpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
                result["scrape_profile"] = scrape_profile
        else:  # wrap — adopt the existing corpus in place. build() acquires into memory
            # BEFORE wiping reference/, so the graph's own reference/ is a safe dir
            # source (no temp copy). Record the upstream web sources (not the local
            # dir) so the wrapped graph stays re-crawlable via status/rebuild.
            specs = [SourceSpec("dir", str(d / "reference"))]
            record_specs = (
                self._specs_from_source_url(info["source_url"], depth) or None
            )
            result = self.build(
                name=d.name,
                specs=specs,
                out_dir=d.parent,
                description=info["description"] or None,
                kg_enrich=kg_enrich,
                version="1.0.0",
                record_specs=record_specs,
            )
        result["migrated_mode"] = chosen
        return result

    # ── periodic refresh (re-download + delta re-ingest) ──────────────────────

    def refresh_one(
        self,
        skill_dir: str | Path,
        *,
        force: bool = False,
        kg_enrich: bool | None = None,
        shrink_guard: bool = True,
    ) -> dict[str, Any]:
        """Re-acquire a managed graph's recorded sources; rewrite + re-ingest if changed.

        Crawls once, compares each source's content hash to ``sources.json``, and skips
        the rewrite + KG re-ingest when nothing changed — so periodic refreshes are
        cheap and only genuinely-changed corpora bump their version and re-embed (the
        KG ingest itself is content-hash delta-skipped, KG-2.8). A crawl failure leaves
        the existing graph untouched (acquire-before-wipe). When ``shrink_guard`` and the
        re-crawl is far smaller than the current corpus (a moved source_url), the existing
        content is kept and the graph reported ``stale_url`` instead of being overwritten.
        """
        d = Path(skill_dir)
        mpath = d / "sources.json"
        if not mpath.exists():
            return {"name": d.name, "status": "skipped", "reason": "not managed"}
        data = json.loads(mpath.read_text(encoding="utf-8"))
        src_dicts = data.get("sources", [])
        specs = [SourceSpec.from_dict(s) for s in src_dicts]
        try:
            bundles = [self.acquire(s) for s in specs]
        except Exception as exc:  # noqa: BLE001 — source unreachable / crawl failed
            return {"name": d.name, "status": "failed", "reason": str(exc)[:200]}
        new_hashes = [b.content_hash for b in bundles]
        old_hashes = [s.get("content_hash") for s in src_dicts]
        if not force and new_hashes == old_hashes:
            return {
                "name": d.name,
                "status": "fresh",
                "version": data.get("skill_graph_version"),
            }
        existing_bytes = _ref_bytes(d / "reference")
        new_bytes = _bundle_bytes(bundles)
        if shrink_guard and _is_shrink(existing_bytes, new_bytes):
            return {
                "name": d.name,
                "status": "stale_url",
                "existing_bytes": existing_bytes,
                "new_bytes": new_bytes,
                "reason": "re-crawl far smaller than existing — source_url likely moved",
            }
        fm = parse_frontmatter((d / "SKILL.md").read_text(encoding="utf-8"))
        version = _bump_patch(data.get("skill_graph_version") or "0.1.0")
        # Delta update: write only the changed/added files, delete removed ones, leave
        # unchanged files (and their bytes/embeddings) untouched, and re-ingest ONLY the
        # changed files into the KG — instead of rewriting + re-embedding the whole tree.
        return self._apply_delta(
            d,
            bundles,
            data=data,
            description=fm.get("description"),
            version=version,
            kg_enrich=kg_enrich,
            force=force,
        )

    def _apply_delta(
        self,
        skill_dir: Path,
        bundles: list[AcquiredBundle],
        *,
        data: dict[str, Any],
        description: str | None,
        version: str,
        kg_enrich: bool | None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Apply only the file-level delta of a re-acquisition to a managed graph.

        Builds the candidate tree in a temp dir, diffs it (by path + sha256) against the
        live ``reference/``, and writes only added/changed files, deletes removed ones,
        and leaves unchanged files in place — then re-ingests ONLY the changed/added
        files (KG-2.8 also content-hash-skips). Avoids rewriting + re-embedding an entire
        corpus when one page moved.
        """
        d = Path(skill_dir)
        ref = d / "reference"
        tmp = Path(tempfile.mkdtemp(prefix="sg_delta_"))
        tmp_ref = tmp / "reference"
        try:
            _write_reference_tree(tmp_ref, bundles, 50)
            new = {
                p.relative_to(tmp_ref).as_posix(): sha256_bytes(p.read_bytes())
                for p in tmp_ref.rglob("*.md")
            }
            old = (
                {
                    p.relative_to(ref).as_posix(): sha256_bytes(p.read_bytes())
                    for p in ref.rglob("*.md")
                }
                if ref.is_dir()
                else {}
            )
            added = sorted(p for p in new if p not in old)
            removed = sorted(p for p in old if p not in new)
            changed = sorted(p for p in new if p in old and new[p] != old[p])
            unchanged = [p for p in new if p in old and new[p] == old[p]]

            if not force and not (added or removed or changed):
                return {
                    "name": d.name,
                    "status": "fresh",
                    "version": data.get("skill_graph_version"),
                    "delta": {
                        "added": 0,
                        "changed": 0,
                        "removed": 0,
                        "unchanged": len(unchanged),
                    },
                }

            ref.mkdir(parents=True, exist_ok=True)
            for rel in (*added, *changed):
                dst = ref / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tmp_ref / rel, dst)
            for rel in removed:
                (ref / rel).unlink(missing_ok=True)

            do_kg = self.kg_enrich if kg_enrich is None else kg_enrich
            # Re-ingest only the changed/added files; --force re-ingests the whole tree.
            ingest_rels = sorted(new) if force else [*added, *changed]
            changed_files = [str(ref / rel) for rel in ingest_rels]
            if do_kg and changed_files:
                kg_result = self._maybe_ingest_kg(changed_files, d.name)
            else:
                kg_result = {
                    "kg_ingested": bool(data.get("kg_ingested")),
                    "kg_ontology": data.get("kg_ontology"),
                    "concepts": data.get("concepts") or [],
                    "domain": f"skillgraph:{d.name}",
                }
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        md_files = sorted(ref.rglob("*.md"))
        source_url = ", ".join(
            b.spec.uri for b in bundles if b.spec.uri.startswith("http")
        )
        self._write_sources_manifest(
            d, d.name, version, bundles, md_files, ref, kg_result
        )
        _write_index_json(d, ref, d.name, version, source_url, kg_result)
        self._render_skill_md(
            d,
            d.name,
            description or _default_description(d.name),
            version,
            bundles,
            md_files,
            ref,
            kg_result,
            source_url=source_url,
        )
        return {
            "name": d.name,
            "status": "refreshed",
            "version": version,
            "file_count": len(md_files),
            "kg_ingested": bool(kg_result.get("kg_ingested")),
            "delta": {
                "added": len(added),
                "changed": len(changed),
                "removed": len(removed),
                "unchanged": len(unchanged),
            },
        }

    def refresh_all(
        self,
        root: str | Path,
        *,
        force: bool = False,
        kg_enrich: bool | None = None,
        only: str = "",
        limit: int = 0,
        shrink_guard: bool = True,
    ) -> dict[str, Any]:
        """Refresh every managed (sources.json) graph under ``root``; report per-graph."""
        names = {n.strip() for n in only.split(",") if n.strip()}
        results: list[dict[str, Any]] = []
        refreshed = 0
        for d in _iter_graph_dirs(Path(root)):
            if not (d / "sources.json").exists():
                continue
            if names and d.name not in names:
                continue
            res = self.refresh_one(
                d, force=force, kg_enrich=kg_enrich, shrink_guard=shrink_guard
            )
            results.append(res)
            if res.get("status") == "refreshed":
                refreshed += 1
                if limit and refreshed >= limit:
                    break
        return {"refreshed": refreshed, "results": results}

    # ── restyle (re-render presentation only) ─────────────────────────────────

    def restyle_one(self, skill_dir: str | Path) -> dict[str, Any]:
        """Re-render SKILL.md + index.json from existing content — no re-crawl/re-ingest.

        Applies the current polished presentation (badges, usage guidance, KG-query
        hints, machine-readable index) to a managed graph using the state already in
        ``sources.json``. Cheap and offline — the way to roll a renderer upgrade across
        every graph without touching content.
        """
        d = Path(skill_dir)
        mpath = d / "sources.json"
        if not mpath.exists():
            return {"name": d.name, "status": "skipped", "reason": "not managed"}
        data = json.loads(mpath.read_text(encoding="utf-8"))
        ref = d / "reference"
        md_files = sorted(ref.rglob("*.md"))
        fm = (
            parse_frontmatter((d / "SKILL.md").read_text(encoding="utf-8"))
            if (d / "SKILL.md").exists()
            else {}
        )
        name = data.get("name", d.name)
        version = data.get("skill_graph_version") or "0.1.0"
        kg_result = {
            "kg_ingested": data.get("kg_ingested"),
            "kg_ontology": data.get("kg_ontology"),
            "kg_manifest": data.get("kg_manifest"),
            "concepts": data.get("concepts") or [],
            "domain": f"skillgraph:{name}",
        }
        source_kinds = sorted(
            {s["kind"] for s in data.get("sources", []) if s.get("kind")}
        )
        source_url = ", ".join(
            s.get("uri", "")
            for s in data.get("sources", [])
            if str(s.get("uri", "")).startswith("http")
        )
        _write_index_json(d, ref, name, version, source_url, kg_result)
        self._render_skill_md(
            d,
            name,
            fm.get("description") or _default_description(name),
            version,
            [],
            md_files,
            ref,
            kg_result,
            source_types_override=source_kinds,
            source_url=source_url,
        )
        return {"name": name, "status": "restyled", "file_count": len(md_files)}

    def restyle_all(self, root: str | Path) -> dict[str, Any]:
        """Restyle every managed graph under ``root`` (presentation refresh, offline)."""
        results = [
            self.restyle_one(d)
            for d in _iter_graph_dirs(Path(root))
            if (d / "sources.json").exists()
        ]
        return {
            "restyled": sum(r["status"] == "restyled" for r in results),
            "results": results,
        }

    # ── distilled-knowledge layer (the 'distilled knowledge' tier) ────────────

    def distill_one(self, skill_dir: str | Path) -> dict[str, Any]:
        """LLM-distill the corpus into an OVERVIEW.md (essence + cheatsheet), then
        re-render so SKILL.md surfaces it.

        Three tiers result: SKILL.md (map) → OVERVIEW.md (distilled essence) →
        reference/ (the full manual). The agent reads the overview first for instant
        grounding and drills into reference only when it needs detail. Needs an LLM;
        injectable via ``distiller_fn`` for offline use.
        """
        d = Path(skill_dir)
        ref = d / "reference"
        if not ref.is_dir() or not any(ref.rglob("*.md")):
            return {
                "name": d.name,
                "status": "skipped",
                "reason": "no reference content",
            }
        digest = _corpus_digest(ref)
        try:
            overview = (
                self.distiller_fn(d.name, digest)
                if self.distiller_fn is not None
                else _default_distill(d.name, digest)
            )
        except Exception as exc:  # noqa: BLE001 — LLM/model unavailable
            return {"name": d.name, "status": "failed", "reason": str(exc)[:200]}
        if not overview.strip():
            return {"name": d.name, "status": "failed", "reason": "empty overview"}
        (d / "OVERVIEW.md").write_text(_optimize_markdown(overview), encoding="utf-8")
        mpath = d / "sources.json"
        if mpath.exists():
            data = json.loads(mpath.read_text(encoding="utf-8"))
            data["distilled"] = True
            mpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.restyle_one(d)  # re-render to link the new overview
        return {
            "name": d.name,
            "status": "distilled",
            "overview_chars": len(overview),
        }

    def distill_all(self, root: str | Path, *, limit: int = 0) -> dict[str, Any]:
        """Distill every managed graph under ``root`` (LLM; bounded by ``limit``)."""
        results: list[dict[str, Any]] = []
        done = 0
        for d in _iter_graph_dirs(Path(root)):
            if not (d / "sources.json").exists():
                continue
            res = self.distill_one(d)
            results.append(res)
            if res.get("status") == "distilled":
                done += 1
                if limit and done >= limit:
                    break
        return {"distilled": done, "results": results}


# ── module helpers (pure / optional-dep guarded) ───────────────────────────────


def _write_reference_tree(
    ref: Path, bundles: list[AcquiredBundle], max_file_kb: int
) -> None:
    """Write the merged + content-optimized reference tree, then split oversized files.

    Shared by the full rebuild and the delta path (which writes into a temp tree to
    diff against the live one). Normalizes whitespace and drops exact-duplicate pages
    (crawls emit the same page under many URLs) and empty shells.
    """
    ref.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    seen_hashes: set[str] = set()
    for bundle in bundles:
        for doc in bundle.docs:
            text = _optimize_markdown(doc.text)
            if not text.strip():
                continue
            h = sha256_text(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            rel = _unique_rel(doc.rel_path, used)
            target = ref / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    if max_file_kb > 0:
        _split_oversized(ref, max_file_kb)


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


def _corpus_digest(ref: Path, max_chars: int = 24000, per_file: int = 900) -> str:
    """A bounded digest of the corpus (path + head of each file) to feed a distiller."""
    parts: list[str] = []
    total = 0
    for p in sorted(ref.rglob("*.md")):
        try:
            head = p.read_text(encoding="utf-8", errors="replace").strip()[:per_file]
        except OSError:
            continue
        block = f"## FILE: {p.relative_to(ref).as_posix()}\n{head}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def _default_distill(name: str, digest: str) -> str:
    """Distill a corpus digest into an OVERVIEW.md via the role-resolved chat model."""
    from pydantic_ai import Agent

    from agent_utilities.core.model_factory import create_model

    title = name.replace("-", " ").replace("docs", "").strip().title()
    agent = Agent(
        create_model(role="generator"),
        system_prompt=(
            "You distill a documentation corpus into a concise, high-signal OVERVIEW "
            "for an AI agent. Output Markdown only; be dense and accurate; never invent."
        ),
    )
    prompt = (
        f"Excerpts from the '{title}' reference corpus follow. Produce OVERVIEW.md with:\n"
        f"# {title} — Distilled Overview\n"
        "## What it is (2-3 sentences)\n"
        "## Core concepts (bulleted, one line each)\n"
        "## Quick reference / cheatsheet (most-used APIs, commands, patterns)\n"
        "## Common tasks (task -> terse how-to)\n"
        "## Gotchas & pitfalls\n"
        "## Where to look (which reference/ files cover what)\n\n"
        "Corpus excerpts:\n\n" + digest
    )
    result = agent.run_sync(prompt)
    return str(getattr(result, "output", None) or getattr(result, "data", "") or "")


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


def _count_leaves(tree: dict[str, Any]) -> int:
    return sum(_count_leaves(v) if isinstance(v, dict) else 1 for v in tree.values())


def _render_toc(tree: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    dirs = sorted(k for k in tree if isinstance(tree[k], dict))
    files = sorted(k for k in tree if not isinstance(tree[k], dict))
    for key in dirs:
        lines.append("  " * indent + f"- 📁 **{key}/** ({_count_leaves(tree[key])})")
        lines.extend(_render_toc(tree[key], indent + 1))
    for key in files:
        title, link = tree[key]
        lines.append("  " * indent + f"- [{title}]({link})")
    return lines


def _optimize_markdown(text: str) -> str:
    """Normalize markdown for denser signal: strip trailing spaces, collapse blank runs."""
    t = text.replace("\r\n", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip() + "\n"


def _section_headings(path: Path, limit: int = 12) -> list[str]:
    """The first few H1-H3 headings of a markdown file (a lightweight symbol index)."""
    out: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^#{1,3}\s+(.+)", line)
            if m:
                out.append(m.group(1).strip())
                if len(out) >= limit:
                    break
    except OSError:
        pass
    return out


def _write_index_json(
    skill_dir: Path,
    ref: Path,
    name: str,
    version: str,
    source_url: str,
    kg_result: dict[str, Any],
) -> None:
    """Emit ``index.json`` — a machine-readable navigation map over the corpus.

    Schema ``skill-graph-index/v1``: per-file path/title/group/bytes/headings + group
    counts, so an agent can jump straight to the right file/section programmatically
    instead of scanning the markdown TOC.
    """
    md_files = sorted(ref.rglob("*.md"))
    groups: dict[str, int] = {}
    sections: list[dict[str, Any]] = []
    for p in md_files:
        group = p.parent.relative_to(ref).as_posix() or "."
        groups[group] = groups.get(group, 0) + 1
        sections.append(
            {
                "path": p.relative_to(skill_dir).as_posix(),
                "title": _extract_title(p),
                "group": group,
                "bytes": p.stat().st_size,
                "headings": _section_headings(p),
            }
        )
    index = {
        "schema": "skill-graph-index/v1",
        "name": name,
        "skill_graph_version": version,
        "kg_ingested": bool(kg_result.get("kg_ingested")),
        "kg_domain": kg_result.get("domain") or f"skillgraph:{name}",
        "concepts": kg_result.get("concepts") or [],
        "source_url": source_url,
        "file_count": len(md_files),
        "groups": dict(sorted(groups.items())),
        "sections": sections,
    }
    (skill_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


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
    guard = not args.no_shrink_guard
    if args.dir:
        result = pipe.migrate_legacy(args.dir, mode=args.mode, shrink_guard=guard)
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
        res = pipe.migrate_legacy(d, mode=args.mode, shrink_guard=guard)
        results.append(res)
        migrated += 0 if res.get("skipped") else 1
        if args.limit and migrated >= args.limit:
            break
    print(json.dumps({"applied": args.apply, "results": results}, indent=2))
    return 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    pipe = SkillGraphPipeline(kg_enrich=not args.no_kg)
    guard = not args.no_shrink_guard
    if args.dir:
        print(
            json.dumps(
                pipe.refresh_one(args.dir, force=args.force, shrink_guard=guard),
                indent=2,
            )
        )
        return 0
    report = pipe.refresh_all(
        args.root,
        force=args.force,
        only=args.only,
        limit=args.limit,
        shrink_guard=guard,
    )
    print(json.dumps(report, indent=2))
    return 0


def _cmd_restyle(args: argparse.Namespace) -> int:
    pipe = SkillGraphPipeline(kg_enrich=False)
    if args.dir:
        print(json.dumps(pipe.restyle_one(args.dir), indent=2))
    else:
        print(json.dumps(pipe.restyle_all(args.root), indent=2))
    return 0


def _cmd_distill(args: argparse.Namespace) -> int:
    pipe = SkillGraphPipeline(kg_enrich=False)
    if args.dir:
        print(json.dumps(pipe.distill_one(args.dir), indent=2))
    else:
        print(json.dumps(pipe.distill_all(args.root, limit=args.limit), indent=2))
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
    m.add_argument(
        "--no-shrink-guard",
        action="store_true",
        help="Overwrite even when a reacquire crawls far less than the existing corpus.",
    )
    m.set_defaults(func=_cmd_migrate)

    rf = sub.add_parser(
        "refresh",
        help="Re-download managed graph(s) and re-ingest only the changed corpora.",
    )
    rfg = rf.add_mutually_exclusive_group(required=True)
    rfg.add_argument("--dir", help="Refresh a single managed skill-graph directory.")
    rfg.add_argument("--root", help="Refresh every managed graph under this root.")
    rf.add_argument(
        "--force",
        action="store_true",
        help="Rewrite + re-ingest even when the re-crawled content is unchanged.",
    )
    rf.add_argument(
        "--only", default="", help="Comma-separated graph names to refresh."
    )
    rf.add_argument(
        "--limit", type=int, default=0, help="Max graphs to refresh (changed ones)."
    )
    rf.add_argument("--no-kg", action="store_true")
    rf.add_argument(
        "--no-shrink-guard",
        action="store_true",
        help="Overwrite even when a re-crawl is far smaller than the existing corpus.",
    )
    rf.set_defaults(func=_cmd_refresh)

    rs = sub.add_parser(
        "restyle",
        help="Re-render SKILL.md + index.json from existing content (no re-crawl).",
    )
    rsg = rs.add_mutually_exclusive_group(required=True)
    rsg.add_argument("--dir", help="Restyle a single managed skill-graph directory.")
    rsg.add_argument("--root", help="Restyle every managed graph under this root.")
    rs.set_defaults(func=_cmd_restyle)

    ds = sub.add_parser(
        "distill",
        help="LLM-distill graph(s) into an OVERVIEW.md (essence + cheatsheet tier).",
    )
    dsg = ds.add_mutually_exclusive_group(required=True)
    dsg.add_argument("--dir", help="Distill a single managed skill-graph directory.")
    dsg.add_argument("--root", help="Distill every managed graph under this root.")
    ds.add_argument("--limit", type=int, default=0, help="Max graphs to distill.")
    ds.set_defaults(func=_cmd_distill)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
