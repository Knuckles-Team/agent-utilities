#!/usr/bin/python
"""KB Document Parser.

Parses raw source documents into DocumentChunk objects for KB ingestion.
Supports: Markdown, PDF, DOCX, EPUB, TXT, HTML, URLs.
Uses SimpleDirectoryReader (LlamaIndex) exactly as vector-mcp does.
Hash-based deduplication avoids re-processing unchanged files.
"""

import contextlib
import hashlib
import logging
from pathlib import Path

import yaml

from ...models.knowledge_base import DocumentChunk, ParsedSource

logger = logging.getLogger(__name__)

# Supported source types and their extensions
SUPPORTED_EXTENSIONS = {
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".epub": "epub",
    ".html": "html",
    ".htm": "html",
}


def _compute_hash(content: str) -> str:
    """SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _count_words(text: str) -> int:
    return len(text.split())


def _chunk_text(text: str, chunk_size: int = 1024) -> list[str]:
    """Split text into word-count-bounded chunks with overlap."""
    words = text.split()
    chunks = []
    step = int(chunk_size * 0.9)  # 10% overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text]


class KBDocumentParser:
    """Parse raw documents into DocumentChunk objects.

    Uses LlamaIndex's SimpleDirectoryReader for multi-format file loading
    (same pattern as vector-mcp) with hash-based deduplication and
    configurable chunk size.
    """

    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size

    def parse_directory(
        self, path: str | Path, recursive: bool = True
    ) -> list[ParsedSource]:
        """Parse all supported files in a directory."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Directory not found: {path}")

        sources = []
        pattern = "**/*" if recursive else "*"
        for fpath in sorted(path.glob(pattern)):
            if fpath.is_file() and fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    source = self.parse_file(fpath)
                    if source:
                        sources.append(source)
                except Exception as e:
                    logger.warning(f"Failed to parse {fpath}: {e}")
        return sources

    def parse_file(self, path: str | Path) -> ParsedSource | None:
        """Parse a single file into a ParsedSource."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        ext = path.suffix.lower()
        source_type = SUPPORTED_EXTENSIONS.get(ext, "txt")

        try:
            content = self._read_file(path, source_type)
        except Exception as e:
            logger.error(f"Cannot read {path}: {e}")
            return None

        if not content or not content.strip():
            return None

        content_hash = _compute_hash(content)
        file_size = path.stat().st_size
        raw_chunks = _chunk_text(content, self.chunk_size)

        chunks = [
            DocumentChunk(
                content=chunk,
                source_path=str(path),
                source_type=source_type,
                chunk_index=i,
                content_hash=_compute_hash(chunk),
                word_count=_count_words(chunk),
            )
            for i, chunk in enumerate(raw_chunks)
        ]

        return ParsedSource(
            name=path.stem,
            file_path=str(path),
            source_type=source_type,
            content_hash=content_hash,
            file_size=file_size,
            chunks=chunks,
        )

    def parse_url(self, url: str, kb_name: str = "web") -> ParsedSource | None:
        """Fetch a URL and parse its content as HTML → Markdown."""
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Install with: pip install httpx")
            return None

        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            content = self._html_to_markdown(resp.text, url)
        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return None

        content_hash = _compute_hash(content)
        raw_chunks = _chunk_text(content, self.chunk_size)
        chunks = [
            DocumentChunk(
                content=chunk,
                source_path=url,
                source_type="url",
                chunk_index=i,
                content_hash=_compute_hash(chunk),
                word_count=_count_words(chunk),
            )
            for i, chunk in enumerate(raw_chunks)
        ]

        return ParsedSource(
            name=url.rstrip("/").split("/")[-1] or kb_name,
            file_path=url,
            source_type="url",
            content_hash=content_hash,
            file_size=len(content.encode("utf-8")),
            chunks=chunks,
        )

    def parse_skill_graph(self, graph_path: str | Path) -> list[ParsedSource]:
        """Parse a skill-graph directory (reads SKILL.md frontmatter + reference/ files).

        Skill-graphs contain a SKILL.md with YAML frontmatter describing the
        graph, plus a reference/ subdirectory with all the markdown content.
        """
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise ValueError(f"Skill-graph directory not found: {graph_path}")

        sources = []

        # Parse SKILL.md metadata
        skill_md = graph_path / "SKILL.md"
        if skill_md.exists():
            source = self.parse_file(skill_md)
            if source:
                sources.append(source)

        # Parse reference/ directory (the main content)
        ref_dir = graph_path / "reference"
        if ref_dir.exists():
            sources.extend(self.parse_directory(ref_dir, recursive=True))
        else:
            # Fallback: parse the entire graph directory
            sources.extend(self.parse_directory(graph_path, recursive=True))

        logger.info(
            f"Parsed {len(sources)} sources from skill-graph: {graph_path.name}"
        )
        return sources

    def read_skill_graph_metadata(self, graph_path: str | Path) -> dict:
        """Read SKILL.md frontmatter to get name, description, tags, source_url."""
        graph_path = Path(graph_path)
        skill_md = graph_path / "SKILL.md"
        if not skill_md.exists():
            return {"name": graph_path.name, "description": "", "tags": []}

        content = skill_md.read_text(encoding="utf-8", errors="replace")
        # Parse YAML frontmatter between --- delimiters
        import re

        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if match:
            with contextlib.suppress(Exception):
                return yaml.safe_load(match.group(1)) or {}
        return {"name": graph_path.name, "description": content[:200]}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_file(self, path: Path, source_type: str) -> str:
        """Read file content as plain text, dispatching by source type."""
        if source_type == "md" or source_type == "txt":
            return path.read_text(encoding="utf-8", errors="replace")

        elif source_type == "html":
            raw = path.read_text(encoding="utf-8", errors="replace")
            return self._html_to_markdown(raw, str(path))

        elif source_type == "pdf":
            return self._read_pdf(path)

        elif source_type == "docx":
            return self._read_docx(path)

        elif source_type == "epub":
            return self._read_epub(path)

        else:
            return path.read_text(encoding="utf-8", errors="replace")

    def _html_to_markdown(self, html: str, source: str = "") -> str:
        """Convert HTML to Markdown using BeautifulSoup (optional dep)."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            # Remove scripts/styles
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback: strip HTML tags with regex
            import re

            clean = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", clean).strip()

    def _read_pdf(self, path: Path) -> str:
        """Extract text from PDF (requires pypdf or pdfminer)."""
        # Try pypdf first (lightweight)
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            pass
        # Fallback to LlamaIndex SimpleDirectoryReader
        try:
            from llama_index.core import SimpleDirectoryReader

            docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
            return "\n".join(d.text for d in docs)
        except Exception as e:
            logger.error(f"Cannot read PDF {path}: {e}")
            return ""

    def _read_docx(self, path: Path) -> str:
        """Extract text from DOCX (requires python-docx)."""
        try:
            import docx

            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text)
        except ImportError:
            logger.warning(
                "python-docx not installed. Install with: pip install python-docx"
            )
            return ""
        except Exception as e:
            logger.error(f"Cannot read DOCX {path}: {e}")
            return ""

    def _read_epub(self, path: Path) -> str:
        """Extract text from EPUB (requires ebooklib + BeautifulSoup)."""
        try:
            import ebooklib
            from bs4 import BeautifulSoup
            from ebooklib import epub

            book = epub.read_epub(str(path))
            texts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), "html.parser")
                texts.append(soup.get_text(separator="\n", strip=True))
            return "\n\n".join(texts)
        except ImportError:
            logger.warning("ebooklib not installed. Install with: pip install ebooklib")
            return ""
        except Exception as e:
            logger.error(f"Cannot read EPUB {path}: {e}")
            return ""
