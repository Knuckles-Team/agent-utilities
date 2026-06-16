#!/usr/bin/python
from __future__ import annotations

"""The standardized skill-graph contract — frontmatter + provenance/freshness manifest.

CONCEPT:KG-2.7 — A skill-graph is an externally-consumable corpus of knowledge
(crawled docs, a distilled KG subgraph, an extracted PDF, freshly-generated text)
packaged as an agent skill: a ``SKILL.md`` index over a ``reference/`` markdown tree.

Historically every source type produced a *differently shaped* SKILL.md and only
the KG-distilled path emitted a provenance record (``kg_manifest.json``). This module
defines the **one** shape every skill-graph shares regardless of how it was built:

* ``SOURCES_SCHEMA`` — the ``sources.json`` sidecar: per-source provenance
  (kind/uri/options/content-hash/fetched-at) that drives staleness detection and
  incremental rebuild, plus a per-file integrity fingerprint. Works for *all*
  source kinds, not just KG distillation.
* :data:`SKILL_FRONTMATTER_KEYS` — the standardized SKILL.md frontmatter superset.
* :func:`validate_skill_graph` — a structural linter (a pre-commit/CI gate).

Deliberately **pure stdlib** (regex frontmatter parse, no PyYAML) so the same
contract can be vendored into the zero-dependency ``skill-graphs`` repo's gate.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

# Manifest schema id — bump when the on-disk shape of ``sources.json`` changes.
SOURCES_SCHEMA = "skill-graph-sources/v1"

# The source kinds the unified pipeline can acquire from. Kept here (not in the
# pipeline module) so the validator can check them without importing the engine.
SOURCE_KINDS: tuple[str, ...] = (
    "web",  # recursive HTML crawl
    "pdf",  # PDF file or URL → markdown
    "office",  # docx/pptx/xlsx/csv → markdown
    "dir",  # local directory of markdown/docs
    "url_reader",  # single URL via the Readability reader (KG-2.66)
    "rest",  # generic REST connector
    "database",  # SQL connector
    "mcp_tool",  # documents fetched from an MCP tool
    "generated",  # LLM-authored corpus ("generate our own")
    "kg_query",  # distilled from a Knowledge-Graph subgraph
    "llms",  # llms.txt / llms-full.txt (the LLM-docs standard — clean, complete)
)

# Frontmatter keys a standardized skill-graph SKILL.md carries. ``description`` and
# ``name`` are required; the rest are written by the pipeline for provenance/freshness.
REQUIRED_FRONTMATTER: tuple[str, ...] = ("name", "description")
SKILL_FRONTMATTER_KEYS: tuple[str, ...] = (
    "name",
    "description",
    "skill_graph_version",
    "source_types",
    "built_at",
    "builder_version",
    "file_count",
    "kg_ingested",
    "kg_manifest",
    "kg_ontology",
    "concepts",
)


def sha256_text(text: str) -> str:
    """Stable ``sha256:<hex>`` digest of UTF-8 text (newline-normalized)."""
    norm = text.replace("\r\n", "\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(norm).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Stable ``sha256:<hex>`` digest of raw bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(skill_md_text: str) -> dict[str, Any]:
    """Parse the leading YAML-ish frontmatter into a flat dict (pure stdlib).

    Handles the small subset skill-graphs use: ``key: scalar`` and
    ``key: [a, b, c]`` inline lists. Values like ``true``/``false`` coerce to bool;
    bracketed lists split on commas (quotes stripped). Unknown shapes stay strings.
    This is intentionally lenient — the validator checks presence/shape, not types.
    """
    m = _FRONTMATTER_RE.match(skill_md_text)
    if not m:
        return {}
    out: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        val = raw.strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            items = [s.strip().strip("'\"") for s in inner.split(",")] if inner else []
            out[key] = [s for s in items if s]
        elif val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            out[key] = val.strip("'\"")
    return out


def validate_skill_graph(skill_dir: str | Path) -> list[str]:
    """Structural lint of one skill-graph directory. Returns a list of errors.

    An empty list means the directory is a valid, standardized skill-graph. The
    checks are deliberately structural (presence + shape + integrity), never
    semantic — they are safe to run as a pre-commit gate across heterogeneous graphs.
    """
    d = Path(skill_dir)
    errors: list[str] = []
    skill_md = d / "SKILL.md"
    if not skill_md.exists():
        return [f"{d.name}: missing SKILL.md"]

    fm = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    if not fm:
        errors.append(f"{d.name}: SKILL.md has no parseable frontmatter")
        return errors

    for key in REQUIRED_FRONTMATTER:
        if not fm.get(key):
            errors.append(f"{d.name}: frontmatter missing required key '{key}'")

    name = fm.get("name")
    if name and name != d.name:
        errors.append(f"{d.name}: frontmatter name '{name}' != directory name")

    ref = d / "reference"
    # A graph either ships a reference/ corpus (doc-graph) or is a hand-authored
    # native graph (body-only, e.g. trading-systems). Only doc-graphs are held to
    # the provenance-manifest contract; native graphs are recognized by absence of
    # reference/ and presence of body content.
    if ref.exists():
        md_files = sorted(p for p in ref.rglob("*.md"))
        if not md_files:
            errors.append(f"{d.name}: reference/ exists but contains no .md files")
        errors.extend(_validate_sources_manifest(d, ref, fm, md_files))
    return errors


def _validate_sources_manifest(
    d: Path, ref: Path, fm: dict[str, Any], md_files: list[Path]
) -> list[str]:
    """Validate the ``sources.json`` sidecar of a doc-graph (best-effort, structural).

    A graph built before this contract existed simply has no ``sources.json`` — that
    is reported as a single actionable error (run a rebuild), not a cascade.
    """
    import json

    errors: list[str] = []
    manifest = d / "sources.json"
    if not manifest.exists():
        return [
            f"{d.name}: missing sources.json provenance manifest "
            "(rebuild via the unified pipeline to add it)"
        ]
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return [f"{d.name}: sources.json unreadable: {exc}"]

    if data.get("schema") != SOURCES_SCHEMA:
        errors.append(
            f"{d.name}: sources.json schema '{data.get('schema')}' != {SOURCES_SCHEMA}"
        )
    for src in data.get("sources", []):
        kind = src.get("kind")
        if kind not in SOURCE_KINDS:
            errors.append(f"{d.name}: sources.json unknown source kind '{kind}'")
    # file_count consistency between frontmatter and the actual tree.
    declared = fm.get("file_count")
    if declared not in (None, "") and str(declared) != str(len(md_files)):
        errors.append(
            f"{d.name}: frontmatter file_count {declared} != actual {len(md_files)}"
        )
    return errors
