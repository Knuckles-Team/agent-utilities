"""Reference markdown fragmenter (CONCEPT:AU-KG.ingest.mapping-dsl).

Turns one markdown file into :class:`~.fragment_contract.Artifact` +
:class:`~.fragment_contract.Fragment` objects: YAML frontmatter keys, GFM pipe
tables (as ``table_row`` fragments, one per row), ATX headings (as
``heading_section`` fragments), and inline links. This is the minimal,
self-contained fragmenter this package's own wiring test drives end to end
(a real file's frontmatter + a real table -> real graph facts); it is
**not** the full "one git-markdown connector" the operator's charter names as
the ultimate proof track (track 10, ``reports/program/universal-ingestion.md``)
— that connector additionally owns repo discovery/watermarking/registration
into ``source_sync``. Whichever connector reads a markdown corpus should
produce fragments in exactly this shape so any mapping DSL rule (frontmatter/
table/heading/link) can be written once against it.

Deliberately simple regex/line-based parsing (no markdown-it/mistune
dependency — see ``dependency-lock``): good enough for GFM frontmatter +
pipe tables + ATX headings + inline links, which is what the mapping DSL's
five rule kinds target. A corpus needing richer markdown should feed the DSL
via a fuller connector without changing the DSL itself.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from .fragment_contract import Artifact, Fragment, content_hash

__all__ = ["fragment_markdown_text", "fragment_markdown_file"]

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _split_frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data, text[match.end() :]


def _is_table_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells)


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def fragment_markdown_text(
    text: str, *, source_path: str, artifact_id: str | None = None
) -> tuple[Artifact, list[Fragment]]:
    """Pure function: markdown text -> ``(Artifact, [Fragment, ...])``."""
    whole_hash = _sha256(text)
    resolved_artifact_id = artifact_id or f"md:{whole_hash[:16]}"
    frontmatter, body = _split_frontmatter(text)
    artifact = Artifact(
        artifact_id=resolved_artifact_id,
        source_path=source_path,
        content_hash=whole_hash,
        connector="git-markdown",
    )

    fragments: list[Fragment] = []

    for key, value in frontmatter.items():
        fragments.append(
            Fragment(
                fragment_id=f"{resolved_artifact_id}#frontmatter:{key}",
                artifact_id=resolved_artifact_id,
                fragment_type="frontmatter_key",
                locator=f"frontmatter.{key}",
                content_hash=content_hash(value),
                value=value,
                metadata={"key": key},
            )
        )

    lines = body.splitlines()
    heading_stack: list[str] = []
    link_index = 0
    row_index = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            heading_path = "/".join(heading_stack)
            fragments.append(
                Fragment(
                    fragment_id=f"{resolved_artifact_id}#heading:{heading_path}",
                    artifact_id=resolved_artifact_id,
                    fragment_type="heading_section",
                    locator=f"heading[{heading_path}]",
                    content_hash=content_hash(title),
                    value=title,
                    metadata={
                        "heading": title,
                        "heading_path": heading_path,
                        "level": level,
                    },
                )
            )
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and _is_table_separator(lines[i + 1]):
            header = _split_row(line)
            i += 2
            heading_path = "/".join(heading_stack)
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells = _split_row(lines[i])
                row = dict(zip(header, cells, strict=False))
                fragments.append(
                    Fragment(
                        fragment_id=f"{resolved_artifact_id}#row:{row_index}",
                        artifact_id=resolved_artifact_id,
                        fragment_type="table_row",
                        locator=f"table[{heading_path}].row[{row_index}]",
                        content_hash=content_hash(row),
                        value=row,
                        metadata={"heading_path": heading_path, "row_index": row_index},
                    )
                )
                row_index += 1
                i += 1
            continue

        for link_match in _LINK_RE.finditer(line):
            text_label, href = link_match.group(1), link_match.group(2)
            fragments.append(
                Fragment(
                    fragment_id=f"{resolved_artifact_id}#link:{link_index}",
                    artifact_id=resolved_artifact_id,
                    fragment_type="link",
                    locator=f"link[{link_index}]",
                    content_hash=content_hash({"text": text_label, "href": href}),
                    value={"text": text_label, "href": href},
                    metadata={"text": text_label, "href": href},
                )
            )
            link_index += 1

        i += 1

    return artifact, fragments


def fragment_markdown_file(path: str | Path) -> tuple[Artifact, list[Fragment]]:
    """Read one markdown file from disk and fragment it."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return fragment_markdown_text(text, source_path=str(file_path))
