"""Reference markdown fragmenter (CONCEPT:AU-KG.ingest.mapping-dsl).

Turns one markdown file into the evidence spine's REAL
:class:`~..ingestion.evidence_spine.Artifact` +
:class:`~..ingestion.evidence_spine.Fragment` objects (CONCEPT:
AU-KG.ingest.evidence-spine-artifact / AU-KG.ingest.stable-fragment-address,
D-GP2-2): YAML frontmatter keys, GFM pipe tables (as ``table_row`` fragments,
one per row), ATX headings (as ``heading`` fragments), and inline links.

**Why this module still exists alongside ``evidence_spine.fragment_markdown``.**
That function already fragments the GENERIC markdown structure (headings,
paragraphs, tables, lists, quotes, code blocks) into the canonical spine. It
does not extract YAML frontmatter or inline links as their own citable units —
those two kinds (plus a JSON/API record's dotted-path fields) are this
package's OWN genuinely additional structural units, declared on the SAME
canonical :data:`~..ingestion.evidence_spine.FRAGMENT_KINDS` vocabulary
(``frontmatter_key``/``link``/``json_field``) rather than a rival, incompatible
one — this module used to define its own provisional ``Artifact``/``Fragment``
stand-in (``fragment_contract.py``, now deleted) before the real evidence-spine
contract published; it now builds the REAL classes directly via
:meth:`~..ingestion.evidence_spine.Fragment.at` so every fragment's
``fragment_id`` is derived exactly the way every other spine-producing path
derives it (CONCEPT:AU-KG.ingest.stable-fragment-address's own identity
scheme — content-independent, structural-path-addressed).

This is the minimal, self-contained fragmenter this package's own wiring test
drives end to end (a real file's frontmatter + a real table -> real graph
facts); it is **not** the full "one git-markdown connector" the operator's
charter names as the ultimate proof track (track 10,
``reports/program/universal-ingestion.md``) — that connector
(``protocols/source_connectors/connectors/git_markdown.py``) additionally owns
repo discovery/watermarking/registration into ``source_sync`` and already
calls ``evidence_spine.fragment_markdown`` directly for the generic structure.
Whichever connector reads a markdown corpus should produce fragments in
exactly this shape so any mapping DSL rule (frontmatter/table/heading/link)
can be written once against it.

Deliberately simple regex/line-based parsing (no markdown-it/mistune
dependency — see ``dependency-lock``): good enough for GFM frontmatter +
pipe tables + ATX headings + inline links, which is what the mapping DSL's
five rule kinds target. A corpus needing richer markdown should feed the DSL
via a fuller connector without changing the DSL itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from ..ingestion.evidence_spine import (
    Artifact,
    Fragment,
    artifact_id_for,
    content_digest,
)

__all__ = ["fragment_markdown_text", "fragment_markdown_file"]

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: This fragmenter's own connector identity for :func:`artifact_id_for` — a
#: standalone caller (the wiring test, a script) that doesn't come through a
#: real connector's :class:`~..ingestion.change_envelope.ChangeEnvelope` still
#: gets a deterministic, source-identity-keyed artifact id.
_CONNECTOR = "domain-pack-markdown-fragmenter"


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
    """Pure function: markdown text -> ``(Artifact, [Fragment, ...])``.

    ``artifact_id`` is keyed to ``source_path`` (source IDENTITY — the evidence
    spine's own scheme, CONCEPT:AU-KG.ingest.evidence-spine-artifact), not to
    content: re-fragmenting the SAME path after an edit updates the SAME
    artifact rather than forking a new one, so a citation into it survives.
    Pass an explicit ``artifact_id`` to override (e.g. a caller that already
    derived one from a real ``ChangeEnvelope``).
    """
    whole_hash = content_digest(text)
    resolved_artifact_id = artifact_id or artifact_id_for(_CONNECTOR, "", source_path)
    frontmatter, body = _split_frontmatter(text)

    fragments: list[Fragment] = []

    for key, value in frontmatter.items():
        fragments.append(
            Fragment.at(
                artifact_id=resolved_artifact_id,
                kind="frontmatter_key",
                label=key,
                text=str(value),
                sequence=len(fragments),
                attributes={"key": key, "value": value},
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
                Fragment.at(
                    artifact_id=resolved_artifact_id,
                    kind="heading",
                    # Anchored by the FULL nested path, not just the title, so
                    # two same-titled headings under different parents (e.g.
                    # "Intro/Details" vs "Setup/Details") never collide onto
                    # one fragment_id.
                    label=heading_path,
                    text=title,
                    sequence=len(fragments),
                    attributes={
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
                    Fragment.at(
                        artifact_id=resolved_artifact_id,
                        kind="table_row",
                        ordinal=row_index,
                        text=lines[i].strip(),
                        sequence=len(fragments),
                        attributes={
                            "heading_path": heading_path,
                            "row_index": row_index,
                            "row": row,
                        },
                    )
                )
                row_index += 1
                i += 1
            continue

        for link_match in _LINK_RE.finditer(line):
            text_label, href = link_match.group(1), link_match.group(2)
            fragments.append(
                Fragment.at(
                    artifact_id=resolved_artifact_id,
                    kind="link",
                    ordinal=link_index,
                    text=text_label,
                    sequence=len(fragments),
                    attributes={"text": text_label, "href": href},
                )
            )
            link_index += 1

        i += 1

    artifact = Artifact(
        artifact_id=resolved_artifact_id,
        connector=_CONNECTOR,
        media_type="text/markdown",
        content_hash=whole_hash,
        source_object_id=source_path,
        fragments=tuple(fragments),
    )
    return artifact, fragments


def fragment_markdown_file(path: str | Path) -> tuple[Artifact, list[Fragment]]:
    """Read one markdown file from disk and fragment it."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return fragment_markdown_text(text, source_path=str(file_path))
