"""Deterministic per-file content classifier for codebase ingestion.

CONCEPT:KG-2.284

When a repository is ingested as a ``CODEBASE`` the structural pipeline only
parses *source* files (``SOURCE_EXTENSIONS``) into ``Code``/``Test``/``Feature``
nodes. A real repo, however, also carries agent **skills**, system **prompts**,
spec/SDD **docs**, and ordinary **markdown** — which should land as their own
native KG types, not be dropped on the floor.

This module is the single deterministic ROUTER that, in ONE walk of the tree,
assigns every file/dir to its correct native :class:`ContentType` using
*extension + path + a lightweight content sniff* — rules/heuristics first, with
an explicit precedence and a confidence. There is **no LLM** here: genuinely
ambiguous files fall through to a conservative default (a ``.json`` that isn't a
recognisable prompt stays unclassified rather than being guessed). The
classifier only *decides*; the engine's existing per-type adaptors do the actual
ingestion (anti-sprawl: this is a router over them, not a new ingest engine).

Precedence (most specific wins; first match decides a file's fate):

1. **Skill** — a directory containing ``SKILL.md`` claims its whole subtree, so a
   skill's ``SKILL.md`` and its ``reference/*.md`` belong to the skill, never a
   generic Document. (Repo-root skills do not claim, so a code repo that ships a
   top-level ``SKILL.md`` still routes its ``docs/``.)
2. **Spec / SDD** — anything under ``.specify/`` or a ``*.spec.md`` file.
3. **Prompt** — ``*.prompt``, or a ``*.json`` under a ``prompts/`` dir, or a
   ``*.json`` whose content sniffs as a prompt template.
4. **Config** — ``config.json`` (model registry) / ``mcp_config.json``.
5. **Document** — any document extension (``.md``/``.rst``/``.txt``/…).
6. **Code** — any source extension (handled by the structural pipeline; reported
   here only for coverage accounting).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..enrichment.pipeline import _SKIP_DIRS, SOURCE_EXTENSIONS
from .engine import ContentType

__all__ = [
    "FileClass",
    "RepoClassification",
    "classify_repo",
]

# Document extensions routed to the DOCUMENT adaptor during a codebase walk.
# Intentionally a TEXT-doc subset of ``IngestionEngine._DOC_EXTENSIONS`` —
# binary/media modalities (png/wav/pdf/…) that live in a source tree are
# overwhelmingly fixtures/assets, not knowledge, so we don't auto-ingest them.
_DOC_EXTS: frozenset[str] = frozenset(
    {".md", ".markdown", ".rst", ".txt", ".text", ".org", ".adoc"}
)

# Prompt template markers — keys/sentinels that distinguish a prompt JSON from an
# arbitrary config/data JSON. A match (or a ``{{ }}`` mustache placeholder) makes
# a ``*.json`` a PROMPT even outside a ``prompts/`` dir.
_PROMPT_JSON_KEYS: frozenset[str] = frozenset(
    {"template", "messages", "system", "system_prompt", "prompt", "instructions"}
)

# How many bytes to sniff from a candidate prompt JSON (cheap, bounded).
_SNIFF_BYTES = 4096


@dataclass(frozen=True)
class FileClass:
    """A single classification decision for one path.

    Attributes:
        path: The file (or skill directory) being classified.
        content_type: The native :class:`ContentType` it routes to.
        reason: Short human-readable rule that fired (for explainability/tests).
        confidence: 1.0 = deterministic rule, <1.0 = heuristic sniff.
    """

    path: Path
    content_type: ContentType
    reason: str
    confidence: float = 1.0


@dataclass
class RepoClassification:
    """The full routing plan produced from one walk of a repo.

    ``code`` is reported for coverage but is consumed by the structural pipeline,
    not the router. ``specs`` are written inline by the engine (there is no SPEC
    adaptor), the rest fan out to their existing adaptors.
    """

    root: Path
    skills: list[FileClass] = field(default_factory=list)
    prompts: list[FileClass] = field(default_factory=list)
    specs: list[FileClass] = field(default_factory=list)
    documents: list[FileClass] = field(default_factory=list)
    configs: list[FileClass] = field(default_factory=list)
    code: list[FileClass] = field(default_factory=list)

    @property
    def all_classes(self) -> list[FileClass]:
        return [
            *self.skills,
            *self.specs,
            *self.prompts,
            *self.configs,
            *self.documents,
            *self.code,
        ]

    def summary(self) -> dict[str, int]:
        return {
            "skills": len(self.skills),
            "specs": len(self.specs),
            "prompts": len(self.prompts),
            "configs": len(self.configs),
            "documents": len(self.documents),
            "code": len(self.code),
        }


def _is_under(path: Path, ancestor_parts: set[str]) -> bool:
    """True if any path component is one of ``ancestor_parts``."""
    return any(part in ancestor_parts for part in path.parts)


def _find_skill_roots(root: Path) -> list[Path]:
    """Directories (excluding the repo root) that contain a ``SKILL.md``.

    A skill root claims its whole subtree so nested ``reference/*.md`` and
    helper files are part of the skill, not separate Documents. The repo root is
    deliberately excluded: a code repo that ships a top-level ``SKILL.md`` must
    still route its ``docs/`` and ``prompts/``.
    """
    roots: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and d != ".git"]
        if any(fn.lower() == "skill.md" for fn in filenames):
            d = Path(dirpath)
            if d.resolve() != root.resolve():
                roots.append(d)
    return roots


def _sniff_prompt_json(path: Path) -> bool:
    """Cheap content sniff: does this ``.json`` look like a prompt template?"""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError:
        return False
    if "{{" in head:  # mustache placeholder — a templating tell
        return True
    # Parse only when the head is a complete object; a truncated read just falls
    # back to the key-substring check below (still deterministic).
    try:
        obj = json.loads(head)
        if isinstance(obj, dict) and (_PROMPT_JSON_KEYS & set(obj.keys())):
            return True
    except (json.JSONDecodeError, ValueError):
        pass
    low = head.lower()
    return any(f'"{k}"' in low for k in _PROMPT_JSON_KEYS)


def _classify_file(path: Path, *, in_prompts_dir: bool) -> FileClass | None:
    """Apply the precedence ladder to a single (non-skill-claimed) file.

    Returns ``None`` when the file matches no rule (e.g. a lockfile, a data
    ``.json``, an image) — those are left unclassified rather than guessed.
    """
    name = path.name.lower()
    suffix = path.suffix.lower()

    # 2. Spec / SDD — under ``.specify/`` or a ``*.spec.md``. Routed to the inline
    #    Spec writer (no SPEC adaptor exists); flagged with a ``SPEC:`` reason so
    #    the router can split it out of the Document stream.
    if _is_under(path, {".specify"}):
        return FileClass(path, ContentType.DOCUMENT, "SPEC:under .specify/", 1.0)
    if name.endswith(".spec.md"):
        return FileClass(path, ContentType.DOCUMENT, "SPEC:*.spec.md", 1.0)

    # 3. Prompt — ``*.prompt``, a ``*.json`` under ``prompts/``, or a sniffed
    #    prompt JSON anywhere. ``config.json``/``mcp_config.json`` are handled by
    #    rule 4 BELOW, so guard them out here first.
    if suffix == ".prompt":
        return FileClass(path, ContentType.PROMPT, "*.prompt extension", 1.0)
    if suffix == ".json" and name not in ("config.json", "mcp_config.json"):
        if in_prompts_dir:
            return FileClass(path, ContentType.PROMPT, "*.json under prompts/", 1.0)
        if _sniff_prompt_json(path):
            return FileClass(path, ContentType.PROMPT, "prompt-template sniff", 0.7)

    # 4. Config — the model-registry config.json / mcp_config.json.
    if name == "mcp_config.json":
        return FileClass(path, ContentType.MCP_SERVER, "mcp_config.json", 1.0)
    if name == "config.json":
        return FileClass(path, ContentType.CONFIG, "config.json", 1.0)

    # 5. Document — text-doc extensions (covers ``.md`` inside a code dir).
    if suffix in _DOC_EXTS:
        return FileClass(path, ContentType.DOCUMENT, f"doc ext {suffix}", 1.0)

    # 6. Code — reported for coverage; the structural pipeline owns it.
    if suffix in SOURCE_EXTENSIONS:
        return FileClass(path, ContentType.CODEBASE, f"source ext {suffix}", 1.0)

    return None


def classify_repo(root: str | Path) -> RepoClassification:
    """Walk ``root`` once and classify every file into its native KG type.

    CONCEPT:KG-2.284 — Deterministic: skill dirs claim their subtree, then the
    precedence ladder (spec → prompt → config → document → code) decides each
    remaining file. The result feeds the codebase-ingest router, which fans the
    non-code classes out to their existing per-type adaptors.
    """
    root = Path(root)
    out = RepoClassification(root=root)
    if not root.exists():
        return out

    if root.is_file():
        fc = _classify_file(root, in_prompts_dir=False)
        if fc is not None:
            _file_into(out, fc)
        return out

    # 1. Skill roots claim their subtrees.
    skill_roots = _find_skill_roots(root)
    resolved_skill_roots = [d.resolve() for d in skill_roots]
    for d in skill_roots:
        out.skills.append(FileClass(d, ContentType.SKILL, "dir contains SKILL.md", 1.0))

    def _claimed(p: Path) -> bool:
        rp = p.resolve()
        return any(rp == sr or sr in rp.parents for sr in resolved_skill_roots)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and d != ".git"]
        dpath = Path(dirpath)
        in_prompts_dir = "prompts" in {p.lower() for p in dpath.parts}
        for fn in filenames:
            fp = dpath / fn
            if _claimed(fp):  # belongs to a skill — already accounted for
                continue
            fc = _classify_file(fp, in_prompts_dir=in_prompts_dir)
            if fc is not None:
                _file_into(out, fc)
    return out


def _file_into(out: RepoClassification, fc: FileClass) -> None:
    """Bucket a classified file into the right list on ``out``."""
    if fc.reason.startswith("SPEC:"):
        out.specs.append(fc)
    elif fc.content_type == ContentType.SKILL:
        out.skills.append(fc)
    elif fc.content_type == ContentType.PROMPT:
        out.prompts.append(fc)
    elif fc.content_type in (ContentType.CONFIG, ContentType.MCP_SERVER):
        out.configs.append(fc)
    elif fc.content_type == ContentType.DOCUMENT:
        out.documents.append(fc)
    elif fc.content_type == ContentType.CODEBASE:
        out.code.append(fc)
