from __future__ import annotations

"""Canonical GraphRAG-style prompt loader.

CONCEPT:AU-KG.retrieval.graph-engineering-canonical-prompts

Loads one of the 5 packaged canonical KG-operation prompt blueprints
(``kg_extraction``, ``kg_normalization``, ``kg_graph_query``,
``kg_grounded_answer``, ``kg_graph_maintenance`` -- ``StructuredPrompt`` JSON,
see ``../prompting/structured.py``) by name and renders its
``instructions.core_directive`` template.

Rendering uses :class:`string.Template` (``$var`` substitution), NOT
``str.format`` -- several of these prompts embed literal JSON examples
(``{"facts": [...]}``) whose braces would otherwise collide with
``str.format``'s field syntax.

This is the single source of truth for the 5 canonical prompts: callers (the
extraction/dedup pipelines, :mod:`..knowledge_graph.retrieval.graph_engineering`)
load their prompt text from here rather than keeping a second inline copy, so
the packaged JSON is genuinely wired, not a parallel unused document. Every
reader degrades to a caller-supplied ``fallback`` on any load error (missing
file, malformed JSON, empty body) -- a packaging edge case never breaks a live
extraction/query/answer path.
"""

import json
import logging
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent

__all__ = ["canonical_prompt_text", "load_canonical_prompt"]


@cache
def _load_prompt_document(name: str) -> dict[str, Any] | None:
    """Parse one packaged prompt JSON file by its filename stem (cached — these
    are packaged, immutable-at-runtime files)."""
    path = _PROMPTS_DIR / f"{name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "canonical prompt %r failed to load from %s: %s", name, path, exc
        )
        return None


def canonical_prompt_text(name: str) -> str | None:
    """Return the raw ``instructions.core_directive`` template for a packaged
    canonical prompt, or ``None`` if it cannot be loaded.

    Callers should fall back to their own inline default rather than raise —
    see :func:`load_canonical_prompt`.
    """
    doc = _load_prompt_document(name)
    if not doc:
        return None
    directive = (doc.get("instructions") or {}).get("core_directive")
    return directive if isinstance(directive, str) and directive.strip() else None


def load_canonical_prompt(
    name: str, *, fallback: str = "", **substitutions: Any
) -> str:
    """Load + render canonical prompt ``name`` with ``$var``-style substitutions.

    Args:
        name: filename stem under ``agent_utilities/prompts/`` (e.g.
            ``"kg_grounded_answer"``).
        fallback: rendered the same way when the packaged prompt is
            missing/unreadable — a best-effort convenience layer over the
            packaged prompt library, never a hard dependency a caller can be
            broken by.
        **substitutions: ``$name``-style values substituted into the template
            via :meth:`string.Template.safe_substitute` (unresolved
            placeholders are left verbatim rather than raising, so a caller
            that only fills some variables still gets a usable string).

    Returns:
        The rendered prompt text, or ``""`` if neither the packaged prompt nor
        ``fallback`` is available.
    """
    text = canonical_prompt_text(name) or fallback
    if not text:
        return ""
    if not substitutions:
        return text
    return Template(text).safe_substitute(**substitutions)
