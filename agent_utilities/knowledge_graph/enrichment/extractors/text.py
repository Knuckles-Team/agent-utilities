"""Generic text → concepts extraction (CONCEPT:KG-2.8).

A single entrypoint that turns ANY text (a chat thread, a prompt, an arbitrary
snippet) into canonical ``Concept`` nodes + ``MENTIONS`` edges from the source.
This is what lets chats, prompts, papers, and code converge on the SAME concept
nodes so the research/self-evolution loop can reason across them — reusing the
document concept extractor (no duplicate prompt/parsing logic).
"""

from __future__ import annotations

from collections.abc import Callable

from ..models import Concept, EnrichmentEdge
from .document import extract_concepts

LLMFn = Callable[[str], str]


def extract_text_concepts(
    text: str,
    source_id: str,
    llm_fn: LLMFn,
    *,
    source_type: str = "text",
    title: str = "",
    limit: int = 12,
) -> tuple[list[Concept], list[EnrichmentEdge]]:
    """Extract ``Concept`` nodes + ``MENTIONS`` edges from arbitrary text.

    Args:
        text: The raw text to mine for concepts.
        source_id: Node id of the source (Thread/prompt/Document) — becomes the
            concept provenance and the ``MENTIONS`` edge source.
        llm_fn: Completion fn (prompt -> JSON). If it yields nothing, returns ([], []).
        source_type: A hint for the prompt ("chat", "prompt", "document", ...).
        title: Optional human title for the prompt context.
        limit: Max concepts to extract.

    Returns:
        ``(concepts, edges)`` where each edge is ``source_id -[:MENTIONS]-> concept``.
    """
    if not text or not text.strip() or llm_fn is None:
        return [], []
    concepts = extract_concepts(
        text, source_id, llm_fn, source_type=source_type, title=title, limit=limit
    )
    edges = [
        EnrichmentEdge(source=source_id, target=c.id, rel_type="MENTIONS")
        for c in concepts
    ]
    return concepts, edges
