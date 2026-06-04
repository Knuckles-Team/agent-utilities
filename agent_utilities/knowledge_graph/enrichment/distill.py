"""Research → feature → spec distillation (CONCEPT:KG-2.8 Phase 3, headline).

Turns ingested research/document concepts into value-ranked enhancement proposals
and SDD-style spec drafts for a target codebase. Pure ranking + injectable LLM, so
it's testable; operates over the enrichment entities/edges (no backend coupling).

Flow: concepts RELATES_TO code-in-codebase (but not yet REALIZES) → rank by value
→ LLM distils enhancements → LLM distils specs → (optionally) write into the
codebase's ``.specify/`` for the SDD skill, then plan + implement.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

from pydantic import BaseModel, Field

from .models import Concept, EnrichmentEdge

LLMFn = Callable[[str], str]


class EnhancementCandidate(BaseModel):
    concept_id: str
    concept_name: str
    summary: str = ""
    value_score: float = 0.0
    relates_to: list[str] = Field(default_factory=list)  # code ids in the codebase
    source_ids: list[str] = Field(default_factory=list)  # docs the concept came from


class SpecDraft(BaseModel):
    title: str
    target_codebase: str
    problem: str = ""
    approach: str = ""
    value: str = ""
    concept_ids: list[str] = Field(default_factory=list)
    value_score: float = 0.0


def gather_enhancement_candidates(
    concepts: list[Concept],
    edges: list[EnrichmentEdge],
    code_file_by_id: dict[str, str],
    codebase_prefix: str,
    limit: int = 20,
) -> list[EnhancementCandidate]:
    """Concepts relevant to the codebase but not yet realized in it, value-ranked.

    value = (# code symbols in the codebase the concept RELATES_TO) × novelty,
    where novelty=1 unless the concept already REALIZES code in the codebase.
    """
    norm = codebase_prefix.replace("\\", "/")

    def in_codebase(code_id: str) -> bool:
        return norm in (code_file_by_id.get(code_id, "").replace("\\", "/"))

    relates: dict[str, set[str]] = {}
    realizes: dict[str, set[str]] = {}
    for e in edges:
        if not e.source.startswith("concept:"):
            continue
        if not in_codebase(e.target):
            continue
        if e.rel_type == "RELATES_TO":
            relates.setdefault(e.source, set()).add(e.target)
        elif e.rel_type == "REALIZES":
            realizes.setdefault(e.source, set()).add(e.target)

    by_id = {c.id: c for c in concepts}
    out: list[EnhancementCandidate] = []
    for cid, code_ids in relates.items():
        c = by_id.get(cid)
        if c is None:
            continue
        novelty = 0.3 if cid in realizes else 1.0
        score = round(len(code_ids) * novelty, 3)
        out.append(
            EnhancementCandidate(
                concept_id=cid,
                concept_name=c.name,
                summary=c.summary,
                value_score=score,
                relates_to=sorted(code_ids),
                source_ids=c.source_ids,
            )
        )
    out.sort(key=lambda x: x.value_score, reverse=True)
    return out[:limit]


_SPEC_PROMPT = """You are proposing high-value enhancements to the codebase
`{codebase}` based on concepts distilled from ingested research/documents.

Candidate concepts (most relevant first), with how many of the codebase's
components they relate to:
{candidates}

Propose the {limit} highest-value, concrete specs to build into this codebase.
For each, give a short title, the problem it solves, the implementation approach,
and the value. Prefer ideas grounded in the candidates.

Output ONLY a JSON array of objects with keys "title", "problem", "approach",
"value", and "concept_names" (array of the candidate names it draws on). No other text."""


def distill_specs(
    codebase: str,
    candidates: list[EnhancementCandidate],
    llm_fn: LLMFn,
    limit: int = 5,
) -> list[SpecDraft]:
    """LLM-distil value-ranked spec drafts from enhancement candidates."""
    if not candidates:
        return []
    cand_text = "\n".join(
        f"- {c.concept_name} (relates to {len(c.relates_to)} components): {c.summary}"
        for c in candidates[:20]
    )
    prompt = _SPEC_PROMPT.format(codebase=codebase, candidates=cand_text, limit=limit)
    try:
        raw = llm_fn(prompt)
        start, end = raw.index("["), raw.rindex("]") + 1
        items = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError, Exception):
        return []

    name_to_cid = {c.concept_name.lower(): c.concept_id for c in candidates}
    name_to_score = {c.concept_name.lower(): c.value_score for c in candidates}
    specs: list[SpecDraft] = []
    for it in items[:limit]:
        if not isinstance(it, dict) or not it.get("title"):
            continue
        names = [str(n).strip() for n in it.get("concept_names", [])]
        cids = [name_to_cid[n.lower()] for n in names if n.lower() in name_to_cid]
        score = round(sum(name_to_score.get(n.lower(), 0.0) for n in names), 3)
        specs.append(
            SpecDraft(
                title=str(it["title"]).strip(),
                target_codebase=codebase,
                problem=str(it.get("problem", "")).strip(),
                approach=str(it.get("approach", "")).strip(),
                value=str(it.get("value", "")).strip(),
                concept_ids=cids,
                value_score=score,
            )
        )
    specs.sort(key=lambda s: s.value_score, reverse=True)
    return specs


def what_specs_could_we_build(
    codebase: str,
    concepts: list[Concept],
    edges: list[EnrichmentEdge],
    code_file_by_id: dict[str, str],
    llm_fn: LLMFn,
    limit: int = 5,
) -> list[SpecDraft]:
    """End-to-end: gather value-ranked candidates → distil spec drafts."""
    candidates = gather_enhancement_candidates(
        concepts, edges, code_file_by_id, codebase
    )
    return distill_specs(codebase, candidates, llm_fn, limit)


def spec_to_markdown(spec: SpecDraft) -> str:
    """Render an SDD-style spec draft for writing into a codebase's .specify/."""
    return (
        f"# Spec: {spec.title}\n\n"
        f"> Auto-distilled by KG-2.8 from ingested research/documents "
        f"(value score {spec.value_score}). Concepts: {', '.join(spec.concept_ids) or 'n/a'}\n\n"
        f"## Problem\n{spec.problem or 'TBD'}\n\n"
        f"## Approach\n{spec.approach or 'TBD'}\n\n"
        f"## Value\n{spec.value or 'TBD'}\n"
    )


def write_spec_drafts(specs: list[SpecDraft], codebase_root: str) -> list[str]:
    """Write spec drafts into ``<codebase_root>/.specify/specs/kg-distilled/``."""
    from .extractors.document import slug

    out_dir = os.path.join(codebase_root, ".specify", "specs", "kg-distilled")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for s in specs:
        path = os.path.join(out_dir, f"{slug(s.title)}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(spec_to_markdown(s))
        paths.append(path)
    return paths
