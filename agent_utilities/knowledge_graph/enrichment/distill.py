"""Research → feature → spec distillation (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 3, headline).

Turns ingested research/document concepts into value-ranked enhancement proposals
and SDD-style spec drafts for a target codebase. Pure ranking + injectable LLM, so
it's testable; operates over the enrichment entities/edges (no backend coupling).

Flow: concepts RELATES_TO code-in-codebase (but not yet REALIZES) → rank by value
→ LLM distils enhancements → LLM distils specs → (optionally) write into the
codebase's ``.specify/`` for the SDD skill, then plan + implement.
"""

from __future__ import annotations

import json
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
    # CONCEPT:AU-AHE.harness.single-file-code-synthesis (Wave-6 D3/WP#1) — the single
    # existing repo-relative .py file this spec most-relevantly targets, resolved from
    # the candidate's RELATES_TO code ids. When populated it threads through
    # persist_spec_proposal → _spec_to_proposal → code_synthesis.synthesize_code so an
    # autonomously-distilled spec emits a REAL single-file edit instead of a prose
    # skeleton. Empty ⇒ the prose SDD fallback runs, exactly as before.
    target_file: str = ""


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

Candidate concepts (most relevant first), each with the existing codebase files
it relates to (edit one of THESE — never invent a path):
{candidates}

Propose the {limit} highest-value, concrete specs to build into this codebase.
For each, give a short title, the problem it solves, the implementation approach,
and the value. Prefer ideas grounded in the candidates.

For "target_file", name the SINGLE existing file (from the related files listed
for the concepts you draw on) that a one-file change should modify — or "" if the
spec needs more than a single-file edit. Prefer a concrete single file so the
change can be synthesized automatically.

Output ONLY a JSON array of objects with keys "title", "problem", "approach",
"value", "concept_names" (array of the candidate names it draws on), and
"target_file" (one of the listed related files, or ""). No other text."""


def distill_specs(
    codebase: str,
    candidates: list[EnhancementCandidate],
    llm_fn: LLMFn,
    limit: int = 5,
    code_file_by_id: dict[str, str] | None = None,
) -> list[SpecDraft]:
    """LLM-distil value-ranked spec drafts from enhancement candidates.

    When ``code_file_by_id`` maps a candidate's ``relates_to`` code ids to file
    paths, each draft is offered those files and the LLM-named ``target_file`` is
    validated against them (never invented) — so a distilled spec can carry a
    resolvable single-file target and reach ``code_synthesis`` (Wave-6 D3). Without
    the map (the loop's cheap intake path) ``target_file`` stays empty ⇒ prose
    fallback, unchanged.
    """
    if not candidates:
        return []
    file_map = code_file_by_id or {}

    def _files_for(c: EnhancementCandidate) -> list[str]:
        seen: list[str] = []
        for cid in c.relates_to:
            f = file_map.get(cid)
            if f and f not in seen:
                seen.append(f.replace("\\", "/"))
        return seen

    files_by_name = {c.concept_name.lower(): _files_for(c) for c in candidates}
    cand_lines = []
    for c in candidates[:20]:
        files = files_by_name.get(c.concept_name.lower(), [])
        files_str = ", ".join(files[:8]) if files else "(no single-file target)"
        cand_lines.append(f"- {c.concept_name}: {c.summary} [files: {files_str}]")
    cand_text = "\n".join(cand_lines)
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
        # Never invent a target: accept the LLM's target_file only if it is one of
        # the files the cited candidates actually relate to.
        allowed = {f for n in names for f in files_by_name.get(n.lower(), [])}
        raw_target = str(it.get("target_file", "") or "").strip().replace("\\", "/")
        target_file = raw_target if raw_target in allowed else ""
        specs.append(
            SpecDraft(
                title=str(it["title"]).strip(),
                target_codebase=codebase,
                problem=str(it.get("problem", "")).strip(),
                approach=str(it.get("approach", "")).strip(),
                value=str(it.get("value", "")).strip(),
                concept_ids=cids,
                value_score=score,
                target_file=target_file,
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
    return distill_specs(
        codebase, candidates, llm_fn, limit, code_file_by_id=code_file_by_id
    )


# NOTE (Wave-6 D2): the raw ``spec_to_markdown``/``write_spec_drafts`` open()/write()
# writer that dumped a prose file into ``.specify/specs/kg-distilled/`` was removed. The
# autonomous loop now authors first-class DSTDD Spec+Tasks through the ONE writer,
# ``agent_utilities.sdd.SDDManager.author_from_draft`` (CONCEPT:AU-AHE.sdd.loop-authored-spec),
# so ``SpecDraft`` is purely the input adapter — one spec model, one writer, one node family.
