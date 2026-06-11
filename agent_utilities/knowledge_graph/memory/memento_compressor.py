"""CONCEPT:KG-2.20 — Mementified Context Management (state compression).

Assimilated from *Memento: Teaching LLMs to Manage Their Own Context* (Kontonis et al.,
Microsoft Research AI Frontiers, 2026). A **memento** is not a human summary — it is a *lemma*:
a terse, information-dense compression of a reasoning/conversation block that preserves exact
formulas, key intermediate values, commands + outcomes, and the current execution state, so a
model can reason *forward* from the memento alone (the block it replaces can be evicted).

This module is the canonical home for the memento compressor (strangled out of the 1,793-line
``agent_context.py`` god-module — ``observer.py`` and ``memory_engine.py`` already imported
``.memento_compressor`` before this module existed, which silently broke the memento write path).

Two pieces, both grounded in the paper:

* :func:`compress_to_memento` — single generation pass that compresses a block into a memento and
  persists it as a KG ``Memento`` node.
* **Judge-refine loop** (CONCEPT:KG-2.20 / paper §Stage 4) — a compressor→judge→recompress cycle.
  The paper measured single-shot mementos at a **28%** rubric pass-rate vs **92%** after two judge
  iterations: initial mementos routinely drop an exact formula or intermediate value a downstream
  block needs. :func:`compress_to_memento` runs this loop by default (``refine=True``), scoring each
  candidate on a six-dimension rubric (formulas-verbatim, values-preserved, methods-named,
  validation, no-hallucination, result-first) with acceptance threshold ``τ=8/10`` and ``≤2`` extra
  iterations, exactly as in the paper.

**Honest limitation (orchestration layer).** We run hosted/API models via pydantic-ai and do not
control the inference engine's KV cache, so an external memento is the paper's *"restart mode"* — it
loses the implicit dual-channel KV side-information the paper measured at −15pp. The mitigation is
lossless recoverability (CONCEPT:KG-2.20 MEM-4): evicted blocks keep a ``SUMMARIZES`` pointer so they
can be re-fetched on demand. This is a substitute for, not an equivalent of, the in-engine channel.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# Compressor prompt — the paper's "STATE-COMPRESSOR": minimise tokens subject to fully capturing
# all logically relevant information (extractive, no new derivations, reason-forward target).
MEMENTO_SYSTEM_PROMPT = """You are a state-compression Memento generator for an autonomous agent.
Your task is to take a block of reasoning and conversation history and compress it into a dense Memento.

## Strict Rules:
1. You are NOT summarizing for a human. You are compressing state for an LLM to reason forward from.
2. You MUST extract exact formulas, key intermediate values, commands executed, and their precise outcomes.
3. Keep the strategic decisions and the current execution state (what succeeded, what failed, what is next).
4. Do NOT hallucinate or add outside knowledge.
5. Provide a terse, information-dense output that can act as a drop-in replacement for the raw block.
6. Output ONLY the memento text.
"""

# Judge prompt — six-dimension rubric from the paper (§Stage 4). The judge returns a 0-10 score and,
# when below threshold, *specific, actionable* feedback (e.g. "missing formula: K^2 - 3K + 3"), never
# vague "add more detail". That specificity is what drives the 28%->92% pass-rate improvement.
MEMENTO_JUDGE_PROMPT = """You are a strict Memento judge. A Memento must let an LLM continue reasoning
WITHOUT seeing the original block. Score the candidate Memento against the ORIGINAL block on a 0-10
scale, summing these six dimensions:
- formulas extracted verbatim (0-3)
- numerical/intermediate values preserved (0-2)
- methods/strategies explicitly named (0-2)
- validation / verification of results included (0-1)
- no hallucinations or invented facts (0-1)
- result-first structure, terse (0-1)

Respond on EXACTLY two lines:
SCORE: <integer 0-10>
FEEDBACK: <if score < 8, the SPECIFIC missing items, e.g. "missing formula: b+7 | 56; missing value b=49". If score >= 8, write "OK".>
"""

# Acceptance threshold and iteration cap — paper §Stage 4 (tau=8/10, max T=2 refine iterations).
MEMENTO_ACCEPT_THRESHOLD = 8
MEMENTO_MAX_REFINE_ITERS = 2


def _memento_llm(system_prompt: str, user_content: str) -> str | None:
    """Run one LLM call for memento compression/judging. Returns text or ``None`` on failure.

    Factored out so the compressor, judge, and tests share one resilient call site (tests
    monkeypatch this rather than standing up a live model).
    """
    try:
        from pydantic_ai import Agent

        from agent_utilities.core.config import (
            DEFAULT_KG_MODEL_ID,
            DEFAULT_LLM_PROVIDER,
        )
        from agent_utilities.core.model_factory import create_model

        model = create_model(
            provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
        )
        agent = Agent(model, system_prompt=system_prompt)

        # Survive being called from inside an already-running event loop —
        # strict no-op (and no global asyncio patching) otherwise.
        from agent_utilities.core.event_loop import allow_nested_run_sync

        allow_nested_run_sync()

        result = agent.run_sync(user_content)
        return str(getattr(result, "data", result)).strip()
    except Exception as e:  # pragma: no cover - exercised via monkeypatch in tests
        logger.warning("Memento LLM call failed: %s", e)
        return None


def _block_text(messages: list[dict[str, str]]) -> str:
    """Render a message block as a role-tagged transcript for compression/judging."""
    lines = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def judge_memento(block_text: str, memento_text: str) -> tuple[int, str]:
    """Score a candidate memento against its source block (CONCEPT:KG-2.20, paper §Stage 4).

    Returns ``(score 0-10, feedback)``. If the judge LLM is unavailable, returns
    ``(MEMENTO_ACCEPT_THRESHOLD, "")`` — i.e. *accept* — so judging never blocks compression in
    environments without a model (graceful degradation to single-shot behaviour).
    """
    user = (
        f"## ORIGINAL BLOCK\n{block_text}\n\n"
        f"## CANDIDATE MEMENTO\n{memento_text}\n\n"
        "Score and give feedback per the rubric."
    )
    raw = _memento_llm(MEMENTO_JUDGE_PROMPT, user)
    if not raw:
        return MEMENTO_ACCEPT_THRESHOLD, ""
    score = 0
    feedback = ""
    m = re.search(r"SCORE:\s*(\d+)", raw, re.IGNORECASE)
    if m:
        score = max(0, min(10, int(m.group(1))))
    fm = re.search(r"FEEDBACK:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if fm:
        feedback = fm.group(1).strip()
    return score, feedback


def compress_to_memento(
    engine: IntelligenceGraphEngine,
    messages: list[dict[str, str]],
    *,
    source: str = "agent_runner",
    dry_run: bool = False,
    refine: bool = True,
    max_refine_iters: int = MEMENTO_MAX_REFINE_ITERS,
    persist_raw: bool = True,
) -> str | None:
    """Compress a block of messages into a dense memento and persist it (CONCEPT:KG-2.20).

    Args:
        engine: IntelligenceGraphEngine instance (for persistence).
        messages: The block of raw messages to compress.
        source: The source agent or component name.
        dry_run: If True, do not persist to the KG.
        refine: Run the compressor→judge→recompress loop (paper §Stage 4). When the judge LLM is
            unavailable this transparently degrades to a single pass.
        max_refine_iters: Max additional compressor passes after the first (paper uses 2).
        persist_raw: Store the raw block as an ``EvictedBlock`` linked ``SUMMARIZES`` for lossless
            recovery (MEM-4). Default on so live eviction is never lossy.

    Returns:
        The accepted memento string, or ``None`` if compression failed.
    """
    if not messages:
        return None

    block_text = _block_text(messages)

    memento_text = _memento_llm(
        MEMENTO_SYSTEM_PROMPT,
        f"## Compress the following block into a Memento:\n\n{block_text}",
    )
    if not memento_text:
        return None

    # Judge-refine loop: recompress with the judge's specific feedback until it clears tau or we
    # exhaust the iteration budget. This is the 28%->92% quality step from the paper.
    if refine:
        for _ in range(max(0, max_refine_iters)):
            score, feedback = judge_memento(block_text, memento_text)
            if score >= MEMENTO_ACCEPT_THRESHOLD:
                break
            retry = _memento_llm(
                MEMENTO_SYSTEM_PROMPT,
                f"## Compress the following block into a Memento:\n\n{block_text}\n\n"
                f"## Your previous Memento was insufficient. Judge feedback (fix exactly this):\n"
                f"{feedback}\n\n## Produce an improved, self-contained Memento:",
            )
            if not retry:
                break
            memento_text = retry

    # Convergence-guaranteed escalation (CONCEPT:KG-2.20 / Root-Theorem F4): the
    # judge-refine loop can terminate with a memento that did NOT actually shrink the
    # block (LLM ignored the budget). Guarantee output < input by deterministic
    # truncation so eviction always reduces footprint (the raw block stays
    # losslessly recoverable via SUMMARIZES).
    memento_text = _guarantee_shorter(memento_text, block_text)

    # External verification gate (CONCEPT:KG-2.20): an *independent*, deterministic
    # faithfulness check (AHE-3.1 FaithfulnessScorer) of the memento against its
    # source block — distinct from the LLM self-judge above. The verdict is stamped
    # as provenance; a failure never blocks persistence (the raw block is kept
    # losslessly via SUMMARIZES so a low-fidelity memento can be re-expanded).
    verdict = verify_memento(block_text, memento_text)
    if not verdict["verified"]:
        logger.warning(
            "[KG-2.20] Memento failed external faithfulness gate "
            "(ratio=%.2f, ungrounded=%s) — persisting with lossless recoverability",
            verdict["faithful_ratio"],
            verdict["ungrounded"],
        )

    if dry_run:
        return memento_text

    raw_block = block_text if persist_raw else None
    _persist_memento(
        engine, memento_text, source=source, raw_block=raw_block, verification=verdict
    )
    return memento_text


def verify_memento(block_text: str, memento_text: str) -> dict[str, Any]:
    """External, deterministic faithfulness check of a memento vs its source block.

    CONCEPT:KG-2.20. Reuses the AHE-3.1 :class:`FaithfulnessScorer` as an
    *external* verifier (independent of the LLM judge-refine loop) so a
    compaction's grounding is auditable and gate-able. Returns
    ``{verified, faithful_ratio, ungrounded, verifier}``.
    """
    from agent_utilities.harness.reliability_scorers import FaithfulnessScorer

    result = FaithfulnessScorer().score("", memento_text, {"evidence": block_text})
    return {
        "verified": result.passed,
        "faithful_ratio": float(result.metrics.get("faithful_ratio", result.score)),
        "ungrounded": int(result.metrics.get("hallucinated", 0)),
        "verifier": result.evaluator,
    }


def _persist_memento(
    engine: IntelligenceGraphEngine,
    memento_text: str,
    *,
    source: str = "unknown",
    raw_block: str | None = None,
    verification: dict[str, Any] | None = None,
) -> str | None:
    """Persist the memento as a ``Memento`` node and return its id.

    CONCEPT:KG-2.20 (MEM-4 lossless recoverability). When ``raw_block`` is provided, the full evicted
    block is stored as an ``EvictedBlock`` node linked ``Memento -[:SUMMARIZES]-> EvictedBlock`` — the
    orchestration-layer substitute for the in-engine implicit KV channel the paper relies on. Eviction
    is therefore never lossy: :func:`recover_evicted_block` can re-fetch the raw block on demand.
    """
    if not engine or not getattr(engine, "backend", None):
        return None

    memento_id = f"mem_{hashlib.md5(memento_text.encode(), usedforsecurity=False).hexdigest()[:10]}"
    current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    props: dict[str, Any] = {
        "name": f"Memento: {current_time}",
        "content": memento_text,
        "source": source,
        "timestamp": current_time,
        "type": "MementoBlock",
        "recoverable": bool(raw_block),
    }
    if verification is not None:
        # Provenance stamp (CONCEPT:KG-2.20) — the external-verification verdict
        # travels with the memento so downstream trust/re-expansion is auditable.
        props["provenance_verified"] = bool(verification.get("verified"))
        props["provenance_faithfulness"] = float(
            verification.get("faithful_ratio", 0.0)
        )
        props["provenance_verifier"] = str(verification.get("verifier", ""))

    try:
        engine.add_node(memento_id, "Memento", properties=props)
        if raw_block:
            block_id = f"evicted_{hashlib.md5(raw_block.encode(), usedforsecurity=False).hexdigest()[:10]}"
            engine.add_node(
                block_id,
                "EvictedBlock",
                properties={
                    "name": f"Evicted block for {memento_id}",
                    "content": raw_block,
                    "source": source,
                    "timestamp": current_time,
                },
            )
            # lossless pointer: the memento SUMMARIZES the raw block it replaced
            engine.link_nodes(memento_id, block_id, "SUMMARIZES")
        logger.info("[KG-2.20] Persisted Memento context block (%s)", memento_id)
        return memento_id
    except Exception as e:
        logger.debug("Failed to persist Memento: %s", e)
        return None


def recover_evicted_block(
    engine: IntelligenceGraphEngine, memento_id: str
) -> str | None:
    """Re-fetch the raw block a memento replaced (CONCEPT:KG-2.20 MEM-4, lossless recall).

    Follows the ``Memento -[:SUMMARIZES]-> EvictedBlock`` pointer. Returns the raw block text, or
    ``None`` if the memento was not persisted with recoverability.
    """
    if not engine or not getattr(engine, "backend", None):
        return None
    try:
        rows = engine.backend.execute(
            "MATCH (m:Memento {id: $id})-[:SUMMARIZES]->(b:EvictedBlock) "
            "RETURN b.content AS content LIMIT 1",
            {"id": memento_id},
        )
        for r in rows or []:
            if r.get("content"):
                return str(r["content"])
    except Exception as e:
        logger.debug("Failed to recover evicted block for %s: %s", memento_id, e)
    return None


def _guarantee_shorter(
    memento_text: str, block_text: str, *, max_ratio: float = 0.9
) -> str:
    """Guarantee a memento is strictly smaller than the block it replaces.

    CONCEPT:KG-2.20 (Root-Theorem F4). If compression failed to reduce size, the
    memento is head-truncated to ``max_ratio`` of the block length with a marker —
    a deterministic terminal guarantee that eviction always shrinks the footprint.
    """
    cap = int(len(block_text) * max_ratio)
    if len(memento_text) <= cap or cap <= 0:
        return memento_text
    marker = " …[truncated:recoverable]"
    keep = max(0, cap - len(marker))
    return memento_text[:keep] + marker


def recover_chain(
    engine: IntelligenceGraphEngine, memento_id: str, *, max_depth: int = 16
) -> str | None:
    """Recover the leaf raw block by walking the SUMMARIZES DAG (CONCEPT:KG-2.20 MEM-4).

    Generalizes :func:`recover_evicted_block` to a hierarchical summary DAG:
    follows ``Memento -[:SUMMARIZES]-> (Memento|EvictedBlock)`` edges down to the
    deepest recoverable content, so multi-level mementos still expand losslessly.
    """
    if not engine or not getattr(engine, "backend", None):
        return None
    current = memento_id
    last_content: str | None = None
    seen: set[str] = set()
    for _ in range(max_depth):
        if current in seen:
            break
        seen.add(current)
        try:
            rows = engine.backend.execute(
                "MATCH (n {id: $id})-[:SUMMARIZES]->(c) "
                "RETURN c.id AS id, c.content AS content LIMIT 1",
                {"id": current},
            )
        except Exception as e:  # pragma: no cover - backend variance
            logger.debug("recover_chain hop failed at %s: %s", current, e)
            break
        row = next(iter(rows or []), None)
        if not row or not row.get("id"):
            break
        if row.get("content"):
            last_content = str(row["content"])
        current = str(row["id"])
    return last_content


def link_parent_memento(
    engine: IntelligenceGraphEngine, parent_id: str, child_ids: list[str]
) -> int:
    """Add a hierarchy level: ``parent_memento -[:SUMMARIZES]-> child_memento`` edges.

    CONCEPT:KG-2.20 — builds the summary DAG (summaries-of-summaries). Returns the
    number of edges linked. The parent's compressed content is produced by
    :func:`compress_to_memento` over the children's text; this wires the lineage.
    """
    if not engine or not getattr(engine, "link_nodes", None):
        return 0
    linked = 0
    for cid in child_ids:
        try:
            engine.link_nodes(parent_id, cid, "SUMMARIZES")
            linked += 1
        except Exception as e:  # pragma: no cover - best-effort
            logger.debug("link_parent_memento failed for %s: %s", cid, e)
    return linked


# ── Semantic-boundary segmentation (CONCEPT:KG-2.20 MEM-3, paper §Stage 1-3) ────────────────────
#
# The paper segments a flat CoT by scoring each inter-unit boundary 0-3 (a *local* question LLMs do
# well) then placing cuts via DP that maximises boundary quality minus a coefficient-of-variation
# size penalty, with a min-block floor (200 tokens). For an agent the natural atomic units are
# already there — each message (an action or observation) is a unit, and an action→observation /
# turn change is the analogue of a "major transition". We therefore score boundaries with the
# paper's signals and place cuts greedily at strong boundaries above the min-block floor (a DP-lite
# realisation; agent histories are short, so the full DP's balance term adds little). This keeps the
# property that matters: never cut mid-derivation, cut at coherent action/observation boundaries.

CONTINUATION_WORDS = ("therefore", "thus", "so ", "hence", "then", "and ", "but ")
BOUNDARY_CUT_THRESHOLD = (
    2.0  # cut only at boundaries scoring >= this (a "major transition")
)
DEFAULT_MIN_BLOCK_TOKENS = 200  # paper Stage 3 floor


def boundary_score(prev_msg: dict[str, Any], next_msg: dict[str, Any]) -> float:
    """Score the boundary *between* two messages 0 (mid-thought) → 3 (major transition).

    Faithful to the paper's local signals: penalise cutting where the prior unit ends with ``:``/``=``
    (mid-calculation) or the next unit opens with a continuation word (Therefore/Thus/So); reward
    role changes (turn boundaries) and action→observation cycles (tool result ↔ assistant).
    """
    prev_role = str(prev_msg.get("role", "")).lower()
    next_role = str(next_msg.get("role", "")).lower()
    prev_content = str(prev_msg.get("content", "")).strip()
    next_content = str(next_msg.get("content", "")).strip()

    score = 1.0
    if prev_role != next_role:
        score += 1.0  # turn boundary
    prev_is_tool = prev_role in ("tool", "function") or prev_msg.get("tool_call_id")
    next_is_tool = next_role in ("tool", "function") or next_msg.get("tool_call_id")
    if prev_is_tool != next_is_tool:
        score += 1.0  # action↔observation cycle boundary
    if prev_content.endswith((":", "=", "+", "-", "(", "[", "{", ",")):
        score -= 1.5  # mid-calculation — never split here
    if next_content.lower().startswith(CONTINUATION_WORDS):
        score -= 1.0  # next unit continues the prior thought
    return max(0.0, min(3.0, score))


def segment_into_blocks(
    messages: list[dict[str, Any]],
    *,
    min_block_tokens: int = DEFAULT_MIN_BLOCK_TOKENS,
) -> list[list[int]]:
    """Partition messages into coherent blocks (lists of indices); first index of each is its start.

    Cuts are placed at boundaries scoring ``>= BOUNDARY_CUT_THRESHOLD`` once the running block clears
    ``min_block_tokens``. A trailing fragment below the floor is attached to the preceding block so we
    never emit a tiny dangling block (the paper's balance intent).
    """
    from .agent_context import estimate_tokens

    n = len(messages)
    if n <= 1:
        return [list(range(n))]

    blocks: list[list[int]] = []
    cur: list[int] = []
    cur_tok = 0
    for i in range(n):
        cur.append(i)
        cur_tok += max(1, estimate_tokens(str(messages[i].get("content", ""))))
        at_strong_boundary = i == n - 1 or (
            boundary_score(messages[i], messages[i + 1]) >= BOUNDARY_CUT_THRESHOLD
        )
        if cur_tok >= min_block_tokens and at_strong_boundary and i != n - 1:
            blocks.append(cur)
            cur = []
            cur_tok = 0
    if cur:
        if blocks and cur_tok < min_block_tokens:
            blocks[-1].extend(cur)  # attach dangling short fragment to the last block
        else:
            blocks.append(cur)
    return blocks


def plan_block_eviction(
    messages: list[dict[str, Any]],
    *,
    budget_tokens: int,
    keep_recent_blocks: int = 1,
    keep_head: int = 1,
    min_block_tokens: int = DEFAULT_MIN_BLOCK_TOKENS,
) -> tuple[list[list[int]], list[int]]:
    """Plan the Memento sawtooth: which *completed* blocks to compress+evict to fit ``budget_tokens``.

    Returns ``(evict_block_groups, kept_indices)``. The head (``keep_head`` messages — typically the
    system prompt) and the most recent ``keep_recent_blocks`` blocks (the block currently being
    reasoned through) are always preserved; older completed blocks are evicted oldest-first only until
    the estimated remaining tokens fall under budget (we never evict more than necessary).
    """
    from .agent_context import estimate_message_tokens

    blocks = segment_into_blocks(messages, min_block_tokens=min_block_tokens)
    if len(blocks) <= keep_recent_blocks:
        return [], list(range(len(messages)))

    evictable = blocks[:-keep_recent_blocks] if keep_recent_blocks else blocks
    # never evict the head messages (system prompt etc.)
    head = set(range(min(keep_head, len(messages))))

    evicted_groups: list[list[int]] = []
    evicted_idx: set[int] = set()
    for block in evictable:
        block_no_head = [i for i in block if i not in head]
        if not block_no_head:
            continue
        remaining = [m for j, m in enumerate(messages) if j not in evicted_idx]
        if estimate_message_tokens(remaining) <= budget_tokens:
            break  # already under budget — stop evicting (minimal eviction)
        evicted_groups.append(block_no_head)
        evicted_idx.update(block_no_head)

    kept = [i for i in range(len(messages)) if i not in evicted_idx]
    return evicted_groups, kept


def get_recent_mementos(
    engine: IntelligenceGraphEngine,
    source: str,
    limit: int = 5,
) -> list[str]:
    """Retrieve the most recent mementos for a given source (oldest-first, for forward reasoning)."""
    if not engine or not getattr(engine, "backend", None):
        return []

    try:
        rows = engine.backend.execute(
            "MATCH (m:Memento {source: $source}) "
            "RETURN m.content AS content "
            "ORDER BY m.timestamp ASC LIMIT $limit",
            {"source": source, "limit": limit},
        )
        return [r.get("content", "") for r in rows if r.get("content")]
    except Exception as e:
        logger.debug("Failed to retrieve Mementos: %s", e)
        return []
