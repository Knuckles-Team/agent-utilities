"""CONCEPT:AU-KG.memory.mementified-context — Mementified Context Management (state compression).

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
* **Judge-refine loop** (CONCEPT:AU-KG.memory.mementified-context / paper §Stage 4) — a compressor→judge→recompress cycle.
  The paper measured single-shot mementos at a **28%** rubric pass-rate vs **92%** after two judge
  iterations: initial mementos routinely drop an exact formula or intermediate value a downstream
  block needs. :func:`compress_to_memento` runs this loop by default (``refine=True``), scoring each
  candidate on a six-dimension rubric (formulas-verbatim, values-preserved, methods-named,
  validation, no-hallucination, result-first) with acceptance threshold ``τ=8/10`` and ``≤2`` extra
  iterations, exactly as in the paper.

**Honest limitation (orchestration layer).** We run hosted/API models via pydantic-ai and do not
control the inference engine's KV cache, so an external memento is the paper's *"restart mode"* — it
loses the implicit dual-channel KV side-information the paper measured at −15pp. Optional raw recall
is disabled by default for privacy. An operator may approve encrypted recovery through a versioned
policy and secret-backed AES-GCM key; plaintext transcripts are never persisted. This is a substitute
for, not an equivalent of, the in-engine channel.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_utilities.core.config import setting
from agent_utilities.security.persistence_privacy import (
    persistence_reference,
    sanitize_for_persistence,
)

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

# Raw blocks are sensitive conversation transcripts.  They are never persisted unless an
# operator explicitly enables this versioned policy *and* supplies a resolvable secret reference.
# The policy token is deliberately exact (rather than a permissive truthy flag) so a typo or an old
# deployment configuration fails closed.
MEMENTO_RAW_RETENTION_POLICY = "approved-encrypted-v1"
MEMENTO_RAW_ENCRYPTION_ALGORITHM = "AES-256-GCM"
_ENCRYPTED_CONTENT_MARKER = "[ENCRYPTED_RAW_BLOCK]"


@dataclass(frozen=True)
class _RawRetentionCipher:
    """Resolved in-memory encryption context; secret material is never persisted."""

    key: bytes
    key_reference: str


def memento_source_reference(source: str) -> str:
    """Return the canonical opaque durable reference for a Memento source."""

    return persistence_reference("memento_source", source, namespace="memory")


def _resolve_secret_reference(reference: str) -> str | None:
    """Resolve a raw-retention key through the configured secrets backend."""

    from agent_utilities.security.secrets_client import create_secrets_client

    return create_secrets_client().resolve_ref(reference)


def _raw_retention_requested(persist_raw: bool | None) -> bool:
    """Resolve the per-call override against the disabled-by-default environment setting."""

    if persist_raw is not None:
        return bool(persist_raw)
    return bool(setting("MEMENTO_RAW_RETENTION_ENABLED", False))


def _raw_retention_cipher() -> _RawRetentionCipher | None:
    """Return an approved, secret-backed cipher context or ``None`` (fail closed)."""

    if not bool(setting("MEMENTO_RAW_RETENTION_ENABLED", False)):
        return None
    policy = str(setting("MEMENTO_RAW_RETENTION_POLICY", "") or "").strip()
    if policy != MEMENTO_RAW_RETENTION_POLICY:
        return None
    reference = str(setting("MEMENTO_RAW_ENCRYPTION_KEY_REF", "") or "").strip()
    if not reference:
        return None
    try:
        secret = _resolve_secret_reference(reference)
    except Exception:  # the reference/error can itself contain deployment details
        logger.warning(
            "Memento raw retention key could not be resolved; retention disabled"
        )
        return None
    if not secret:
        return None

    # Accept any high-entropy secret representation from the configured backend while deriving the
    # fixed-width AES key with domain separation.  Only an opaque reference fingerprint is stored.
    key = hashlib.sha256(
        b"agent-utilities:memento-raw-key:v1\x00" + str(secret).encode("utf-8")
    ).digest()
    return _RawRetentionCipher(
        key=key,
        key_reference=persistence_reference(
            "memento_raw_key", reference, namespace="memory"
        ),
    )


def _raw_aad(block_id: str) -> bytes:
    return f"agent-utilities:memento-raw:v1\x00{block_id}".encode()


def _encrypt_raw_block(
    raw_block: str, block_id: str, cipher: _RawRetentionCipher
) -> dict[str, Any]:
    """Encrypt one transcript with an authenticated, block-bound envelope."""

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(cipher.key).encrypt(
        nonce, raw_block.encode("utf-8"), _raw_aad(block_id)
    )
    return {
        "content": _ENCRYPTED_CONTENT_MARKER,
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "encryption_algorithm": MEMENTO_RAW_ENCRYPTION_ALGORITHM,
        "encryption_version": 1,
        "key_reference": cipher.key_reference,
    }


def _decrypt_raw_block(row: dict[str, Any]) -> str | None:
    """Decrypt an encrypted EvictedBlock only while its retention policy remains approved."""

    cipher = _raw_retention_cipher()
    if cipher is None:
        return None
    if str(row.get("encryption_algorithm") or "") != MEMENTO_RAW_ENCRYPTION_ALGORITHM:
        return None
    if int(row.get("encryption_version") or 0) != 1:
        return None
    stored_reference = str(row.get("key_reference") or "")
    if stored_reference and not hmac.compare_digest(
        stored_reference, cipher.key_reference
    ):
        return None
    block_id = str(row.get("id") or "")
    if not block_id:
        return None
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        plaintext = AESGCM(cipher.key).decrypt(
            base64.b64decode(str(row.get("nonce") or ""), validate=True),
            base64.b64decode(str(row.get("ciphertext") or ""), validate=True),
            _raw_aad(block_id),
        )
        return plaintext.decode("utf-8")
    except Exception:
        # Authentication failures and malformed encrypted records are indistinguishable.
        logger.warning("Encrypted Memento raw block could not be authenticated")
        return None


def _memento_llm(system_prompt: str, user_content: str) -> str | None:
    """Run one LLM call for memento compression/judging. Returns text or ``None`` on failure.

    Factored out so the compressor, judge, and tests share one resilient call site (tests
    monkeypatch this rather than standing up a live model).
    """
    try:
        from agent_utilities.core.config import (
            DEFAULT_KG_MODEL_ID,
            DEFAULT_LLM_PROVIDER,
        )
        from agent_utilities.core.contextual_model import create_context_agent
        from agent_utilities.core.model_factory import create_model

        model = create_model(
            provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
        )
        agent = create_context_agent(model, system_prompt=system_prompt)

        # Survive being called from inside an already-running event loop —
        # strict no-op (and no global asyncio patching) otherwise.
        from agent_utilities.core.event_loop import allow_nested_run_sync

        allow_nested_run_sync()

        result = agent.run_sync(user_content)
        return str(getattr(result, "data", result)).strip()
    except Exception:  # pragma: no cover - exercised via monkeypatch in tests
        logger.warning("Memento LLM call failed")
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
    """Score a candidate memento against its source block (CONCEPT:AU-KG.memory.mementified-context, paper §Stage 4).

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
    persist_raw: bool | None = None,
) -> str | None:
    """Compress a block of messages into a dense memento and persist it (CONCEPT:AU-KG.memory.mementified-context).

    Args:
        engine: IntelligenceGraphEngine instance (for persistence).
        messages: The block of raw messages to compress.
        source: The source agent or component name.
        dry_run: If True, do not persist to the KG.
        refine: Run the compressor→judge→recompress loop (paper §Stage 4). When the judge LLM is
            unavailable this transparently degrades to a single pass.
        max_refine_iters: Max additional compressor passes after the first (paper uses 2).
        persist_raw: Request encrypted raw-block retention. The default reads
            ``MEMENTO_RAW_RETENTION_ENABLED`` (which defaults to false). A request is honored only
            when the versioned retention policy is explicitly approved and
            ``MEMENTO_RAW_ENCRYPTION_KEY_REF`` resolves through the secrets backend. Plaintext raw
            blocks are never persisted.

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

    # Convergence-guaranteed escalation (CONCEPT:AU-KG.memory.mementified-context / Root-Theorem F4): the
    # judge-refine loop can terminate with a memento that did NOT actually shrink the
    # block (LLM ignored the budget). Guarantee output < input by deterministic
    # truncation so eviction always reduces footprint. Optional raw recovery is handled separately
    # by the explicit encrypted-retention policy.
    memento_text = _guarantee_shorter(memento_text, block_text)

    # External verification gate (CONCEPT:AU-KG.memory.mementified-context): an *independent*, deterministic
    # faithfulness check (AHE-3.1 FaithfulnessScorer) of the memento against its
    # source block — distinct from the LLM self-judge above. The verdict is stamped
    # as provenance; a failure never blocks privacy-safe memento persistence.
    verdict = verify_memento(block_text, memento_text)
    if not verdict["verified"]:
        logger.warning(
            "[KG-2.20] Memento failed external faithfulness gate "
            "(ratio=%.2f, ungrounded=%s) — persisting privacy-safe memento",
            verdict["faithful_ratio"],
            verdict["ungrounded"],
        )

    if dry_run:
        return memento_text

    raw_block = block_text if _raw_retention_requested(persist_raw) else None
    _persist_memento(
        engine, memento_text, source=source, raw_block=raw_block, verification=verdict
    )
    return memento_text


def verify_memento(block_text: str, memento_text: str) -> dict[str, Any]:
    """External, deterministic faithfulness check of a memento vs its source block.

    CONCEPT:AU-KG.memory.mementified-context. Reuses the AHE-3.1 :class:`FaithfulnessScorer` as an
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

    Memento content passes through the persistence privacy guard and ``source`` is converted to an
    opaque durable reference. Raw content is disabled by default; when requested, it is retained
    only as an authenticated AES-GCM envelope under the explicit versioned policy and secret-backed
    key. No plaintext transcript or secret reference is written to the graph.
    """
    if not engine or not getattr(engine, "backend", None):
        return None

    source_ref = memento_source_reference(source)
    clean_value, privacy_report = sanitize_for_persistence(memento_text)
    clean_memento = str(clean_value)
    memento_id = "memento:" + persistence_reference(
        "memento", clean_memento, namespace=source_ref
    )
    current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    block_id: str | None = None
    block_props: dict[str, Any] | None = None
    if raw_block:
        cipher = _raw_retention_cipher()
        if cipher is None:
            logger.warning(
                "Memento raw retention was requested without an approved secret-backed policy; "
                "retention disabled"
            )
        else:
            # A keyed content reference avoids exposing even a dictionary-testable plaintext digest.
            block_digest = hmac.new(
                cipher.key, raw_block.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            block_id = "evicted:" + persistence_reference(
                "evicted_block", block_digest, namespace=source_ref
            )
            try:
                block_props = {
                    "name": "Encrypted evicted Memento block",
                    "source": source_ref,
                    "timestamp": current_time,
                    **_encrypt_raw_block(raw_block, block_id, cipher),
                }
            except Exception:
                # Encryption is mandatory for this path. Never fall back to plaintext.
                logger.warning(
                    "Memento raw-block encryption failed; retention disabled"
                )
                block_id = None
                block_props = None

    props: dict[str, Any] = {
        "name": f"Memento: {current_time}",
        "content": clean_memento,
        "source": source_ref,
        "timestamp": current_time,
        "recoverable": bool(block_props),
        "privacy_redactions": privacy_report.redactions,
        "privacy_categories": list(privacy_report.detected_types),
    }
    if verification is not None:
        # Provenance stamp (CONCEPT:AU-KG.memory.mementified-context) — the external-verification verdict
        # travels with the memento so downstream trust/re-expansion is auditable.
        props["provenance_verified"] = bool(verification.get("verified"))
        props["provenance_faithfulness"] = float(
            verification.get("faithful_ratio", 0.0)
        )
        props["provenance_verifier"] = str(verification.get("verifier", ""))

    try:
        engine.add_node(memento_id, "Memento", properties=props)
        if block_id and block_props:
            engine.add_node(
                block_id,
                "EvictedBlock",
                properties=block_props,
            )
            # The pointer exposes neither source identity nor plaintext content.
            engine.link_nodes(memento_id, block_id, "SUMMARIZES")
        logger.info("[KG-2.20] Persisted Memento context block (%s)", memento_id)
        return memento_id
    except Exception:
        logger.debug("Failed to persist privacy-safe Memento")
        return None


def _recover_evicted_row(
    row: dict[str, Any],
) -> str | None:
    if not row.get("ciphertext"):
        return None
    return _decrypt_raw_block(row)


def recover_evicted_block(
    engine: IntelligenceGraphEngine, memento_id: str
) -> str | None:
    """Recover an approved encrypted block (CONCEPT:AU-KG.memory.mementified-context MEM-4).

    Follows the ``Memento -[:SUMMARIZES]-> EvictedBlock`` pointer. Encrypted records are decrypted
    only while the explicit retention policy and referenced secret remain available. Any record
    outside the current encrypted schema fails closed.
    """
    if not engine or not getattr(engine, "backend", None):
        return None
    try:
        rows = engine.backend.execute(
            "MATCH (m:Memento {id: $id})-[:SUMMARIZES]->(b:EvictedBlock) "
            "RETURN b.id AS id, b.content AS content, b.source AS source, "
            "b.timestamp AS timestamp, b.ciphertext AS ciphertext, b.nonce AS nonce, "
            "b.encryption_algorithm AS encryption_algorithm, "
            "b.encryption_version AS encryption_version, b.key_reference AS key_reference LIMIT 1",
            {"id": memento_id},
        )
        for r in rows or []:
            recovered = _recover_evicted_row(dict(r))
            if recovered is not None:
                return recovered
    except Exception:
        logger.debug("Failed to recover encrypted Memento raw block")
    return None


def _guarantee_shorter(
    memento_text: str, block_text: str, *, max_ratio: float = 0.9
) -> str:
    """Guarantee a memento is strictly smaller than the block it replaces.

    CONCEPT:AU-KG.memory.mementified-context (Root-Theorem F4). If compression failed to reduce size, the
    memento is head-truncated to ``max_ratio`` of the block length with a marker —
    a deterministic terminal guarantee that eviction always shrinks the footprint.
    """
    cap = int(len(block_text) * max_ratio)
    if len(memento_text) <= cap or cap <= 0:
        return memento_text
    marker = " …[truncated:memento]"
    keep = max(0, cap - len(marker))
    return memento_text[:keep] + marker


def recover_chain(
    engine: IntelligenceGraphEngine, memento_id: str, *, max_depth: int = 16
) -> str | None:
    """Recover an approved leaf by walking the SUMMARIZES DAG (CONCEPT:AU-KG.memory.mementified-context MEM-4).

    Generalizes :func:`recover_evicted_block` to a hierarchical summary DAG. Memento summaries are
    read normally; an ``EvictedBlock`` leaf is subject to the same encrypted-retention policy as
    direct recovery.
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
                "RETURN c.id AS id, c.node_type AS node_type, c.content AS content, "
                "c.source AS source, c.timestamp AS timestamp, c.ciphertext AS ciphertext, "
                "c.nonce AS nonce, c.encryption_algorithm AS encryption_algorithm, "
                "c.encryption_version AS encryption_version, "
                "c.key_reference AS key_reference LIMIT 1",
                {"id": current},
            )
        except Exception:  # pragma: no cover - backend variance
            logger.debug("Encrypted Memento recovery-chain hop failed")
            break
        row = next(iter(rows or []), None)
        if not row or not row.get("id"):
            break
        node_type = str(row.get("node_type") or "")
        is_evicted = bool(row.get("ciphertext")) or node_type == "EvictedBlock"
        if is_evicted:
            return _recover_evicted_row(dict(row))
        if row.get("content"):
            clean, _ = sanitize_for_persistence(str(row["content"]))
            last_content = str(clean)
        current = str(row["id"])
    return last_content


def link_parent_memento(
    engine: IntelligenceGraphEngine, parent_id: str, child_ids: list[str]
) -> int:
    """Add a hierarchy level: ``parent_memento -[:SUMMARIZES]-> child_memento`` edges.

    CONCEPT:AU-KG.memory.mementified-context — builds the summary DAG (summaries-of-summaries). Returns the
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
        except Exception:  # pragma: no cover - best-effort
            logger.debug("Memento hierarchy link failed")
    return linked


# ── Semantic-boundary segmentation (CONCEPT:AU-KG.memory.mementified-context MEM-3, paper §Stage 1-3) ────────────────────
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
    """Retrieve recent mementos using the current opaque source reference."""
    if not engine or not getattr(engine, "backend", None):
        return []

    source_ref = memento_source_reference(source)
    records: list[dict[str, str]] = []

    def _append_row(row: dict[str, Any]) -> None:
        raw_content = str(row.get("content") or "")
        if not raw_content:
            return
        clean_value, report = sanitize_for_persistence(raw_content)
        clean_content = str(clean_value)
        node_id = str(row.get("id") or "")
        if node_id and report.changed:
            try:
                engine.add_node(
                    node_id,
                    "Memento",
                    properties={
                        "content": clean_content,
                        "source": source_ref,
                        "privacy_redactions": report.redactions,
                        "privacy_categories": list(report.detected_types),
                    },
                )
            except Exception:
                logger.warning(
                    "Memento content could not be privacy-sanitized in place"
                )
        records.append(
            {
                "id": node_id,
                "content": clean_content,
                "timestamp": str(row.get("timestamp") or ""),
            }
        )

    try:
        rows = engine.backend.execute(
            "MATCH (m:Memento {source: $source}) "
            "RETURN m.id AS id, m.content AS content, m.timestamp AS timestamp "
            "ORDER BY m.timestamp ASC LIMIT $limit",
            {"source": source_ref, "limit": limit},
        )
        for row in rows or []:
            _append_row(dict(row))
        return [row["content"] for row in records]
    except Exception:
        logger.debug("Failed to retrieve privacy-safe Mementos")
        return []
