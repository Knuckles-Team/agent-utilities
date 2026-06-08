#!/usr/bin/python
from __future__ import annotations

"""Provenance-completeness critic gate — deterministic accept/revise/escalate.

CONCEPT:AHE-3.13 — Layered Pre-Emit Gate (provenance check)

A deterministic pre-emit gate distilled from the MAKA "traceable, risk-aware
decision support" research (``.specify/specs/research-evolution-20260606/`` plan
b4-04). Before an answer is emitted it verifies *provenance completeness*:

* every **numeric** claim must trace to a tool output / supplied value, or sit in
  a sentence carrying a valid citation;
* every **substantive** sentence (carries a number or is non-trivially long) must
  carry a citation ``[id]`` that references a known source.

It then returns ``accept`` / ``revise`` / ``escalate`` under a revise budget —
so under-grounded answers are sent back for repair, and persistently
under-grounded ones are escalated rather than silently shipped.

This is **complementary** to :mod:`agent_utilities.capabilities.adversarial_verifier`
(which uses an LLM to *refute* a claim): this gate is a cheap, deterministic
provenance accounting pass with no model or network.

Concept: provenance-gate
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[([^\]]+)\]")
# Numbers: integers/decimals/percentages with optional thousands separators,
# excluding bracketed citation ids (handled separately).
_NUMBER = re.compile(r"(?<![\[\w])\d[\d,]*(?:\.\d+)?%?")

Decision = Literal["accept", "revise", "escalate"]


class ProvenanceVerdict(BaseModel):
    """Outcome of a provenance-completeness check."""

    decision: Decision
    completeness: float = Field(ge=0.0, le=1.0)
    numeric_grounded: float = Field(ge=0.0, le=1.0)
    claim_grounded: float = Field(ge=0.0, le=1.0)
    ungrounded_numbers: list[str] = Field(default_factory=list)
    uncited_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def _norm_num(tok: str) -> str:
    return tok.replace(",", "").rstrip("%")


@dataclass
class ProvenanceCriticGate:
    """Deterministic provenance-completeness gate.

    Args:
        accept_threshold: Completeness at/above which the answer is accepted.
        max_revise: Number of revise attempts before escalation.
        min_claim_words: Sentences shorter than this (and number-free) are treated
            as non-substantive and exempt from the citation requirement.
    """

    accept_threshold: float = 0.9
    max_revise: int = 2
    min_claim_words: int = 6

    def evaluate(
        self,
        answer: str,
        *,
        sources: list[str] | dict[str, Any] | None = None,
        tool_values: list[Any] | None = None,
        attempt: int = 0,
    ) -> ProvenanceVerdict:
        """Check provenance completeness and return a gated decision.

        Args:
            answer: The answer text about to be emitted.
            sources: Valid source ids citations may reference (list or dict keys).
            tool_values: Values produced by tools that numbers may trace to.
            attempt: How many revise cycles have already happened.
        """
        valid_sources = set(
            sources.keys() if isinstance(sources, dict) else (sources or [])
        )
        valid_sources = {str(s) for s in valid_sources}
        tool_norm = {_norm_num(str(v)) for v in (tool_values or [])}

        sentences = _sentences(answer)

        # --- numeric grounding -------------------------------------------------
        all_numbers: list[str] = []
        ungrounded_numbers: list[str] = []
        invalid_citations: list[str] = []
        for sent in sentences:
            cites = _CITATION.findall(sent)
            sent_has_valid_cite = any(c.strip() in valid_sources for c in cites)
            for c in cites:
                if c.strip() not in valid_sources:
                    invalid_citations.append(c.strip())
            for num in _NUMBER.findall(sent):
                all_numbers.append(num)
                grounded = _norm_num(num) in tool_norm or sent_has_valid_cite
                if not grounded:
                    ungrounded_numbers.append(num)

        numeric_grounded = (
            1.0 - len(ungrounded_numbers) / len(all_numbers) if all_numbers else 1.0
        )

        # --- claim (sentence) grounding ---------------------------------------
        substantive = [s for s in sentences if self._is_substantive(s)]
        uncited_claims: list[str] = []
        for sent in substantive:
            cites = _CITATION.findall(sent)
            if not any(c.strip() in valid_sources for c in cites):
                uncited_claims.append(sent[:120])
        claim_grounded = (
            1.0 - len(uncited_claims) / len(substantive) if substantive else 1.0
        )

        completeness = 0.5 * numeric_grounded + 0.5 * claim_grounded

        reasons: list[str] = []
        if ungrounded_numbers:
            reasons.append(f"{len(ungrounded_numbers)} ungrounded numeric claim(s)")
        if uncited_claims:
            reasons.append(f"{len(uncited_claims)} uncited substantive claim(s)")
        if invalid_citations:
            reasons.append(
                f"citation(s) to unknown sources: {sorted(set(invalid_citations))}"
            )

        if completeness >= self.accept_threshold and not invalid_citations:
            decision: Decision = "accept"
        elif attempt < self.max_revise:
            decision = "revise"
        else:
            decision = "escalate"

        return ProvenanceVerdict(
            decision=decision,
            completeness=round(completeness, 6),
            numeric_grounded=round(numeric_grounded, 6),
            claim_grounded=round(claim_grounded, 6),
            ungrounded_numbers=ungrounded_numbers,
            uncited_claims=uncited_claims,
            invalid_citations=sorted(set(invalid_citations)),
            reasons=reasons,
        )

    def _is_substantive(self, sentence: str) -> bool:
        if _NUMBER.search(sentence):
            return True
        words = [w for w in re.findall(r"[A-Za-z0-9']+", sentence)]
        return len(words) >= self.min_claim_words
