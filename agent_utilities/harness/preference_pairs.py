#!/usr/bin/python
from __future__ import annotations

"""First-class preference-pair corpus — the DPO-family substrate.

CONCEPT:AU-AHE.harness.preference-corpus-reliability — Preference-Corpus Reliability

We already generate *implicit* preference signal in three places, but never
consolidated it into one clean, DPO-ready store:

* :class:`~agent_utilities.harness.eval_corpus.EvalCorpus` — regression cases
  (``query`` → ``expected_output``); a case carrying an ``actual``/``rejected`` in
  its metadata is a ready (chosen, rejected) pair.
* the trace distiller's ``EpisodeToPreferenceRule`` — writes ``preference`` nodes
  (a successful vs a failed episode over the same context).
* :class:`~agent_utilities.knowledge_graph.adaptation.feedback.FeedbackService` —
  a human correction is ``corrected_value`` (chosen) vs the original (rejected).

:class:`PreferencePairExporter` consolidates all three into deduplicated
:class:`PreferencePair` records. The DPO-family refinements layer on top:

* **RAPPO** (arXiv OR LrHfYPFTtg) — :func:`reliability_filter` drops ambiguous /
  low-margin pairs ("keep the best, forget the rest"), logging the drop count.
* **TI-DPO** (arXiv:2505.19653) — per-token ``token_weights`` on the chosen response.
* **InSPO** (arXiv:2512.23126) — an optional ``alternative`` the policy is
  conditioned on (reflective preference optimization).

Source: ``.specify/specs/reasoning-rl-2026/`` (W1.1 + W3.1–3.3).
"""

import hashlib
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PreferencePair(BaseModel):
    """A single (chosen ≻ rejected) preference example over one prompt."""

    id: str = Field(default="", description="content hash; stable + dedup key")
    prompt: str
    chosen: str
    rejected: str
    source: str = Field(default="", description="eval_corpus | distilled | correction")
    # RAPPO: preference margin / confidence in [0, 1]; lower = more ambiguous.
    margin: float = Field(default=1.0, ge=0.0, le=1.0)
    # TI-DPO: optional per-token importance weights over the chosen response.
    token_weights: list[float] = Field(default_factory=list)
    # InSPO: optional alternative response to condition on (reflective DPO).
    alternative: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _ctx: Any) -> None:
        if not self.id:
            raw = f"{self.prompt}{self.chosen}{self.rejected}".encode(errors="replace")
            object.__setattr__(
                self, "id", f"pref-{hashlib.sha256(raw).hexdigest()[:12]}"
            )


def _coerce_str(v: Any) -> str:
    return v if isinstance(v, str) else ("" if v is None else str(v))


class PreferencePairExporter:
    """Consolidate preference signal from the eval corpus, distilled episodes, and
    human corrections into one deduplicated :class:`PreferencePair` list.

    Backend-agnostic: pass explicit source rows (``export_from``) for testing, or use
    :meth:`from_engine` to gather them from the live KG.
    """

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    @classmethod
    def from_engine(cls, engine: Any) -> PreferencePairExporter:
        backend = getattr(engine, "backend", None) or engine
        return cls(backend=backend)

    # --- normalization of each source shape into PreferencePair ----------------

    @staticmethod
    def _from_eval_case(row: dict[str, Any]) -> PreferencePair | None:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = {}
        rejected = meta.get("rejected") or meta.get("actual")
        chosen = row.get("expected_output")
        prompt = row.get("query")
        if not (prompt and chosen and rejected):
            return None
        return PreferencePair(
            prompt=_coerce_str(prompt),
            chosen=_coerce_str(chosen),
            rejected=_coerce_str(rejected),
            source="eval_corpus",
            metadata={"case_id": row.get("id")},
        )

    @staticmethod
    def _from_preference_node(row: dict[str, Any]) -> PreferencePair | None:
        prompt = row.get("prompt") or row.get("context") or row.get("query")
        chosen = row.get("chosen") or row.get("winner") or row.get("preferred")
        rejected = row.get("rejected") or row.get("loser") or row.get("dispreferred")
        if not (prompt and chosen and rejected):
            return None
        return PreferencePair(
            prompt=_coerce_str(prompt),
            chosen=_coerce_str(chosen),
            rejected=_coerce_str(rejected),
            source="distilled",
            metadata={"node_id": row.get("id")},
        )

    @staticmethod
    def _from_correction(row: dict[str, Any]) -> PreferencePair | None:
        prompt = row.get("target") or row.get("query")
        chosen = row.get("corrected_value") or row.get("expected")
        rejected = row.get("original") or row.get("rejected")
        if not (prompt and chosen and rejected):
            return None
        return PreferencePair(
            prompt=_coerce_str(prompt),
            chosen=_coerce_str(chosen),
            rejected=_coerce_str(rejected),
            source="correction",
            metadata={"correction_id": row.get("id")},
        )

    def export_from(
        self,
        *,
        eval_cases: list[dict[str, Any]] | None = None,
        preference_nodes: list[dict[str, Any]] | None = None,
        corrections: list[dict[str, Any]] | None = None,
    ) -> list[PreferencePair]:
        """Normalize + dedup explicit source rows into PreferencePairs."""
        pairs: list[PreferencePair] = []
        for row in eval_cases or []:
            if (p := self._from_eval_case(row)) is not None:
                pairs.append(p)
        for row in preference_nodes or []:
            if (p := self._from_preference_node(row)) is not None:
                pairs.append(p)
        for row in corrections or []:
            if (p := self._from_correction(row)) is not None:
                pairs.append(p)
        # Dedup by content id, preferring the first (richer) source seen.
        seen: dict[str, PreferencePair] = {}
        for p in pairs:
            seen.setdefault(p.id, p)
        return list(seen.values())

    def export(self) -> list[PreferencePair]:
        """Gather all three sources from the live KG backend and consolidate."""
        if self.backend is None or not hasattr(self.backend, "execute"):
            return []

        def _q(cypher: str) -> list[dict[str, Any]]:
            try:
                return self.backend.execute(cypher) or []
            except Exception as e:  # noqa: BLE001 — one source failing never blocks others
                logger.debug("preference export query failed: %s", e)
                return []

        eval_cases = _q(
            "MATCH (c) WHERE c.type = 'eval_case' "
            "RETURN c.id as id, c.query as query, c.expected_output as expected_output, "
            "c.metadata as metadata"
        )
        preference_nodes = _q(
            "MATCH (p) WHERE p.type = 'preference' "
            "RETURN p.id as id, p.prompt as prompt, p.context as context, "
            "p.chosen as chosen, p.rejected as rejected"
        )
        corrections = _q(
            "MATCH (c) WHERE c.type = 'correction' "
            "RETURN c.id as id, c.target as target, c.corrected_value as corrected_value, "
            "c.original as original"
        )
        return self.export_from(
            eval_cases=eval_cases,
            preference_nodes=preference_nodes,
            corrections=corrections,
        )


# --- DPO-family refinements (W3.1–3.3) -----------------------------------------


def reliability_filter(
    pairs: list[PreferencePair], *, min_margin: float = 0.1
) -> tuple[list[PreferencePair], int]:
    """RAPPO: drop ambiguous / low-margin pairs ("keep the best, forget the rest").

    Also drops degenerate pairs whose ``chosen == rejected`` (zero information).
    Returns ``(kept, dropped_count)``; callers should ``log`` the drop count — no
    silent truncation. (arXiv OR LrHfYPFTtg)
    """
    kept: list[PreferencePair] = []
    dropped = 0
    for p in pairs:
        if p.chosen.strip() == p.rejected.strip() or p.margin < min_margin:
            dropped += 1
            continue
        kept.append(p)
    return kept, dropped


def attach_token_weights(pair: PreferencePair, weights: list[float]) -> PreferencePair:
    """TI-DPO: attach per-token importance weights over the chosen response so a
    downstream DPO loss can focus on the tokens that drive the preference.
    (arXiv:2505.19653)
    """
    return pair.model_copy(update={"token_weights": list(weights)})


def with_reflection(pair: PreferencePair, alternative: str) -> PreferencePair:
    """InSPO: condition the pair on an *alternative* response (reflective DPO).

    Off by default — callers opt in by supplying an alternative (e.g. an
    adversarial-critic rewrite). (arXiv:2512.23126)
    """
    return pair.model_copy(update={"alternative": alternative})
