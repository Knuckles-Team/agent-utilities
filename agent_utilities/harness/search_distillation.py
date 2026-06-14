#!/usr/bin/python
from __future__ import annotations

"""Search-distillation harvester: turn solved reasoning into training data.

CONCEPT:OS-5.36 — a search-distillation harvester that converts the reasoning router's verified high-scoring results and best-of-k candidate sets into a versioned SFT and preference-pair corpus, collapse-guarded, so test-time search compute becomes better training data

The paper's answer to the data wall (§5.1/§5.3) is to *convert test-time compute into
better data* — distil search-improved outputs (AlphaZero-style) back into a corpus that
trains the next prior. AU now produces exactly that signal: the KG-2.68 reasoning router
runs paradigms and scores each result, and test-time-diversity produces scored best-of-k
candidate sets. This harvester taps those, rejection-samples the winners into
``(prompt → completion)`` SFT rows and ``(chosen, rejected)`` preference pairs, gates each
through the SAFE-1.4 model-collapse guard, and persists a versioned ``SyntheticCorpus`` the
in-house trainer (ML-001..007) can consume — closing the test-time-compute → training-data
loop. The fine-tune consumption itself needs external compute; the harvest + curation is
pure and in-repo.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_SEP = "\x1f"  # prompt/completion key separator


@dataclass
class SFTRow:
    """One supervised fine-tuning row distilled from a solved task."""

    prompt: str
    completion: str
    score: float
    source: str = ""
    synthetic: bool = True


@dataclass
class PreferencePair:
    """A ``(chosen ≻ rejected)`` pair distilled from a scored candidate set."""

    prompt: str
    chosen: str
    rejected: str


def _stringify(answer: Any) -> str:
    """Render a paradigm answer to a stable completion string."""
    if answer is None:
        return ""
    if hasattr(answer, "render") and callable(answer.render):
        return str(answer.render())
    if isinstance(answer, (set, frozenset)):
        return ", ".join(sorted(str(x) for x in answer))
    return str(answer)


@dataclass
class SearchDistillationHarvester:
    """Harvest the router's verified winners into a collapse-guarded corpus."""

    engine: Any = None
    min_score: float = 0.8
    guard: Any = None
    _corpus: list[SFTRow] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.guard is None:
            from agent_utilities.harness.corpus_collapse_guard import CorpusCollapseGuard

            self.guard = CorpusCollapseGuard()

    # ── from a single routed result (the live router path) ───────────
    def harvest_result(self, task: Any, result: Any) -> SFTRow | None:
        """Mint an SFT row from a high-scoring routed reasoning result, or ``None``."""
        if result is None or float(getattr(result, "score", 0.0)) < self.min_score:
            return None
        completion = _stringify(getattr(result, "answer", None))
        if not completion:
            return None
        prompt = self._prompt_of(task)
        ok, reason = self.guard.admit(
            f"{prompt}{_SEP}{completion}",
            embedding=getattr(task, "embedding", None),
            score=float(result.score),
            synthetic=True,
        )
        if not ok:
            logger.debug("[OS-5.36] collapse guard rejected row: %s", reason)
            return None
        row = SFTRow(prompt, completion, float(result.score), str(getattr(result, "reasoner", "")))
        self._corpus.append(row)
        self._persist(row)
        return row

    # ── from a scored candidate set (best-of-k) ──────────────────────
    def harvest_candidates(
        self, prompt: str, candidates: list[tuple[Any, float]]
    ) -> tuple[list[SFTRow], list[PreferencePair]]:
        """Rejection-sample a scored candidate set into SFT rows + preference pairs."""
        scored = sorted(
            ((_stringify(a), float(s)) for a, s in candidates),
            key=lambda t: t[1],
            reverse=True,
        )
        rows: list[SFTRow] = []
        pairs: list[PreferencePair] = []
        if not scored:
            return rows, pairs
        best_text, best_score = scored[0]
        if best_score >= self.min_score and best_text:
            ok, _ = self.guard.admit(
                f"{prompt}{_SEP}{best_text}", score=best_score, synthetic=True
            )
            if ok:
                row = SFTRow(prompt, best_text, best_score, "best_of_k")
                rows.append(row)
                self._corpus.append(row)
                self._persist(row)
        for losing_text, _ in scored[1:]:
            if losing_text and losing_text != best_text:
                pairs.append(PreferencePair(prompt, best_text, losing_text))
        return rows, pairs

    # ── corpus access + persistence ──────────────────────────────────
    def corpus(self) -> list[SFTRow]:
        return list(self._corpus)

    @staticmethod
    def _prompt_of(task: Any) -> str:
        payload = getattr(task, "payload", None)
        if isinstance(payload, dict) and payload.get("prompt"):
            return str(payload["prompt"])
        return str(getattr(task, "goal", task))

    def _persist(self, row: SFTRow) -> None:
        if self.engine is None:
            return
        import uuid

        try:
            self.engine.add_node(
                f"synthetic_corpus:{uuid.uuid4().hex[:12]}",
                "SyntheticCorpus",
                properties={
                    "prompt": row.prompt[:2000],
                    "completion": row.completion[:4000],
                    "score": row.score,
                    "source": row.source,
                    "synthetic": True,
                    "recorded_at": _now_iso(),
                },
            )
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.debug("[OS-5.36] could not persist corpus row: %s", exc)


def _now_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
