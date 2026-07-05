#!/usr/bin/python
from __future__ import annotations

"""Human-correction → rule/outcome/eval feedback loop (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

The compounding layer the "Company Brain" was missing: a single entry point where
a human says "this was wrong, here's the fix" and the correction becomes
persistent future behaviour. Three correction types:

* ``outcome`` — adjust the reward EMA of a designated entity so routing/retrieval
  prefers/avoids it next time (reuses :meth:`CapabilityIndex.record_outcome`).
* ``rule`` — persist a ``Correction`` node (+ ``CORRECTS`` edge) and a durable,
  active governance/voice/source rule. Because a *human* asserted it, the rule is
  authoritative immediately — no synthesis threshold needed. It is consumed at
  retrieval time by :func:`apply_governance_rules`.
* ``eval`` — append a regression case to the eval corpus so the mistake is caught
  automatically from then on.

Dependencies are injected so the service is unit-testable without a live engine;
:meth:`from_engine` wires it from a running :class:`IntelligenceGraphEngine`.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_VALID = {
    "outcome",
    "rule",
    "eval",
    "reads_avoided",
    "action_outcome",
    "gotcha",
    "selective_erasure",
}

# Map a free-form rule scope to the persisted node type consumed by
# governance_rules.load_active_rules.
_RULE_TYPE = {
    "voice": "voice_rule",
    "source": "source_rule",
    "governance": "governance_rule",
    "preference": "preference",
}


@dataclass
class CorrectionResult:
    correction_type: str
    target_id: str
    applied: bool
    detail: str
    created_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "correction_type": self.correction_type,
            "target_id": self.target_id,
            "applied": self.applied,
            "detail": self.detail,
            "created_ids": self.created_ids,
        }


class FeedbackService:
    """Turn human corrections into durable behaviour change.

    Args:
        backend: anything exposing ``add_node``/``add_edge`` (graph writer).
        capability_index: optional :class:`CapabilityIndex` for outcome rewards.
        eval_corpus: optional :class:`EvalCorpus` for eval corrections.
    """

    def __init__(
        self,
        backend: Any = None,
        capability_index: Any = None,
        eval_corpus: Any = None,
    ) -> None:
        self.backend = backend
        self.capability_index = capability_index
        self.eval_corpus = eval_corpus

    @classmethod
    def from_engine(cls, engine: Any) -> FeedbackService:
        backend = getattr(engine, "backend", None) or getattr(engine, "store", None)
        kg = getattr(engine, "knowledge_graph", None) or getattr(engine, "kg", None)
        cap = getattr(kg, "retrieval", None) if kg is not None else None
        corpus = None
        try:
            from ...harness.eval_corpus import EvalCorpus

            corpus = EvalCorpus(backend)
        except Exception:  # pragma: no cover - corpus optional
            corpus = None
        return cls(backend=backend, capability_index=cap, eval_corpus=corpus)

    # ------------------------------------------------------------------
    def record_correction(
        self,
        correction_type: str,
        target_id: str,
        corrected_value: Any = None,
        reason: str = "",
        *,
        actor_id: str = "human",
        rule_scope: str = "governance",
        rule_kind: str = "forbid",
        reward: float | None = None,
    ) -> CorrectionResult:
        """Record a human correction and apply it durably."""
        ctype = correction_type.strip().lower()
        if ctype not in _VALID:
            return CorrectionResult(
                ctype, target_id, False, f"unknown correction_type {correction_type!r}"
            )
        if ctype == "reads_avoided":
            return self.record_reads_avoided(
                target_id, corrected_value=corrected_value, reason=reason
            )
        if ctype == "action_outcome":
            return self.record_action_outcome(
                target_id, corrected_value=corrected_value, reason=reason
            )
        if ctype == "gotcha":
            return self.record_gotcha(
                target_id, str(corrected_value or reason or ""), actor_id=actor_id
            )
        if ctype == "selective_erasure":
            return self._apply_selective_erasure(target_id, corrected_value, reason)
        if ctype == "outcome":
            return self._apply_outcome(target_id, reward, corrected_value, reason)
        if ctype == "rule":
            return self._apply_rule(
                target_id, corrected_value, reason, actor_id, rule_scope, rule_kind
            )
        return self._apply_eval(target_id, corrected_value, reason)

    # ------------------------------------------------------------------
    def export_preference_pairs(self, *, min_margin: float = 0.1) -> list[Any]:
        """Consolidate eval corpus + distilled episodes + corrections into a
        reliability-filtered, DPO-ready preference-pair corpus (CONCEPT:AU-AHE.harness.preference-corpus-reliability).

        This is the read-side of the feedback loop: every correction/eval recorded
        through this service flows back out as clean (chosen ≻ rejected) pairs, with
        RAPPO ambiguous-pair filtering applied. Layers TI-DPO token weights / InSPO
        reflection are opt-in on the returned pairs.
        """
        from agent_utilities.harness.preference_pairs import (
            PreferencePairExporter,
            reliability_filter,
        )

        exporter = PreferencePairExporter(backend=self.backend)
        kept, dropped = reliability_filter(exporter.export(), min_margin=min_margin)
        if dropped:
            logger.info(
                "[AHE-3.17] preference export: kept=%d dropped=%d (ambiguous/degenerate)",
                len(kept),
                dropped,
            )
        return kept

    # ------------------------------------------------------------------
    def _apply_outcome(self, target_id, reward, corrected_value, reason):
        if self.capability_index is None or not hasattr(
            self.capability_index, "record_outcome"
        ):
            return CorrectionResult(
                "outcome", target_id, False, "no capability_index available"
            )
        r = reward
        if r is None and corrected_value is not None:
            try:
                r = float(corrected_value)
            except (TypeError, ValueError):
                r = None
        if r is None:
            return CorrectionResult(
                "outcome",
                target_id,
                False,
                "outcome correction needs reward/corrected_value",
            )
        new = self.capability_index.record_outcome(target_id, reward=r)
        return CorrectionResult(
            "outcome", target_id, True, f"reward updated to {new:.3f} ({reason})"
        )

    # ------------------------------------------------------------------
    def _apply_selective_erasure(
        self, target_id: str, corrected_value: Any, reason: str
    ) -> CorrectionResult:
        """Provenance-scoped reward erasure (CONCEPT:AU-KG.memory.generation-scoped-selective-reward).

        Forget the learned reward EMA for one or more superseded designations,
        so the retrieval router re-learns them from the neutral prior instead of
        carrying utility scored under a now-displaced source/impl/model regime.
        This is the Red Queen Gödel Machine's *selective erasure*
        (arXiv:2606.26294) on the memory router's utility records.

        ``target_id`` is the primary id; ``corrected_value`` may carry extra ids
        (a JSON list, or a comma/whitespace-separated string) so a whole
        superseded generation is forgotten in one call.
        """
        index = self.capability_index
        if index is None or not hasattr(index, "selective_erase_rewards"):
            return CorrectionResult(
                "selective_erasure", target_id, False, "no capability_index available"
            )
        ids: list[str] = [target_id] if target_id else []
        if corrected_value is not None:
            payload = corrected_value
            if isinstance(payload, str):
                text = payload.strip()
                if text.startswith("["):
                    try:
                        import json as _json

                        payload = _json.loads(text)
                    except Exception:
                        payload = text.replace(",", " ").split()
                else:
                    payload = text.replace(",", " ").split()
            if isinstance(payload, list | tuple | set):
                ids.extend(str(p) for p in payload)
        erased = index.selective_erase_rewards(ids)
        return CorrectionResult(
            "selective_erasure",
            target_id,
            erased > 0,
            f"erased {erased} reward record(s) ({reason})"
            if reason
            else f"erased {erased} reward record(s)",
        )

    # ------------------------------------------------------------------
    def record_reads_avoided(
        self,
        capability_id: str,
        *,
        reads_avoided: bool = True,
        files_read: int = 0,
        correct: bool = True,
        query: str = "",
        corrected_value: Any = None,
        reason: str = "",
    ) -> CorrectionResult:
        """Close the reads-avoided measurement loop (CONCEPT:AU-AHE.evaluation.reads-avoided-feedback).

        When a ``code_context`` answer is served the agent reports back whether the
        KG answer **replaced a file read** (``reads_avoided``), how many files it
        had to read anyway (``files_read``), and whether the answer was ``correct``.
        That triple becomes a reward on the answer's ``capability_id`` reward-EMA
        (so the code-context retriever GEPA-optimizes toward answers that replace a
        read) *and*, when the agent supplies the right answer, an eval-corpus case
        so the same question is graded automatically thereafter.

        ``corrected_value`` may carry a JSON/dict ``{reads_avoided, files_read,
        correct, query}`` (the on-the-wire form from ``graph_feedback``).
        """
        ra, fr, ok, q = reads_avoided, files_read, correct, query
        if corrected_value is not None:
            payload = corrected_value
            if isinstance(payload, str):
                try:
                    import json as _json

                    payload = _json.loads(payload)
                except Exception:
                    payload = {}
            if isinstance(payload, dict):
                ra = bool(payload.get("reads_avoided", ra))
                fr = int(payload.get("files_read", fr) or 0)
                ok = bool(payload.get("correct", ok))
                q = str(payload.get("query", q) or q)

        if not ok:
            reward = 0.0
        elif ra and fr <= 0:
            reward = 1.0  # answer fully replaced the read
        elif ra:
            reward = 0.7  # helped, but some files still read
        else:
            reward = 0.3  # read anyway despite the answer

        outcome = self._apply_outcome(
            capability_id, reward, None, reason or "reads_avoided"
        )
        created = list(outcome.created_ids)
        # Persist the graded case so the answer is regression-checked from now on.
        if (
            ok
            and q
            and self.eval_corpus is not None
            and hasattr(self.eval_corpus, "add_case")
        ):
            try:
                case_id = self.eval_corpus.add_case(
                    query=q,
                    expected_output=capability_id,
                    tags=["code_context", "reads_avoided"],
                    reason=reason or "code_context answer replaced a read",
                )
                created.append(case_id)
            except Exception as exc:  # pragma: no cover - corpus optional
                logger.debug("reads_avoided eval case failed: %s", exc)
        return CorrectionResult(
            "reads_avoided",
            capability_id,
            outcome.applied,
            f"reward={reward:.2f} reads_avoided={ra} files_read={fr} correct={ok}",
            created,
        )

    # ------------------------------------------------------------------
    def record_action_outcome(
        self,
        action_id: str,
        *,
        success: bool = True,
        reward: float | None = None,
        expected: str = "",
        observed: str = "",
        query: str = "",
        corrected_value: Any = None,
        reason: str = "",
        agent_id: str = "",
    ) -> CorrectionResult:
        """Close the loop on ANY autonomous action (CONCEPT:AU-AHE.evaluation.action-outcome-feedback).

        The general form of :meth:`record_reads_avoided`: an executed action — a
        ``code_context`` answer, a deploy, a ticket close, a routing choice — reports
        ``{success, optional reward, expected vs observed}``. That becomes a reward on
        the action's ``action_id`` reward-EMA (so routing / retrieval / playbooks
        prefer actions that *achieve their goal*) and, when ``expected`` + ``query``
        are given, an eval-corpus case so the outcome is regression-checked from then
        on. This is the substrate the autonomy ramp and goal-SLA loops build on:
        outcomes, measured, compounding — the missing back-half of the operating loop.

        ``corrected_value`` may carry JSON/dict ``{success, reward, expected, observed,
        query}`` (the on-the-wire form from ``graph_feedback``).
        """
        s, r, exp, obs, q = success, reward, expected, observed, query
        if corrected_value is not None:
            payload = corrected_value
            if isinstance(payload, str):
                try:
                    import json as _json

                    payload = _json.loads(payload)
                except Exception:
                    payload = {}
            if isinstance(payload, dict):
                s = bool(payload.get("success", s))
                if payload.get("reward") is not None:
                    try:
                        r = float(payload["reward"])
                    except (TypeError, ValueError):
                        r = None
                exp = str(payload.get("expected", exp) or exp)
                obs = str(payload.get("observed", obs) or obs)
                q = str(payload.get("query", q) or q)

        if r is None:
            r = 1.0 if s else 0.0
        r = max(0.0, min(1.0, r))
        outcome = self._apply_outcome(action_id, r, None, reason or "action_outcome")
        created = list(outcome.created_ids)
        # CONCEPT:AU-ORCH.routing.route-outcome-feedback — a model-route outcome also trains the adaptive router's
        # per-role confidence so the cheapest model that keeps working wins next time.
        if action_id.startswith("model_route:"):
            try:
                from agent_utilities.core.model_router import record_model_outcome

                record_model_outcome(action_id, reward=r)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("model-route outcome update failed: %s", exc)
        # CONCEPT:AU-OS.governance.autonomy-change-proposer — a "trust:<actor>:<kind>" outcome trains the autonomy ramp
        # so a consistently-correct actor earns wider governance scope for that kind.
        elif action_id.startswith("trust:"):
            try:
                from agent_utilities.orchestration.autonomy_ramp import record_trust

                _, actor, kind = (action_id.split(":", 2) + ["", ""])[:3]
                record_trust(self.backend, actor, kind, success=s)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("trust outcome update failed: %s", exc)
        # CONCEPT:AU-AHE.org.role-experience — a "role_experience:<role_id>" outcome
        # accrues into the owning :Employee's experience profile (successes/
        # partials/failures + score + seniority), so the org recruiter reuses
        # proven staff on the next synthesis (the Self-Grown loop). The optional
        # corrected_value carries {employee_id, domains}.
        elif action_id.startswith("role_experience:"):
            try:
                from agent_utilities.orchestration.org_runtime import (
                    record_role_experience,
                )

                role_id = action_id.split(":", 1)[1]
                emp_id = ""
                domains: list[str] = []
                payload = corrected_value
                if isinstance(payload, str):
                    try:
                        import json as _json

                        payload = _json.loads(payload)
                    except Exception:
                        payload = {}
                if isinstance(payload, dict):
                    emp_id = str(payload.get("employee_id", "") or "")
                    doms = payload.get("domains") or []
                    if isinstance(doms, list):
                        domains = [str(d) for d in doms]
                record_role_experience(
                    self.backend, role_id, employee_id=emp_id,
                    success=s, reward=r, domains=domains,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("role experience outcome update failed: %s", exc)
        if (
            q
            and exp
            and self.eval_corpus is not None
            and hasattr(self.eval_corpus, "add_case")
        ):
            try:
                # CONCEPT:AU-AHE.harness.when-outcome-names-agent — when the outcome names the agent that produced it,
                # tag the eval case ``agent:<id>`` so the per-agent trainset
                # (build_agent_trainset) can pool THIS agent's real metrics for its own
                # DSPy optimization (attribution by agent, not just trace signature).
                tags = ["action_outcome"]
                if agent_id:
                    tags.append(f"agent:{agent_id}")
                case_id = self.eval_corpus.add_case(
                    query=q,
                    expected_output=exp,
                    tags=tags,
                    reason=reason or "action outcome",
                    metadata={"agent_id": agent_id} if agent_id else None,
                )
                created.append(case_id)
            except Exception as exc:  # pragma: no cover - corpus optional
                logger.debug("action_outcome eval case failed: %s", exc)
        return CorrectionResult(
            "action_outcome",
            action_id,
            outcome.applied,
            f"reward={r:.2f} success={s}" + (f" observed={obs[:40]}" if obs else ""),
            created,
        )

    # ------------------------------------------------------------------
    def agent_eval_cases(self, agent_id: str, *, limit: int = 500) -> list[Any]:
        """The eval-corpus slice attributed to one agent (CONCEPT:AU-AHE.harness.when-outcome-names-agent).

        The per-agent attribution the hardening loop optimizes against: every case the
        agent's own ``record_action_outcome`` calls tagged ``agent:<id>``. These ARE the
        agent's measured executions (expected vs the goal that was reached), so they double
        as the training signal and the held-out scoring slice for its prompt.
        """
        if self.eval_corpus is None or not hasattr(self.eval_corpus, "load_cases"):
            return []
        tag = f"agent:{agent_id}"
        out: list[Any] = []
        try:
            for case in self.eval_corpus.load_cases():
                if tag in (getattr(case, "tags", []) or []):
                    out.append(case)
                    if len(out) >= limit:
                        break
        except Exception as exc:  # pragma: no cover - corpus optional
            logger.debug("agent_eval_cases failed: %s", exc)
        return out

    def build_agent_trainset(self, agent_id: str, *, limit: int = 500) -> list[Any]:
        """Pool an agent's outcomes into a DSPy trainset (CONCEPT:AU-AHE.harness.when-outcome-names-agent).

        Turns :meth:`agent_eval_cases` into ``dspy.Example(context, task) -> response``
        rows (``task`` = the query, ``response`` = the outcome that was reached), so the
        DSPy optimizer for THIS agent is steered by ITS real execution metrics. Degrades to
        plain dicts when DSPy is not importable, so the caller (build_hardened_prompt) works
        offline.
        """
        cases = self.agent_eval_cases(agent_id, limit=limit)
        try:
            import dspy

            return [
                dspy.Example(
                    context="",
                    task=getattr(c, "query", "") or "",
                    response=getattr(c, "expected_output", "") or "",
                ).with_inputs("context", "task")
                for c in cases
                if getattr(c, "expected_output", "")
            ]
        except ImportError:
            return [
                {
                    "context": "",
                    "task": getattr(c, "query", "") or "",
                    "response": getattr(c, "expected_output", "") or "",
                }
                for c in cases
                if getattr(c, "expected_output", "")
            ]

    # ------------------------------------------------------------------
    def record_gotcha(
        self,
        target_id: str,
        note: str,
        *,
        severity: str = "warn",
        actor_id: str = "human",
    ) -> CorrectionResult:
        """Pin a hard-won gotcha to a file/module so it's inherited, not relearned.

        CONCEPT:AU-KG.ingest.gotcha-feedback-capture — a ``:Gotcha`` node keyed by a (normalized) path + note,
        surfaced by ``code_context`` when an agent touches that area. The dogfood
        fix: traps like "gen scripts import the canonical copy, not the worktree" or
        "``_get_engine()`` hangs in a one-off host process" live IN the KG attached
        to the code, surfaced on touch — instead of being rediscovered every session.
        """
        if self.backend is None or not hasattr(self.backend, "add_node"):
            return CorrectionResult("gotcha", target_id, False, "no backend to persist")
        if not note.strip():
            return CorrectionResult("gotcha", target_id, False, "empty gotcha note")
        from agent_utilities.core.source_paths import normalize_path

        path = normalize_path(target_id) or target_id
        gid = f"gotcha:{uuid.uuid5(uuid.NAMESPACE_URL, path + '|' + note).hex[:12]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.backend.add_node(
            gid,
            type="Gotcha",
            path=path,
            note=note.strip(),
            severity=severity,
            actor_id=actor_id,
            timestamp=ts,
        )
        return CorrectionResult(
            "gotcha", target_id, True, f"pinned gotcha to {path}", [gid]
        )

    def _apply_rule(
        self, target_id, corrected_value, reason, actor_id, rule_scope, rule_kind
    ):
        if self.backend is None or not hasattr(self.backend, "add_node"):
            return CorrectionResult(
                "rule", target_id, False, "no backend to persist rule"
            )
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        corr_id = f"correction:{uuid.uuid4().hex[:12]}"
        self.backend.add_node(
            corr_id,
            type="correction",
            target=target_id,
            reason=reason,
            corrected_value=str(corrected_value or ""),
            actor_id=actor_id,
            assertion_type="human_judgment",
            timestamp=ts,
        )
        created = [corr_id]
        if target_id and hasattr(self.backend, "add_edge"):
            try:
                self.backend.add_edge(corr_id, target_id, rel_type="corrects")
            except Exception as exc:  # pragma: no cover
                logger.debug("corrects edge failed: %s", exc)
        rule_id = f"rule:{uuid.uuid4().hex[:12]}"
        rule_type = _RULE_TYPE.get(rule_scope, "governance_rule")
        self.backend.add_node(
            rule_id,
            type=rule_type,
            kind=rule_kind,
            target=target_id,
            weight=0.5,
            reason=reason,
            active=True,
            assertion_type="human_judgment",
            source_correction=corr_id,
            timestamp=ts,
        )
        created.append(rule_id)
        return CorrectionResult(
            "rule",
            target_id,
            True,
            f"persisted {rule_type} ({rule_kind}) for {target_id}",
            created,
        )

    def _apply_eval(self, target_id, corrected_value, reason):
        if self.eval_corpus is None or not hasattr(self.eval_corpus, "add_case"):
            return CorrectionResult(
                "eval", target_id, False, "no eval corpus available"
            )
        case_id = self.eval_corpus.add_case(
            query=target_id,
            expected_output=str(corrected_value or ""),
            tags=["from_correction"],
            reason=reason,
        )
        return CorrectionResult(
            "eval", target_id, True, f"added eval case {case_id}", [case_id]
        )
