#!/usr/bin/python
from __future__ import annotations

"""Production promotion-governance validator (CONCEPT:AU-AHE.harness.promotion-governance-validator).

:class:`~agent_utilities.knowledge_graph.research.auto_merge.GovernedAutoMerger`
always accepted an injected ``governance_validator`` — but until now only test
mocks were ever injected, so a "governed" auto-merge in production either held
every proposal (validator required but absent) or validated nothing. This
module is the real validator, and the merger now constructs it as its DEFAULT
whenever an engine is available.

A promotion candidate must clear four rules to be governance-valid:

1. **MergePolicy thresholds** — quality score at/above the policy threshold
   plus structural completeness (a named spec with a stated goal).
2. **SHACL governance shapes** — the spec, materialized as an RDF node in the
   ``kg#`` namespace, must conform to the bundled
   ``shapes/governance.shapes.ttl`` (via the existing
   :class:`~agent_utilities.knowledge_graph.core.shacl_validator.SHACLValidator`)
   where a shape targets its class; classes without a shape conform vacuously.
3. **Regression gate recorded** — when the failure analyzer's regression gate
   (AHE-3.18) has recorded a verdict for this proposal
   (``RegressionGateResult`` nodes), the latest one must be a ``pass``; a
   recorded ``hold`` blocks promotion. No record ⇒ the rule defers to the
   merger's live ``regression_check``.
4. **Constitution rules** — active ``forbid``-kind governance rules in the KG
   (``ConstitutionRule``/``Policy``/``governance_rule`` nodes, ingested from
   ``.specify/memory/constitution.md`` by the policy ingestor or recorded as
   human corrections) must not match the proposal text.

The validator is deliberately conservative where it *can* observe (a matching
forbid rule, a recorded gate hold, or a SHACL violation always blocks) and
non-blocking where governance data simply does not exist — the master switch
``KG_GOLDEN_AUTO_MERGE`` stays False by default regardless.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SHAPES = Path(__file__).parent.parent / "shapes" / "governance.shapes.ttl"

# Rule kinds that constitute a prohibition when matched against a proposal.
_FORBID_KINDS = {"forbid", "prohibit", "prohibition", "constraint"}


@dataclass
class GovernanceCheck:
    """One named rule outcome inside a :class:`PromotionVerdict`."""

    name: str
    passed: bool
    reason: str = ""


@dataclass
class PromotionVerdict:
    """The full per-rule verdict for one promotion candidate."""

    proposal_id: str
    checks: list[GovernanceCheck] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[str]:
        return [f"{c.name}: {c.reason}" for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "valid": self.valid,
            "checks": [
                {"name": c.name, "passed": c.passed, "reason": c.reason}
                for c in self.checks
            ],
        }


def _spec_text(spec: Any) -> str:
    """Lowercased searchable text of a proposal (name + goal + description)."""
    parts = []
    for attr in ("name", "goal", "description"):
        val = getattr(spec, attr, None)
        if val is None and isinstance(spec, dict):
            val = spec.get(attr)
        if val:
            parts.append(str(val))
    return " ".join(parts).lower()


def _spec_class(spec: Any) -> str:
    """Map a proposal to its kg# class name for SHACL targeting.

    Pydantic/dataclass specs map by class name with a ``Spec`` suffix stripped
    (``TeamSpec`` → ``Team``, ``AgentSpec`` → ``Agent``); dicts may carry an
    explicit ``type``; anything else is a generic ``Proposal`` (no shape
    targets it, so SHACL conforms vacuously).
    """
    if isinstance(spec, dict):
        return str(spec.get("type") or "Proposal")
    cls = type(spec).__name__
    return cls[:-4] if cls.endswith("Spec") and len(cls) > 4 else cls


class _OneNodeGraph:
    """Minimal ``graph.nodes(data=True)`` shim for ``build_data_graph``."""

    def __init__(self, node_id: str, data: dict[str, Any]):
        self._node_id = node_id
        self._data = data

    def nodes(self, data: bool = False):
        return [(self._node_id, self._data)] if data else [self._node_id]


class PromotionGovernanceValidator:
    """Validate golden-loop promotion candidates against live governance.

    CONCEPT:AU-AHE.harness.promotion-governance-validator — Promotion Governance Validator.

    Callable: ``validator(spec) -> bool`` (the ``governance_validator``
    contract of :class:`GovernedAutoMerger`); :meth:`validate` returns the
    full per-rule :class:`PromotionVerdict` for audit/debugging.

    Args:
        engine: KG engine (``query_cypher``) for recorded regression-gate
            verdicts and constitution rules. ``None`` ⇒ those rules are
            not applicable (structural + SHACL rules still run).
        policy: The :class:`MergePolicy` whose thresholds rule (c) enforces.
            Defaults to the env-resolved conservative policy.
        shapes_path: SHACL shapes file (defaults to the bundled
            ``governance.shapes.ttl``).
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        policy: Any = None,
        shapes_path: str | Path | None = None,
    ) -> None:
        from .auto_merge import MergePolicy

        self.engine = engine
        self.policy = policy or MergePolicy.from_env()
        self.shapes_path = str(shapes_path or _DEFAULT_SHAPES)

    def __call__(self, spec: Any) -> bool:
        verdict = self.validate(spec)
        if not verdict.valid:
            logger.info(
                "[AHE-3.20] promotion governance held %s: %s",
                verdict.proposal_id,
                "; ".join(verdict.failures),
            )
        return verdict.valid

    # ── rules ──────────────────────────────────────────────────────────
    def validate(self, spec: Any) -> PromotionVerdict:
        """Run every governance rule; the verdict is valid only if all pass."""
        from .auto_merge import GovernedAutoMerger

        verdict = PromotionVerdict(proposal_id=GovernedAutoMerger._spec_id(spec))
        verdict.checks.append(self._check_merge_policy(spec))
        verdict.checks.append(self._check_shacl(spec))
        verdict.checks.append(self._check_regression_gate(spec, verdict.proposal_id))
        verdict.checks.append(self._check_capability_ratchet(spec, verdict.proposal_id))
        verdict.checks.append(self._check_constitution(spec))
        return verdict

    def _check_merge_policy(self, spec: Any) -> GovernanceCheck:
        """(c) MergePolicy thresholds + structural completeness."""
        from .auto_merge import GovernedAutoMerger

        quality = GovernedAutoMerger.score_proposal(spec)
        if quality < self.policy.quality_threshold:
            return GovernanceCheck(
                "merge_policy",
                False,
                f"quality {quality:.2f} < threshold "
                f"{self.policy.quality_threshold:.2f}",
            )
        name = getattr(spec, "name", None) or (
            spec.get("name") if isinstance(spec, dict) else None
        )
        goal = getattr(spec, "goal", None) or (
            spec.get("goal") if isinstance(spec, dict) else None
        )
        if not name or not goal:
            return GovernanceCheck(
                "merge_policy", False, "proposal lacks a name and/or a stated goal"
            )
        return GovernanceCheck("merge_policy", True, f"quality {quality:.2f}")

    def _check_shacl(self, spec: Any) -> GovernanceCheck:
        """(a) SHACL governance shapes, where a shape targets the spec class."""
        try:
            from ..pipeline.phases.shacl_gate import SHACL_SUPPORT, build_data_graph

            if not SHACL_SUPPORT:
                return GovernanceCheck(
                    "shacl", True, "pyshacl/rdflib not installed — not applicable"
                )
            if not Path(self.shapes_path).exists():
                return GovernanceCheck(
                    "shacl", True, f"shapes file not found: {self.shapes_path}"
                )

            from ..core.shacl_validator import SHACLValidator

            data = spec.model_dump() if hasattr(spec, "model_dump") else None
            if data is None:
                data = dict(spec) if isinstance(spec, dict) else {}
                for attr in ("name", "goal", "description", "lead"):
                    val = getattr(spec, attr, None)
                    if val is not None:
                        data.setdefault(attr, val)
            data["type"] = _spec_class(spec)
            node_id = f"proposal_{abs(hash(_spec_text(spec))) % 10**8}"
            graph = build_data_graph(_OneNodeGraph(node_id, data))
            report = SHACLValidator().validate(graph, self.shapes_path)
            if report.get("conforms", False):
                return GovernanceCheck("shacl", True, "conforms")
            messages = "; ".join(
                str(v.get("message", "")) for v in report.get("violations", [])[:3]
            )
            return GovernanceCheck("shacl", False, messages or "shape violation")
        except Exception as exc:  # noqa: BLE001 — cannot prove conformance ⇒ hold
            return GovernanceCheck("shacl", False, f"validation error: {exc}")

    def _check_regression_gate(self, spec: Any, proposal_id: str) -> GovernanceCheck:
        """(b) the latest *recorded* regression-gate verdict must be a pass.

        The failure analyzer's gate (``make_regression_check``, AHE-3.18)
        records every verdict as a ``RegressionGateResult`` node. A recorded
        ``hold`` blocks promotion; no record defers to the merger's live
        ``regression_check`` (which still gates the merge independently).
        """
        if self.engine is None:
            return GovernanceCheck(
                "regression_gate", True, "no engine — no recorded gate to consult"
            )
        try:
            rows = self.engine.query_cypher(
                "MATCH (r:RegressionGateResult) WHERE r.proposal_id = $pid "
                "RETURN r.result AS result, r.timestamp AS timestamp",
                {"pid": proposal_id},
            )
        except Exception as exc:  # noqa: BLE001 — no gate store yet
            logger.debug("regression gate lookup failed: %s", exc)
            rows = []
        rows = [r for r in rows or [] if isinstance(r, dict) and r.get("result")]
        if not rows:
            return GovernanceCheck(
                "regression_gate",
                True,
                "no recorded verdict (live regression check still gates the merge)",
            )
        latest = max(rows, key=lambda r: str(r.get("timestamp") or ""))
        if str(latest["result"]).lower() == "pass":
            return GovernanceCheck("regression_gate", True, "recorded pass")
        return GovernanceCheck(
            "regression_gate", False, f"recorded {latest['result']} verdict"
        )

    def _check_capability_ratchet(self, spec: Any, proposal_id: str) -> GovernanceCheck:
        """(e) the latest *recorded* capability-ratchet verdict must not be a hold.

        The capability ratchet (``capability_ratchet.py``, AHE-3.24) records a
        ``CapabilityRatchetResult`` whenever it measures a published worktree. A
        recorded ``hold`` (a measured capability regression / ManifestVerifier
        revert, AHE-3.23) blocks re-promotion; no record defers to the live
        post-publish ratchet, which abandons the branch on regression anyway.
        """
        from .capability_ratchet import latest_ratchet_result

        if self.engine is None:
            return GovernanceCheck(
                "capability_ratchet", True, "no engine — no recorded verdict to consult"
            )
        result = latest_ratchet_result(self.engine, proposal_id)
        if result is None:
            return GovernanceCheck(
                "capability_ratchet",
                True,
                "no recorded verdict (live post-publish ratchet still gates the merge)",
            )
        if result == "pass":
            return GovernanceCheck("capability_ratchet", True, "recorded pass")
        return GovernanceCheck(
            "capability_ratchet", False, f"recorded {result} verdict"
        )

    def _check_constitution(self, spec: Any) -> GovernanceCheck:
        """(d) no active forbid-kind constitution/governance rule matches."""
        if self.engine is None:
            return GovernanceCheck(
                "constitution", True, "no engine — rules not applicable"
            )
        try:
            rows = self.engine.query_cypher(
                "MATCH (r) WHERE r:ConstitutionRule OR r:Policy "
                "OR r:governance_rule "
                "RETURN r.id AS id, r.kind AS kind, r.target AS target, "
                "r.description AS description, r.active AS active LIMIT 200"
            )
        except Exception as exc:  # noqa: BLE001 — unreadable rules ≠ a match
            logger.debug("constitution rule lookup failed: %s", exc)
            return GovernanceCheck(
                "constitution", True, "rules not queryable — not applicable"
            )

        text = _spec_text(spec)
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if row.get("active") is False:
                continue
            if str(row.get("kind") or "").lower() not in _FORBID_KINDS:
                continue
            target = str(row.get("target") or "").strip().lower()
            if target and target in text:
                return GovernanceCheck(
                    "constitution",
                    False,
                    f"forbid rule {row.get('id')} matches target {target!r}",
                )
        return GovernanceCheck("constitution", True, "no forbid rule matched")
