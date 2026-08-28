from __future__ import annotations

"""AHE Manifest Verifier.

CONCEPT:AU-AHE.harness.harness-evolution — Agentic Harness Engineering (Decision Observability)

Verifies whether the Evolve Agent's predictions came true by comparing
a ChangeManifest's self-declared predictions against actual outcomes
from the next evaluation round.

This is distinct from the graph ``verifier_step`` which validates
*task output quality*. The ManifestVerifier validates *evolution
predictions* — did the edit actually fix what we predicted it would?

Verification calculates:
    - fix_precision: Of predicted fixes, what fraction actually fixed?
    - fix_recall: Of actual fixes, what fraction were predicted?
    - regression_precision: Of predicted regressions, what fraction occurred?
    - unexpected_regressions: Tasks that regressed but were NOT predicted

Auto-revert is triggered when:
    - Unexpected regressions are detected (not self-declared)
    - Overall score delta is negative
"""


import logging
from typing import Any

from .component_registry import HarnessComponentRegistry
from .evidence_corpus import EvidenceCorpus
from .manifest import ChangeManifest, VerificationResult

logger = logging.getLogger(__name__)


def _entry_haystack(entry: Any) -> str:
    """Flatten one evidence entry's content/tags/metadata into one searchable string."""
    return " ".join(
        [
            entry.content or "",
            " ".join(entry.tags or []),
            " ".join(f"{k}={v}" for k, v in (entry.metadata or {}).items()),
        ]
    )


class ManifestVerifier:
    """Verifies Evolve Agent predictions against actual outcomes.

    Args:
        registry: The harness component registry for rollback operations.
        knowledge_engine: Optional KG engine for recording verification
            results as edges (CAUSED_REGRESSION, CONFIRMED_FIX).
    """

    def __init__(
        self,
        registry: HarnessComponentRegistry,
        knowledge_engine: Any = None,
        reliability_multiple: float = 3.0,
    ) -> None:
        self.registry = registry
        self.knowledge_engine = knowledge_engine
        # Plan b7-04 F7: fix-precision must beat random by this factor to be "reliable".
        self.reliability_multiple = reliability_multiple

    @staticmethod
    def _signature_fired(
        signature: dict[str, Any], new_evidence: EvidenceCorpus
    ) -> bool:
        """Did the edit's declared trace feature actually appear (CONCEPT:AU-AHE.evaluation.edit-claims-fix)?

        The ``attribution_signature`` names a feature/value pair (e.g.
        ``{"tool_call": "WikiTextFetch", "min_count": 1}``). We scan the new
        evidence entries' content/tags/metadata for the named value and require it
        to appear at least ``min_count`` times. An empty signature trivially fires
        (back-compat for edits that declare none).
        """
        if not signature:
            return True
        min_count = int(signature.get("min_count", 1) or 1)
        # The named feature is any non-control value in the signature.
        needles = [
            str(v)
            for k, v in signature.items()
            if k != "min_count" and v not in (None, "")
        ]
        if not needles:
            return True
        count = sum(
            1
            for entry in new_evidence.entries
            if all(n in _entry_haystack(entry) for n in needles)
        )
        return count >= min_count

    @staticmethod
    def _actual_outcome_deltas(
        baseline_outcomes: dict[str, bool], new_outcomes: dict[str, bool]
    ) -> tuple[set[str], set[str]]:
        """Return ``(actual_fixes, actual_regressions)`` from before/after pass maps."""
        actual_fixes = {
            task_id
            for task_id, passed in new_outcomes.items()
            if passed and not baseline_outcomes.get(task_id, True)
        }
        actual_regressions = {
            task_id
            for task_id, passed in new_outcomes.items()
            if not passed and baseline_outcomes.get(task_id, False)
        }
        return actual_fixes, actual_regressions

    @staticmethod
    def _precision_recall(
        confirmed_fixes: set[str],
        predicted_fixes: set[str],
        actual_fixes: set[str],
        confirmed_regressions: set[str],
        predicted_regressions: set[str],
    ) -> tuple[float, float, float]:
        fix_precision = (
            len(confirmed_fixes) / len(predicted_fixes) if predicted_fixes else 0.0
        )
        fix_recall = len(confirmed_fixes) / len(actual_fixes) if actual_fixes else 0.0
        regression_precision = (
            len(confirmed_regressions) / len(predicted_regressions)
            if predicted_regressions
            else 0.0
        )
        return fix_precision, fix_recall, regression_precision

    def _attribution_reliability(
        self,
        fix_precision: float,
        actual_fixes: set[str],
        universe: int,
        predicted_fixes: set[str],
    ) -> tuple[float, float, bool]:
        """Would random predictions of the same scope do as well (plan b7-04 F7)?

        Random precision = base rate of actual fixes among all evaluated tasks;
        the harness is only "reliable" when its fix_precision beats that base
        rate by ``self.reliability_multiple``.
        """
        random_baseline_precision = len(actual_fixes) / universe if universe else 0.0
        attribution_lift = (
            fix_precision / random_baseline_precision
            if random_baseline_precision > 0.0
            else 0.0
        )
        attribution_reliable = bool(predicted_fixes) and (
            attribution_lift >= self.reliability_multiple
        )
        return random_baseline_precision, attribution_lift, attribution_reliable

    @staticmethod
    def _verification_recommendation(
        unexpected_regressions: set[str],
        overall_delta: float,
        unattributed_edits: list[str],
    ) -> str:
        if unexpected_regressions:
            return "partial_revert"
        if overall_delta < 0:
            return "full_revert"
        if unattributed_edits:
            # Apparent gain with no evidence the edit fired → do not confirm.
            return "partial_revert"
        return "confirm"

    async def verify(
        self,
        manifest: ChangeManifest,
        baseline_evidence: EvidenceCorpus,
        new_evidence: EvidenceCorpus,
    ) -> VerificationResult:
        """Compare predicted fixes/regressions with actual task deltas.

        Args:
            manifest: The ChangeManifest to verify.
            baseline_evidence: Evidence from before the edits.
            new_evidence: Evidence from after the edits.

        Returns:
            A VerificationResult with precision/recall metrics.
        """
        logger.info(
            f"ManifestVerifier: Verifying manifest {manifest.round_id} "
            f"({len(manifest.edits)} edits)"
        )

        # Build task-level outcome maps
        baseline_outcomes = {e.task_id: e.pass_fail for e in baseline_evidence.entries}
        new_outcomes = {e.task_id: e.pass_fail for e in new_evidence.entries}
        actual_fixes, actual_regressions = self._actual_outcome_deltas(
            baseline_outcomes, new_outcomes
        )

        # Compare with predictions
        predicted_fixes = set(manifest.get_all_predicted_fixes())
        predicted_regressions = set(manifest.get_all_predicted_regressions())

        confirmed_fixes = predicted_fixes & actual_fixes
        false_positive_fixes = predicted_fixes - actual_fixes
        unexpected_regressions = actual_regressions - predicted_regressions
        confirmed_regressions = predicted_regressions & actual_regressions

        # Calculate precision/recall
        fix_precision, fix_recall, regression_precision = self._precision_recall(
            confirmed_fixes,
            predicted_fixes,
            actual_fixes,
            confirmed_regressions,
            predicted_regressions,
        )

        # Self-attribution reliability (CONCEPT:AU-AHE.harness.harness-evolution, plan b7-04 F7).
        random_baseline_precision, attribution_lift, attribution_reliable = (
            self._attribution_reliability(
                fix_precision, actual_fixes, len(new_outcomes), predicted_fixes
            )
        )

        # Calculate overall score delta
        overall_delta = new_evidence.benchmark_score - baseline_evidence.benchmark_score

        # Signature attribution (CONCEPT:AU-AHE.evaluation.edit-claims-fix): an edit that declared an
        # ``attribution_signature`` must have actually FIRED in the new trace — else
        # any apparent fix is unattributed (coincidence / reward-hacking) and the
        # edit cannot be credited. HarnessX never checks this; we make every edit
        # falsifiable.
        unattributed_edits = [
            edit.id
            for edit in manifest.edits
            if edit.attribution_signature
            and not self._signature_fired(edit.attribution_signature, new_evidence)
        ]

        recommendation = self._verification_recommendation(
            unexpected_regressions, overall_delta, unattributed_edits
        )

        result = VerificationResult(
            unattributed_edits=sorted(unattributed_edits),
            fix_precision=fix_precision,
            fix_recall=fix_recall,
            regression_precision=regression_precision,
            unexpected_regressions=sorted(unexpected_regressions),
            confirmed_fixes=sorted(confirmed_fixes),
            confirmed_regressions=sorted(confirmed_regressions),
            false_positive_fixes=sorted(false_positive_fixes),
            overall_delta=overall_delta,
            recommendation=recommendation,
            random_baseline_precision=random_baseline_precision,
            attribution_lift=attribution_lift,
            attribution_reliable=attribution_reliable,
        )

        # Update the manifest
        manifest.actual_score = new_evidence.benchmark_score
        manifest.verification_result = result
        manifest.verification_status = (
            "confirmed" if recommendation == "confirm" else "needs_revert"
        )

        logger.info(
            f"ManifestVerifier: Verification complete. "
            f"Fix precision: {fix_precision:.2f}, "
            f"Regressions: {len(unexpected_regressions)} unexpected, "
            f"Delta: {overall_delta:+.2f}, "
            f"Recommendation: {recommendation}"
        )

        # Record in KG
        await self._record_verification_in_kg(manifest, result)

        return result

    def _revert_all_edits(self, manifest: ChangeManifest) -> list[str]:
        reverted_files: list[str] = []
        for edit in manifest.edits:
            if edit.git_commit_sha:
                success = self.registry.rollback_component(
                    edit.file_path, f"{edit.git_commit_sha}~1"
                )
                if success:
                    reverted_files.append(edit.file_path)
        return reverted_files

    def _revert_regressed_edits(
        self, manifest: ChangeManifest, unexpected_regressions: list[str]
    ) -> list[str]:
        """Partial revert: only revert edits whose predicted fixes regressed."""
        reverted_files: list[str] = []
        regression_set = set(unexpected_regressions)
        for edit in manifest.edits:
            predicted_set = set(edit.predicted_fixes)
            overlap = predicted_set & regression_set
            # If any predicted fix is actually a regression, revert this edit
            if not (overlap and edit.git_commit_sha):
                continue
            logger.warning(
                "ManifestVerifier: reverting component due to "
                "unexpected regressions count=%d",
                len(overlap),
            )
            success = self.registry.rollback_component(
                edit.file_path, f"{edit.git_commit_sha}~1"
            )
            if success:
                reverted_files.append(edit.file_path)
        return reverted_files

    async def auto_revert(
        self,
        manifest: ChangeManifest,
        verification: VerificationResult,
    ) -> list[str]:
        """Automatically revert component edits that caused regressions.

        Only reverts edits whose predicted_fixes overlap with the
        unexpected_regressions — i.e., edits that made things worse
        in an unpredicted way.

        Args:
            manifest: The manifest containing edits to potentially revert.
            verification: The verification result with regression data.

        Returns:
            List of file paths that were reverted.
        """
        reverted_files: list[str] = []

        if not verification.unexpected_regressions:
            logger.info(
                "ManifestVerifier: No unexpected regressions — nothing to revert."
            )
            return reverted_files

        if verification.recommendation == "full_revert":
            logger.warning(
                "ManifestVerifier: Full revert recommended. "
                "Reverting all edits in this manifest."
            )
            reverted_files = self._revert_all_edits(manifest)
        else:
            reverted_files = self._revert_regressed_edits(
                manifest, verification.unexpected_regressions
            )

        if reverted_files:
            manifest.verification_status = "reverted"
            logger.info(
                f"ManifestVerifier: Reverted {len(reverted_files)} files: "
                f"{reverted_files}"
            )

        return reverted_files

    async def _record_verification_in_kg(
        self,
        manifest: ChangeManifest,
        result: VerificationResult,
    ) -> None:
        """Record verification results in the Knowledge Graph.

        Creates edges:
            - CONFIRMED_FIX: edit → fixed task
            - CAUSED_REGRESSION: edit → regressed task
        """
        if not self.knowledge_engine:
            return

        try:
            # Record as a memory node for searchability
            self.knowledge_engine.add_memory(
                content=(
                    f"Verification of manifest {manifest.round_id}: "
                    f"fix_precision={result.fix_precision:.2f}, "
                    f"unexpected_regressions={result.unexpected_regressions}, "
                    f"recommendation={result.recommendation}"
                ),
                name=f"Verification {manifest.round_id}",
                category="ahe_verification",
                tags=["ahe", "verification", manifest.round_id, result.recommendation],
            )
        except Exception as e:
            logger.warning(f"ManifestVerifier: KG recording failed: {e}")
