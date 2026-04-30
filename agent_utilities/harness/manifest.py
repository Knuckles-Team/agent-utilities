"""Harness Change Manifest Models.

CONCEPT:AU-012 — Agentic Harness Engineering (Component & Decision Observability)

Provides Pydantic models for the AHE change manifest system. Each manifest
records edits to harness components with self-declared predictions that are
later verified against actual outcomes.

The manifest serves as the **causal boundary** in the hybrid model:
    - Epistemic State: Knowledge Graph (what the agent knows)
    - Normative State: Filesystem (what the agent is allowed to do)
    - **Causal Boundary: Change Manifest (what caused improvement)**

Manifests are stored in two locations:
    - ``.specify/manifests/<round_id>.json`` — git-diffable normative state
    - Knowledge Graph ``ChangeManifest`` nodes — epistemic queries
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ComponentType(StrEnum):
    """AHE harness component types.

    Each type corresponds to an independently editable, versionable
    file or set of files that can be rolled back cleanly via git.
    """

    SYSTEM_PROMPT = "system_prompt"
    TOOL_DESCRIPTION = "tool_description"
    TOOL_IMPLEMENTATION = "tool_implementation"
    MIDDLEWARE = "middleware"
    SKILL = "skill"
    SUB_AGENT = "sub_agent"
    LONG_TERM_MEMORY = "long_term_memory"


class ComponentEdit(BaseModel):
    """A single edit to a harness component.

    Each edit maps to exactly one git commit and contains falsifiable
    predictions about which tasks the edit will fix and which might regress.

    Attributes:
        component_type: The AHE component category being edited.
        file_path: Relative path to the edited file.
        edit_summary: Human-readable description of what was changed.
        predicted_fixes: Task IDs expected to be fixed by this edit.
        predicted_regressions: Task IDs at risk of regressing.
        evidence_references: IDs linking to EvidenceCorpus entries that
            motivated this edit.
        git_commit_sha: The git commit SHA after the edit is applied.
        timestamp: ISO 8601 timestamp of when the edit was recorded.
    """

    id: str = Field(default_factory=lambda: f"edit:{uuid.uuid4().hex[:8]}")
    component_type: ComponentType
    file_path: str
    edit_summary: str
    diff_content: str | None = None
    predicted_fixes: list[str] = Field(default_factory=list)
    predicted_regressions: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    git_commit_sha: str | None = None
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result of verifying a manifest's predictions against actual outcomes.

    Attributes:
        fix_precision: Of the predicted fixes, what fraction actually fixed?
        fix_recall: Of the actual fixes, what fraction were predicted?
        regression_precision: Of predicted regressions, what fraction occurred?
        unexpected_regressions: Task IDs that regressed but were NOT predicted.
        confirmed_fixes: Task IDs that were predicted to fix AND actually fixed.
        confirmed_regressions: Task IDs that regressed as predicted.
        false_positive_fixes: Predicted to fix but did NOT fix.
    """

    fix_precision: float = 0.0
    fix_recall: float = 0.0
    regression_precision: float = 0.0
    unexpected_regressions: list[str] = Field(default_factory=list)
    confirmed_fixes: list[str] = Field(default_factory=list)
    confirmed_regressions: list[str] = Field(default_factory=list)
    false_positive_fixes: list[str] = Field(default_factory=list)
    overall_delta: float = 0.0
    recommendation: str = ""  # "confirm", "partial_revert", "full_revert"


class ChangeManifest(BaseModel):
    """A versioned manifest of all edits in one evolution round.

    The manifest is the primary artifact for **decision observability**.
    It pairs every edit with a self-declared prediction, which is later
    verified by the ``ManifestVerifier`` to determine whether the edit
    actually improved performance.

    Lifecycle:
        1. Created by the Evolve Agent with ``verification_status='pending'``
        2. Edits applied → git commits recorded
        3. Evaluation benchmark run → new ``EvidenceCorpus`` generated
        4. ``ManifestVerifier`` compares predictions vs outcomes
        5. Status updated to ``confirmed`` or ``reverted``

    Attributes:
        round_id: Unique identifier for this evolution round.
        edits: List of component edits made in this round.
        baseline_score: Benchmark score before this round's edits.
        predicted_score: Expected benchmark score after edits.
        actual_score: Measured benchmark score (filled after verification).
        verification_status: Current lifecycle state.
        verification_result: Detailed verification outcome.
    """

    round_id: str = Field(default_factory=lambda: f"round:{uuid.uuid4().hex[:8]}")
    edits: list[ComponentEdit] = Field(default_factory=list)
    baseline_score: float | None = None
    predicted_score: float | None = None
    actual_score: float | None = None
    verification_status: str = "pending"  # pending | confirmed | reverted
    verification_result: VerificationResult | None = None
    parent_round_id: str | None = None
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_edit(self, edit: ComponentEdit) -> None:
        """Add a component edit to this manifest."""
        self.edits.append(edit)

    def get_all_predicted_fixes(self) -> list[str]:
        """Aggregate all predicted fix task IDs across edits."""
        fixes: list[str] = []
        for edit in self.edits:
            fixes.extend(edit.predicted_fixes)
        return list(set(fixes))

    def get_all_predicted_regressions(self) -> list[str]:
        """Aggregate all predicted regression task IDs across edits."""
        regressions: list[str] = []
        for edit in self.edits:
            regressions.extend(edit.predicted_regressions)
        return list(set(regressions))

    def get_edits_by_type(self, component_type: ComponentType) -> list[ComponentEdit]:
        """Filter edits by component type."""
        return [e for e in self.edits if e.component_type == component_type]

    def to_sdd_path(self, workspace_path: str) -> str:
        """Return the file path for SDD manifest storage."""
        import os

        manifests_dir = os.path.join(workspace_path, ".specify", "manifests")
        os.makedirs(manifests_dir, exist_ok=True)
        return os.path.join(manifests_dir, f"{self.round_id}.json")
