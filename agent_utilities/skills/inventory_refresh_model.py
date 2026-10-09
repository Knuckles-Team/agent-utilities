"""Typed model for a skill-inventory refresh entry.

CONCEPT:AU-DEV.skills.inventory-refresh-determinism

AU-DEV-R003 requires that agent-utilities' source-to-installed skill
inventory refresh is idempotent after a provider skill moves or is renamed:
it must print its plan before mutating anything, then verify the resulting
hashes and IDs.

This is the ``.1`` slice for that row: a typed model of one inventory entry
that refuses a refresh which mutates before a plan was printed, or which
leaves the installed hash mismatched with the source after claiming to have
refreshed. Wiring this into the real Codex / XDG skill-path refresh command
is the remaining behavior (AU-DEV-R003.2+); see
``specs/au-developer-environment/tasks.md`` for the recorded split.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class SkillInventoryRefreshEntry(BaseModel):
    """One source-to-installed skill inventory entry after a refresh.

    Construction refuses a refresh result that mutated the installed copy
    without first printing a plan, or that reports completion while the
    installed hash still disagrees with the source hash, per AU-DEV-R003.
    """

    skill_id: str
    source_hash: str
    installed_hash: str
    plan_printed: bool
    mutated: bool

    @model_validator(mode="after")
    def _refuse_unplanned_or_unverified_refresh(self) -> SkillInventoryRefreshEntry:
        if self.mutated and not self.plan_printed:
            raise ValueError(
                "skill inventory refresh refuses to mutate "
                f"{self.skill_id!r} before printing its plan (AU-DEV-R003)"
            )
        if self.mutated and self.source_hash != self.installed_hash:
            raise ValueError(
                "skill inventory refresh refuses to report completion for "
                f"{self.skill_id!r} while installed_hash still disagrees "
                "with source_hash (AU-DEV-R003)"
            )
        return self
