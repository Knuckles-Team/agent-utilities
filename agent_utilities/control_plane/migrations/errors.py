"""Fail-closed errors for migration authority transitions."""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    "MigrationAuthorizationError",
    "MigrationCasConflictError",
    "MigrationConflictError",
    "MigrationDomainError",
    "MigrationGateError",
    "MigrationNotFoundError",
    "MigrationPrerequisiteError",
    "MigrationReplayError",
]


class MigrationDomainError(ValueError):
    """Stable, privacy-safe domain error with no source payload."""

    code: ClassVar[str] = "migration_domain_error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.code)


class MigrationAuthorizationError(MigrationDomainError):
    code = "migration_authorization_denied"


class MigrationCasConflictError(MigrationDomainError):
    code = "migration_cas_conflict"


class MigrationConflictError(MigrationDomainError):
    code = "migration_conflict"


class MigrationGateError(MigrationDomainError):
    code = "migration_gate_blocked"


class MigrationNotFoundError(MigrationDomainError):
    code = "migration_not_found"


class MigrationPrerequisiteError(MigrationDomainError):
    code = "migration_prerequisite_missing"


class MigrationReplayError(MigrationDomainError):
    code = "migration_replay_drift"
