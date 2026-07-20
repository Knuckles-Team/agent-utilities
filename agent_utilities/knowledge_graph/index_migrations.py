"""Current one-time persisted-state index migration registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


class IndexMigrationBackend(Protocol):
    """Authority operation required by the current index migration."""

    def hydrate_engine_embeddings(self, batch_log_every: int = 5000) -> int: ...


@dataclass(frozen=True, slots=True)
class IndexMigrationSpec:
    """Deterministic release description for one executable migration."""

    migrationId: str
    version: str
    mode: str
    executor: str
    fromState: str
    toState: str
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    rollback: str

    def as_release_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["preconditions"] = list(self.preconditions)
        value["postconditions"] = list(self.postconditions)
        return value


EMBEDDING_AUTHORITY_ANN = IndexMigrationSpec(
    migrationId="embedding-authority-ann-v1",
    version="1",
    mode="one-time-persisted-state",
    executor="epistemic-graph-authority.hydrate-engine-embeddings",
    fromState="node-property-embeddings",
    toState="authority-ann-index",
    preconditions=(
        "signed-pre-cutover-snapshot",
        "exact-release-engine-ready",
        "exclusive-migration-lease",
    ),
    postconditions=(
        "all-eligible-embeddings-indexed",
        "hydration-count-reconciled",
        "semantic-query-smoke-pass",
    ),
    rollback="restore-signed-pre-cutover-snapshot",
)

INDEX_MIGRATIONS: tuple[IndexMigrationSpec, ...] = (EMBEDDING_AUTHORITY_ANN,)


def index_migration_catalog() -> dict[str, object]:
    """Return the exact deterministic catalog consumed by release assembly."""

    return {
        "apiVersion": "graphos.io/v1",
        "kind": "IndexMigrationCatalog",
        "catalogVersion": 1,
        "migrationMode": "one-time-persisted-state",
        "entryCount": len(INDEX_MIGRATIONS),
        "entries": [migration.as_release_dict() for migration in INDEX_MIGRATIONS],
    }


def run_index_migration(
    migration_id: str,
    backend: IndexMigrationBackend,
    *,
    batch_log_every: int = 5000,
) -> int:
    """Execute the one current registered migration through the authority API."""

    if migration_id != EMBEDDING_AUTHORITY_ANN.migrationId:
        raise ValueError("index migration is not registered in the exact release catalog")
    if batch_log_every < 1:
        raise ValueError("index migration log interval must be positive")
    return int(backend.hydrate_engine_embeddings(batch_log_every=batch_log_every))
