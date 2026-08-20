"""Persistence-independent repository seams for connector control state.

The protocol deliberately separates operator desired state from append-only
probe observations.  An engine/SQL/KG adapter can implement this interface,
but this domain module does not select or import a database.  Implementations
must apply tenant, principal and grant scope before filtering, ordering,
pagination or counting.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    AccessScope,
    AuthorizationDecision,
    AuthorizationSet,
    ConnectorIdentity,
    ConnectorListRequest,
    ConnectorPage,
    ConnectorVersion,
    DesiredConnectorState,
    InventoryReference,
    KeysetCursor,
    Observation,
    ServerIdentity,
)


class RepositoryUnavailable(RuntimeError):
    """The authoritative repository could not complete an operation."""


class RepositoryContractError(RuntimeError):
    """A repository returned data outside the typed privacy/scope contract."""


@runtime_checkable
class ConnectorRepository(Protocol):
    """Typed owner-independent repository interface.

    ``put_desired`` is the only desired-state mutation.  ``append_observation``
    is append-only and may not remove, disable or replace a desired record.
    Authorization decisions are stored independently by ``kind``.  A concrete
    adapter may use the native engine, relational catalog or another approved
    authority, but it must preserve these operation boundaries.
    """

    def put_connector(self, identity: ConnectorIdentity) -> None:
        """Create or idempotently retain one immutable connector identity."""

    def put_server(self, server: ServerIdentity) -> None:
        """Create or idempotently retain one stable server identity."""

    def put_inventory(self, inventory: InventoryReference) -> None:
        """Record one controlled package inventory reference."""

    def put_version(self, version: ConnectorVersion) -> None:
        """Record one immutable release; mutation of an existing version is forbidden."""

    def put_desired(self, desired: DesiredConnectorState) -> None:
        """Replace desired state only through an explicit versioned change."""

    def append_observation(self, observation: Observation) -> None:
        """Append one immutable probe result; never tombstone desired state."""

    def put_authorization(self, decision: AuthorizationDecision) -> None:
        """Record one approval/install/credential/enable decision independently."""

    def get_server(self, scope: AccessScope, server_id: str) -> ServerIdentity | None:
        """Read one server only inside the verified scope."""

    def get_version(
        self, scope: AccessScope, version_id: str
    ) -> ConnectorVersion | None:
        """Read one immutable version only inside the verified scope."""

    def get_desired(
        self,
        scope: AccessScope,
        server_id: str,
    ) -> DesiredConnectorState | None:
        """Read operator desired state; an absent observation does not erase it."""

    def get_latest_observation(
        self,
        scope: AccessScope,
        server_id: str,
    ) -> Observation | None:
        """Read the latest honest observation, including failure/empty outcomes."""

    def get_authorizations(
        self,
        scope: AccessScope,
        server_id: str,
        version_id: str,
    ) -> AuthorizationSet | None:
        """Read independently stored decisions for one exact scope."""

    def list_connectors(self, request: ConnectorListRequest) -> ConnectorPage:
        """Return a scope-filtered bounded keyset page.

        Scope filtering must occur before any sort, cursor application, count,
        or page construction.  A returned cursor must carry the same
        ``request.scope.scope_digest``.
        """

    def cursor_for(
        self,
        scope: AccessScope,
        after_server_id: str,
        after_version_id: str,
    ) -> KeysetCursor:
        """Create a scope-bound keyset position without exposing a database token."""
