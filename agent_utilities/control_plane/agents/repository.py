"""Persistence-independent repository seams for the agent control plane.

The implementation may be backed by the authoritative knowledge engine, an
approved relational read model, or another repository owned outside this
domain.  This package deliberately does not select storage or dispatch an
agent.  Implementations must keep release pointers CAS-atomic and enforce
scope before filtering or pagination.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    AccessScope,
    AgentIdentity,
    AgentKeysetCursor,
    AgentListRequest,
    AgentPage,
    AgentRegistration,
    AgentReleasePointer,
    AgentVersion,
    ApprovalRecord,
    ReleaseMutation,
    ReleaseTrack,
)


class RepositoryUnavailable(RuntimeError):
    """The authoritative agent repository could not complete an operation."""


class RepositoryContractError(RuntimeError):
    """An adapter returned data outside the typed identity/privacy contract."""


@runtime_checkable
class AgentRepository(Protocol):
    """Typed repository authority for identity, release, and approval state."""

    def put_registration(self, registration: AgentRegistration) -> None:
        """Atomically retain one immutable identity and version bundle."""

    def put_identity(self, identity: AgentIdentity) -> None:
        """Create or idempotently retain one immutable agent identity."""

    def put_version(self, version: AgentVersion) -> None:
        """Create or idempotently retain one immutable digest-pinned version."""

    def get_version(self, scope: AccessScope, version_id: str) -> AgentVersion | None:
        """Read one release only inside the verified visibility scope."""

    def get_release_pointer(
        self, scope: AccessScope, agent_id: str, channel: ReleaseTrack
    ) -> AgentReleasePointer | None:
        """Return the sole CAS pointer for an agent/channel key."""

    def get_release_pointer_for_version(
        self, scope: AccessScope, agent_id: str, version_id: str
    ) -> AgentReleasePointer | None:
        """Resolve one explicitly pinned version to its sole active pointer."""

    def compare_and_swap_release(
        self,
        scope: AccessScope,
        mutation: ReleaseMutation,
        pointer: AgentReleasePointer,
    ) -> AgentReleasePointer:
        """Atomically apply a revision/version fenced promotion or rollback."""

    def get_approval(
        self,
        scope: AccessScope,
        agent_id: str,
        version_id: str,
        approval_id: str,
    ) -> ApprovalRecord | None:
        """Read one exact approval; missing state must not authorize resolution."""

    def list_agents(self, request: AgentListRequest) -> AgentPage:
        """Return a bounded scope-filtered keyset page.

        Scope filtering must precede ordering, cursor application, counting,
        and page construction.  A returned cursor must carry the request scope
        digest, and duplicate agent/channel pointers are a contract error.
        """

    def cursor_for(
        self,
        scope: AccessScope,
        after_agent_id: str,
        after_version_id: str,
    ) -> AgentKeysetCursor:
        """Create a scope-bound keyset position without exposing storage tokens."""
