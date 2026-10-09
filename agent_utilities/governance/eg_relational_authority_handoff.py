"""AU-SEMANTIC-R019.1: typed delegation claim for relational-authority
governance moving to EG.

Split from AU-SEMANTIC-R019 ("Remaining EG-bound standardization, infra and
governance modules move out"): this slice ships the typed EG-delegation
claim and the refusal for an undelegated claim. Removing
``governance/relational_authority.py``'s local structural gate and routing
the check through EG land in a later slice.
"""

from __future__ import annotations

from dataclasses import dataclass


class RelationalAuthorityDelegationError(RuntimeError):
    """Raised when relational-authority governance is asserted without an
    explicit EG delegation claim.

    AU holds no local authority over the engine fleet catalog, usage store,
    or state store schemas; the check must be delegated to epistemic-graph.
    """


@dataclass(frozen=True)
class RelationalAuthorityDelegationClaim:
    """Typed claim that EG, not AU, performs a relational-authority
    governance check for the named domain."""

    domain: str
    delegated_to: str = "epistemic-graph"

    def require_delegated(self) -> None:
        """Refuse unless this claim delegates the check to epistemic-graph."""

        if self.delegated_to != "epistemic-graph":
            raise RelationalAuthorityDelegationError(
                "AU-SEMANTIC-R019: relational-authority governance for "
                f"domain {self.domain!r} has no local AU authority; it must "
                "be delegated to epistemic-graph, not "
                f"{self.delegated_to!r}."
            )
