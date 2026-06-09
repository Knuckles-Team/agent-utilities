from __future__ import annotations

"""External permission sync — source ACLs → KG-2.46 fine-grained permissioning.

CONCEPT:ECO-4.28 — External Permission Sync

Connectors report each document's :class:`ExternalAccess` (the source system's
groups/users/markings). This module maps that onto the *existing* KG-2.46
permissioning engine — it introduces **no new permission store**:

  * ``group_ids`` / ``user_emails`` → discretionary ``read_roles`` on a
    :class:`NodeACL` registered via :func:`ontology.permissioning.build_acl`.
  * public vs restricted → a :class:`DataClassification` (``PUBLIC`` /
    ``INTERNAL``).
  * each marking → a mandatory :class:`ontology.permissioning.Marking` applied via
    :func:`apply_marking`.
  * the document's ACL + markings then **propagate to its chunks** along the
    ``HAS_CHUNK`` edges (:func:`propagate_over_edges`), so a restricted document
    cannot be laundered through its chunk objects.

Because KG-2.46's :func:`enforce` is the default-on read gate, a document synced
here is automatically filtered at retrieval time for actors who lack the access —
the Onyx "external permission sync" feature, but enforced by an entailment-aware
graph gate rather than an index ACL field.
"""

import logging

from ...knowledge_graph.ontology.permissioning import (
    Marking,
    apply_marking,
    build_acl,
    propagate_over_edges,
)
from ...models.company_brain import DataClassification, NodeACL
from .base import ExternalAccess

logger = logging.getLogger(__name__)

__all__ = ["sync_access"]


def _read_roles(access: ExternalAccess) -> list[str]:
    """Map source principals to KG-2.46 ``read_roles`` tokens.

    Groups become ``group:<id>`` and users ``user:<email>`` so they namespace
    cleanly against marking roles (``marking:<name>``) on an actor.
    """
    roles: list[str] = [f"group:{g}" for g in access.group_ids if g]
    roles += [f"user:{e}" for e in access.user_emails if e]
    return roles


def sync_access(
    document_id: str,
    access: ExternalAccess | None,
    chunk_edges: list[tuple[str, str]] | None = None,
    *,
    data_owner: str = "",
) -> NodeACL | None:
    """Mirror a source document's external access into the KG-2.46 model.

    CONCEPT:ECO-4.28.

    Args:
        document_id: The ``Document`` node id the access applies to.
        access: The connector-reported :class:`ExternalAccess`. ``None`` or a
            ``is_public`` descriptor means "no restriction" → no ACL is written
            (the default-allow gate passes it through), but any markings are still
            applied + propagated.
        chunk_edges: ``(document_id, chunk_id)`` edges (the ``HAS_CHUNK`` set from
            ``DocumentProcessor``). The document's markings/classification
            propagate onto each chunk so chunk-level retrieval is governed too.
        data_owner: Optional data-owner principal recorded on the ACL.

    Returns:
        The registered :class:`NodeACL` when a restriction was applied, else
        ``None`` (public / no principals). Markings are applied regardless.
    """
    if access is None:
        return None

    acl: NodeACL | None = None
    roles = _read_roles(access)

    # Only register a discretionary ACL when the source actually restricts the
    # document. A public doc with no principals stays open (default-allow gate).
    if not access.is_public and roles:
        acl = build_acl(
            document_id,
            classification=DataClassification.INTERNAL,
            read_roles=roles,
            data_owner=data_owner,
        )

    # Mandatory markings always apply (they cannot be relaxed by an ACL).
    for name in access.markings:
        if name:
            apply_marking(document_id, Marking(name))

    # Propagate the document's mandatory controls onto its chunks so chunk-level
    # retrieval honours the same access (KG-2.46 propagation; classification flows
    # to the strictest along the edge).
    if chunk_edges and (acl is not None or access.markings):
        propagate_over_edges(list(chunk_edges))

    logger.debug(
        "[ECO-4.28] synced access for %s: roles=%d markings=%d public=%s",
        document_id,
        len(roles),
        len(access.markings),
        access.is_public,
    )
    return acl
