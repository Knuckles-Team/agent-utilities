from __future__ import annotations

"""External permission sync — source ACLs → KG-2.46 fine-grained permissioning.

CONCEPT:AU-ECO.connector.external-permission-sync — External Permission Sync

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
    roles += [str(role).strip() for role in access.read_roles if str(role).strip()]
    return list(dict.fromkeys(roles))


def sync_access(
    document_id: str,
    access: ExternalAccess | None,
    chunk_edges: list[tuple[str, str]] | None = None,
    *,
    data_owner: str = "",
    classification: DataClassification | None = None,
) -> NodeACL | None:
    """Mirror a source document's external access into the KG-2.46 model.

    CONCEPT:AU-ECO.connector.external-permission-sync.

    Args:
        document_id: The ``Document`` node id the access applies to.
        access: The connector-reported :class:`ExternalAccess`. ``None`` means no
            descriptor was supplied. Every supplied descriptor receives an
            explicit ACL: public records get a ``PUBLIC`` ACL, while non-public
            records get an ``INTERNAL`` ACL, including a deny-all ACL when they
            have no principals. Markings are applied + propagated.
        chunk_edges: ``(document_id, chunk_id)`` edges (the ``HAS_CHUNK`` set from
            ``DocumentProcessor``). The document's markings/classification
            propagate onto each chunk so chunk-level retrieval is governed too.
        data_owner: Optional data-owner principal recorded on the ACL.
        classification: Optional authoritative local classification. When
            omitted it is derived from ``access.is_public``. A public source
            assertion and local classification must agree.

    Returns:
        The registered :class:`NodeACL` when a descriptor was supplied, else
        ``None``. Markings are applied whenever a descriptor is present.
    """
    if access is None:
        return None

    acl: NodeACL | None = None
    roles = _read_roles(access)
    resolved_classification = classification or (
        DataClassification.PUBLIC if access.is_public else DataClassification.INTERNAL
    )
    if access.is_public != (resolved_classification == DataClassification.PUBLIC):
        raise ValueError("source access and local classification disagree")

    # Every governed object gets an explicit ACL. Public is a positive source
    # assertion, not an absent policy row; restricted with no principals is the
    # intentional deny-all case.
    acl = build_acl(
        document_id,
        classification=resolved_classification,
        read_roles=roles,
        data_owner=data_owner,
    )

    # Mandatory markings always apply (they cannot be relaxed by an ACL).
    for name in access.markings:
        if name:
            apply_marking(document_id, Marking(name))

    # Discretionary ACLs do not inherit read_roles through classification-only
    # propagation, so register the same source ACL on every materialized child.
    # This is especially important for quarantine: an empty role list is an
    # intentional deny-all decision, not an absent ACL.
    if chunk_edges:
        for _source_id, target_id in chunk_edges:
            build_acl(
                target_id,
                classification=acl.classification,
                read_roles=roles,
                data_owner=data_owner,
            )

    # Propagate the document's mandatory controls onto its chunks so chunk-level
    # retrieval honours the same access (KG-2.46 propagation; classification flows
    # to the strictest along the edge).
    if chunk_edges:
        propagate_over_edges(list(chunk_edges))

    logger.debug(
        "[ECO-4.28] synced access for %s: roles=%d markings=%d public=%s",
        document_id,
        len(roles),
        len(access.markings),
        access.is_public,
    )
    return acl
