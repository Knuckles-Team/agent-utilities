"""Activating an APPROVED schema repair (AU-SEC-R005).

A repair proposal becomes the source's contract only after a human approved
its ``action.approval`` lease (``active -> consumed``). :func:`activate_approved`
finds such a proposal, attaches its exact typed contract (EG renders the
SHACL) to the live graph through
EG ``GraphSchema.AttachApproved`` -- which re-reads the lease and refuses on
any mismatch, so an unapproved or altered candidate can never reach the
ontology from here or anywhere else -- then advances the approved contract to
the candidate's shape and closes the approval (``consumed -> expired``).

A refused approval (``revoked``) retires its proposal; a pending one is left
alone. The sync that drives this runs it before measuring drift, so the next
drain after an approval is measured against the repaired contract and the
held delta flows.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .candidate import APPROVED_SOURCE_PREFIX, approved_candidate_digest, record_contract
from .contract_store import ApprovedContract, ContractStoreUnavailable, save_contract
from .eg_port import SchemaRepairPort
from .repair import PROPOSAL_LABEL
from .shape import RecordShape

logger = logging.getLogger(__name__)

_PROPOSALS = (
    "MATCH (n:SchemaRepairProposal {source: $source}) "
    "RETURN n.id AS id, n.status AS status, n.approval_lease_id AS lease_id, "
    "n.candidate_digest AS digest, n.shape AS shape, n.contract AS contract"
)


@dataclass(frozen=True, slots=True)
class Activation:
    """One approved repair that is now the live contract."""

    proposal_id: str
    approval_lease_id: str
    candidate_digest: str
    closed: bool

    def summary(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "approval_lease_id": self.approval_lease_id,
            "candidate_digest": self.candidate_digest,
            "approval_closed": self.closed,
        }


def proposals(engine: Any, source: str) -> list[Mapping[str, Any]]:
    """Every recorded repair proposal of the source, whatever its status."""
    execute = getattr(getattr(engine, "backend", None), "execute", None)
    if not callable(execute):
        raise ContractStoreUnavailable("the engine exposes no proposal read")
    rows = execute(_PROPOSALS, {"source": source}) or []
    return [row for row in rows if isinstance(row, Mapping)]


def pending_proposals(engine: Any, source: str) -> list[Mapping[str, Any]]:
    """The source's proposals still waiting on their approval."""
    return [
        row
        for row in proposals(engine, source)
        if row.get("status") == "pending_approval" and row.get("lease_id")
    ]


def _mark(engine: Any, proposal_id: str, status: str) -> None:
    try:
        engine.add_node(proposal_id, PROPOSAL_LABEL, properties={"status": status})
    except Exception as exc:
        logger.warning(
            "schema repair proposal %s not marked %s: %s", proposal_id, status, exc
        )


def _stored_contract(row: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        contract = json.loads(str(row.get("contract") or ""))
    except ValueError:
        return None
    return contract if isinstance(contract, dict) else None


def _intact(source: str, row: Mapping[str, Any]) -> bool:
    """The stored contract still hashes to the digest that was approved."""
    contract = _stored_contract(row)
    if contract is None:
        return False
    try:
        stored_shape = json.loads(str(row.get("shape") or ""))
    except ValueError:
        return False
    if not isinstance(stored_shape, Mapping):
        return False
    # The AU projection must describe the exact contract the operator approved,
    # not independent mutable proposal data that could widen the next ingest.
    if record_contract(RecordShape.from_json(stored_shape)) != contract:
        return False
    source_id = f"{APPROVED_SOURCE_PREFIX}{source}"
    return approved_candidate_digest(source_id, contract) == row.get("digest")


def _close(port: SchemaRepairPort, tenant: str, lease: Mapping[str, Any]) -> bool:
    try:
        port.close_approval(
            tenant, str(lease["lease_id"]), int(lease.get("revision") or 0)
        )
    except Exception as exc:
        logger.warning(
            "approval %s not closed after activation: %s", lease.get("lease_id"), exc
        )
        return False
    return True


def _activate(
    engine: Any,
    port: SchemaRepairPort,
    tenant: str,
    source: str,
    row: Mapping[str, Any],
) -> Activation:
    lease_id = str(row["lease_id"])
    port.attach_approved(
        f"{APPROVED_SOURCE_PREFIX}{source}", _stored_contract(row) or {}, lease_id
    )
    shape = RecordShape.from_json(json.loads(str(row["shape"])))
    save_contract(engine, ApprovedContract(source, shape, lease_id))
    lease = port.get_lease(tenant, lease_id) or {"lease_id": lease_id}
    closed = _close(port, tenant, lease)
    _mark(engine, str(row["id"]), "activated")
    return Activation(str(row["id"]), lease_id, str(row["digest"]), closed)


def activate_approved(
    engine: Any, port: SchemaRepairPort, tenant: str, source: str
) -> Activation | None:
    """Activate the source's approved proposal, if one is approved now."""
    for row in pending_proposals(engine, source):
        lease = port.get_lease(tenant, str(row["lease_id"]))
        status = (lease or {}).get("status")
        if status in {"revoked", "expired"} or lease is None:
            _mark(engine, str(row["id"]), "refused")
        elif status == "consumed" and _intact(source, row):
            return _activate(engine, port, tenant, source, row)
    return None


__all__ = ["Activation", "activate_approved", "pending_proposals", "proposals"]
