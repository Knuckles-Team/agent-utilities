"""The approved record contract of each source, kept in the knowledge graph.

One ``SourceRecordContract`` node per source (``schema-contract:<source>``)
holds the approved :class:`~.shape.RecordShape`, its digest, and who approved
it: ``bootstrap`` (the first observation of a source, trust on first use),
``policy`` (a declared evolution policy absorbed a compatible drift), or the
id of the approval lease that activated a repair (EH-403).

The contract advances only together with the source checkpoint: it is written
after a delta was applied, never for a quarantined one. A store that cannot be
read or written fails the sync closed -- drift that cannot be measured is not
treated as no drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .shape import RecordShape, shape_digest

#: Node label of an approved contract.
CONTRACT_LABEL = "SourceRecordContract"
_READ = (
    "MATCH (n:SourceRecordContract {id: $id}) "
    "RETURN n.shape AS shape, n.approved_by AS approved_by"
)


class ContractStoreUnavailable(RuntimeError):
    """The approved contract could not be read or written."""


@dataclass(frozen=True, slots=True)
class ApprovedContract:
    """A source's approved record shape and who approved it."""

    source: str
    shape: RecordShape
    approved_by: str

    @property
    def digest(self) -> str:
        return shape_digest(self.shape)


def contract_id(source: str) -> str:
    return f"schema-contract:{source}"


def load_contract(engine: Any, source: str) -> ApprovedContract | None:
    """The approved contract, ``None`` before a source's first observation."""
    execute = getattr(getattr(engine, "backend", None), "execute", None)
    if not callable(execute):
        raise ContractStoreUnavailable("the engine exposes no contract read")
    try:
        rows = execute(_READ, {"id": contract_id(source)}) or []
    except Exception as exc:
        raise ContractStoreUnavailable(f"contract read failed: {exc}") from exc
    row = next((r for r in rows if isinstance(r, dict) and r.get("shape")), None)
    if row is None:
        return None
    try:
        shape = RecordShape.from_json(json.loads(str(row["shape"])))
    except (TypeError, ValueError) as exc:
        raise ContractStoreUnavailable(f"stored contract is unreadable: {exc}") from exc
    return ApprovedContract(source, shape, str(row.get("approved_by") or ""))


def save_contract(engine: Any, contract: ApprovedContract) -> None:
    """Write (replace) the approved contract of ``contract.source``."""
    properties = {
        "name": f"record contract of {contract.source}",
        "source": contract.source,
        "shape": json.dumps(contract.shape.to_json(), sort_keys=True),
        "shape_digest": contract.digest,
        "approved_by": contract.approved_by,
    }
    try:
        engine.add_node(
            contract_id(contract.source), CONTRACT_LABEL, properties=properties
        )
    except Exception as exc:
        raise ContractStoreUnavailable(f"contract write failed: {exc}") from exc


__all__ = [
    "CONTRACT_LABEL",
    "ApprovedContract",
    "ContractStoreUnavailable",
    "contract_id",
    "load_contract",
    "save_contract",
]
