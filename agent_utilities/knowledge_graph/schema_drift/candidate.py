"""A schema-repair candidate: the new record contract, as typed data (EH-403).

The candidate is what an approval approves. EG owns every SHACL/RDF document
(DECISIONS 2026-09-24, AUD-27), so AU never renders one: the candidate is the
proposed record shape as a typed ``RecordContract`` (field -> required + JSON
type set), which EG renders into SHACL itself when it validates
(``GraphSchema.ValidateRepair``) or attaches (``GraphSchema.AttachApproved``) it
under ``approved:<source>``. :func:`approved_candidate_digest` is byte-for-byte
the digest EG recomputes from the same contract
(``eg_types::graph_schema::approval``) -- a hash over data, not over RDF.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .shape import JSON_TYPES, RecordShape

#: Key prefix EG reserves for approval-bound schema sources.
APPROVED_SOURCE_PREFIX = "approved:"
#: Domain separator EG frames the candidate digest with.
APPROVED_CANDIDATE_DOMAIN = "eg/approved-schema-candidate/v1"


def record_contract(shape: RecordShape) -> dict[str, Any]:
    """EG's ``RecordContract`` wire form of a record shape (types in EG's order)."""
    return {
        "fields": {
            name: {
                "required": spec.required,
                "types": [kind for kind in JSON_TYPES if kind in spec.types],
            }
            for name, spec in shape.fields
        }
    }


def canonical_contract_json(contract: Mapping[str, Any]) -> str:
    """The exact text EG's ``RecordContract::canonical_json`` produces."""
    return json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def approved_candidate_digest(source_id: str, contract: Mapping[str, Any]) -> str:
    """EG's candidate identity: domain, key and canonical contract, NUL-terminated."""
    parts = (APPROVED_CANDIDATE_DOMAIN, source_id, canonical_contract_json(contract))
    return hashlib.sha256("".join(f"{part}\0" for part in parts).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RepairCandidate:
    """The proposed contract and its identity."""

    source: str
    shape: RecordShape
    renames: Mapping[str, str] = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return f"{APPROVED_SOURCE_PREFIX}{self.source}"

    @property
    def contract(self) -> dict[str, Any]:
        return record_contract(self.shape)

    @property
    def digest(self) -> str:
        return approved_candidate_digest(self.source_id, self.contract)


def build_candidate(
    source: str, shape: RecordShape, renames: Mapping[str, str]
) -> RepairCandidate:
    return RepairCandidate(source, shape, dict(renames))


__all__ = [
    "APPROVED_CANDIDATE_DOMAIN",
    "APPROVED_SOURCE_PREFIX",
    "RepairCandidate",
    "approved_candidate_digest",
    "build_candidate",
    "canonical_contract_json",
    "record_contract",
]
