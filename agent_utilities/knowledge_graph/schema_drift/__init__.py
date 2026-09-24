"""Self-healing schema drift for connector ingestion (EH-402, EH-403).

"Self-healing" here means detect, contain, propose and verify automatically;
activation is always approved.

* :mod:`.shape` / :mod:`.classify` -- deterministic record-shape drift
  classification into six classes.
* :mod:`.policy` -- the operator-declared ``ContractEvolutionPolicy`` (only
  compatible classes may ever auto-continue).
* :mod:`.report` -- the typed ``SchemaDriftReport``, recorded as observation
  evidence.
* :mod:`.gate` -- the sync-side gate: quarantine without checkpoint advance,
  a Gap/WorkItem, and a repair proposal.
* :mod:`.repair` / :mod:`.candidate` -- Decide-mapped candidate contract,
  shadow-graph verification, approval queued on the fleet approvals lease.
* :mod:`.activation` -- an approved candidate attached through EG
  ``GraphSchema.AttachApproved`` (refused by EG without the approval).
"""

from __future__ import annotations

from .classify import COMPATIBLE_CLASSES, DriftChange, DriftClass, classify
from .gate import GateOutcome, GateRequest, commit_contract, run_gate
from .policy import ContractEvolutionPolicy, declared_policy, parse_policy
from .report import SchemaDriftReport, Verdict
from .shape import FieldShape, RecordShape, infer_shape, shape_digest

__all__ = [
    "COMPATIBLE_CLASSES",
    "ContractEvolutionPolicy",
    "DriftChange",
    "DriftClass",
    "FieldShape",
    "GateOutcome",
    "GateRequest",
    "RecordShape",
    "SchemaDriftReport",
    "Verdict",
    "classify",
    "commit_contract",
    "declared_policy",
    "infer_shape",
    "parse_policy",
    "run_gate",
    "shape_digest",
]
