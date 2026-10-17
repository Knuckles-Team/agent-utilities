"""In-run harness mechanisms: edits, tool contracts, constraints, pre-emit gates.

Each mechanism applies a rule or a limit it is given inside one run. Thresholds
and registries come from the caller (the host passes epistemic-graph decisions);
the defaults here are fallbacks only.
"""

from .constraint_engine import (
    ConstraintEngine,
    ConstraintLevel,
    ConstraintViolation,
    HierarchicalConstraint,
)
from .contract_validator import ContractValidator, ToolContract
from .edit_engine import (
    Edit,
    EditOutcome,
    EditResult,
    apply_edits,
    apply_with_reflection,
    parse_edits,
)
from .quality_gates import (
    Antipattern,
    AntipatternRegistry,
    MultiDimensionalCritique,
    PreEmitGate,
    PreflightGate,
)

__all__ = [
    "Antipattern",
    "AntipatternRegistry",
    "ConstraintEngine",
    "ConstraintLevel",
    "ConstraintViolation",
    "ContractValidator",
    "Edit",
    "EditOutcome",
    "EditResult",
    "HierarchicalConstraint",
    "MultiDimensionalCritique",
    "PreEmitGate",
    "PreflightGate",
    "ToolContract",
    "apply_edits",
    "apply_with_reflection",
    "parse_edits",
]
