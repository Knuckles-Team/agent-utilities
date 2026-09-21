"""Layer and cross-product dependency-direction gate for agent-utilities.

The target points inward::

    contracts <- domain <- ports <- application <- adapters <- composition

A module may import its own layer or a layer to its left. Adapters may not call
other adapter groups. This package measures both rules across every static import,
including function-local and ``TYPE_CHECKING`` imports. It separately labels the
eager subset so its result is not confused with the zero-tolerance eager-cycle gate.

The internal six-layer result remains an operator/refactor-planning census while
the package is decomposed. The accepted four-owner contract is not legacy debt:
imports of GraphOS, SDK implementation modules, or EG private/server modules are
blocking at an absolute zero. There is no threshold, baseline, update flag, or
compatibility exception. Discovery or parsing failures exit 2 because an
incomplete scan cannot claim coverage.

Layer assignment follows the canonical agent-utilities-development inventory with
one explicit, reported exception: ``core.config`` is treated as the innermost typed
configuration contract. The canonical table calls it composition while the mandatory
configuration discipline requires all modules to depend on it and forbids direct
environment reads elsewhere. The canonical rule remains recorded beside the effective
exception so the disagreement stays visible.

The parser remains independent of ``check_import_cycles.py``. The blocking cycle gate
keeps only eager imports and deliberately over-produces candidate graph edges; this
census retains every statement and resolves one target per alias so it can count import
statements. Sharing private parsing policy would couple an advisory report to a
zero-tolerance commit gate.

Usage::

    python3 scripts/check_layer_direction.py
    python3 scripts/check_layer_direction.py --list
    python3 scripts/check_layer_direction.py path/to/package
"""

from scripts.layer_direction.imports import ImportWalker
from scripts.layer_direction.model import (
    CLASSIFICATION_EXCEPTIONS,
    LAYER_RULES,
    BoundaryViolation,
    ScanIncomplete,
    Violation,
    classify,
)
from scripts.layer_direction.report import main
from scripts.layer_direction.scan import scan

__all__ = [
    "CLASSIFICATION_EXCEPTIONS",
    "ImportWalker",
    "LAYER_RULES",
    "ScanIncomplete",
    "Violation",
    "BoundaryViolation",
    "classify",
    "main",
    "scan",
]
