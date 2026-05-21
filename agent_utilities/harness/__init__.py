"""Harness subpackage for evaluation and backtesting. CONCEPT:AHE-3.4

Exposes OptimisticStateLocker, BranchMergeStateLocker, ToolContract, and ContractValidator.
"""

from .contract_validator import ContractValidator, ToolContract
from .distributed_state_manager import BranchMergeStateLocker, OptimisticStateLocker

__all__ = [
    "OptimisticStateLocker",
    "BranchMergeStateLocker",
    "ToolContract",
    "ContractValidator",
]
