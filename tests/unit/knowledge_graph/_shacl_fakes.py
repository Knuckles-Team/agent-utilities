"""Shared test double for the committed EG SHACL authority.

43197d7c6 ("refactor: move semantic authority to epistemic graph") moved
``PromotionGovernanceValidator``'s governance-shape check onto a committed EG
authority (``engine.shacl_validate_committed``); ``_validate_shacl_spec``
fails that check closed unconditionally whenever the engine lacks this
method at all, so a stub engine without it can never clear
``verdict.valid`` and a gated canary/apply/promote/route branch never fires.

Several stub engines across this test suite need the same always-conforms
behavior for their synthetic specs; share one mixin instead of repeating the
method body per class.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = ["AlwaysConformsShaclMixin"]


class AlwaysConformsShaclMixin:
    """Mix into a stub engine to make it pass PromotionGovernanceValidator's
    SHACL check unconditionally."""

    def shacl_validate_committed(self, _document: str) -> Any:
        return SimpleNamespace(conforms=True, results=[])
