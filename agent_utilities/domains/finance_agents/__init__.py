"""AU-owned finance agent roles (AU-CONTEXT-R007).

Sibling to ``agent_utilities/domains/finance/`` rather than nested under it:
AU-CONTEXT-R007.1 pins ``domains/finance/*.py`` as a shrink-only set of
finance *math* modules being migrated to epistemic-graph's finance core.
Agent roles like this package's ``flip_explainer`` are not finance math --
they are AU-owned decision/explanation logic that calls through the typed
delegation seam (``agent_utilities/api/finance_delegation.py``) for any
underlying calculation -- so they live outside that pinned set entirely.
"""

from __future__ import annotations
