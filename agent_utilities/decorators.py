"""Backward-compatibility shim for ``agent_utilities.decorators``.

The ``require_auth`` decorator was moved to ``agent_utilities.core.decorators``
during the AU-011 refactoring. This module re-exports it to avoid breaking
downstream consumers that import from the legacy path.

.. deprecated:: 0.3.0
    Import from ``agent_utilities.core.decorators`` or
    ``agent_utilities.api_utilities`` instead.
"""

from agent_utilities.core.decorators import require_auth

__all__ = ["require_auth"]
