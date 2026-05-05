"""Backward-compatible re-export of exception classes.

Canonical location: agent_utilities.core.exceptions
This module exists solely so that downstream agents using
``from agent_utilities.exceptions import AuthError`` continue to work.
"""

from agent_utilities.core.exceptions import *  # noqa: F401, F403
from agent_utilities.core.exceptions import (  # explicit re-exports for type checkers
    AuthError,
    MissingParameterError,
    ParameterError,
    UnauthorizedError,
)

__all__ = [
    "AuthError",
    "UnauthorizedError",
    "ParameterError",
    "MissingParameterError",
]
