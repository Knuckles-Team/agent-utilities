#!/usr/bin/python
"""API Utilities Module.

This module serves as a central export point for API-related
exceptions and authentication decorators used throughout the
agent ecosystem.
"""

__version__ = "0.2.42"

__all__ = [
    "require_auth",
    "ApiError",
    "AuthError",
    "LoginRequiredError",
    "MissingParameterError",
    "ParameterError",
    "UnauthorizedError",
]

from agent_utilities.core.decorators import require_auth
from agent_utilities.core.exceptions import (
    ApiError,
    AuthError,
    LoginRequiredError,
    MissingParameterError,
    ParameterError,
    UnauthorizedError,
)
