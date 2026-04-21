#!/usr/bin/python
"""API Utilities Module.

This module serves as a central export point for API-related
exceptions and authentication decorators used throughout the
agent ecosystem.
"""

__version__ = "0.2.40"

from .decorators import require_auth  # noqa: F401
from .exceptions import (  # noqa: F401
    ApiError,
    AuthError,
    LoginRequiredError,
    MissingParameterError,
    ParameterError,
    UnauthorizedError,
)
