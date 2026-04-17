#!/usr/bin/python
# coding: utf-8
"""API Utilities Module.

This module serves as a central export point for API-related
exceptions and authentication decorators used throughout the
agent ecosystem.
"""

__version__ = "0.2.40"

from .exceptions import (  # noqa: F401
    AuthError,
    ApiError,
    UnauthorizedError,
    MissingParameterError,
    ParameterError,
    LoginRequiredError,
)
from .decorators import require_auth  # noqa: F401
