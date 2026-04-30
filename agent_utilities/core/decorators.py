#!/usr/bin/python
"""Agent Decorators Module.

CONCEPT:AU-011 — Secrets & Authentication

This module provides reusable decorators for agent logic. It includes
authentication guards and other functional wrappers used to enforce
pre-conditions on agent actions or API calls.
"""

import functools

from agent_utilities.core.exceptions import LoginRequiredError


def require_auth(function):
    """Decorator to enforce authentication on API wrapper methods.

    Checks if 'self.headers' is populated before executing the wrapped
    function. Typically used in API client wrappers to ensure credentials
    are present.

    Raises:
        LoginRequiredError: If headers are missing.

    """

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if not self.headers:
            raise LoginRequiredError
        return function(self, *args, **kwargs)

    return wrapper
