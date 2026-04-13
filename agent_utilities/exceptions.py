#!/usr/bin/python
# coding: utf-8
"""Custom Exceptions Module.

This module defines specialized exception classes used across the agent
ecosystem to handle errors related to authentication, API communication,
and parameter validation.
"""


class AuthError(Exception):
    """Base exception for all authentication-related errors."""

    pass


class ApiError(Exception):
    """Raised when an external API call fails or returns an error."""

    pass


class UnauthorizedError(AuthError):
    """Raised when access is denied due to insufficient permissions or invalid credentials."""

    pass


class MissingParameterError(Exception):
    """Raised when a required parameter is missing from a function call or API request."""

    pass


class ParameterError(Exception):
    """Raised when a provided parameter is invalid or has an incorrect format."""

    pass


class LoginRequiredError(Exception):
    """Raised when an action requires authentication but no token is provided."""

    pass
