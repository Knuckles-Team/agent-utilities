#!/usr/bin/python
# coding: utf-8

import functools

__version__ = "0.1.6"


class AuthError(Exception):
    """
    Authentication error
    """

    pass


class UnauthorizedError(AuthError):
    """
    Unauthorized error
    """

    pass


class MissingParameterError(Exception):
    """
    Missing Parameter error
    """

    pass


class ParameterError(Exception):
    """
    Parameter error
    """

    pass


class LoginRequiredError(Exception):
    """
    Authentication error
    """

    pass


def require_auth(function):
    """
    Wraps API calls in function that ensures headers are passed
    with a token
    """

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if not self.headers:
            raise LoginRequiredError
        return function(self, *args, **kwargs)

    return wrapper
