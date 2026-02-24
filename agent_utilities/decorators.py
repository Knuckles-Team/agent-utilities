#!/usr/bin/python
# coding: utf-8

import functools
from agent_utilities.exceptions import LoginRequiredError


def require_auth(function):
    """
    Wraps API calls in function that ensures headers are passed
    with a token or API key
    """

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if not (
            "Authorization" in self.headers or "X-ArchiveBox-API-Key" in self.headers
        ):
            raise LoginRequiredError
        return function(self, *args, **kwargs)

    return wrapper
