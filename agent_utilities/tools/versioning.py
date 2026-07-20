import functools


def tool_version(version: str):
    """Decorator to attach version metadata to agent tools.

    This attaches an __agentic_version__ attribute to the function and
    injects the version into the docstring for LLMs to read.
    """

    def decorator(func):
        func.__agentic_version__ = version
        if func.__doc__:
            func.__doc__ = f"{func.__doc__.rstrip()}\n\n    Version: {version}\n"
        else:
            func.__doc__ = f"Version: {version}\n"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator
