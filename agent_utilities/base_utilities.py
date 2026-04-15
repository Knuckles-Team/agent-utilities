#!/usr/bin/python
# coding: utf-8
"""Base Utilities Module.

This module provides a collection of low-level helper functions and classes used across
the agent-utilities package for type conversion, environment variable expansion,
package management, and dynamic object patching.
"""

import inspect
import json
import logging
from urllib.parse import urlparse
import os
import pickle
import re
import sys

from importlib.resources import as_file, files
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
)

from packaging import version
from typing_extensions import ParamSpec
from dotenv import load_dotenv

if TYPE_CHECKING:
    pass

T = TypeVar("T")
P = ParamSpec("P")
F = TypeVar("F", bound=Callable[..., Any])

try:
    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:
    AsyncOpenAI = None
    OpenAIProvider = None

try:
    from groq import AsyncGroq
    from pydantic_ai.providers.groq import GroqProvider
except ImportError:
    AsyncGroq = None
    GroqProvider = None

try:
    from mistralai import Mistral
    from pydantic_ai.providers.mistral import MistralProvider
except ImportError:
    Mistral = None
    MistralProvider = None

try:
    from pydantic_ai.models.anthropic import AnthropicModel
    from anthropic import AsyncAnthropic
    from pydantic_ai.providers.anthropic import AnthropicProvider
except ImportError:
    AnthropicModel = None
    AsyncAnthropic = None
    AnthropicProvider = None

__version__ = "0.2.39"


def to_float(string=None) -> float:
    """Convert a value to a float.

    Args:
        string: The value to convert. Defaults to None.

    Returns:
        The converted float value, or 0.0 if conversion fails or input is empty.

    """
    if isinstance(string, float):
        return string
    if not string:
        return 0.0
    try:
        return float(str(string).strip())
    except (ValueError, TypeError):
        return 0.0


def to_boolean(string=None) -> bool:
    """Convert a value to a boolean.

    Args:
        string: The value to convert. Defaults to None.

    Returns:
        True if the value is one of {"t", "true", "y", "yes", "1"}, False otherwise.

    """
    if isinstance(string, bool):
        return string
    if not string:
        return False
    normalized = str(string).strip().lower()
    return normalized in {"t", "true", "y", "yes", "1"}


def to_integer(string=None) -> int:
    """Convert a value to an integer.

    Args:
        string: The value to convert. Defaults to None.

    Returns:
        The converted integer value, or 0 if conversion fails or input is empty.

    """
    if isinstance(string, int):
        return string
    if not string:
        return 0
    try:
        return int(str(string).strip())
    except (ValueError, TypeError):
        return 0


def to_list(string: Union[str, list] = None) -> list:
    """Convert a value to a list.

    Args:
        string: The value to convert (JSON string or list). Defaults to None.

    Returns:
        A list of values. If the input is a JSON string, it's parsed;
        otherwise, it's split by commas. Returns an empty list if input is empty.

    """
    if isinstance(string, list):
        return string
    if not string:
        return []
    try:
        return json.loads(string)
    except Exception:
        return string.split(",")


def to_dict(string: Union[str, dict] = None) -> dict:
    """Convert a value to a dictionary.

    Args:
        string: The value to convert (JSON string or dict). Defaults to None.

    Returns:
        The converted dictionary.

    Raises:
        ValueError: If the input cannot be converted to a dictionary.

    """
    if isinstance(string, dict):
        return string
    if not string:
        return {}
    try:
        return json.loads(string)
    except Exception:
        raise ValueError(f"Cannot convert '{string}' to dict")


def expand_env_vars(text: str) -> str:
    """Expand environment variables in a string, supporting ${VAR:-DEFAULT} syntax.

    Args:
        text: The string containing possible environment variables.

    Returns:
        The string with variables expanded.

    """
    if not text:
        return text

    pattern = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")

    def replace(match):
        var_name = match.group(1)
        default_value = match.group(2)

        val = os.getenv(var_name)
        if val is not None:
            return val.strip("\r")

        if default_value is not None:
            return default_value

        # VALIDATION_MODE: Return a dummy value instead of the original placeholder
        # to prevent startup crashes in environments where secrets are not set.
        if to_boolean(os.getenv("VALIDATION_MODE", "False")):
            # Check if this is likely a token or secret to provide a more realistic dummy
            is_secret = any(
                k in var_name.upper()
                for k in ["TOKEN", "SECRET", "PASSWORD", "KEY", "AUTH", "API"]
            )
            return (
                f"dummy_{var_name.lower()}"
                if is_secret
                else f"validation_{var_name.lower()}"
            )

        # Default fallback: return empty string if no default provided
        return ""

    return pattern.sub(replace, text)


def is_loopback_url(
    url: str, current_host: str = None, current_port: int = None
) -> bool:
    """Check if a URL is a loopback to the current agent's process.

    Args:
        url: The candidate MCP URL.
        current_host: The host addressing the current agent.
        current_port: The port the current agent is running on.

    Returns:
        True if the URL is a loopback to this process, False otherwise.

    """
    if not url:
        return False

    try:
        parsed = urlparse(url)
        if not parsed.port or not current_port or parsed.port != int(current_port):
            return False

        hostname = parsed.hostname.lower() if parsed.hostname else ""
        loopback_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
        if hostname in loopback_hosts:
            return True

        if current_host and hostname == current_host.lower():
            return True

        return False
    except Exception:
        return False


def GET_DEFAULT_SSL_VERIFY() -> bool:
    """Read SSL verification setting from the environment.

    Returns:
        The value of the 'SSL_VERIFY' environment variable as a boolean.
        Defaults to True if not set.

    """
    return to_boolean(os.getenv("SSL_VERIFY", "true"))


def ensure_package_installed(package_name: str, auto_install: bool = False) -> bool:
    """Check if a package is installed, optionally attempting to install it.

    Args:
        package_name: The name of the package to check (e.g. 'gitlab-api').
        auto_install: Whether to attempt installation via pip if missing.

    Returns:
        True if installed (or successfully installed), False otherwise.

    """
    import importlib.util
    import logging
    import sys

    logger = logging.getLogger(__name__)

    spec = importlib.util.find_spec(package_name.replace("-", "_"))
    if spec:
        return True

    if not auto_install:
        logger.warning(f"Package '{package_name}' is not installed.")
        return False

    logger.info(f"Attempting to install package '{package_name}'...")
    try:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Successfully installed '{package_name}'.")
        return True
    except Exception as e:
        logger.error(f"Failed to install '{package_name}': {e}")
        return False


def load_env_vars(override: bool = False) -> None:
    """Load environment variables from a .env file located in the caller's directory.

    This function searches for a .env file by traversing up to 5 levels of parent
    directories from the calling module's location.

    Args:
        override: Whether to override existing environment variables. Defaults to False.

    """
    try:
        package_name = retrieve_package_name()
        if package_name and package_name != "unknown_package":
            stack = inspect.stack()
            caller_file = None
            for frame in stack:
                if "agent_utilities" not in frame.filename:
                    caller_file = frame.filename
                    break

            if caller_file:
                start_dir = Path(caller_file).parent
                dotenv_path = None
                curr = start_dir

                for _ in range(5):
                    candidate = curr / ".env"
                    if candidate.exists():
                        dotenv_path = str(candidate)
                        break
                    if curr == curr.parent:
                        break
                    curr = curr.parent

                if dotenv_path:
                    load_dotenv(dotenv_path, override=override)

                else:
                    pass

    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading .env file: {e}")


def save_model(model: Any, file_name: str = "model", file_path: str = ".") -> str:
    """Serialize a model object to a pickle file.

    Args:
        model: The model object to serialize.
        file_name: Name of the output file without extension. Defaults to "model".
        file_path: Directory path where the file should be saved. Defaults to ".".

    Returns:
        The absolute or relative path to the saved pickle file.

    """
    pickle_file = os.path.join(file_path, f"{file_name}.pkl")
    with open(pickle_file, "wb") as file:
        pickle.dump(model, file)
    return pickle_file


def load_model(file: str) -> Any:
    """Deserialize a model object from a pickle file.

    Args:
        file: Path to the pickle file.

    Returns:
        The deserialized model object.

    """
    with open(file, "rb") as model_file:
        model = pickle.load(model_file)
    return model


def retrieve_package_name() -> str:
    """Returns the top-level package name of the module that imported this utility.

    This function inspects the call stack to find the first external frame
    that belongs to a valid Python package (identified by pyproject.toml,
    setup.py, or __init__.py).

    Returns:
        The name of the top-level package as a string. Defaults to 'agent_utilities'.

    """
    first_external_frame_package = None

    skip_packages = (
        "agent_utilities",
        "universal_skills",
        "agent-utilities",
        "universal-skills",
        "tmp",
        "__main__",
        "env",
        "venv",
        "fastapi",
        "starlette",
        "uvicorn",
        "pydantic",
        "pydantic_ai",
        "inspect",
        "importlib",
        "contextlib",
        "logging",
        "asyncio",
    )
    try:
        stack = inspect.stack()
        for i, frame_info in enumerate(stack):
            frame_file = frame_info.filename
            if not frame_file or not os.path.exists(frame_file):
                continue

            path = Path(frame_file).resolve()

            is_skipped = False
            for part in path.parts:
                if part in skip_packages:
                    is_skipped = True
                    break
            if is_skipped:
                continue

            curr = path.parent
            for _ in range(4):
                if (curr / "pyproject.toml").is_file() or (curr / "setup.py").is_file():
                    pkg_name = curr.name.replace("-", "_")
                    if pkg_name not in skip_packages:
                        return pkg_name

                if (curr / "__init__.py").is_file():
                    pkg_name = curr.name.replace("-", "_")
                    if pkg_name not in skip_packages:
                        if not first_external_frame_package:
                            first_external_frame_package = pkg_name

                if curr == curr.parent:
                    break
                curr = curr.parent

            if not first_external_frame_package:
                pkg_name = path.parent.name.replace("-", "_")
                if pkg_name not in skip_packages:
                    first_external_frame_package = pkg_name

    except Exception:
        pass

    if first_external_frame_package:
        return first_external_frame_package

    if __package__:
        top = __package__.partition(".")[0]
        if top and top not in skip_packages and top != "__main__":
            return top

    return "agent_utilities"


def get_library_file_path(file: str) -> str:
    """Get the absolute path to a file within the current library package.

    Args:
        file: The relative path to the file within the package.

    Returns:
        The absolute string path to the file on the filesystem.

    """
    library_file = files(retrieve_package_name()).joinpath(file)
    with as_file(library_file) as path:
        library_file_path = str(path)
    return library_file_path


def get_logger(name: str):
    """Standardized logger setup for the agent ecosystem.

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        A configured logging.Logger instance with INFO level and a StreamHandler.

    """
    logger = getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@dataclass
class ModuleInfo:
    """Represents a Python module and its version constraints.

    Attributes:
        name: The name of the module.
        min_version: Minimum required version. Defaults to None.
        max_version: Maximum allowed version. Defaults to None.
        min_inclusive: Whether min_version is inclusive (>=). Defaults to False (>).
        max_inclusive: Whether max_version is inclusive (<=). Defaults to False (<).

    """

    name: str
    min_version: str | None = None
    max_version: str | None = None
    min_inclusive: bool = False
    max_inclusive: bool = False

    def is_in_sys_modules(self) -> str | None:
        """Check if the module is installed and satisfies version constraints.

        Returns:
            None if the module is correctly installed; otherwise, a string
            describing why it's missing or invalid.

        """
        import importlib.util
        from importlib.metadata import version as get_version, PackageNotFoundError

        if not importlib.util.find_spec(self.name):
            return f"'{self.name}' is not installed."

        if self.min_version or self.max_version:
            try:
                installed_version = get_version(self.name)
            except PackageNotFoundError:
                try:
                    module = importlib.import_module(self.name)
                    installed_version = getattr(module, "__version__", None)
                except ImportError:
                    return f"'{self.name}' is found but version could not be retrieved."

            if installed_version is None:
                return f"'{self.name}' is installed, but the version is not available."

            installed_ver = version.parse(installed_version)

            if self.min_version:
                min_ver = version.parse(self.min_version)
                msg = f"'{self.name}' is installed, but the installed version {installed_version} is too low (required '{self}')."
                if not self.min_inclusive and installed_ver == min_ver:
                    return msg
                if self.min_inclusive and installed_ver < min_ver:
                    return msg

            if self.max_version:
                max_ver = version.parse(self.max_version)
                msg = f"'{self.name}' is installed, but the installed version {installed_version} is too high (required '{self}')."
                if not self.max_inclusive and installed_ver == max_ver:
                    return msg
                if self.max_inclusive and installed_ver > max_ver:
                    return msg

        return None

    def __repr__(self) -> str:
        s = self.name
        if self.min_version:
            s += (
                f">={self.min_version}"
                if self.min_inclusive
                else f">{self.min_version}"
            )
        if self.max_version:
            s += (
                f"<={self.max_version}"
                if self.max_inclusive
                else f"<{self.max_version}"
            )
        return s

    @classmethod
    def from_str(cls, module_info: str) -> "ModuleInfo":
        """Parse a module information string into a ModuleInfo object.

        The expected format is 'package_name[>=version, <version]'.

        Args:
            module_info: The string to parse.

        Returns:
            A ModuleInfo instance.

        Raises:
            ValueError: If the input string format is invalid.

        """
        pattern = re.compile(r"^(?P<name>[a-zA-Z0-9-_]+)(?P<constraint>.*)$")
        match = pattern.match(module_info.strip())
        if not match:
            raise ValueError(f"Invalid package information: {module_info}")

        name = match.group("name")
        constraints = match.group("constraint").strip()
        min_version = max_version = None
        min_inclusive = max_inclusive = False

        if constraints:
            constraint_pattern = re.findall(r"(>=|<=|>|<)([0-9\.]+)?", constraints)
            for operator, ver in constraint_pattern:
                if not ver:
                    continue
                if operator == ">=":
                    min_version = ver
                    min_inclusive = True
                elif operator == "<=":
                    max_version = ver
                    max_inclusive = True
                elif operator == ">":
                    min_version = ver
                    min_inclusive = False
                elif operator == "<":
                    max_version = ver
                    max_inclusive = False
        return ModuleInfo(
            name=name,
            min_version=min_version,
            max_version=max_version,
            min_inclusive=min_inclusive,
            max_inclusive=max_inclusive,
        )


class Result:
    """Simple container for tracking the success or failure of an operation."""

    def __init__(self) -> None:
        self._failed: bool | None = None

    @property
    def is_successful(self) -> bool:
        """Check if the operation was successful.

        Returns:
            True if not failed, False otherwise.

        Raises:
            ValueError: If the result has not yet been set.

        """
        if self._failed is None:
            raise ValueError("Result not set")
        return not self._failed


@contextmanager
def optional_import_block() -> Generator[Result, None, None]:
    """Context manager to wrap optional imports.

    Allows a block of code to be attempted; if an ImportError occurs,
    the Result object will mark the operation as failed rather than crashing.

    Yields:
        A Result object that tracks whether the import succeeded.

    """
    result = Result()
    try:
        yield result
        result._failed = False
    except ImportError as e:
        getLogger(__name__).debug(f"Ignoring ImportError: {e}")
        result._failed = True


def get_missing_imports(modules: str | Iterable[str]) -> dict[str, str]:
    """Identify which required modules are missing or have version mismatches.

    Args:
        modules: A single module string or an iterable of module strings
            with optional version constraints (e.g., 'requests>=2.0').

    Returns:
        A dictionary mapping package names to their error messages.
        Empty if all modules are present and valid.

    """
    if isinstance(modules, str):
        modules = [modules]
    module_infos = [ModuleInfo.from_str(module) for module in modules]
    x = {m.name: m.is_in_sys_modules() for m in module_infos}
    return {k: v for k, v in x.items() if v}


class PatchObject(ABC, Generic[T]):
    """Abstract base class for patching objects to handle missing dependencies.

    This class provides a mechanism to wrap an object and replace its members
    with "patched" versions that raise ImportError if required modules are missing.

    Attributes:
        o: The original object being patched.
        missing_modules: A dictionary of missing modules and their status messages.
        dep_target: The target dependency identifier for the patch.

    """

    def __init__(self, o: T, missing_modules: dict[str, str], dep_target: str):
        if not self.accept(o):
            raise ValueError(f"Cannot patch object of type {type(o)}")
        self.o = o
        self.missing_modules = missing_modules
        self.dep_target = dep_target

    @classmethod
    @abstractmethod
    def accept(cls, o: Any) -> bool:
        """Determine if the given object can be patched by this PatchObject subclass.

        Args:
            o: The object to evaluate.

        Returns:
            True if the object is acceptable for patching, False otherwise.

        """
        ...

    @abstractmethod
    def patch(self, except_for: Iterable[str]) -> T:
        """Apply the patch to the object, replacing members that rely on missing modules.

        Args:
            except_for: An iterable of member names that should be excluded from patching.

        Returns:
            The patched object.

        """
        ...

    def get_object_with_metadata(self) -> Any:
        """Retrieve the original object associated with this patcher.

        Returns:
            The original unpatched object.

        """
        return self.o

    @property
    def msg(self) -> str:
        """Generate a detailed error message listing the missing modules for this object.

        Returns:
            A formatted string describing the missing dependencies and installation advice.

        """
        o = self.get_object_with_metadata()
        plural = len(self.missing_modules) > 1
        fqn = f"{o.__module__}.{o.__name__}" if hasattr(o, "__module__") else o.__name__
        msg = f"{'Modules' if plural else 'A module'} needed for {fqn} {'are' if plural else 'is'} missing:\n"
        for _, status in self.missing_modules.items():
            msg += f" - {status}\n"
        msg += f"Please install {'them' if plural else 'it'} using appropriate extras."
        return msg

    def copy_metadata(self, retval: T) -> None:
        """Copy metadata (docstrings, name, module) from the original object to a replacement.

        Args:
            retval: The replacement object to receive the metadata.

        """
        o = self.o
        if hasattr(o, "__doc__"):
            retval.__doc__ = o.__doc__
        if hasattr(o, "__name__"):
            retval.__name__ = o.__name__
        if hasattr(o, "__module__"):
            retval.__module__ = o.__module__

    _registry: list[type["PatchObject[Any]"]] = []

    @classmethod
    def register(cls) -> Callable[[type["PatchObject[Any]"]], type["PatchObject[Any]"]]:
        """Decorator to register a PatchObject subclass in the global registry.

        Returns:
            A decorator function that adds the subclass to the _registry list.

        """

        def decorator(subclass: type["PatchObject[Any]"]) -> type["PatchObject[Any]"]:
            cls._registry.append(subclass)
            return subclass

        return decorator

    @classmethod
    def create(
        cls, o: T, *, missing_modules: dict[str, str], dep_target: str
    ) -> Optional["PatchObject[T]"]:
        """Factory method to create the appropriate PatchObject for a given object.

        Args:
            o: The object to patch.
            missing_modules: Dictionary of missing modules and their status.
            dep_target: Target dependency identifier.

        Returns:
            A PatchObject instance if a suitable subclass is found, otherwise None.

        """
        for subclass in cls._registry:
            if subclass.accept(o):
                return subclass(o, missing_modules, dep_target)
        return None


@PatchObject.register()
class PatchCallable(PatchObject[F]):
    """Patcher for callable objects (functions and methods)."""

    @classmethod
    def accept(cls, o: Any) -> bool:
        """Check if the object is a function or method."""
        return inspect.isfunction(o) or inspect.ismethod(o)

    def patch(self, except_for: Iterable[str]) -> F:
        if self.o.__name__ in except_for:
            return self.o
        f: Callable[..., Any] = self.o

        @wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)
        return _call


@PatchObject.register()
class PatchClass(PatchObject[type[Any]]):
    """Patcher for class objects."""

    @classmethod
    def accept(cls, o: Any) -> bool:
        """Check if the object is a class."""
        return inspect.isclass(o)

    def patch(self, except_for: Iterable[str]) -> type[Any]:
        if self.o.__name__ in except_for:
            return self.o
        for name, member in inspect.getmembers(self.o):
            if name.startswith("__") and name != "__init__":
                continue
            patched = patch_object(
                member,
                missing_modules=self.missing_modules,
                dep_target=self.dep_target,
                fail_if_not_patchable=False,
                except_for=except_for,
            )
            with suppress(AttributeError):
                setattr(self.o, name, patched)
        return self.o


def patch_object(
    o: T,
    *,
    missing_modules: dict[str, str],
    dep_target: str,
    fail_if_not_patchable: bool = True,
    except_for: str | Iterable[str] | None = None,
) -> T:
    """Apply a patch to an object to handle missing dependencies.

    Args:
        o: The object to patch.
        missing_modules: Dictionary of missing modules and their status messages.
        dep_target: The target dependency identifier for the patch.
        fail_if_not_patchable: Whether to raise an error if no patcher fits. Defaults to True.
        except_for: Member names to exclude from patching.

    Returns:
        The patched object (or the original if no patcher was applied).

    """
    patcher = PatchObject.create(
        o, missing_modules=missing_modules, dep_target=dep_target
    )
    if fail_if_not_patchable and patcher is None:
        raise ValueError(f"Cannot patch object of type {type(o)}")
    except_for = [except_for] if isinstance(except_for, str) else (except_for or [])
    return patcher.patch(except_for=except_for) if patcher else o


def require_optional_import(
    modules: str | Iterable[str],
    dep_target: str,
    *,
    except_for: Optional[Union[str, Iterable[str]]] = None,
) -> Callable[[F], F]:
    """Decorator to mark an object as requiring optional dependencies.

    If the specified modules are missing, the decorated object is wrapped
    with a PatchObject that raises ImportError when accessed or called.

    Args:
        modules: A single module string or list of module requirements.
        dep_target: Identifier for the dependency group (e.g. 'openai').
        except_for: List of member names that should remain accessible.

    Returns:
        A decorator that applies the dependency patch if necessary.

    """
    missing_modules = get_missing_imports(modules)

    def decorator(o: F) -> F:
        if not missing_modules:
            return o
        return patch_object(
            o,
            missing_modules=missing_modules,
            dep_target=dep_target,
            except_for=except_for,
        )

    return decorator
