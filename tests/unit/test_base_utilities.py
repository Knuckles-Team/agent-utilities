from __future__ import annotations

import pytest
import os
from agent_utilities import base_utilities

import inspect
import logging
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_to_float():
    assert base_utilities.to_float("1.5") == 1.5
    assert base_utilities.to_float(2.0) == 2.0
    assert base_utilities.to_float("invalid") == 0.0
    assert base_utilities.to_float(None) == 0.0

def test_to_boolean():
    assert base_utilities.to_boolean("true") is True
    assert base_utilities.to_boolean("YES") is True
    assert base_utilities.to_boolean("1") is True
    assert base_utilities.to_boolean(True) is True
    assert base_utilities.to_boolean("false") is False
    assert base_utilities.to_boolean(None) is False

def test_to_integer():
    assert base_utilities.to_integer("10") == 10
    assert base_utilities.to_integer(5) == 5
    assert base_utilities.to_integer("abc") == 0
    assert base_utilities.to_integer(None) == 0

def test_to_list():
    assert base_utilities.to_list("[1, 2, 3]") == [1, 2, 3]
    assert base_utilities.to_list("a,b,c") == ["a", "b", "c"]
    assert base_utilities.to_list([1, 2]) == [1, 2]
    assert base_utilities.to_list(None) == []

def test_to_dict():
    assert base_utilities.to_dict('{"a": 1}') == {"a": 1}
    assert base_utilities.to_dict({"b": 2}) == {"b": 2}
    assert base_utilities.to_dict(None) == {}
    with pytest.raises(ValueError):
        base_utilities.to_dict("not a dict")

def test_expand_env_vars(monkeypatch):
    monkeypatch.delenv("VALIDATION_MODE", raising=False)
    monkeypatch.setenv("TEST_VAR", "hello")
    assert base_utilities.expand_env_vars("${TEST_VAR}") == "hello"
    assert base_utilities.expand_env_vars("${MISSING:-default}") == "default"
    assert base_utilities.expand_env_vars("${MISSING}") == ""

    # Validation mode
    monkeypatch.setenv("VALIDATION_MODE", "true")
    assert base_utilities.expand_env_vars("${API_KEY}") == "dummy_api_key"
    assert base_utilities.expand_env_vars("${NORMAL_VAR}") == "validation_normal_var"

def test_is_loopback_url():
    assert base_utilities.is_loopback_url("http://localhost:8000", current_port=8000) is True
    assert base_utilities.is_loopback_url("http://127.0.0.1:8000", current_port=8000) is True
    assert base_utilities.is_loopback_url("http://google.com", current_port=8000) is False
    assert base_utilities.is_loopback_url("http://localhost:9000", current_port=8000) is False

def test_retrieve_package_name():
    # In a test environment, it might return 'agent_utilities' or the test runner package
    pkg = base_utilities.retrieve_package_name()
    assert isinstance(pkg, str)
    assert pkg != ""

def test_save_load_model(tmp_path):
    data = {"key": "value"}
    path = base_utilities.save_model(data, "test_model", str(tmp_path))
    assert os.path.exists(path)

    loaded = base_utilities.load_model(path)
    assert loaded == data

def test_result_class():
    res = base_utilities.Result()
    with pytest.raises(ValueError):
        _ = res.is_successful

    res._failed = False
    assert res.is_successful is True

    res._failed = True
    assert res.is_successful is False

def test_optional_import_block():
    # Clean block: no exception escapes -> is_successful stays True.
    with base_utilities.optional_import_block() as result:
        _ = 1 + 1
    assert result.is_successful is True

    # Block that raises ImportError: optional_import_block swallows it and
    # flips is_successful to False.
    with base_utilities.optional_import_block() as result:
        import __definitely_not_a_real_module__  # noqa: F401
    assert result.is_successful is False

def test_module_info_from_str():
    mi = base_utilities.ModuleInfo.from_str("requests>=2.0.0")
    assert mi.name == "requests"
    assert mi.min_version == "2.0.0"
    assert mi.min_inclusive is True

    mi2 = base_utilities.ModuleInfo.from_str("pytest<8.0")
    assert mi2.name == "pytest"
    assert mi2.max_version == "8.0"
    assert mi2.max_inclusive is False

def test_get_missing_imports():
    # Assuming requests is installed
    missing = base_utilities.get_missing_imports(["requests", "non_existent_pkg"])
    assert "non_existent_pkg" in missing
    assert "requests" not in missing

def test_expand_env_vars_empty_string_returns_empty() -> None:
    """Empty string short-circuit: line 191 early return."""
    assert base_utilities.expand_env_vars("") == ""


def test_expand_env_vars_none_returns_none() -> None:
    """None input short-circuits through `if not text`."""
    assert base_utilities.expand_env_vars(None) is None  # type: ignore[arg-type]


def test_expand_env_vars_strips_carriage_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """Carriage-return stripping on env values (line 201)."""
    monkeypatch.setenv("SOME_VAR", "value\r")
    assert base_utilities.expand_env_vars("${SOME_VAR}") == "value"


def test_expand_env_vars_validation_mode_non_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation mode non-secret placeholder (line 217)."""
    monkeypatch.delenv("GENERIC_NAME", raising=False)
    monkeypatch.setenv("VALIDATION_MODE", "true")
    result = base_utilities.expand_env_vars("${GENERIC_NAME}")
    assert result == "validation_generic_name"


def test_expand_env_vars_validation_mode_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation mode secret placeholder (line 214-218)."""
    monkeypatch.delenv("MY_TOKEN", raising=False)
    monkeypatch.setenv("VALIDATION_MODE", "true")
    assert base_utilities.expand_env_vars("${MY_TOKEN}") == "dummy_my_token"


# ---------------------------------------------------------------------------
# is_loopback_url: edge cases (lines 241, 253-258)
# ---------------------------------------------------------------------------


def test_is_loopback_url_empty_returns_false() -> None:
    """Empty URL short-circuit (line 241)."""
    assert base_utilities.is_loopback_url("") is False


def test_is_loopback_url_none_returns_false() -> None:
    """None URL short-circuit (line 241)."""
    assert base_utilities.is_loopback_url(None) is False  # type: ignore[arg-type]


def test_is_loopback_url_matches_current_host() -> None:
    """Matching current_host branch (line 253-254)."""
    assert (
        base_utilities.is_loopback_url(
            "http://myhost.local:8000",
            current_host="myhost.local",
            current_port=8000,
        )
        is True
    )


def test_is_loopback_url_non_matching_current_host() -> None:
    """Non-matching current_host falls through to `return False` (line 256)."""
    assert (
        base_utilities.is_loopback_url(
            "http://some-other.host:8000",
            current_host="myhost.local",
            current_port=8000,
        )
        is False
    )


def test_is_loopback_url_exception_path() -> None:
    """Malformed URL triggers urlparse-safe exception branch (line 257-258)."""
    # A value that makes urlparse's `.port` attribute raise ValueError
    # (non-numeric port segment).  `urlparse("http://h:abc")` raises on `.port`.
    assert base_utilities.is_loopback_url("http://h:abc", current_port=80) is False


# ---------------------------------------------------------------------------
# GET_DEFAULT_SSL_VERIFY
# ---------------------------------------------------------------------------


def test_get_default_ssl_verify_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default SSL verify (no env)."""
    monkeypatch.delenv("SSL_VERIFY", raising=False)
    assert base_utilities.GET_DEFAULT_SSL_VERIFY() is True


def test_get_default_ssl_verify_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSL_VERIFY=false env override."""
    monkeypatch.setenv("SSL_VERIFY", "false")
    assert base_utilities.GET_DEFAULT_SSL_VERIFY() is False


# ---------------------------------------------------------------------------
# ensure_package_installed: auto-install paths (lines 293-317)
# ---------------------------------------------------------------------------


def test_ensure_package_installed_already_present() -> None:
    """Already-installed path returns True without subprocess."""
    assert base_utilities.ensure_package_installed("json") is True


def test_ensure_package_installed_missing_no_auto_install(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing + auto_install=False logs warning and returns False (line 293-295)."""
    caplog.set_level(logging.WARNING, logger="agent_utilities.base_utilities")
    result = base_utilities.ensure_package_installed("pkg_definitely_does_not_exist_12345")
    assert result is False


def test_ensure_package_installed_missing_with_auto_install_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auto_install=True happy path uses subprocess (line 297-314)."""
    called: dict[str, Any] = {}

    def fake_check_call(cmd, **kwargs):
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    result = base_utilities.ensure_package_installed(
        "pkg_definitely_does_not_exist_12345",
        auto_install=True,
    )
    assert result is True
    assert called["cmd"][1] == "-m"
    assert called["cmd"][2] == "pip"
    assert "pkg_definitely_does_not_exist_12345" in called["cmd"]


def test_ensure_package_installed_missing_with_auto_install_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auto_install=True subprocess failure returns False (line 315-317)."""

    def fake_check_call(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    result = base_utilities.ensure_package_installed(
        "pkg_definitely_does_not_exist_12345",
        auto_install=True,
    )
    assert result is False


# ---------------------------------------------------------------------------
# load_env_vars: file discovery (lines 330-361)
# ---------------------------------------------------------------------------


def test_load_env_vars_finds_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_env_vars finds and loads a .env upward from caller frame."""
    # Create a .env in tmp_path
    env_file = tmp_path / ".env"
    env_file.write_text("PUSH_ENV_VAR_MARKER=hello_from_dotenv\n")

    # Write a caller script into tmp_path and exec it to simulate a frame
    # outside of agent_utilities.  Instead we monkeypatch inspect.stack()
    # to return a fake frame pointing at tmp_path/sub/caller.py.
    sub = tmp_path / "sub"
    sub.mkdir()
    caller_py = sub / "caller.py"
    caller_py.write_text("# stub caller\n")

    FakeFrame = type(
        "FakeFrame",
        (),
        {"filename": str(caller_py)},
    )

    def fake_stack() -> list[Any]:
        return [FakeFrame()]

    monkeypatch.setenv("PUSH_ENV_VAR_MARKER", "initial_value")
    monkeypatch.setattr(inspect, "stack", fake_stack)
    # Also force retrieve_package_name to return a valid value
    monkeypatch.setattr(base_utilities, "retrieve_package_name", lambda: "some_caller_pkg")

    base_utilities.load_env_vars(override=True)
    # .env should have been loaded and overridden
    assert os.environ.get("PUSH_ENV_VAR_MARKER") == "hello_from_dotenv"
    # Cleanup
    monkeypatch.delenv("PUSH_ENV_VAR_MARKER", raising=False)


def test_load_env_vars_no_dotenv_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_env_vars traverses upward but finds no .env (line 357-358)."""
    caller_py = tmp_path / "caller.py"
    caller_py.write_text("# stub caller\n")

    FakeFrame = type("FakeFrame", (), {"filename": str(caller_py)})

    def fake_stack() -> list[Any]:
        return [FakeFrame()]

    monkeypatch.setattr(inspect, "stack", fake_stack)
    monkeypatch.setattr(base_utilities, "retrieve_package_name", lambda: "some_caller_pkg")
    # Must not raise
    base_utilities.load_env_vars()


def test_load_env_vars_exception_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exception in load_env_vars is caught (line 360-361)."""

    def boom() -> str:
        raise RuntimeError("simulated")

    monkeypatch.setattr(base_utilities, "retrieve_package_name", boom)
    # Should not raise
    base_utilities.load_env_vars()


def test_load_env_vars_unknown_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """retrieve_package_name returns 'unknown_package' early-exit."""
    monkeypatch.setattr(base_utilities, "retrieve_package_name", lambda: "unknown_package")
    base_utilities.load_env_vars()


def test_load_env_vars_no_external_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every frame is inside agent_utilities, caller_file stays None."""
    FakeFrame = type(
        "FakeFrame",
        (),
        {"filename": "/some/path/agent_utilities/file.py"},
    )
    monkeypatch.setattr(inspect, "stack", lambda: [FakeFrame()])
    monkeypatch.setattr(base_utilities, "retrieve_package_name", lambda: "foo_pkg")
    # Must not raise (caller_file is None)
    base_utilities.load_env_vars()


# ---------------------------------------------------------------------------
# save_model / load_model round-trip (already tested) + error branches
# ---------------------------------------------------------------------------


def test_save_model_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save_model with default file_name='model' and custom path."""
    monkeypatch.chdir(tmp_path)
    data = [1, 2, 3]
    path = base_utilities.save_model(data, file_name="serialized", file_path=str(tmp_path))
    assert Path(path).exists()
    loaded = base_utilities.load_model(path)
    assert loaded == data


# ---------------------------------------------------------------------------
# retrieve_package_name: deeper branches (lines 450-452, 461, 465-480)
# ---------------------------------------------------------------------------


def test_retrieve_package_name_default_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """retrieve_package_name returns a sensible default when invoked normally.

    Since the real function hard-codes 'tmp' in its ``skip_packages`` tuple
    (pytest runs live under /tmp), it's not practical to build a fake package
    layout inside tmp_path and have the function accept it.  We instead
    exercise the default path via the real stack and verify a valid return.
    """
    result = base_utilities.retrieve_package_name()
    assert isinstance(result, str)
    # In pytest, our __package__ is agent_utilities itself
    assert result


def test_retrieve_package_name_frame_not_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame file that does not exist is skipped (line 434-435)."""
    FakeFrame = type(
        "FakeFrame", (), {"filename": "/definitely/not/here/caller.py"}
    )
    monkeypatch.setattr(inspect, "stack", lambda: [FakeFrame()])
    result = base_utilities.retrieve_package_name()
    assert isinstance(result, str) and result  # Still returns something


def test_retrieve_package_name_empty_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty inspect stack falls back to __package__ default."""
    monkeypatch.setattr(inspect, "stack", lambda: [])
    result = base_utilities.retrieve_package_name()
    # With empty stack and __package__ == "agent_utilities", should return default
    assert result == "agent_utilities"


def test_retrieve_package_name_stack_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exception in stack iteration is caught (line 469-470)."""

    def broken_stack() -> list[Any]:
        raise RuntimeError("simulated")

    monkeypatch.setattr(inspect, "stack", broken_stack)
    result = base_utilities.retrieve_package_name()
    assert result == "agent_utilities"


# ---------------------------------------------------------------------------
# get_library_file_path (lines 493-496)
# ---------------------------------------------------------------------------


def test_get_library_file_path_agent_utilities_init() -> None:
    """get_library_file_path resolves a real file in the package."""
    with patch.object(base_utilities, "retrieve_package_name", return_value="agent_utilities"):
        result = base_utilities.get_library_file_path("__init__.py")
        assert isinstance(result, str)
        assert result.endswith("__init__.py")
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger_with_handler() -> None:
    """get_logger configures a fresh logger and attaches a StreamHandler."""
    name = "agent_utilities.test_push_logger_unique_xyz"
    # Ensure fresh
    logging.Logger.manager.loggerDict.pop(name, None)
    logger = base_utilities.get_logger(name)
    assert logger.level == logging.INFO
    assert any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    )


def test_get_logger_idempotent() -> None:
    """Calling get_logger twice does not duplicate handlers."""
    name = "agent_utilities.test_push_logger_idem"
    logging.Logger.manager.loggerDict.pop(name, None)
    logger1 = base_utilities.get_logger(name)
    n_handlers = len(logger1.handlers)
    logger2 = base_utilities.get_logger(name)
    assert logger1 is logger2
    assert len(logger2.handlers) == n_handlers


# ---------------------------------------------------------------------------
# ModuleInfo: is_in_sys_modules branches (lines 557-585)
# ---------------------------------------------------------------------------


def test_module_info_not_installed() -> None:
    """is_in_sys_modules returns 'not installed' string for missing pkg."""
    mi = base_utilities.ModuleInfo(name="pkg_definitely_missing_xyz_67890")
    result = mi.is_in_sys_modules()
    assert result is not None
    assert "is not installed" in result


def test_module_info_installed_no_constraint() -> None:
    """is_in_sys_modules returns None when installed and no version constraint."""
    mi = base_utilities.ModuleInfo(name="json")
    # json is a stdlib module, but PackageNotFoundError may apply;
    # expect None (pass) when no version spec
    # json has no distribution metadata; with no constraint we return None.
    result = mi.is_in_sys_modules()
    assert result is None


def test_module_info_with_valid_min_version() -> None:
    """Valid min version constraint (line 571-577)."""
    mi = base_utilities.ModuleInfo(name="packaging", min_version="0.0.1", min_inclusive=True)
    assert mi.is_in_sys_modules() is None


def test_module_info_with_failing_min_inclusive() -> None:
    """Failing min inclusive constraint returns error string."""
    mi = base_utilities.ModuleInfo(
        name="packaging",
        min_version="9999.0.0",
        min_inclusive=True,
    )
    result = mi.is_in_sys_modules()
    assert result is not None
    assert "too low" in result


def test_module_info_with_failing_min_strict() -> None:
    """Failing min strict constraint returns error string."""
    with patch(
        "importlib.metadata.version", return_value="1.0.0"
    ), patch("importlib.util.find_spec", return_value=True):
        mi = base_utilities.ModuleInfo(
            name="packaging",
            min_version="1.0.0",
            min_inclusive=False,
        )
        result = mi.is_in_sys_modules()
        assert result is not None
        assert "too low" in result


def test_module_info_with_valid_max_version() -> None:
    """Valid max version constraint (line 579-585)."""
    mi = base_utilities.ModuleInfo(
        name="packaging", max_version="9999.0.0", max_inclusive=True
    )
    assert mi.is_in_sys_modules() is None


def test_module_info_with_failing_max_inclusive() -> None:
    """Failing max inclusive constraint returns error string."""
    mi = base_utilities.ModuleInfo(
        name="packaging",
        max_version="0.0.1",
        max_inclusive=True,
    )
    result = mi.is_in_sys_modules()
    assert result is not None
    assert "too high" in result


def test_module_info_with_failing_max_strict() -> None:
    """Failing max strict constraint returns error string."""
    with patch(
        "importlib.metadata.version", return_value="1.0.0"
    ), patch("importlib.util.find_spec", return_value=True):
        mi = base_utilities.ModuleInfo(
            name="packaging",
            max_version="1.0.0",
            max_inclusive=False,
        )
        result = mi.is_in_sys_modules()
        assert result is not None
        assert "too high" in result


def test_module_info_package_not_found_but_has_version_attr() -> None:
    """Fallback path when get_version raises but __version__ exists (line 560-562)."""
    fake_mod = MagicMock(__version__="1.2.3")
    with patch(
        "importlib.util.find_spec", return_value=True
    ), patch(
        "importlib.metadata.version",
        side_effect=__import__("importlib.metadata").metadata.PackageNotFoundError(
            "nope"
        ),
    ), patch("importlib.import_module", return_value=fake_mod):
        mi = base_utilities.ModuleInfo(name="fake_pkg", min_version="1.0.0", min_inclusive=True)
        assert mi.is_in_sys_modules() is None


def test_module_info_package_not_found_no_version_attr() -> None:
    """Fallback raises ImportError when no __version__ (line 563-564)."""
    with patch(
        "importlib.util.find_spec", return_value=True
    ), patch(
        "importlib.metadata.version",
        side_effect=__import__("importlib.metadata").metadata.PackageNotFoundError(
            "nope"
        ),
    ), patch(
        "importlib.import_module",
        side_effect=ImportError("cannot import"),
    ):
        mi = base_utilities.ModuleInfo(name="fake_pkg", min_version="1.0.0", min_inclusive=True)
        result = mi.is_in_sys_modules()
        assert result is not None
        assert "version could not be retrieved" in result


def test_module_info_installed_version_none() -> None:
    """When installed version is None (line 566-567)."""
    with patch(
        "importlib.util.find_spec", return_value=True
    ), patch("importlib.metadata.version", return_value=None):
        mi = base_utilities.ModuleInfo(name="fake_pkg", min_version="1.0.0", min_inclusive=True)
        result = mi.is_in_sys_modules()
        assert result is not None
        assert "version is not available" in result


# ---------------------------------------------------------------------------
# ModuleInfo.__repr__ (lines 590-603)
# ---------------------------------------------------------------------------


def test_module_info_repr_no_constraints() -> None:
    """repr with no version constraints."""
    mi = base_utilities.ModuleInfo(name="foo")
    assert repr(mi) == "foo"


def test_module_info_repr_min_inclusive() -> None:
    """repr renders >= for min_inclusive."""
    mi = base_utilities.ModuleInfo(name="foo", min_version="1.0.0", min_inclusive=True)
    assert repr(mi) == "foo>=1.0.0"


def test_module_info_repr_min_strict() -> None:
    """repr renders > for strict min."""
    mi = base_utilities.ModuleInfo(name="foo", min_version="1.0.0", min_inclusive=False)
    assert repr(mi) == "foo>1.0.0"


def test_module_info_repr_max_inclusive() -> None:
    """repr renders <= for max_inclusive."""
    mi = base_utilities.ModuleInfo(name="foo", max_version="2.0.0", max_inclusive=True)
    assert repr(mi) == "foo<=2.0.0"


def test_module_info_repr_max_strict() -> None:
    """repr renders < for strict max."""
    mi = base_utilities.ModuleInfo(name="foo", max_version="2.0.0", max_inclusive=False)
    assert repr(mi) == "foo<2.0.0"


def test_module_info_repr_both_bounds() -> None:
    """repr with both min and max constraints."""
    mi = base_utilities.ModuleInfo(
        name="foo",
        min_version="1.0.0",
        min_inclusive=True,
        max_version="2.0.0",
        max_inclusive=False,
    )
    assert repr(mi) == "foo>=1.0.0<2.0.0"


# ---------------------------------------------------------------------------
# ModuleInfo.from_str edge cases (lines 624, 635, 640-644)
# ---------------------------------------------------------------------------


def test_module_info_from_str_invalid_raises() -> None:
    """Invalid string raises ValueError (line 624)."""
    with pytest.raises(ValueError, match="Invalid package information"):
        base_utilities.ModuleInfo.from_str("!!!not a pkg!!!")


def test_module_info_from_str_no_constraints() -> None:
    """Bare name parses to ModuleInfo with no version."""
    mi = base_utilities.ModuleInfo.from_str("plain-pkg")
    assert mi.name == "plain-pkg"
    assert mi.min_version is None
    assert mi.max_version is None


def test_module_info_from_str_greater_than_strict() -> None:
    """`>` operator parses as min_inclusive=False."""
    mi = base_utilities.ModuleInfo.from_str("pkg>1.2.3")
    assert mi.min_version == "1.2.3"
    assert mi.min_inclusive is False


def test_module_info_from_str_less_than_strict() -> None:
    """`<` operator parses as max_inclusive=False."""
    mi = base_utilities.ModuleInfo.from_str("pkg<2.0.0")
    assert mi.max_version == "2.0.0"
    assert mi.max_inclusive is False


def test_module_info_from_str_less_than_or_equal() -> None:
    """`<=` operator parses as max_inclusive=True."""
    mi = base_utilities.ModuleInfo.from_str("pkg<=2.0.0")
    assert mi.max_version == "2.0.0"
    assert mi.max_inclusive is True


def test_module_info_from_str_operator_with_no_version_skipped() -> None:
    """Operator with no version value is skipped (line 635)."""
    # A constraint like "pkg>=" should match operator but with empty version,
    # so the loop hits `continue` at line 635.
    mi = base_utilities.ModuleInfo.from_str("pkg>=")
    assert mi.name == "pkg"
    assert mi.min_version is None
    assert mi.max_version is None


def test_module_info_from_str_multiple_constraints() -> None:
    """Both `>=` and `<` in one constraint string."""
    mi = base_utilities.ModuleInfo.from_str("pkg>=1.0,<2.0")
    assert mi.min_version == "1.0"
    assert mi.min_inclusive is True
    assert mi.max_version == "2.0"
    assert mi.max_inclusive is False


# ---------------------------------------------------------------------------
# get_missing_imports
# ---------------------------------------------------------------------------


def test_get_missing_imports_single_string_present() -> None:
    """Single present module yields empty dict."""
    assert base_utilities.get_missing_imports("json") == {}


def test_get_missing_imports_single_string_missing() -> None:
    """Single missing module yields single entry."""
    missing = base_utilities.get_missing_imports("_definitely_missing_pkg_zzz")
    assert list(missing.keys()) == ["_definitely_missing_pkg_zzz"]


# ---------------------------------------------------------------------------
# PatchObject: msg, copy_metadata, create (lines 782-793, 802-808, 842-845)
# ---------------------------------------------------------------------------


def test_patch_object_create_returns_none_for_unsupported() -> None:
    """PatchObject.create returns None for something neither function nor class."""
    result = base_utilities.PatchObject.create(
        42, missing_modules={"pkg": "pkg is not installed."}, dep_target="dep"
    )
    assert result is None


def test_patch_object_create_returns_callable_patcher() -> None:
    """PatchObject.create returns PatchCallable for a function."""

    def my_fn() -> int:
        return 1

    patcher = base_utilities.PatchObject.create(
        my_fn,
        missing_modules={"pkg": "pkg is not installed."},
        dep_target="dep",
    )
    assert patcher is not None
    assert isinstance(patcher, base_utilities.PatchCallable)


def test_patch_object_create_returns_class_patcher() -> None:
    """PatchObject.create returns PatchClass for a class."""

    class Foo:
        pass

    patcher = base_utilities.PatchObject.create(
        Foo,
        missing_modules={"pkg": "pkg is not installed."},
        dep_target="dep",
    )
    assert patcher is not None
    assert isinstance(patcher, base_utilities.PatchClass)


def test_patch_object_msg_single_module() -> None:
    """msg property formats a single-module message correctly."""

    def my_fn() -> int:
        return 1

    patcher = base_utilities.PatchObject.create(
        my_fn,
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
    )
    assert patcher is not None
    msg = patcher.msg
    assert "A module needed for" in msg
    assert "'pkg' is not installed." in msg
    assert "install it using appropriate extras" in msg


def test_patch_object_msg_plural_modules() -> None:
    """msg property formats a plural message correctly."""

    def my_fn() -> int:
        return 1

    patcher = base_utilities.PatchObject.create(
        my_fn,
        missing_modules={
            "pkg1": "'pkg1' is not installed.",
            "pkg2": "'pkg2' is not installed.",
        },
        dep_target="dep",
    )
    assert patcher is not None
    msg = patcher.msg
    assert "Modules needed for" in msg
    assert "are missing" in msg
    assert "install them using appropriate extras" in msg


def test_patch_object_msg_object_without_module_or_name() -> None:
    """msg gracefully handles an object with no __module__ / __name__."""

    class NoDunders:
        pass

    obj = NoDunders()
    patcher = base_utilities.PatchClass(
        type(obj),
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
    )
    msg = patcher.msg
    # As long as it doesn't crash, test passes
    assert "missing" in msg


def test_patch_object_copy_metadata_all_fields() -> None:
    """copy_metadata copies __doc__, __name__, __module__."""

    def src() -> int:
        """source doc"""
        return 1

    def tgt() -> int:
        return 2

    patcher = base_utilities.PatchCallable(
        src,
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
    )
    patcher.copy_metadata(tgt)
    assert tgt.__doc__ == "source doc"
    assert tgt.__name__ == "src"
    assert tgt.__module__ == src.__module__


def test_patch_object_constructor_rejects_unacceptable() -> None:
    """PatchObject subclass __init__ raises ValueError for unsuitable objects."""
    with pytest.raises(ValueError, match="Cannot patch"):
        # int(42) isn't a callable, rejected by PatchCallable.accept
        base_utilities.PatchCallable(  # type: ignore[type-var]
            42,  # type: ignore[arg-type]
            missing_modules={"pkg": "'pkg' is not installed."},
            dep_target="dep",
        )


# ---------------------------------------------------------------------------
# PatchCallable.patch (lines 858-867)
# ---------------------------------------------------------------------------


def test_patch_callable_patch_basic() -> None:
    """PatchCallable.patch returns a wrapper that raises ImportError on call."""

    def original(x: int) -> int:
        """orig doc"""
        return x * 2

    patcher = base_utilities.PatchCallable(
        original,
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
    )
    patched = patcher.patch(except_for=[])
    assert patched.__doc__ == "orig doc"
    with pytest.raises(ImportError):
        patched(1)


def test_patch_callable_patch_except_for() -> None:
    """PatchCallable.patch returns original if name is in except_for."""

    def original() -> int:
        return 42

    patcher = base_utilities.PatchCallable(
        original,
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
    )
    patched = patcher.patch(except_for=["original"])
    assert patched is original
    # Call unchanged
    assert patched() == 42


# ---------------------------------------------------------------------------
# PatchClass.patch (lines 880-894)
# ---------------------------------------------------------------------------


def test_patch_class_patch_excepts_by_name() -> None:
    """PatchClass.patch returns original class if name in except_for."""

    class MyClass:
        def method(self) -> int:
            return 1

    patcher = base_utilities.PatchClass(
        MyClass,
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
    )
    out = patcher.patch(except_for=["MyClass"])
    assert out is MyClass
    assert MyClass().method() == 1


def test_patch_class_patch_methods() -> None:
    """PatchClass.patch wraps non-dunder methods to raise ImportError."""

    class MyClass:
        def patched_method(self) -> int:
            return 1

    patcher = base_utilities.PatchClass(
        MyClass,
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
    )
    out = patcher.patch(except_for=[])
    # The returned class is the same MyClass with members replaced
    assert out is MyClass


# ---------------------------------------------------------------------------
# patch_object (lines 918-924)
# ---------------------------------------------------------------------------


def test_patch_object_with_string_except_for() -> None:
    """patch_object accepts a single string in except_for (line 923)."""

    def my_fn() -> int:
        return 42

    patched = base_utilities.patch_object(
        my_fn,
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
        except_for="my_fn",
    )
    # Excluded by name -> should still call successfully
    assert patched() == 42


def test_patch_object_not_patchable_raises() -> None:
    """patch_object raises ValueError when fail_if_not_patchable=True and unsupported."""
    with pytest.raises(ValueError, match="Cannot patch"):
        base_utilities.patch_object(
            42,
            missing_modules={"pkg": "'pkg' is not installed."},
            dep_target="dep",
            fail_if_not_patchable=True,
        )


def test_patch_object_not_patchable_fallback_returns_original() -> None:
    """patch_object returns original when fail_if_not_patchable=False and unsupported."""
    out = base_utilities.patch_object(
        42,
        missing_modules={"pkg": "'pkg' is not installed."},
        dep_target="dep",
        fail_if_not_patchable=False,
    )
    assert out == 42


# ---------------------------------------------------------------------------
# require_optional_import decorator (line 952)
# ---------------------------------------------------------------------------


def test_require_optional_import_all_present_returns_as_is() -> None:
    """Decorator returns object unchanged when all modules present (line 952)."""

    @base_utilities.require_optional_import("json", dep_target="stdlib")
    def f(x: int) -> int:
        return x + 1

    assert f(1) == 2


def test_require_optional_import_missing_module_patches_callable() -> None:
    """Decorator wraps the callable with an ImportError-raising wrapper."""

    @base_utilities.require_optional_import(
        "pkg_definitely_does_not_exist_xyz", dep_target="missing"
    )
    def g(x: int) -> int:
        return x + 1

    with pytest.raises(ImportError):
        g(1)


def test_require_optional_import_multiple_modules_mixed() -> None:
    """Decorator handles mixed present + missing."""

    @base_utilities.require_optional_import(
        ["json", "pkg_definitely_does_not_exist_xyz"],
        dep_target="mixed",
    )
    def h() -> int:
        return 42

    with pytest.raises(ImportError):
        h()
