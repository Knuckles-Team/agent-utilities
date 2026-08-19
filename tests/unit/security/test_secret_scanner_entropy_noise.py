"""Regression guard for ``check_secret_history``'s entropy noise filter.

The scanner's ``_ENTROPY_TOKEN_RE`` admits ``.``, so a Sphinx cross-reference
such as ``agent_utilities.security.system_rbac_admission.ensure_...`` scores as
one high-entropy token and every docstring reads as a credential. Whitelisting
"dotted path" naively is a real weakening, because a JWT is *also* a dotted
token whose segments are alphanumeric -- a planted-JWT canary caught that
regression while the filter was written. These tests pin BOTH halves: the
false positives stay suppressed, and real credential shapes stay detected.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCANNER = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "security"
    / "check_secret_history.py"
)


def _synthetic_value(*parts: str) -> str:
    """Keep credential-shape canaries exact at runtime without source literals."""
    return "".join(parts)


def _load():
    spec = importlib.util.spec_from_file_location("_check_secret_history", _SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_check_secret_history"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "token",
    [
        "agent_utilities.security.system_rbac_admission.ensure_system_principal_access",
        "agent_webui.graph_admission.ensure_tenant_admission",
        "agent_utilities.security.SystemAdmissionError",
        "agent_utilities.security.CONTROL_ROLE_NAME",
    ],
)
def test_dotted_source_paths_are_noise(token: str) -> None:
    assert _load()._is_entropy_noise(token) is True


@pytest.mark.parametrize(
    "token",
    [
        # A JWT is dotted too -- this is the canary that caught a real weakening.
        _synthetic_value(
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            ".",
            "eyJzdWIiOiIxMjM0NQ",
        ),
        _synthetic_value(
            "eyJhbGciOiJIUzI1NiJ9",
            ".",
            "eyJzdWIiOiJhIn0",
            ".",
            "dBjftJeZ4CVPmB92K",
        ),
        _synthetic_value(
            "AKIA",
            "IOSFODNN7EXAMPLE",
            "/",
            "wJalrXUtnX",
            "tnFEMI+",
            "K7MDENG",
        ),
        _synthetic_value(
            "ghp_",
            "1a2B3c4D5e6F7g8H9i0JkLmNoPqRsTuVwXyZ==",
        ),
        _synthetic_value(
            "xoxb-",
            "1234567890-",
            "abcdefghijklmnopqrst",
            ".",
            "uvwxyz",
        ),
    ],
)
def test_real_credential_shapes_are_never_noise(token: str) -> None:
    assert _load()._is_entropy_noise(token) is False
