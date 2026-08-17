"""BUG-241: bare-hostname detection in ``check_tracked_privacy.py``.

The runtime-source internal-endpoint pass (``classify_runtime_source_line``)
used to require a URL scheme before the host, so a bare hostname literal
with no scheme prefix at all was structurally invisible to it. Proven live
with a planted two-line canary in a scanned file: a schemed URL form was
caught while an otherwise-identical bare-hostname form on the adjacent line
was not.

These tests drive the gate's own classification function directly (the same
approach ``tests/gates/test_wheel_privacy_gate.py`` uses for its gate), and
must cover BOTH the newly-fixed detection AND the RFC-reserved/labelled-fake
composition it must not re-flood past (BUG-228's ~130-false-positive
regression is exactly what re-lands if that composition breaks).

Every non-reserved "bad" hostname fixture below is built at runtime via
string concatenation rather than written as one matchable source literal --
the pattern this gate's own ``_is_runtime_source_path`` docstring recommends
for a fixture that must stay leak-shaped on purpose. Without that, this file
would itself become a new tracked-source finding under its own gate the
moment it is committed (self-referentially proven while writing it).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _gate_module() -> ModuleType:
    source = Path(__file__).parents[2] / "scripts" / "check_tracked_privacy.py"
    spec = importlib.util.spec_from_file_location("check_tracked_privacy", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The gate module defines a frozen @dataclass (Violation); dataclasses'
    # own string-annotation resolution looks the module up via
    # sys.modules[cls.__module__], so it must be registered before
    # exec_module runs the class body, or that lookup raises AttributeError
    # on None.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flags_internal_endpoint(gate: ModuleType, line: str) -> bool:
    categories = gate.classify_runtime_source_line(line, identifiers=frozenset())
    return any("internal endpoint" in category for category in categories)


def _non_reserved_host(*labels: str, suffix: str) -> str:
    """Build a real-shaped, non-reserved hostname without a matchable literal."""
    return ".".join(labels) + "." + suffix


# --------------------------------------------------------------------------- #
# The four known-bad/known-good cases the bug report requires proof for.
# --------------------------------------------------------------------------- #


def test_bare_internal_hostname_fails() -> None:
    gate = _gate_module()
    host = _non_reserved_host("real-host", suffix="arpa")
    assert _flags_internal_endpoint(gate, f'BARE = "{host}"') is True


def test_schemed_internal_hostname_fails() -> None:
    gate = _gate_module()
    host = _non_reserved_host("real-host", suffix="arpa")
    assert _flags_internal_endpoint(gate, f'ENDPOINT = "http://{host}"') is True


def test_rfc_reserved_documentation_domain_passes() -> None:
    gate = _gate_module()
    assert (
        _flags_internal_endpoint(gate, 'CONTACT = "user@example.invalid"') is False
    )


def test_labelled_fake_internal_hostname_passes() -> None:
    gate = _gate_module()
    assert (
        _flags_internal_endpoint(
            gate, '_FAKE_HOST_IDENTITY = "example-host-prod.internal.arpa"'
        )
        is False
    )


# --------------------------------------------------------------------------- #
# The two-line canary from the bug report, reproduced as a regression test.
# --------------------------------------------------------------------------- #


def test_planted_canary_both_forms_are_caught() -> None:
    gate = _gate_module()
    host = _non_reserved_host("canary-service", suffix="arpa")
    schemed = f'SCHEMED = "http://{host}:8080/v1"'
    bare = f'BARE = "{host}"'
    assert _flags_internal_endpoint(gate, schemed) is True
    assert _flags_internal_endpoint(gate, bare) is True


# --------------------------------------------------------------------------- #
# False-positive guard: a Python enum-member attribute access shaped like
# ``label.SUFFIX`` (found live in this corpus as a DataClassification enum
# member reference, 18 occurrences across the runtime tree) is not a
# hostname. Detection is case-sensitive on the fixed suffix word
# specifically to exclude this shape without a Python-syntax-aware parse --
# every real/synthetic hostname literal in this corpus is written lowercase,
# while enum members are UPPER_SNAKE_CASE by convention.
# --------------------------------------------------------------------------- #


def test_enum_member_access_is_not_flagged() -> None:
    gate = _gate_module()
    line = "classification: DataClassification = DataClassification.INTERNAL"
    assert _flags_internal_endpoint(gate, line) is False


def test_svc_cluster_local_bare_and_schemed_both_fail() -> None:
    gate = _gate_module()
    # The Kubernetes internal-DNS suffix itself matches with ZERO preceding
    # labels (same as the docs-path pattern), so even the bare suffix word
    # must be built at runtime here, not written as one matchable literal.
    svc_suffix = ".".join(("svc", "cluster", "local"))
    host = _non_reserved_host("coordinator", "cell", suffix=svc_suffix)
    assert _flags_internal_endpoint(gate, f'HOST = "{host}"') is True
    assert _flags_internal_endpoint(gate, f'ENDPOINT = "http://{host}:8080"') is True


def test_full_corpus_scan_is_clean() -> None:
    """The whole tracked tree must scan clean against the frozen baseline.

    Mirrors ``main()``'s own new-vs-baseline diff without invoking the CLI,
    so a regression here fails as a normal pytest assertion (with the
    offending findings in the message) instead of only showing up as a
    pre-commit/CI gate failure.
    """
    gate = _gate_module()
    violations = gate.scan()
    baseline = gate._load_baseline()
    new = [v for v in violations if gate._baseline_key(v) not in baseline]
    assert new == [], [v.render() for v in new]
