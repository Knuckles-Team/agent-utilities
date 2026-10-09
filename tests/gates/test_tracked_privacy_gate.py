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
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from scripts._git_subprocess_env import sanitized_git_env


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


def _synthetic_identity(term: str) -> tuple[bytes, ...]:
    return (term.casefold().encode("ascii"),)


def test_model_specific_identity_catalog_is_external_and_versioned(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    catalog = tmp_path / "identity-policy.json"
    catalog.write_text(
        json.dumps({"version": "fixture-v1", "identities": ["deny"]}),
        encoding="utf-8",
    )

    assert gate.load_identity_catalog(catalog) == (b"deny",)


def test_absent_identity_policy_skips_that_pass_locally_and_fails_in_ci(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    gate = _gate_module()
    absent = tmp_path / "absent.json"

    monkeypatch.delenv("CI", raising=False)
    assert gate._required_identity_catalog(absent) == ()
    assert "SKIPPED (tracked-privacy identity pass)" in capsys.readouterr().out

    monkeypatch.setenv("CI", "true")
    try:
        gate._required_identity_catalog(absent)
    except SystemExit as exit_:
        assert exit_.code == 2
    else:
        raise AssertionError("an absent policy must fail closed under CI")


def test_malformed_identity_policy_still_fails_locally(tmp_path: Path, monkeypatch) -> None:
    gate = _gate_module()
    catalog = tmp_path / "identity-policy.json"
    catalog.write_text("not json", encoding="utf-8")
    monkeypatch.delenv("CI", raising=False)
    try:
        gate._required_identity_catalog(catalog)
    except SystemExit as exit_:
        assert exit_.code == 2
    else:
        raise AssertionError("a malformed policy must never be skipped")


def test_model_specific_identity_rejects_embedded_case_variants(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    source = tmp_path / "arbitrary.data"
    variants = (
        "deny",
        "DENY",
        "dEnY",
        "denys",
        "predeny",
        "denypost",
        "pre_deny_post",
        "pre-deny-post",
        "de-ny",
        "de_ny",
        "9deny2",
        "myDenyAgent",
    )
    source.write_text("\n".join(variants), encoding="utf-8")

    findings = [
        finding
        for finding in gate._prohibited_identity_violations(
            tmp_path, {}, _synthetic_identity("deny")
        )
        if finding.category == "model-specific identity in tracked artifact"
    ]

    assert [(finding.path, finding.line) for finding in findings] == [
        ("arbitrary.data", line) for line in range(1, len(variants) + 1)
    ]


def test_model_specific_identity_allows_cross_camel_boundary_coincidence(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    source = tmp_path / "arbitrary.data"
    source.write_text("ModeNylon\nMode_Nylon\n", encoding="utf-8")

    findings = gate._prohibited_identity_violations(
        tmp_path, {}, _synthetic_identity("deny")
    )

    assert findings == []


def test_model_specific_identity_rejects_tracked_path(tmp_path: Path) -> None:
    gate = _gate_module()
    source = tmp_path / "role-deny-worker.data"
    source.write_text("neutral content\n", encoding="utf-8")

    findings = gate._prohibited_identity_violations(
        tmp_path, {}, _synthetic_identity("deny")
    )

    assert [(finding.path, finding.line) for finding in findings] == [
        ("role-deny-worker.data", 0)
    ]


def test_model_specific_identity_scan_uses_bounded_stream_reads() -> None:
    gate = _gate_module()
    scanner = sys.modules[gate.scan_prohibited_identities.__module__]

    class GuardedStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert 0 < size <= scanner.READ_SIZE
            return super().read(size)

    prefix = b" " * (scanner.READ_SIZE - 2)
    findings = list(
        scanner._matching_stream_groups(
            GuardedStream(prefix + b"de-ny"), _synthetic_identity("deny")
        )
    )

    assert findings == [(1, b"de-ny")]


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
    assert _flags_internal_endpoint(gate, 'CONTACT = "user@example.invalid"') is False


def test_labelled_fake_internal_hostname_passes() -> None:
    gate = _gate_module()
    assert (
        _flags_internal_endpoint(
            gate, '_FAKE_HOST_IDENTITY = "example-host-prod.internal.arpa"'
        )
        is False
    )


# --------------------------------------------------------------------------- #
# D-W12-AU-EXCEPTIONS-3: "someone" was a stray, undocumented entry in
# ``_RESERVED_HOME_USERS`` (BUG-228, ``ee3814af7``) -- not one of that set's
# own documented categories (generic role noun / "example" family /
# alice-bob personas / ``*-account`` idiom / single-letter stand-in) -- which
# silently hid tests/gates/test_docs_contract_gate.py's positive
# "/home/" + "someone" detection fixture. Both directions must hold: a
# generic, non-reserved home-path username is still DETECTED (the fix), and
# a genuinely-documented reserved placeholder is still NOT flagged (BUG-228's
# ~130-false-positive flood must not re-land, per that fix's own lesson).
# --------------------------------------------------------------------------- #


def _flags_home_path(gate: ModuleType, line: str) -> bool:
    categories = gate.classify_runtime_source_line(line, identifiers=frozenset())
    return any("machine-specific home path" in category for category in categories)


def test_generic_home_username_is_detected_not_reserved() -> None:
    """The exact regression: a home path under an arbitrary, non-reserved
    username must be flagged. Built via runtime concatenation, not one
    matchable source literal, so this test file does not itself become a
    tracked-source finding the moment it is committed (the same
    self-referential trap ``test_docs_contract_gate.py``'s sibling fixture
    documents)."""
    gate = _gate_module()
    home_user = "some" + "one"
    assert _flags_home_path(gate, f"path: /home/{home_user}/state/tree") is True


def test_documented_reserved_home_username_still_passes() -> None:
    """A genuinely-documented reserved placeholder (the RFC 2606 "example"
    word, explicitly covered by ``_RESERVED_HOME_USERS``'s own docstring)
    must stay exempt -- proves the fix narrowed the stray entry only, not the
    whole reserved-placeholder mechanism BUG-228 relies on."""
    gate = _gate_module()
    assert _flags_home_path(gate, "path: /home/example/state/tree") is False


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


# --------------------------------------------------------------------------- #
# EH-467: a checkout configured with the canonical Claude/Codex commit
# identity (operator ruling, plans/refactor/DECISIONS.md 2026-09-24) must not
# turn every ordinary "Claude"/"Codex" mention in tracked prose into a
# manufactured local-identifier leak.
# --------------------------------------------------------------------------- #


def _init_repo(root: Path, *, name: str, email: str) -> None:
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", name],
        ["git", "config", "user.email", email],
        ["git", "config", "commit.gpgsign", "false"],
    ):
        subprocess.run(args, cwd=root, check=True, capture_output=True, env=sanitized_git_env())


def test_derive_local_identifiers_excludes_the_canonical_agent_identities(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    _init_repo(tmp_path, name="Claude", email="noreply@anthropic.com")

    identifiers = gate.derive_local_identifiers(tmp_path)

    assert "claude" not in identifiers
    assert "noreply@anthropic.com" not in identifiers


def test_privacy_gate_does_not_flag_an_ambient_canonical_identity_mention(
    tmp_path: Path,
) -> None:
    """Committing as ``user.name=Claude`` must not flood every "Claude" mention."""
    gate = _gate_module()
    _init_repo(tmp_path, name="Claude", email="noreply@anthropic.com")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text(
        "Claude reviewed and approved this change.\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", "docs/notes.md"],
        cwd=tmp_path, check=True, capture_output=True, env=sanitized_git_env(),
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "add notes"],
        cwd=tmp_path, check=True, capture_output=True, env=sanitized_git_env(),
    )

    violations = gate.scan(tmp_path)

    assert violations == [], [v.render() for v in violations]


def test_full_corpus_scan_is_clean() -> None:
    """The whole tracked tree must scan clean against the absolute ``MAX``.

    CX-RAT-09: the baseline/ratchet mechanism this test used to diff against
    is deleted -- a count-based allowance is the wrong instrument for a
    leak-prevention gate on a repo that publishes to a public GitHub org (see
    ``scripts/check_tracked_privacy.py``'s module-level comment). Mirrors
    ``main()``'s own MAX comparison without invoking the CLI, so a
    regression here fails as a normal pytest assertion (with the offending
    findings in the message) instead of only showing up as a pre-commit/CI
    gate failure.
    """
    gate = _gate_module()
    violations = gate.scan()
    assert len(violations) <= gate.MAX, [v.render() for v in violations]
