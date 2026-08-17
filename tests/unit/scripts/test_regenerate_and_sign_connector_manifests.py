"""Contracts for the controlled-release connector-manifest regenerate+sign orchestrator
(GOC-16/BUG-234 custody-path mechanism).

Proves the freeze-verification and built-artifact-verification gates fail closed on
each of the conditions the operator runbook (``docs/release/connector-manifest-
signing-custody.md``) describes, WITHOUT touching real git state, real OpenBao, or
real key material. Subprocess-level generator/sign behavior is exercised separately by
the generator scripts' own test suites (``test_generate_connector_manifests.py``,
``test_generate_native_connector_manifest.py``) and by
``test_connector_manifest_signing_known_bad.py``; this file is scoped to the
orchestrator's own control flow.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "regenerate_and_sign_connector_manifests",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "release"
    / "regenerate_and_sign_connector_manifests.py",
)
orch = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(orch)


class TestVerifyFreeze:
    def test_clean_tree_and_matching_sha_has_no_problems(self, monkeypatch):
        monkeypatch.setattr(orch, "_git", lambda *a: "" if a[0] == "status" else "abc123")
        problems = orch.verify_freeze(frozen_sha="abc123", expected_lock_digest=None)
        assert problems == []

    def test_dirty_tree_is_a_problem(self, monkeypatch):
        monkeypatch.setattr(
            orch, "_git", lambda *a: "M some/file.py" if a[0] == "status" else "abc123"
        )
        problems = orch.verify_freeze(frozen_sha=None, expected_lock_digest=None)
        assert any("not clean" in p for p in problems)

    def test_sha_mismatch_is_a_problem(self, monkeypatch):
        monkeypatch.setattr(orch, "_git", lambda *a: "" if a[0] == "status" else "def456")
        problems = orch.verify_freeze(frozen_sha="abc123", expected_lock_digest=None)
        assert any("!= expected frozen commit" in p for p in problems)

    def test_dependency_lock_drift_is_a_problem(self, monkeypatch):
        monkeypatch.setattr(orch, "_git", lambda *a: "" if a[0] == "status" else "abc123")
        monkeypatch.setattr(
            "agent_utilities.knowledge_graph.ontology.ontology_integrity.dependency_lock_digest",
            lambda: "f" * 64,
        )
        problems = orch.verify_freeze(
            frozen_sha="abc123", expected_lock_digest="0" * 64
        )
        assert any("dependency lock moved" in p for p in problems)

    def test_unreadable_lock_is_a_problem_not_a_crash(self, monkeypatch):
        from agent_utilities.knowledge_graph.ontology import ontology_integrity

        monkeypatch.setattr(orch, "_git", lambda *a: "" if a[0] == "status" else "abc123")

        def _raise():
            raise ontology_integrity.ReleaseSigningError("dependency lock is unreadable")

        monkeypatch.setattr(
            "agent_utilities.knowledge_graph.ontology.ontology_integrity.dependency_lock_digest",
            _raise,
        )
        problems = orch.verify_freeze(
            frozen_sha="abc123", expected_lock_digest="0" * 64
        )
        assert any("could not be read" in p for p in problems)


class TestVerifyBuiltArtifact:
    def test_missing_distribution_is_a_problem(self, monkeypatch):
        import importlib.metadata as metadata

        def _raise(_name):
            raise metadata.PackageNotFoundError

        monkeypatch.setattr(metadata, "distribution", _raise)
        problems = orch.verify_built_artifact()
        assert any("not installed as a distribution" in p for p in problems)

    def test_editable_install_is_refused(self, monkeypatch):
        import importlib.metadata as metadata

        fake_dist = mock.Mock()
        fake_dist.read_text.return_value = json.dumps(
            {"dir_info": {"editable": True}, "url": "file:///repo"}
        )
        monkeypatch.setattr(metadata, "distribution", lambda _name: fake_dist)
        problems = orch.verify_built_artifact()
        assert any("EDITABLE" in p for p in problems)

    def test_non_editable_install_passes(self, monkeypatch):
        import importlib.metadata as metadata

        fake_dist = mock.Mock()
        fake_dist.read_text.return_value = None
        monkeypatch.setattr(metadata, "distribution", lambda _name: fake_dist)
        problems = orch.verify_built_artifact()
        assert problems == []


class TestMainRefusesOnFreezeFailureBeforeRegenerating:
    def test_frozen_sha_mismatch_exits_nonzero_without_calling_regenerate(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(orch, "_git", lambda *a: "" if a[0] == "status" else "wrong-sha")
        regenerate_called = mock.Mock()
        monkeypatch.setattr(orch, "regenerate", regenerate_called)
        monkeypatch.setattr(
            "sys.argv",
            [
                "regenerate_and_sign_connector_manifests.py",
                "--frozen-sha",
                "expected-sha",
            ],
        )

        exit_code = orch.main()

        assert exit_code == 1
        regenerate_called.assert_not_called()
        out = json.loads(capsys.readouterr().out)
        assert out["freeze"]["ok"] is False

    def test_required_built_artifact_failure_exits_nonzero_without_regenerating(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(orch, "_git", lambda *a: "" if a[0] == "status" else "sha")
        monkeypatch.setattr(
            orch, "verify_built_artifact", lambda: ["not a built artifact"]
        )
        regenerate_called = mock.Mock()
        monkeypatch.setattr(orch, "regenerate", regenerate_called)
        monkeypatch.setattr(
            "sys.argv",
            [
                "regenerate_and_sign_connector_manifests.py",
                "--frozen-sha",
                "sha",
                "--require-built-artifact",
            ],
        )

        exit_code = orch.main()

        assert exit_code == 1
        regenerate_called.assert_not_called()
        out = json.loads(capsys.readouterr().out)
        assert out["built_artifact"]["ok"] is False
