"""Behavioral contracts for the exact local release promoter."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts.release import promote_local_release as promoter

_DIGEST = "sha256:" + "1" * 64
_RAW_DIGEST = "2" * 64


def _record_digest(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _release_wheel(
    root: Path, *, duplicate_top_level_metadata: bool = False
) -> tuple[Path, str]:
    path = root / "agent_utilities-1.0.0-py3-none-any.whl"
    dist_info = "agent_utilities-1.0.0.dist-info"
    vendored_dist_info = "agent_utilities/_vendor/dependency-2.0.dist-info"
    members = {
        "agent_utilities/__init__.py": b"",
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.1\nName: agent-utilities\nVersion: 1.0.0\n\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
        ),
        f"{vendored_dist_info}/METADATA": (
            b"Metadata-Version: 2.1\nName: dependency\nVersion: 2.0\n\n"
        ),
        f"{vendored_dist_info}/WHEEL": b"vendored wheel metadata\n",
        f"{vendored_dist_info}/RECORD": b"vendored record metadata\n",
    }
    if duplicate_top_level_metadata:
        members["duplicate-1.0.0.dist-info/METADATA"] = (
            b"Metadata-Version: 2.1\nName: duplicate\nVersion: 1.0.0\n\n"
        )
    record_lines = [
        f"{name},sha256={_record_digest(payload)},{len(payload)}"
        for name, payload in members.items()
    ]
    record_name = f"{dist_info}/RECORD"
    members[record_name] = (
        "\n".join((*record_lines, f"{record_name},,")) + "\n"
    ).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


@pytest.fixture
def wheel_with_vendored_dist_info(tmp_path: Path) -> tuple[Path, str]:
    return _release_wheel(tmp_path)


@pytest.fixture
def wheel_with_duplicate_top_level_metadata(tmp_path: Path) -> tuple[Path, str]:
    return _release_wheel(tmp_path, duplicate_top_level_metadata=True)


def _spec() -> promoter.ReleaseSpec:
    packages = {
        name: promoter.PackagePin(name, "1.0.0", f"{name}-1.0.0.whl", _DIGEST)
        for name in promoter._REQUIRED_PACKAGES
    }
    commands = {
        "canary": promoter.CommandSpec("graph-os-release-canary", ("--json",), 30),
        "doctor": promoter.CommandSpec(
            "agent-utilities-doctor",
            (
                "--json",
                "--live",
                "--only",
                *promoter._DOCTOR_CHECKS,
            ),
            120,
        ),
    }
    return promoter.ReleaseSpec(
        release_id="release-test",
        requirements_file="release-requirements.txt",
        requirements_digest=_DIGEST,
        packages=packages,
        native_artifacts={name: _DIGEST for name in promoter._NATIVE_ARTIFACTS},
        toolchain={
            "python": promoter.ToolPin("3.12.0", _DIGEST),
            "uv": promoter.ToolPin("1.0.0", _DIGEST),
        },
        commands=commands,
        digest=_DIGEST,
    )


def test_wheel_inspection_ignores_nested_vendored_dist_info(
    wheel_with_vendored_dist_info: tuple[Path, str],
) -> None:
    path, digest = wheel_with_vendored_dist_info

    name, version, record, _count, _size, _scripts = promoter._inspect_wheel(
        path, digest=digest
    )

    assert (name, version) == ("agent-utilities", "1.0.0")
    assert "agent_utilities/_vendor/dependency-2.0.dist-info/METADATA" in record
    assert "agent_utilities/_vendor/dependency-2.0.dist-info/WHEEL" in record
    assert "agent_utilities/_vendor/dependency-2.0.dist-info/RECORD" in record


def test_wheel_inspection_rejects_duplicate_top_level_metadata(
    wheel_with_duplicate_top_level_metadata: tuple[Path, str],
) -> None:
    path, digest = wheel_with_duplicate_top_level_metadata

    with pytest.raises(promoter.ReleaseError, match="incomplete-wheel-metadata"):
        promoter._inspect_wheel(path, digest=digest)


def _promoted_evidence(spec: promoter.ReleaseSpec) -> dict[str, Any]:
    evidence = promoter._base_evidence(spec)
    evidence.update(
        {
            "status": "promoted",
            "errorCode": None,
            "certificationArtifacts": {
                "agentUtilitiesSha256": _RAW_DIGEST,
                "agentUtilitiesFileCount": 20,
                "distributionClosureSha256": _RAW_DIGEST,
                "releasePythonSha256": _RAW_DIGEST,
                "graphosSha256": _RAW_DIGEST,
                "engineSha256": _RAW_DIGEST,
            },
        }
    )
    evidence["closure"].update(
        {
            "distributionCount": 3,
            "recordVerified": True,
            "nativeArtifactCount": 2,
            "dependencyEdgeCount": 2,
            "releaseTreeEntryCount": 20,
            "immutableAfterProof": True,
        }
    )
    evidence["processGate"] = {"beforePromotion": 0, "afterVerification": 0}
    evidence["commands"] = {
        role: {
            "status": "passed",
            "exitCode": 0,
            "outputDigest": _DIGEST,
        }
        for role in ("venv", "install", "canary", "doctor")
    }
    return evidence


def _external_signing_stub(name: str, payload: bytes) -> dict[str, Any]:
    if name == promoter._SIGNER_ENV:
        return {
            "algorithm": "ed25519",
            "keyId": "key:" + "3" * 64,
            "signature": "A" * 86,
            "subjectDigest": promoter._sha256(payload),
        }
    signed = json.loads(payload)
    return {
        "verified": True,
        "subjectDigest": signed["signature"]["subjectDigest"],
        "keyId": signed["signature"]["keyId"],
    }


def test_installed_release_attestor_recomputes_every_promotion_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_root = (tmp_path / "release").absolute()
    site_packages = release_root / "runtime" / "site-packages"
    distribution = site_packages / "agent_utilities-1.0.0.dist-info"
    distribution.mkdir(parents=True)
    metadata = distribution / "METADATA"
    metadata.write_text("Name: agent-utilities\nVersion: 1.0.0\n", encoding="utf-8")
    metadata_resolved = metadata.resolve(strict=True)
    seal_checks: list[Path] = []
    installed_record = {"agent_utilities/__init__.py": ("digest", 1)}
    expected = {
        "agentUtilitiesSha256": "1" * 64,
        "agentUtilitiesFileCount": 10,
        "distributionClosureSha256": "2" * 64,
        "releasePythonSha256": "3" * 64,
        "graphosSha256": "4" * 64,
        "engineSha256": "5" * 64,
    }

    monkeypatch.setattr(
        promoter, "_verify_release_sealed", lambda root: seal_checks.append(root)
    )
    monkeypatch.setattr(promoter, "_scan_regular_tree", lambda root: {metadata})
    monkeypatch.setattr(promoter, "_site_packages", lambda runtime: site_packages)
    monkeypatch.setattr(
        promoter,
        "_verify_record",
        lambda candidate, *, site_packages, release_root: (
            {metadata_resolved},
            installed_record,
        ),
    )
    monkeypatch.setattr(
        promoter,
        "_installed_agent_tree_identity",
        lambda **kwargs: ("1" * 64, 10),
    )
    monkeypatch.setattr(
        promoter, "_installed_closure_identity", lambda **kwargs: "2" * 64
    )

    def certify(
        root: Path,
        *,
        agent_utilities_sha256: str,
        agent_utilities_file_count: int,
        distribution_closure_sha256: str,
    ) -> dict[str, Any]:
        assert root == release_root
        assert agent_utilities_sha256 == "1" * 64
        assert agent_utilities_file_count == 10
        assert distribution_closure_sha256 == "2" * 64
        return expected

    monkeypatch.setattr(promoter, "_certification_artifacts", certify)

    assert promoter.attest_installed_release(release_root) == expected
    assert seal_checks == [release_root, release_root]


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda value: value["closure"].update(distributionCount=0),
            "promoted-evidence-invariant-failed",
        ),
        (
            lambda value: value["commands"]["doctor"].update(status="failed"),
            "promoted-evidence-invariant-failed",
        ),
        (
            lambda value: value["activation"].update(rollback="completed"),
            "evidence-activation-invalid",
        ),
    ],
)
def test_promoted_evidence_rejects_false_positive_states(mutation, code) -> None:
    spec = _spec()
    evidence = _promoted_evidence(spec)
    mutation(evidence)

    with pytest.raises(promoter.ReleaseError, match=code):
        promoter._validate_evidence_semantics(evidence, spec)


def test_evidence_publication_is_signed_private_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    destination = parent / "promotion.json"
    spec = _spec()
    monkeypatch.setattr(promoter, "_external_json", _external_signing_stub)
    signed = promoter._sign_evidence(_promoted_evidence(spec), spec)

    promoter._write_evidence(destination, signed, spec=spec)
    original = destination.read_bytes()

    with pytest.raises(promoter.ReleaseError, match="evidence-destination-must-be-new"):
        promoter._write_evidence(destination, signed, spec=spec)
    assert destination.read_bytes() == original
    assert os.stat(destination).st_mode & 0o077 == 0


def test_evidence_parent_fsync_fault_preserves_committed_bytes_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir(mode=0o700)
    destination = parent / "promotion.json"
    spec = _spec()
    monkeypatch.setattr(promoter, "_external_json", _external_signing_stub)
    signed = promoter._sign_evidence(_promoted_evidence(spec), spec)
    real_fsync = promoter.os.fsync
    calls = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected-parent-fsync-fault")
        real_fsync(descriptor)

    monkeypatch.setattr(promoter.os, "fsync", fail_parent_fsync)

    with pytest.raises(promoter.EvidencePublicationUncertain):
        promoter._write_evidence(destination, signed, spec=spec)
    assert json.loads(destination.read_text(encoding="utf-8")) == signed


def test_evidence_destination_rejects_release_tree_collision(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    wheelhouse = tmp_path / "wheelhouse"
    private = tmp_path / "private"
    for directory in (root, wheelhouse, private):
        directory.mkdir(mode=0o700)
    spec_path = private / "spec.json"
    spec_path.write_text("{}", encoding="utf-8")

    with pytest.raises(promoter.ReleaseError, match="evidence-input-collision"):
        promoter._validate_evidence_destination(
            root / "evidence.json",
            spec_path=spec_path,
            wheelhouse=wheelhouse,
            releases_root=root,
        )


def test_root_and_candidate_bindings_reject_path_replacement(tmp_path: Path) -> None:
    root = (tmp_path / "releases").absolute()
    root.mkdir(mode=0o700)
    candidate = root / "release-test"
    candidate.mkdir(mode=0o700)
    root_fd = promoter._open_releases_root(root)
    candidate_fd = promoter._open_candidate(root_fd, "release-test")
    try:
        parked_candidate = root / "release-parked"
        candidate.rename(parked_candidate)
        candidate.mkdir(mode=0o700)
        fd_root = promoter._fd_path(candidate_fd)
        (fd_root / "bound.txt").write_text("held-descriptor", encoding="ascii")
        assert (parked_candidate / "bound.txt").read_text(encoding="ascii") == (
            "held-descriptor"
        )
        assert not (candidate / "bound.txt").exists()
        with pytest.raises(promoter.ReleaseError, match="release-stage-changed"):
            promoter._assert_candidate_binding(root, "release-test", candidate_fd)

        candidate.rmdir()
        parked_candidate.rename(candidate)
        parked_root = tmp_path / "releases-parked"
        root.rename(parked_root)
        root.mkdir(mode=0o700)
        with pytest.raises(promoter.ReleaseError, match="releases-root-changed"):
            promoter._assert_root_binding(root, root_fd)
    finally:
        os.close(candidate_fd)
        os.close(root_fd)


def test_external_json_rejects_oversized_signer_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "external-signer"
    executable.write_bytes(b"signer")
    executable.chmod(0o700)
    monkeypatch.setenv(promoter._SIGNER_ENV, json.dumps([str(executable)]))

    def reject_overflow(_command, **kwargs):
        assert kwargs["input_payload"] == b"{}"
        assert kwargs["max_output_bytes"] == 128 * 1024
        raise promoter.ReleaseError("evidence-signing-output-invalid")

    monkeypatch.setattr(promoter, "_invoke_bounded", reject_overflow)

    with pytest.raises(promoter.ReleaseError, match="evidence-signing-failed"):
        promoter._external_json(promoter._SIGNER_ENV, b"{}")


def test_bounded_runner_enforces_output_limit_during_execution(tmp_path: Path) -> None:
    with pytest.raises(promoter.ReleaseError, match="proof-output-invalid"):
        promoter._invoke_bounded(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            cwd=tmp_path,
            environment=dict(os.environ),
            timeout_seconds=10,
            role="proof",
            max_output_bytes=1024,
        )


def test_wheelhouse_byte_budget_rejects_before_artifact_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)

    class _Entry:
        def __init__(self, name: str, size: int) -> None:
            self.name = name
            self.path = str(wheelhouse / name)
            self._size = size

        def is_symlink(self) -> bool:
            return False

        def stat(self, *, follow_symlinks: bool):
            assert follow_symlinks is False
            return type(
                "EntryMetadata",
                (),
                {"st_mode": promoter.stat.S_IFREG | 0o400, "st_size": self._size},
            )()

    entries = [
        _Entry("release-requirements.txt", 1),
        _Entry("oversized-1.0.0-py3-none-any.whl", promoter._MAX_WHEELHOUSE_BYTES + 1),
    ]
    monkeypatch.setattr(promoter.os, "scandir", lambda _path: entries)
    monkeypatch.setattr(
        promoter,
        "_read_regular",
        lambda *_args, **_kwargs: pytest.fail(
            "oversized wheelhouse must fail before lock or wheel inspection"
        ),
    )

    with pytest.raises(promoter.ReleaseError, match="wheelhouse-byte-budget-exceeded"):
        promoter.validate_wheelhouse(wheelhouse, _spec())


@pytest.mark.parametrize(
    ("member_count", "uncompressed_bytes", "error_code"),
    [
        (promoter._MAX_RELEASE_FILES + 1, 1, "release-file-count-limit"),
        (1, promoter._MAX_RELEASE_BYTES + 1, "release-byte-budget-exceeded"),
    ],
)
def test_aggregate_wheel_archive_budget_rejects_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_count: int,
    uncompressed_bytes: int,
    error_code: str,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    lock_payload = b"agent-utilities[serving]==1.0.0 --hash=sha256:" + b"1" * 64 + b"\n"
    base = _spec()
    spec = promoter.ReleaseSpec(
        release_id=base.release_id,
        requirements_file=base.requirements_file,
        requirements_digest=promoter._sha256(lock_payload),
        packages=base.packages,
        native_artifacts=base.native_artifacts,
        toolchain=base.toolchain,
        commands=base.commands,
        digest=base.digest,
    )

    class _Entry:
        def __init__(self, name: str) -> None:
            self.name = name
            self.path = str(wheelhouse / name)

        def is_symlink(self) -> bool:
            return False

        def stat(self, *, follow_symlinks: bool):
            assert follow_symlinks is False
            return type(
                "EntryMetadata",
                (),
                {"st_mode": promoter.stat.S_IFREG | 0o400, "st_size": 1},
            )()

    wheel_name = "agent_utilities-1.0.0-py3-none-any.whl"
    monkeypatch.setattr(
        promoter.os,
        "scandir",
        lambda _path: [_Entry(spec.requirements_file), _Entry(wheel_name)],
    )
    monkeypatch.setattr(
        promoter, "_read_regular", lambda *_args, **_kwargs: lock_payload
    )
    monkeypatch.setattr(
        promoter,
        "_hash_regular",
        lambda *_args, **_kwargs: (_DIGEST, 1),
    )
    monkeypatch.setattr(
        promoter,
        "_inspect_wheel",
        lambda *_args, **_kwargs: (
            "agent-utilities",
            "1.0.0",
            {},
            member_count,
            uncompressed_bytes,
            frozenset(),
        ),
    )

    with pytest.raises(promoter.ReleaseError, match=error_code):
        promoter.validate_wheelhouse(wheelhouse, spec)


def test_existing_evidence_without_committed_journal_fails_before_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "releases"
    wheelhouse = tmp_path / "wheelhouse"
    inputs = tmp_path / "inputs"
    evidence_parent = tmp_path / "evidence"
    for directory in (root, wheelhouse, inputs, evidence_parent):
        directory.mkdir(mode=0o700)
    spec_path = inputs / "spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    destination = (evidence_parent / "promotion.json").absolute()
    destination.write_text("{}", encoding="utf-8")
    destination.chmod(0o600)
    monkeypatch.setattr(promoter, "load_spec", lambda *_a, **_k: _spec())
    monkeypatch.setattr(
        promoter,
        "_verify_toolchain",
        lambda _spec: pytest.fail("toolchain must not run for an occupied destination"),
    )

    status, evidence = promoter.promote(
        spec_path=spec_path,
        release_id="release-test",
        wheelhouse=wheelhouse,
        releases_root=root.absolute(),
        evidence_path=destination,
    )

    assert status == 1
    assert evidence["status"] == "rejected"
    assert evidence["errorCode"] == "evidence-destination-must-be-new"


def test_interrupted_activation_is_rolled_back_before_new_work(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    root.mkdir(mode=0o700)
    for release in ("release-old", "release-test"):
        (root / release).mkdir(mode=0o700)
        (root / release).chmod(0o555)
    (root / "current").symlink_to("release-test")
    spec = _spec()
    root_fd = promoter._open_releases_root(root)
    try:
        promoter._write_journal(
            root_fd,
            promoter._journal_payload(spec, previous="release-old", phase="activated"),
        )
        status, evidence = promoter._recover_activation(
            root_fd,
            spec=spec,
            evidence_path=(tmp_path / "unused.json").absolute(),
        )
    finally:
        os.close(root_fd)

    assert status == "rolled-back"
    assert evidence is None
    assert os.readlink(root / "current") == "release-old"
    assert not (root / promoter._JOURNAL_NAME).exists()


@pytest.mark.parametrize("already_published", [False, True])
def test_committed_activation_recovery_publishes_exact_signed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_published: bool,
) -> None:
    root = tmp_path / "releases"
    evidence_parent = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    evidence_parent.mkdir(mode=0o700)
    (root / "release-test").mkdir(mode=0o700)
    (root / "release-test").chmod(0o555)
    (root / "current").symlink_to("release-test")
    spec = _spec()
    monkeypatch.setattr(promoter, "_external_json", _external_signing_stub)
    signed = promoter._sign_evidence(_promoted_evidence(spec), spec)
    destination = (evidence_parent / "promotion.json").absolute()
    if already_published:
        promoter._write_evidence(destination, signed, spec=spec)
    root_fd = promoter._open_releases_root(root.absolute())
    try:
        promoter._write_journal(
            root_fd,
            promoter._journal_payload(
                spec,
                previous=None,
                phase="committed",
                evidence=signed,
            ),
        )
        status, recovered = promoter._recover_activation(
            root_fd,
            spec=spec,
            evidence_path=destination,
        )
    finally:
        os.close(root_fd)

    assert status == "committed"
    assert recovered == signed
    assert json.loads(destination.read_text(encoding="utf-8")) == signed
    assert not (root / promoter._JOURNAL_NAME).exists()


def test_recovery_conflict_fails_closed_and_preserves_journal(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    root.mkdir(mode=0o700)
    for release in ("release-old", "release-test", "release-other"):
        (root / release).mkdir(mode=0o700)
        (root / release).chmod(0o555)
    (root / "current").symlink_to("release-other")
    spec = _spec()
    root_fd = promoter._open_releases_root(root.absolute())
    try:
        promoter._write_journal(
            root_fd,
            promoter._journal_payload(
                spec,
                previous="release-old",
                phase="activated",
            ),
        )
        with pytest.raises(promoter.ReleaseError, match="activation-recovery-conflict"):
            promoter._recover_activation(
                root_fd,
                spec=spec,
                evidence_path=(tmp_path / "unused.json").absolute(),
            )
    finally:
        os.close(root_fd)

    assert os.readlink(root / "current") == "release-other"
    assert (root / promoter._JOURNAL_NAME).is_file()


def test_loader_injection_variables_are_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(name, "injected")

    installer = promoter._installer_environment()
    runtime = promoter._runtime_environment(Path("runtime"))

    for environment in (installer, runtime):
        assert "LD_PRELOAD" not in environment
        assert "LD_LIBRARY_PATH" not in environment
        assert "DYLD_INSERT_LIBRARIES" not in environment
        assert "PYTHONPATH" not in environment


def test_native_promoter_rejects_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(promoter.sys, "platform", "darwin")

    with pytest.raises(promoter.ReleaseError, match="unsupported-platform"):
        promoter._require_supported_platform()


def test_promote_rolls_back_failed_doctor_and_clears_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    releases = tmp_path / "releases"
    wheelhouse = tmp_path / "wheelhouse"
    inputs = tmp_path / "inputs"
    evidence_parent = tmp_path / "evidence"
    for directory in (releases, wheelhouse, inputs, evidence_parent):
        directory.mkdir(mode=0o700)
    (releases / "release-old").mkdir(mode=0o700)
    (releases / "release-old").chmod(0o555)
    (releases / "current").symlink_to("release-old")
    spec_path = inputs / "spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    evidence_path = (evidence_parent / "result.json").absolute()
    spec = _spec()
    proof = {
        "status": "passed",
        "exitCode": 0,
        "outputDigest": _DIGEST,
    }

    monkeypatch.setattr(promoter, "load_spec", lambda *_a, **_k: spec)

    def bound_tools(_spec):
        source = Path("/dev/null")
        descriptors = (os.open(source, os.O_RDONLY), os.open(source, os.O_RDONLY))
        return tuple(
            promoter.BoundExecutable(
                descriptor,
                promoter._fd_path(descriptor),
                source,
            )
            for descriptor in descriptors
        )

    monkeypatch.setattr(promoter, "_verify_toolchain", bound_tools)
    monkeypatch.setattr(
        promoter,
        "validate_wheelhouse",
        lambda *_a, **_k: (
            {"a": object(), "b": object(), "c": object()},
            {"a": object(), "b": object(), "c": object()},
            b"",
        ),
    )

    def stage(release_root, *_args):
        staged = release_root / ".wheelhouse"
        staged.mkdir()
        return staged

    def remove(staged):
        staged.rmdir()

    def create_runtime(release_root, *_args, **_kwargs):
        binary_root = release_root / "runtime" / "bin"
        binary_root.mkdir(parents=True)
        for name in ("python", "graph-os", "epistemic-graph-server"):
            path = binary_root / name
            path.write_bytes(name.encode())
            path.chmod(0o700)
        return {"venv": dict(proof), "install": dict(proof)}

    monkeypatch.setattr(promoter, "_stage_wheelhouse", stage)
    monkeypatch.setattr(promoter, "_remove_staged_wheelhouse", remove)
    monkeypatch.setattr(promoter, "_create_runtime", create_runtime)
    monkeypatch.setattr(
        promoter,
        "verify_installed_release",
        lambda *_a, **_k: {
            "distributionCount": 3,
            "recordVerified": True,
            "nativeArtifactCount": 2,
            "dependencyEdgeCount": 2,
            "agentUtilitiesSha256": _RAW_DIGEST,
            "agentUtilitiesFileCount": 20,
            "distributionClosureSha256": _RAW_DIGEST,
        },
    )
    monkeypatch.setattr(
        promoter,
        "_certification_artifacts",
        lambda *_a, **_k: {
            "agentUtilitiesSha256": _RAW_DIGEST,
            "agentUtilitiesFileCount": 20,
            "distributionClosureSha256": _RAW_DIGEST,
            "releasePythonSha256": _RAW_DIGEST,
            "graphosSha256": _RAW_DIGEST,
            "engineSha256": _RAW_DIGEST,
        },
    )
    monkeypatch.setattr(promoter, "running_graph_process_count", lambda: 0)

    def command_proof(_root, _command, *, role):
        if role == "doctor":
            failed = {**proof, "status": "failed", "exitCode": 1}
            raise promoter.CommandProofError("doctor-failed", failed)
        return dict(proof)

    monkeypatch.setattr(promoter, "_command_proof", command_proof)
    monkeypatch.setattr(
        promoter, "_sign_evidence", lambda value, _spec: {**value, "signature": {}}
    )
    published: list[dict[str, Any]] = []
    monkeypatch.setattr(
        promoter,
        "_write_evidence",
        lambda _path, value, **_kwargs: published.append(value),
    )

    status, evidence = promoter.promote(
        spec_path=spec_path,
        release_id=spec.release_id,
        wheelhouse=wheelhouse,
        releases_root=releases.absolute(),
        evidence_path=evidence_path,
    )

    assert status == 1
    assert evidence["errorCode"] == "doctor-failed"
    assert evidence["status"] == "rolled-back"
    assert evidence["activation"]["rollback"] == "completed"
    assert os.readlink(releases / "current") == "release-old"
    assert not (releases / promoter._JOURNAL_NAME).exists()
    assert published[-1]["status"] == "rolled-back"


def test_version_parser_rejects_non_string_and_oversized_values() -> None:
    with pytest.raises(promoter.ReleaseError, match="bad-version"):
        promoter._version_value(1, "bad-version")
    with pytest.raises(promoter.ReleaseError, match="bad-version"):
        promoter._version_value("1" * 129, "bad-version")


def test_signed_evidence_rejects_unknown_signature_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    monkeypatch.setattr(promoter, "_external_json", _external_signing_stub)
    signed = promoter._sign_evidence(_promoted_evidence(spec), spec)
    signed["signature"]["unexpected"] = True

    with pytest.raises(promoter.ReleaseError, match="evidence-signature-invalid"):
        promoter._verify_signed_evidence(signed, spec)
