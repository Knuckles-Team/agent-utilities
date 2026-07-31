"""OpenBao (Vault) KV v2 custody for the release-signing key (D-DP-2 / D-OC-1).

CONCEPT:AU-KG.ontology.release-key-rotation — proves the read path end to end:

    SECRETS_BACKEND=vault, SECRETS_VAULT_URL, SECRETS_VAULT_MOUNT, VAULT_TOKEN
        -> VaultBackend (KV v2, optionally version-pinned via `path#field@version`)
        -> SecretsClient.resolve_ref
        -> resolve_runtime_secret_reference (agent_utilities.security.cli_secrets)
        -> ontology_integrity.release_signer_for_publication

against a FAKE ``hvac.Client`` (an in-memory stand-in for OpenBao's KV v2 HTTP API,
never a real network call) so the whole custody path is proven without a live
server. What is under test is the resolver's own logic — parsing the reference,
selecting a KV version, deriving and checking the public key — which is identical
whether the HTTP transport underneath is real or faked.

Never asserts on private key material anywhere — only on the DERIVED PUBLIC key,
exactly like ``test_release_key_custody.py``.
"""

from __future__ import annotations

import base64
import secrets as secrets_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

hvac = pytest.importorskip("hvac")

from agent_utilities.knowledge_graph.ontology import ontology_integrity
from agent_utilities.security.secrets_client import (
    SecretsClient,
    VaultBackend,
    _split_versioned_key,
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _keypair() -> tuple[str, str]:
    """A fresh, throwaway Ed25519 (seed, derived_public_key) pair, both b64url.

    Generated in-process for the test only — never the real production seed.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    seed = secrets_module.token_bytes(32)
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return _b64url(seed), _b64url(public)


def _lock(tmp_path: Path, public_key: str) -> Path:
    path = tmp_path / "ontology.lock"
    path.write_text(
        yaml.safe_dump(
            {
                "agents/connector-0/connector_manifest.yml": {
                    "hash": "0" * 64,
                    "signing_public_key": public_key,
                    "certification_signing_public_key": public_key,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


class _FakeKvV2:
    """In-memory stand-in for OpenBao's KV v2 API — every version kept, like the real thing."""

    def __init__(self) -> None:
        self._versions: dict[str, dict[int, dict[str, str]]] = {}

    def seed(self, path: str, version: int, data: dict[str, str]) -> None:
        self._versions.setdefault(path, {})[version] = data

    def read_secret_version(
        self, *, path: str, mount_point: str, version: int | None = None
    ) -> dict[str, Any]:
        entry = self._versions.get(path)
        if not entry:
            raise hvac.exceptions.InvalidPath(f"no such path: {mount_point}/{path}")
        chosen = version if version is not None else max(entry)
        if chosen not in entry:
            raise hvac.exceptions.InvalidPath(f"no such version: {chosen}")
        return {"data": {"data": dict(entry[chosen]), "metadata": {"version": chosen}}}


def _fake_hvac_client_factory(kv2: _FakeKvV2):
    """Build the ``hvac.Client`` replacement bound to one fake KV v2 store.

    Authenticates via the plain static-token branch of ``VaultBackend._authenticate``
    (``is_authenticated()`` returning ``True``) so construction never performs a real
    network call.
    """

    def _factory(*, url: str, token: str | None = None) -> Any:
        return SimpleNamespace(
            url=url,
            token=token,
            is_authenticated=lambda: True,
            secrets=SimpleNamespace(kv=SimpleNamespace(v2=kv2)),
        )

    return _factory


@pytest.fixture
def fake_openbao(monkeypatch: pytest.MonkeyPatch) -> _FakeKvV2:
    kv2 = _FakeKvV2()
    monkeypatch.setattr(hvac, "Client", _fake_hvac_client_factory(kv2))
    return kv2


def _vault_backend(fake_openbao: _FakeKvV2, *, mount: str = "apps") -> VaultBackend:
    return VaultBackend(
        url="http://openbao.fake.invalid:8200",
        token="fake-test-token",
        mount_point=mount,
    )


# ---------------------------------------------------------------------------
# VaultBackend.get() — version-pinned reads (the D-OC-1 mechanism itself)
# ---------------------------------------------------------------------------


class TestVersionedKvRead:
    def test_split_versioned_key_parses_trailing_digits(self) -> None:
        assert _split_versioned_key("agent-utilities#SEED@2") == (
            "agent-utilities#SEED",
            2,
        )
        assert _split_versioned_key("agent-utilities#SEED") == (
            "agent-utilities#SEED",
            None,
        )
        # Not-a-version suffix is left as part of the literal key (safe fallback).
        assert _split_versioned_key("agent-utilities#weird@field") == (
            "agent-utilities#weird@field",
            None,
        )

    def test_unpinned_read_returns_the_latest_version(
        self, fake_openbao: _FakeKvV2
    ) -> None:
        fake_openbao.seed("agent-utilities", 2, {"SEED": "v2-value"})
        fake_openbao.seed("agent-utilities", 3, {"SEED": "v3-value"})
        backend = _vault_backend(fake_openbao)

        assert backend.get("agent-utilities#SEED") == "v3-value"

    def test_pinned_read_returns_exactly_that_version_even_though_latest_moved(
        self, fake_openbao: _FakeKvV2
    ) -> None:
        """The whole point: a later overwrite of "latest" cannot reach a pinned reader."""
        fake_openbao.seed("agent-utilities", 2, {"SEED": "v2-value"})
        fake_openbao.seed("agent-utilities", 3, {"SEED": "v3-value-bad-overwrite"})
        backend = _vault_backend(fake_openbao)

        assert backend.get("agent-utilities#SEED@2") == "v2-value"

    def test_pinning_a_missing_version_fails_closed(
        self, fake_openbao: _FakeKvV2
    ) -> None:
        fake_openbao.seed("agent-utilities", 2, {"SEED": "v2-value"})
        backend = _vault_backend(fake_openbao)

        assert backend.get("agent-utilities#SEED@99") is None

    def test_resolve_ref_through_the_client_honours_the_version_pin(
        self, fake_openbao: _FakeKvV2
    ) -> None:
        fake_openbao.seed("agent-utilities", 2, {"SEED": "good"})
        fake_openbao.seed("agent-utilities", 3, {"SEED": "bad"})
        client = SecretsClient(backend=_vault_backend(fake_openbao))

        assert client.resolve_ref("vault://agent-utilities#SEED@2") == "good"
        assert client.resolve_ref("vault://agent-utilities#SEED") == "bad"


# ---------------------------------------------------------------------------
# End-to-end: settings -> factory -> resolver -> release_signer_for_publication
# ---------------------------------------------------------------------------


class TestReleaseSigningThroughOpenBaoCustody:
    """Exercises the exact D-DP-2 fix-shape settings against the fake transport."""

    def _configure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        ref: str,
        mount: str = "apps",
    ) -> None:
        monkeypatch.setenv("SECRETS_BACKEND", "vault")
        monkeypatch.setenv("SECRETS_VAULT_URL", "http://openbao.fake.invalid:8200")
        monkeypatch.setenv("SECRETS_VAULT_MOUNT", mount)
        monkeypatch.setenv("VAULT_TOKEN", "fake-test-token")
        monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF", ref)

    def test_signing_succeeds_end_to_end_through_a_version_pinned_openbao_read(
        self,
        tmp_path: Path,
        fake_openbao: _FakeKvV2,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        good_seed, good_public_key = _keypair()
        bad_seed, _bad_public_key = _keypair()
        fake_openbao.seed("agent-utilities", 2, {"SEED": good_seed})
        # A later, unrelated write landed in "latest" — exactly the D-OB-5 shape.
        fake_openbao.seed("agent-utilities", 3, {"SEED": bad_seed})

        self._configure(monkeypatch, ref="vault://agent-utilities#SEED@2")
        lock_path = _lock(tmp_path, good_public_key)

        signer = ontology_integrity.release_signer_for_publication(lock_path=lock_path)

        assert signer.public_key == good_public_key
        # The signer actually works (a real signature verifies against the same key).
        digest = "0" * 64
        signature = signer.sign(digest)
        assert ontology_integrity.verify_release_signature(
            digest,
            signature,
            signer_id=signer.signer_id,
            algorithm=signer.algorithm,
            public_key=signer.public_key,
            trusted_public_keys=(good_public_key,),
        )

    def test_reading_unpinned_latest_after_a_bad_overwrite_is_refused_at_signing_time(
        self,
        tmp_path: Path,
        fake_openbao: _FakeKvV2,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the version pin, "latest" is whatever was written last — refused."""
        good_seed, good_public_key = _keypair()
        bad_seed, _bad_public_key = _keypair()
        fake_openbao.seed("agent-utilities", 2, {"SEED": good_seed})
        fake_openbao.seed("agent-utilities", 3, {"SEED": bad_seed})

        self._configure(monkeypatch, ref="vault://agent-utilities#SEED")  # no @version
        lock_path = _lock(tmp_path, good_public_key)

        with pytest.raises(
            ontology_integrity.ReleaseSigningError, match="does not match"
        ):
            ontology_integrity.release_signer_for_publication(lock_path=lock_path)

    def test_a_disagreeing_seed_is_refused_at_signing_time_never_at_admission(
        self,
        tmp_path: Path,
        fake_openbao: _FakeKvV2,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The D-OB-5 regression test: OpenBao custody holds a seed the lock does not
        pin. Refusal must happen before any artifact is produced, not at admission."""
        _pinned_seed, pinned_public_key = _keypair()
        wrong_seed, _wrong_public_key = _keypair()
        fake_openbao.seed("agent-utilities", 1, {"SEED": wrong_seed})

        self._configure(monkeypatch, ref="vault://agent-utilities#SEED@1")
        lock_path = _lock(tmp_path, pinned_public_key)

        with pytest.raises(
            ontology_integrity.ReleaseSigningError, match="does not match"
        ):
            ontology_integrity.release_signer_for_publication(lock_path=lock_path)

    def test_recovering_from_a_bad_overwrite_is_repointing_the_version_pin_not_a_vault_mutation(
        self,
        tmp_path: Path,
        fake_openbao: _FakeKvV2,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The "rollback": OpenBao KV v2 never deletes version 2, so recovery is
        re-declaring `ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF` to pin it again — no
        write to OpenBao, no data movement, nothing to undo."""
        good_seed, good_public_key = _keypair()
        bad_seed, _bad_public_key = _keypair()
        fake_openbao.seed("agent-utilities", 2, {"SEED": good_seed})
        fake_openbao.seed("agent-utilities", 3, {"SEED": bad_seed})
        lock_path = _lock(tmp_path, good_public_key)

        # Operator notices the bad "latest" and re-points the ref at the known-good
        # version — the fake store above is never mutated between these two calls.
        self._configure(monkeypatch, ref="vault://agent-utilities#SEED")
        with pytest.raises(ontology_integrity.ReleaseSigningError):
            ontology_integrity.release_signer_for_publication(lock_path=lock_path)

        self._configure(monkeypatch, ref="vault://agent-utilities#SEED@2")
        signer = ontology_integrity.release_signer_for_publication(lock_path=lock_path)
        assert signer.public_key == good_public_key

    def test_an_env_reference_is_still_refused_even_with_vault_configured(
        self,
        tmp_path: Path,
        fake_openbao: _FakeKvV2,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The local file/env tier never becomes the source of truth for release
        signing, even when OpenBao is fully configured and reachable — only
        `vault://`/`secret://` (versioned custody) is accepted."""
        good_seed, good_public_key = _keypair()
        fake_openbao.seed("agent-utilities", 2, {"SEED": good_seed})
        monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", good_seed)

        self._configure(monkeypatch, ref="env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL")
        lock_path = _lock(tmp_path, good_public_key)

        with pytest.raises(ontology_integrity.ReleaseSigningError, match="versioned"):
            ontology_integrity.release_signer_for_publication(lock_path=lock_path)
