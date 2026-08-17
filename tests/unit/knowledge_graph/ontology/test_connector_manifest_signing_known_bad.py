"""Known-bad proofs for the connector-manifest signing custody path (GOC-16/BUG-234).

BUG-234 records that GOC-84 (connector manifest regeneration/signing) is hard-blocked
on the release signing key not being reachable from any working environment in this
session, and that the lane *correctly declined* to regenerate/sign the real bundled
manifests without it — a signed-but-stale manifest is strictly worse than an unsigned
one. This lane (GOC-16) builds the custody-path *mechanism* the operator can run once
they supply the key through OpenBao (never here), and is required to prove the gate the
mechanism feeds actually catches drift, not merely that the gate code exists.

Each test below builds its OWN small, correctly signed manifest (a throwaway Ed25519
keypair via ``ReleaseSigner.from_runtime()`` against an ``env://`` test-only reference —
exactly the pattern ``test_connector_manifest_gate.py`` already uses), tampers with
exactly ONE of the four GOC-84/GOC-16-named adversarial dimensions, and proves the gate
refuses to let the source activate — with a bounded, specific diagnostic, never a bare
crash or a silent pass:

  1. one-bit source change  -- a native connector's own code changes after signing
  2. schema change          -- a resource/field mapping changes after signing
  3. alias change            -- a sync preset's `server` alias changes after signing
  4. dependency-lock drift   -- the frozen `uv.lock` moves after signing

No real key material, real OpenBao access, or the real bundled fleet manifests are
touched by any test here.
"""

from __future__ import annotations

import base64
import secrets
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.ontology import connector_manifest_gate as gate
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceSpec,
    SchemaMapping,
    SyncSpec,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (
    compile_manifest,
    export_manifest_ttl,
)
from agent_utilities.knowledge_graph.ontology.ontology_integrity import (
    DEFAULT_SIGNER_ID,
    ReleaseSigner,
    canonical_hash,
    canonical_manifest_hash,
    dependency_lock_digest,
)

_WIDGET_SCHEMA_SHA256 = "1" * 64


@pytest.fixture(autouse=True)
def release_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A throwaway, test-only Ed25519 signer — never the real release key."""
    private_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", private_key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    signer = ReleaseSigner.from_runtime()
    monkeypatch.setenv("ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS", signer.public_key)


def _install_widget_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_presets",
        lambda provider: (
            {
                "widget-records": {
                    "server": "widget-mcp",
                    "tool": "widget_records",
                    "action": "list",
                    "records_path": "items",
                    "id_field": "id",
                    "text_field": "summary",
                }
            }
            if provider == "widget-mcp"
            else None
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_schema_fingerprints",
        lambda provider: (
            {"widget_records": _WIDGET_SCHEMA_SHA256} if provider == "widget-mcp" else None
        ),
    )


def _write_signed_widget_manifest(
    root: Path,
    pkg: str,
    *,
    fields: dict[str, str] | None = None,
    server: str = "widget-mcp",
    dependency_lock: str | None = None,
) -> Path:
    """A freshly generated, correctly signed manifest for a throwaway ``widget-mcp``
    connector — parameterised so each test can vary exactly the one dimension it is
    proving the gate catches (schema fields, sync-preset server alias, or the pinned
    dependency-lock digest), while everything else is generated genuinely correctly.
    """
    (root / pkg).mkdir(parents=True)
    manifest = ConnectorManifest(
        connector=pkg,
        resources=[ResourceSpec(name="Widget", id_prefix="widget")],
        schema_mappings={"Widget": SchemaMapping(fields=fields or {"name": "xsd:string"})},
        sync=[
            SyncSpec(
                preset="widget-records",
                server=server,
                tool="widget_records",
                action="list",
                records_path="items",
                id_field="id",
                text_field="summary",
                tool_schema_sha256=_WIDGET_SCHEMA_SHA256,
                raw={
                    "server": server,
                    "tool": "widget_records",
                    "action": "list",
                    "records_path": "items",
                    "id_field": "id",
                    "text_field": "summary",
                },
            )
        ],
        provenance=ProvenanceSpec(
            integrity=IntegrityInfo(hash="0" * 64), signer=DEFAULT_SIGNER_ID
        ),
    )
    spec = compile_manifest(manifest)
    ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)
    import rdflib

    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    digest, n = canonical_hash(g)
    signer = ReleaseSigner.from_runtime()
    unsigned = manifest.model_copy(
        update={
            "provenance": ProvenanceSpec(
                integrity=IntegrityInfo(hash=digest, triple_count=n),
                signer=signer.signer_id,
                signature_algorithm=signer.algorithm,
                signing_public_key=signer.public_key,
                dependency_lock_digest=dependency_lock,
            )
        }
    )
    manifest = unsigned.model_copy(
        update={
            "provenance": unsigned.provenance.model_copy(
                update={"signature": signer.sign(canonical_manifest_hash(unsigned))}
            )
        }
    )

    import yaml

    path = root / pkg / "connector_manifest.yml"
    path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _load_native_manifest_generator():
    """Load ``scripts/generate_native_connector_manifest.py`` as a module (it is a
    script, not an importable package member) — the same technique
    ``test_connector_manifest_gate.py`` uses.
    """
    import importlib.util

    repo_root = Path(gate.__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "generate_native_connector_manifest",
        repo_root / "scripts" / "generate_native_connector_manifest.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_single_source_native_bundle(tmp_path: Path, monkeypatch, *, source: str) -> Path:
    """A self-consistent, correctly signed native ``connector_manifest.yml`` (plus its
    ``tool_schema_fingerprints.json`` sidecar) whose ``sync`` list covers exactly
    ``{source}`` — generated against the REAL, currently-installed native connector
    code, so the recorded fingerprint is the true one for today's source tree.
    """
    gen = _load_native_manifest_generator()
    single_source_map = {source: "native-source-connectors"}
    monkeypatch.setattr(gate, "SOURCE_TO_CONNECTOR_PACKAGE", single_source_map)
    monkeypatch.setattr(gen, "SOURCE_TO_CONNECTOR_PACKAGE", single_source_map)
    output_dir = tmp_path / "native-source-connectors"
    gen.write_bundle(output_dir)
    return output_dir / "connector_manifest.yml"


# ---------------------------------------------------------------------------
# 1. One-bit source change (native connector code changes after signing)
# ---------------------------------------------------------------------------


def test_one_bit_source_change_blocks_activation(tmp_path: Path, monkeypatch) -> None:
    """A native connector's own Python source diverging by even one byte from what
    was fingerprinted and signed must block that source's activation — the exact
    "one-bit source change" case GOC-84's acceptance gates name explicitly.

    Simulated without mutating the real installed package (unsafe under a shared test
    run) by monkeypatching the fingerprint *evidence function* to return what a live
    recompute would yield if the source had changed by one byte since generation —
    exercising the real ``pinned != actual`` comparison in
    ``_native_provider_violations`` against a genuinely different digest.
    """
    path = _write_single_source_native_bundle(tmp_path, monkeypatch, source="web")

    real_evidence = gate._native_activation_fingerprint_evidence

    def _tampered_evidence(source_type, **kwargs):
        digest, modules = real_evidence(source_type, **kwargs)
        if source_type == "web":
            # A one-bit change in the connector's own source changes its fingerprint;
            # flip the last hex character to produce a different, well-formed digest.
            tampered_char = "0" if digest[-1] != "0" else "1"
            digest = digest[:-1] + tampered_char
        return digest, modules

    monkeypatch.setattr(
        gate, "_native_activation_fingerprint_evidence", _tampered_evidence
    )

    violations = gate.check_manifest_bytes(path, require_provider=True)

    assert any(
        "[tool-schema]" in v and "differs from its signed code fingerprint" in v
        for v in violations
    ), violations


# ---------------------------------------------------------------------------
# 2. Schema change (a resource/field mapping changes after signing)
# ---------------------------------------------------------------------------


def test_schema_change_blocks_activation(tmp_path: Path, monkeypatch) -> None:
    """A hand-edited schema field (post-signing) must block activation — via
    ``precheck_source`` end to end, proving the refusal happens before any tool is
    exposed, not merely that ``check_manifest_bytes`` alone would notice it.
    """
    path = _write_signed_widget_manifest(tmp_path, "widget-mcp")
    _install_widget_provider(monkeypatch)

    # Confirm the clean manifest activates first, so the failure below is
    # attributable to the schema edit and nothing else.
    clean = gate.precheck_source("widget", agents_root=tmp_path)
    assert clean["ok"] is True, clean["violations"]

    text = path.read_text(encoding="utf-8")
    tampered = text.replace("xsd:string", "xsd:integer", 1)
    assert tampered != text
    path.write_text(tampered, encoding="utf-8")

    result = gate.precheck_source("widget", agents_root=tmp_path)

    assert result["checked"] is True
    assert result["ok"] is False
    assert any(
        "[integrity]" in v or "[signature]" in v for v in result["violations"]
    ), result["violations"]


# ---------------------------------------------------------------------------
# 3. Alias change (a sync preset's ``server`` alias changes after signing)
# ---------------------------------------------------------------------------


def test_alias_change_blocks_activation(tmp_path: Path, monkeypatch) -> None:
    """A sync preset's ``server`` alias hand-edited after signing (the shape of the
    27-provider MCP-alias drift ``docs/architecture/drift_proof_release.md`` §2
    records) must block activation — proven through the full ``precheck_source`` path
    a caller actually goes through before a tool is exposed.
    """
    path = _write_signed_widget_manifest(tmp_path, "widget-mcp")
    _install_widget_provider(monkeypatch)

    clean = gate.precheck_source("widget", agents_root=tmp_path)
    assert clean["ok"] is True, clean["violations"]

    text = path.read_text(encoding="utf-8")
    tampered = text.replace("server: widget-mcp", "server: renamed-widget-mcp")
    assert tampered != text
    assert "renamed-widget-mcp" in tampered
    path.write_text(tampered, encoding="utf-8")

    result = gate.precheck_source("widget", agents_root=tmp_path)

    assert result["checked"] is True
    assert result["ok"] is False
    assert any("[signature]" in v for v in result["violations"]), result["violations"]


# ---------------------------------------------------------------------------
# 4. Dependency-lock drift (the frozen ``uv.lock`` moves after signing)
# ---------------------------------------------------------------------------


def test_dependency_lock_drift_blocks_activation(tmp_path: Path, monkeypatch) -> None:
    """A manifest signed against a specific frozen ``uv.lock`` must refuse to
    activate once the live lock disagrees — the GOC-84/GOC-16-named "dependency-lock
    drift" case, which neither the manifest's own schema hash nor its signature
    alone can see (both are internally self-consistent; only a live-vs-pinned lock
    comparison catches this).
    """
    frozen_digest = dependency_lock_digest()
    path = _write_signed_widget_manifest(
        tmp_path, "widget-mcp", dependency_lock=frozen_digest
    )
    _install_widget_provider(monkeypatch)

    clean = gate.precheck_source("widget", agents_root=tmp_path)
    assert clean["ok"] is True, clean["violations"]

    def _drifted_digest(*_args, **_kwargs):
        # A dependency version bump (or add/remove) after the manifest was signed —
        # simulated without touching the real repo's uv.lock.
        return "f" * 64 if frozen_digest != "f" * 64 else "0" * 64

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.ontology_integrity.dependency_lock_digest",
        _drifted_digest,
    )

    result = gate.precheck_source("widget", agents_root=tmp_path)

    assert result["checked"] is True
    assert result["ok"] is False
    assert any(
        "[dependency-lock]" in v and "drifted since this manifest was generated" in v
        for v in result["violations"]
    ), result["violations"]


def test_dependency_lock_digest_is_stable_and_sensitive_to_real_drift(
    tmp_path: Path,
) -> None:
    """Sanity-checks :func:`dependency_lock_digest` itself: stable across re-parses
    of the identical lock, and genuinely different when a pinned version moves —
    independent of the gate wiring the other tests exercise.
    """
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "widget"\nversion = "1.0.0"\n\n'
        '[[package]]\nname = "gadget"\nversion = "2.3.1"\n',
        encoding="utf-8",
    )
    first = dependency_lock_digest(lock)
    second = dependency_lock_digest(lock)
    assert first == second

    lock.write_text(
        '[[package]]\nname = "widget"\nversion = "1.0.1"\n\n'
        '[[package]]\nname = "gadget"\nversion = "2.3.1"\n',
        encoding="utf-8",
    )
    drifted = dependency_lock_digest(lock)
    assert drifted != first
