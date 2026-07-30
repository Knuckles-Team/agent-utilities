"""Signed hydration manifest — absent-vs-hidden resolution tests.

CONCEPT:AU-KG.audit.hydration-manifest-signed
CONCEPT:AU-KG.audit.hydration-absent-vs-hidden
"""

from __future__ import annotations

import base64
import secrets

import pytest

from agent_utilities.knowledge_graph.ingestion import hydration_manifest as hm


@pytest.fixture(autouse=True)
def release_signing_key(monkeypatch):
    """The SAME env://-referenced Ed25519 seed pattern
    ``scripts/generate_native_connector_manifest.py`` tests already use — no
    second signing scheme, no key material printed or committed."""
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )


class FakeReader:
    """A scripted :class:`hm.GraphReader`-shaped stub: fixed per-label counts
    and per-label hydration-step records, optionally raising."""

    def __init__(self, counts=None, steps=None, raises_for=()):
        self._counts = counts or {}
        self._steps = steps or {}
        self._raises_for = set(raises_for)

    def _query(self, cypher: str):
        for label in self._raises_for:
            if f":{label})" in cypher or f'"boot-hydration:{label}"' in cypher:
                raise ConnectionError(f"simulated failure for {label}")
        for label, count in self._counts.items():
            if f"MATCH (n:{label})" in cypher:
                return [{"c": count}]
        for step, record in self._steps.items():
            if f'"boot-hydration:{step}"' in cypher:
                return [
                    {
                        "status": record.get("status"),
                        "updated_at": record.get("updated_at"),
                    }
                ]
        return []

    def as_reader(self, authority: str) -> hm.GraphReader:
        return hm.GraphReader(query=self._query, authority=authority)


# ---------------------------------------------------------------------------
# GraphReader
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_graph_reader_count_label_success():
    reader = FakeReader(counts={"Skill": 275}).as_reader("serving")
    assert reader.count_label("Skill") == 275


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_graph_reader_count_label_zero_when_absent():
    reader = FakeReader(counts={}).as_reader("serving")
    assert reader.count_label("Server") == 0


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_graph_reader_count_label_raises_with_cause_chained():
    reader = FakeReader(raises_for={"Server"}).as_reader("serving")
    with pytest.raises(hm.GraphReaderError) as excinfo:
        reader.count_label("Server")
    assert isinstance(excinfo.value.__cause__, ConnectionError)


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_graph_reader_hydration_step_parses_record():
    reader = FakeReader(
        steps={
            "capabilities": {
                "status": "completed",
                "updated_at": "2026-07-29T00:00:00Z",
            }
        }
    ).as_reader("serving")
    record = reader.hydration_step("capabilities")
    assert record == {"status": "completed", "updated_at": "2026-07-29T00:00:00Z"}


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_graph_reader_hydration_step_none_when_absent():
    reader = FakeReader().as_reader("serving")
    assert reader.hydration_step("capabilities") is None


# ---------------------------------------------------------------------------
# Verdict fusion — the absent-vs-hidden decision rule
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_verdict_hidden_not_absent_when_service_authority_diverges():
    """THE required contract: expected>0, 0 under the serving principal, >0
    under service authority -> ``blocked`` (an RLS visibility gap), never
    ``not_started`` (which would falsely claim the class was never ingested)."""
    verdict, reason = hm._fuse_verdict(
        expected=62, actual_serving=0, actual_service=62, service_attempted=True
    )
    assert verdict == "blocked"
    assert "RLS visibility gap" in reason


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_verdict_not_started_when_both_readers_agree_on_zero():
    verdict, reason = hm._fuse_verdict(
        expected=62, actual_serving=0, actual_service=0, service_attempted=True
    )
    assert verdict == "not_started"
    assert "never ingested" in reason


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_verdict_blocked_when_service_authority_unavailable_and_actual_zero():
    """Without a service-authority read, a real gap and an RLS hide are
    indistinguishable — the verdict must stay honest (``blocked``), not
    default to the more comfortable ``not_started``."""
    verdict, reason = hm._fuse_verdict(
        expected=62, actual_serving=0, actual_service=None, service_attempted=False
    )
    assert verdict == "blocked"
    assert "cannot be ruled out" in reason


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_verdict_complete_when_actual_meets_expected():
    verdict, reason = hm._fuse_verdict(
        expected=62, actual_serving=62, actual_service=62, service_attempted=True
    )
    assert verdict == "complete"


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_verdict_partial_when_some_but_not_all_present():
    verdict, reason = hm._fuse_verdict(
        expected=62, actual_serving=30, actual_service=30, service_attempted=True
    )
    assert verdict == "partial"
    assert "30/62" in reason


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_verdict_not_started_with_no_declared_universe_and_no_service_read():
    verdict, reason = hm._fuse_verdict(
        expected=None, actual_serving=0, actual_service=None, service_attempted=False
    )
    assert verdict == "not_started"


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_verdict_blocked_when_serving_read_itself_fails():
    verdict, reason = hm._fuse_verdict(
        expected=62, actual_serving=None, actual_service=None, service_attempted=False
    )
    assert verdict == "blocked"


# ---------------------------------------------------------------------------
# build_hydration_manifest — end-to-end fusion over the registered classes
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_build_hydration_manifest_covers_every_registered_class():
    serving = FakeReader(counts={"Skill": 275, "Prompt": 96}).as_reader("serving")
    service = FakeReader(counts={"Skill": 275, "Prompt": 96}).as_reader("service")

    manifest = hm.build_hydration_manifest(serving=serving, service=service)

    assert manifest.service_authority_available is True
    by_class = {c.source_class: c for c in manifest.coverage}
    assert set(by_class) == {spec.name for spec in hm.SOURCE_CLASSES}
    assert by_class["Skill"].actual_serving == 275
    assert by_class["Prompt"].actual_serving == 96
    # Concept has no declared universe anywhere in this codebase.
    assert by_class["Concept"].expected is None


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_build_hydration_manifest_flags_hidden_server_class(monkeypatch):
    """Reproduces the bug report's exact shape: Server=0 under the serving
    principal while the declared 62-server universe says >0 are expected, and
    a service-authority read proves rows exist -> blocked, not not_started."""

    def fixture_expected() -> hm.ExpectedResult:
        return hm.ExpectedResult(62, "fixture universe", "deadbeef")

    monkeypatch.setattr(
        hm,
        "SOURCE_CLASSES",
        tuple(
            hm.SourceClassSpec("Server", "Server", fixture_expected)
            if spec.name == "Server"
            else spec
            for spec in hm.SOURCE_CLASSES
        ),
    )
    serving = FakeReader(counts={"Server": 0}).as_reader("serving")
    service = FakeReader(counts={"Server": 62}).as_reader("service")

    manifest = hm.build_hydration_manifest(serving=serving, service=service)
    server_row = next(c for c in manifest.coverage if c.source_class == "Server")

    assert server_row.expected == 62
    assert server_row.actual_serving == 0
    assert server_row.actual_service == 62
    assert server_row.verdict == "blocked"
    assert "RLS visibility gap" in server_row.reason


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_build_hydration_manifest_no_service_authority_is_honest():
    serving = FakeReader(counts={"Skill": 0}).as_reader("serving")

    manifest = hm.build_hydration_manifest(serving=serving, service=None)

    assert manifest.service_authority_available is False
    skill_row = next(c for c in manifest.coverage if c.source_class == "Skill")
    assert skill_row.actual_service is None
    assert skill_row.service_authority_attempted is False


def _verified_service_session():
    """The shape ``start_background_daemons`` captures: a verified, authenticated
    process authority — the SAME factory pattern the other GraphSession tests use."""
    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.security.brain_context import ActorContext, ActorType

    return GraphSession(
        actor=ActorContext(
            actor_id="graph-os-background",
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id="tenant-a",
            authenticated=True,
        ),
        tenant="tenant-a",
        scopes=frozenset({"kg:read"}),
        graph="tenant-a",
    )


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_service_authority_reader_is_none_without_a_distinct_session(monkeypatch):
    """A second reader over the SAME ambient identity is NOT a second authority.

    Engine reads resolve their principal from the task-local ``GraphSession``
    (``GraphComputeClient._send``), not from the engine object, so wrapping a
    second engine handle with no session of its own would produce an identical
    read while the manifest advertised an independent confirmation. When the
    active engine has captured no background session, the service reader must
    be honestly absent.
    """
    from agent_utilities.knowledge_graph.core import engine as engine_mod

    class _EngineWithoutBackgroundSession:
        def query_cypher(self, _cypher):
            return [{"c": 7}]

    monkeypatch.setattr(
        engine_mod.IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda cls: _EngineWithoutBackgroundSession()),
    )

    assert hm.resolve_service_authority_reader() is None


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_service_authority_reader_binds_the_background_session(monkeypatch):
    """With a captured background session it IS a second authority: the reader
    must bind that session for the duration of the read, so the divergence
    check compares two genuinely different identities."""
    from agent_utilities.knowledge_graph.core import engine as engine_mod
    from agent_utilities.knowledge_graph.core import session as session_mod

    service_session = _verified_service_session()
    seen: list[object] = []

    class _EngineWithBackgroundSession:
        _background_worker_session = service_session

        def query_cypher(self, _cypher):
            seen.append(session_mod.current_session())
            return [{"c": 3}]

    monkeypatch.setattr(
        engine_mod.IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda cls: _EngineWithBackgroundSession()),
    )

    reader = hm.resolve_service_authority_reader()
    assert reader is not None
    assert reader.count_label("Server") == 3
    assert seen == [service_session]
    # ...and it restores the ambient session afterwards.
    assert session_mod.current_session() is not service_session


@pytest.mark.concept("AU-KG.audit.hydration-absent-vs-hidden")
def test_service_authority_reader_is_none_when_it_would_be_the_same_identity(
    monkeypatch,
):
    """If the ambient serving session already IS the background session, the
    two reads cannot differ — claiming an independent confirmation would be a
    fabrication, so the reader is honestly absent."""
    from agent_utilities.knowledge_graph.core import engine as engine_mod
    from agent_utilities.knowledge_graph.core import session as session_mod

    service_session = _verified_service_session()

    class _EngineWithBackgroundSession:
        _background_worker_session = service_session

        def query_cypher(self, _cypher):
            return [{"c": 3}]

    monkeypatch.setattr(
        engine_mod.IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda cls: _EngineWithBackgroundSession()),
    )

    token = session_mod.set_session(service_session)
    try:
        assert hm.resolve_service_authority_reader() is None
    finally:
        session_mod.reset_session(token)


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_error_summary_classifies_by_stage_and_exception_type():
    serving = FakeReader(raises_for={"Server"}).as_reader("serving")
    manifest = hm.build_hydration_manifest(serving=serving, service=None)
    summary = manifest.error_summary
    assert any(key.startswith("serving_read:") for key in summary)


# ---------------------------------------------------------------------------
# Signing — reuses the existing Ed25519 release authority
# ---------------------------------------------------------------------------


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_sign_and_verify_round_trip():
    serving = FakeReader(counts={"Skill": 275}).as_reader("serving")
    manifest = hm.build_hydration_manifest(serving=serving, service=None)

    signed = hm.sign_hydration_manifest(manifest)

    assert signed["signature"]
    assert signed["signer"] == "ontology-manifest-generator"
    assert signed["signature_algorithm"] == "ed25519"
    from agent_utilities.knowledge_graph.ontology import ontology_integrity

    trusted = (signed["signing_public_key"],)
    assert hm.verify_hydration_manifest(signed, trusted_public_keys=trusted) is True
    # sanity: the shared verification primitive agrees independently.
    assert ontology_integrity.verify_release_signature(
        signed["digest"],
        signed["signature"],
        signer_id=signed["signer"],
        algorithm=signed["signature_algorithm"],
        public_key=signed["signing_public_key"],
        trusted_public_keys=trusted,
    )


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_verify_rejects_tampered_document():
    serving = FakeReader(counts={"Skill": 275}).as_reader("serving")
    manifest = hm.build_hydration_manifest(serving=serving, service=None)
    signed = hm.sign_hydration_manifest(manifest)

    tampered = dict(signed)
    tampered["coverage"] = []  # attacker "hides" every completeness gap

    trusted = (signed["signing_public_key"],)
    assert hm.verify_hydration_manifest(tampered, trusted_public_keys=trusted) is False


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_verify_rejects_untrusted_public_key():
    serving = FakeReader(counts={"Skill": 275}).as_reader("serving")
    manifest = hm.build_hydration_manifest(serving=serving, service=None)
    signed = hm.sign_hydration_manifest(manifest)

    assert hm.verify_hydration_manifest(signed, trusted_public_keys=()) is False


# ---------------------------------------------------------------------------
# Persistence — best-effort, mirrors _record_boot_hydration_step
# ---------------------------------------------------------------------------


class _RecordingEngine:
    def __init__(self):
        self.calls = []

    def add_node(self, node_id, label, properties):
        self.calls.append((node_id, label, properties))


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_persist_hydration_manifest_writes_one_stable_node():
    engine = _RecordingEngine()
    serving = FakeReader(counts={"Skill": 275}).as_reader("serving")
    manifest = hm.build_hydration_manifest(serving=serving, service=None)
    signed = hm.sign_hydration_manifest(manifest)

    hm.persist_hydration_manifest(engine, signed)

    assert len(engine.calls) == 1
    node_id, label, props = engine.calls[0]
    assert node_id == f"hydration-manifest:{manifest.generated_at}"
    assert label == "HydrationManifest"
    assert props["signer"] == "ontology-manifest-generator"


@pytest.mark.concept("AU-KG.audit.hydration-manifest-signed")
def test_persist_hydration_manifest_is_a_noop_without_add_node():
    hm.persist_hydration_manifest(object(), {"generated_at": "x"})  # must not raise


# ---------------------------------------------------------------------------
# _first_int must never fabricate a count (waves 1-5 gate regression)
# ---------------------------------------------------------------------------


def test_first_int_reads_a_dict_row():
    assert hm._first_int([{"c": 7}], "Server") == 7


def test_first_int_returns_zero_only_for_a_genuinely_empty_result():
    assert hm._first_int([], "Server") == 0


def test_first_int_raises_on_a_dict_row_with_no_integer_value():
    """A malformed row must NOT be reported as a real count of 0.

    Regression (waves 1-5 gate): the dict branch returned 0 when no value was
    int-convertible, while the tuple/list branch raised on the same class of
    malformed row. That asymmetry made "the count query came back malformed"
    indistinguishable from "this label genuinely has 0 nodes" — the exact
    absent-vs-hidden ambiguity this module exists to resolve.
    """
    with pytest.raises(hm.GraphReaderError):
        hm._first_int([{"c": None}], "Server")


def test_first_int_raises_on_a_malformed_sequence_row():
    with pytest.raises(hm.GraphReaderError):
        hm._first_int([("not-a-number",)], "Server")
