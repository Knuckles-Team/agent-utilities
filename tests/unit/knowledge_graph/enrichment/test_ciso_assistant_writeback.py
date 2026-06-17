"""CISO Assistant write-back sink tests (CONCEPT:KG-2.111).

Exercises the unified, fail-closed, dry-run-first write-back core with the
CISO Assistant sink: KG-derived governance entities → CISO objects via the
generated ``api_*_create`` methods, gated by ``CISO_ASSISTANT_ENABLE_WRITE``.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback


class _Resp:
    def __init__(self, data):
        self.data = data


class FakeCisoClient:
    def __init__(self):
        self.created: list[tuple[str, dict]] = []

    def api_policies_create(self, body=None):
        self.created.append(("policy", body or {}))
        return _Resp({"id": "new-policy-1", **(body or {})})

    def api_applied_controls_create(self, body=None):
        self.created.append(("control", body or {}))
        return _Resp({"id": "new-control-1", **(body or {})})


def test_dry_run_proposes_without_writing():
    client = FakeCisoClient()
    out = run_writeback(
        "ciso_assistant",
        client=client,
        creations=[{"type": "Policy", "name": "Data Retention Policy"}],
        dry_run=True,
    )
    assert out["status"] == "completed"
    assert out["dry_run"] is True
    assert out["created"] == 0
    assert client.created == []
    assert out["proposals"][0]["op"] == "create_policy"


def test_live_write_refused_without_enable_flag(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    out = run_writeback(
        "ciso_assistant",
        client=FakeCisoClient(),
        creations=[{"type": "Policy", "name": "P"}],
        dry_run=False,
    )
    assert out["status"] == "refused"
    assert "CISO_ASSISTANT_ENABLE_WRITE" in out["reason"]


def test_live_write_when_enabled(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeCisoClient()
    out = run_writeback(
        "ciso_assistant",
        client=client,
        folder="folder-root",
        creations=[
            {"type": "Policy", "name": "Data Retention Policy", "ref_id": "DRP-1"},
            {"type": "Control", "name": "Encrypt At Rest"},
        ],
        dry_run=False,
    )
    assert out["status"] == "completed"
    assert out["created"] == 2
    kinds = [k for k, _ in client.created]
    assert kinds == ["policy", "control"]
    # folder is required by CISO and threaded into every create
    assert all(payload["folder"] == "folder-root" for _, payload in client.created)


def test_creation_without_folder_is_skipped(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeCisoClient()
    out = run_writeback(
        "ciso_assistant",
        client=client,
        creations=[{"type": "Policy", "name": "No Folder Policy"}],
        dry_run=False,
    )
    assert out["created"] == 0
    assert out["skipped"] >= 1
    assert client.created == []
