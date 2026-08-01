"""Meta-proof for the governed outbound HTTP construction gate."""

from __future__ import annotations

from pathlib import Path

from scripts.check_http_egress_boundary import validate


def test_direct_http_client_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "agent_utilities"
    root.mkdir()
    (root / "unsafe.py").write_text(
        "import httpx\nclient = httpx.Client()\n", encoding="utf-8"
    )

    errors = validate(root)

    assert len(errors) == 1
    assert "direct httpx.Client" in errors[0]
