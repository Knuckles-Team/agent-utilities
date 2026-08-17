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


def test_direct_httpx2_client_is_rejected(tmp_path: Path) -> None:
    """GOC-87: the egress boundary gate covers httpx2 exactly like httpx.

    Known-bad input: a file that constructs ``httpx2.AsyncClient()`` outside
    the sanctioned ``httpsupport.httpx2_adapter``/``transport_factory`` seam
    must be rejected, proving the gate actually enforces the staged
    migration's single-construction-point invariant rather than merely
    existing.
    """
    root = tmp_path / "agent_utilities"
    root.mkdir()
    (root / "unsafe2.py").write_text(
        "import httpx2\nclient = httpx2.AsyncClient()\n", encoding="utf-8"
    )

    errors = validate(root)

    assert len(errors) == 1
    assert "direct httpx2.AsyncClient" in errors[0]
