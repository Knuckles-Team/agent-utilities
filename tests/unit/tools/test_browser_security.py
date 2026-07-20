from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.tools.browser import browser_interactions, browser_screenshot
from agent_utilities.tools.browser.browser_manager import require_safe_browser_url


def _context(tmp_path):
    return SimpleNamespace(deps=SimpleNamespace(workspace_path=str(tmp_path)))


def test_browser_url_policy_rejects_non_http_and_private_destinations(monkeypatch):
    from agent_utilities.tools.browser import browser_manager

    monkeypatch.setattr(browser_manager, "browser_fetch_enabled", lambda: True)
    with pytest.raises(ValueError):
        require_safe_browser_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        require_safe_browser_url("http://169.254.169.254/latest/meta-data")


def test_screenshot_writer_is_workspace_confined_private_and_non_overwriting(tmp_path):
    ctx = _context(tmp_path)
    asset_ref = browser_screenshot._write_screenshot(
        ctx, None, b"png-bytes", prefix="page"
    )
    assert asset_ref.startswith("browser-screenshot://page_")
    assert str(tmp_path) not in asset_ref

    written = list((tmp_path / ".agents" / "browser").glob("*.png"))
    assert len(written) == 1
    assert written[0].read_bytes() == b"png-bytes"

    existing = tmp_path / "existing.png"
    existing.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        browser_screenshot._write_screenshot(
            ctx, "existing.png", b"replacement", prefix="page"
        )
    assert existing.read_bytes() == b"original"

    with pytest.raises(ValueError):
        browser_screenshot._write_screenshot(
            ctx, str(tmp_path / "outside.png"), b"png", prefix="page"
        )


@pytest.mark.asyncio
async def test_browser_interaction_is_opt_in_and_sanitizes_bounded_text(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(browser_interactions, "browser_fetch_enabled", lambda: False)
    denied = await browser_interactions.get_element_text(_context(tmp_path), "#content")
    assert denied["success"] is False

    class Page:
        async def inner_text(self, _selector):
            return "contact person@example.test " + ("x" * (70 * 1024))

    class Manager:
        async def get_current_page(self):
            return Page()

    monkeypatch.setattr(browser_interactions, "browser_fetch_enabled", lambda: True)
    monkeypatch.setattr(browser_interactions, "get_browser_manager", lambda: Manager())
    result = await browser_interactions.get_element_text(_context(tmp_path), "#content")
    assert result["success"] is True
    assert result["truncated"] is True
    assert "person@example.test" not in result["text"]
    assert result["privacy_redactions"] >= 1
