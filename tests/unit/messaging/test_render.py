"""Markdown → Telegram-HTML rendering (CONCEPT:ECO-4.0 entrypoint rendering)."""

from __future__ import annotations

from agent_utilities.messaging.render import markdown_to_telegram_html as render


def test_inline_emphasis_and_code() -> None:
    out = render("**bold** and *italic* and `code`")
    assert out == "<b>bold</b> and <i>italic</i> and <code>code</code>"


def test_headers_become_bold_and_bullets() -> None:
    out = render("## Stacks\n- portainer\n- graph-os")
    assert "<b>Stacks</b>" in out
    assert "• portainer" in out and "• graph-os" in out
    assert "#" not in out and out.count("- ") == 0


def test_code_block_is_escaped_verbatim() -> None:
    out = render("```python\nprint(1 < 2 & 3)\n```")
    # HTML specials inside code are escaped; the body is wrapped in <pre>, no inner markdown.
    assert "<pre>" in out and "</pre>" in out
    assert "1 &lt; 2 &amp; 3" in out


def test_links_render_as_anchors() -> None:
    out = render("see [the repo](https://example.com/x?a=1)")
    assert '<a href="https://example.com/x?a=1">the repo</a>' in out


def test_plain_text_html_specials_are_escaped() -> None:
    out = render("a < b & c > d, no markdown")
    assert out == "a &lt; b &amp; c &gt; d, no markdown"


def test_empty_is_passthrough() -> None:
    assert render("") == ""
