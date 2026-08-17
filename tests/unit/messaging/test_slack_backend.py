"""Tests for the Slack backend's file-attachment parsing (CONCEPT:AU-ECO.messaging.voice-attachment-fallback).

Only exercises the pure ``_slack_attachments`` helper — no ``slack-bolt`` import/network
required — so this runs in the base test environment regardless of whether the optional
``agent-utilities[messaging-slack]`` extra is installed.
"""

from __future__ import annotations

from agent_utilities.messaging.backends.slack import _slack_attachments
from agent_utilities.messaging.models import MediaType


def test_audio_file_becomes_audio_attachment_with_auth_header() -> None:
    """A Slack voice-clip/audio upload (an ordinary ``audio/*`` file in Slack's Events
    API — there is no distinct "voice message" object) must be classified so the core
    transcription path picks it up, and carry the bot's bearer token since Slack serves
    file bytes from an authenticated endpoint (unlike Telegram's pre-signed URL)."""
    message = {
        "channel": "C1",
        "files": [
            {
                "id": "F1",
                "name": "voice-message.ogg",
                "mimetype": "audio/ogg",
                "url_private_download": "https://files.slack.com/files-pri/T1-F1/voice.ogg",
                "size": 4096,
            }
        ],
    }
    attachments = _slack_attachments(message, "xoxb-bot-token")
    assert len(attachments) == 1
    att = attachments[0]
    assert att.media_type == MediaType.AUDIO
    assert att.url == "https://files.slack.com/files-pri/T1-F1/voice.ogg"
    assert att.mime_type == "audio/ogg"
    assert att.filename == "voice-message.ogg"
    assert att.auth_header == {"Authorization": "Bearer xoxb-bot-token"}


def test_non_audio_file_classified_as_file_not_audio() -> None:
    message = {
        "files": [
            {
                "id": "F2",
                "name": "report.pdf",
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/files-pri/T1-F2/report.pdf",
            }
        ]
    }
    attachments = _slack_attachments(message, "xoxb-bot-token")
    assert attachments[0].media_type == MediaType.FILE


def test_no_files_key_yields_no_attachments() -> None:
    assert _slack_attachments({"channel": "C1", "text": "hi"}, "xoxb-bot-token") == []


def test_file_without_url_is_skipped() -> None:
    message = {"files": [{"id": "F3", "mimetype": "audio/mp4"}]}
    assert _slack_attachments(message, "xoxb-bot-token") == []


def test_missing_bot_token_yields_no_auth_header() -> None:
    message = {
        "files": [
            {
                "id": "F1",
                "mimetype": "audio/ogg",
                "url_private_download": "https://files.slack.com/x",
            }
        ]
    }
    attachments = _slack_attachments(message, "")
    assert attachments[0].auth_header == {}
