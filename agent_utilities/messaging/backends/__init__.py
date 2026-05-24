"""Messaging Backend Implementations (CONCEPT:ECO-4.5).

Each backend module in this package implements the ``MessagingBackend`` ABC
for a specific messaging platform. Backends are lazily imported via
``importlib.metadata.entry_points`` to avoid pulling in platform-specific
dependencies unless explicitly installed.

Supported backends:
    - ``discord`` — Discord via ``discord.py``
    - ``slack`` — Slack via ``slack-bolt``
    - ``telegram`` — Telegram via ``python-telegram-bot``
    - ``whatsapp`` — WhatsApp via ``neonize`` + Business API
    - ``teams`` — Microsoft Teams via ``botbuilder-core``
    - ``googlechat`` — Google Chat via ``google-api-python-client``
    - ``googlemeet`` — Google Meet via ``google-api-python-client``
    - ``mattermost`` — Mattermost via ``mattermostdriver``
    - ``matrix`` — Matrix via ``matrix-nio``
    - ``irc`` — IRC via ``irc``
    - ``signal`` — Signal via ``semaphore-bot``
    - ``imessage`` — iMessage via AppleScript bridge
    - ``line`` — LINE via ``line-bot-sdk``
    - ``twitch`` — Twitch via ``twitchio``
    - ``synology`` — Synology Chat via webhook/httpx
    - ``voicecall`` — Voice Call via ``twilio``
    - ``nextcloud`` — Nextcloud Talk via httpx REST

Install individual backends with::

    pip install agent-utilities[messaging-discord]

Or install all with::

    pip install agent-utilities[messaging]
"""

# CONCEPT:ECO-4.5 — Native Messaging Backend Abstraction
