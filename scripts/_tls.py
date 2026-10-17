"""Shared TLS context for the operator scripts that fetch over HTTPS."""

from __future__ import annotations

import os
import ssl


def tls_context() -> ssl.SSLContext:
    """Default TLS context honoring SSL_CERT_FILE/REQUESTS_CA_BUNDLE/SSL_CERT_DIR."""
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    capath = os.environ.get("SSL_CERT_DIR")
    return ssl.create_default_context(cafile=cafile or None, capath=capath or None)
