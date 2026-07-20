"""Lifecycle-owned HTTPS OIDC authority for exact skill certification.

The authority exists only inside the certification orchestrator.  It binds an
ephemeral loopback port, generates a private CA and leaf certificate for that
run, verifies its own TLS endpoint before exposing runtime environment
references, and removes its private work directory during shutdown.  Durable
configuration and evidence receive only the fixed authority mode, bounded
token lifetime, booleans, and aggregate counts.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import socketserver
import ssl
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

AUTHORITY_MODE = "ephemeral-https-loopback"
DEFAULT_TOKEN_TTL_SECONDS = 300
MIN_TOKEN_TTL_SECONDS = 180
MAX_TOKEN_TTL_SECONDS = 3_600

_BIND_HOST = "127.0.0.1"
_AUDIENCE = "graph-os-skill-certification"
_CLIENT_SECRET_ENV = "GRAPHOS_SKILL_CERT_OIDC_CLIENT_SECRET"
_TLS_PROFILE_ENV = "GRAPHOS_SKILL_CERT_OIDC_TLS_PROFILE"
_CLIENT_SECRET_REF = f"env://{_CLIENT_SECRET_ENV}"
_TLS_PROFILE_REF = f"env://{_TLS_PROFILE_ENV}"
_SOCKET_DEADLINE_SECONDS = 3.0
_MAX_REQUESTS = 256
_MAX_REQUEST_LINE_BYTES = 2_048
_MAX_HEADER_BYTES = 16_384
_MAX_HEADER_COUNT = 32
_MAX_HEADER_LINE_BYTES = 2_048
_MAX_BODY_BYTES = 8_192
_MAX_FORM_FIELDS = 8
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


class CertificationAuthorityError(RuntimeError):
    """The ephemeral authority violated its bounded lifecycle contract."""


class _RequestError(CertificationAuthorityError):
    def __init__(
        self, status: HTTPStatus, oauth_error: str = "invalid_request"
    ) -> None:
        super().__init__(oauth_error)
        self.status = status
        self.oauth_error = oauth_error


@dataclass(frozen=True)
class _Request:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


def validated_token_ttl_seconds(value: int | None) -> int:
    """Return the current bounded certification token lifetime."""

    token_ttl_seconds = DEFAULT_TOKEN_TTL_SECONDS if value is None else value
    if (
        isinstance(token_ttl_seconds, bool)
        or not isinstance(token_ttl_seconds, int)
        or not MIN_TOKEN_TTL_SECONDS <= token_ttl_seconds <= MAX_TOKEN_TTL_SECONDS
    ):
        raise CertificationAuthorityError(
            "token lifetime is outside the certification range"
        )
    return token_ttl_seconds


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_uint(value: int) -> str:
    width = max(1, (value.bit_length() + 7) // 8)
    return _base64url(value.to_bytes(width, "big"))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class _Authority:
    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        issuer: str,
        token_ttl_seconds: int,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self.issuer = issuer
        self._token_ttl_seconds = validated_token_ttl_seconds(token_ttl_seconds)
        self._subject = f"subject:opaque:{secrets.token_hex(16)}"
        self._tenant = f"tenant:opaque:{secrets.token_hex(16)}"
        self._private_key: rsa.RSAPrivateKey | None = rsa.generate_private_key(
            public_exponent=65_537,
            key_size=2_048,
        )
        public_key = self._private_key.public_key()
        public_der = public_key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self._kid = _base64url(hashlib.sha256(public_der).digest())
        numbers = public_key.public_numbers()
        self.jwks = {
            "keys": [
                {
                    "alg": "RS256",
                    "e": _base64url_uint(numbers.e),
                    "kid": self._kid,
                    "kty": "RSA",
                    "n": _base64url_uint(numbers.n),
                    "use": "sig",
                }
            ]
        }
        self._mint_count = 0
        self._mint_lock = threading.Lock()

    @property
    def mint_count(self) -> int:
        with self._mint_lock:
            return self._mint_count

    def close(self) -> None:
        self._client_id = ""
        self._client_secret = ""
        self._subject = ""
        self._tenant = ""
        self._private_key = None

    def discovery(self) -> dict[str, Any]:
        return {
            "grant_types_supported": ["client_credentials"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "issuer": self.issuer,
            "jwks_uri": f"{self.issuer}/jwks",
            "scopes_supported": ["kg:admin"],
            "subject_types_supported": ["public"],
            "token_endpoint": f"{self.issuer}/token",
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
            ],
        }

    def _authenticate(self, authorization: str) -> bool:
        scheme, separator, encoded = authorization.partition(" ")
        if scheme.lower() != "basic" or not separator or not encoded:
            return False
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("ascii")
        except (ValueError, UnicodeDecodeError):
            return False
        client_id, separator, client_secret = decoded.partition(":")
        return bool(
            separator
            and secrets.compare_digest(client_id, self._client_id)
            and secrets.compare_digest(client_secret, self._client_secret)
        )

    def mint(self) -> str:
        key = self._private_key
        if key is None:
            raise CertificationAuthorityError("signing authority is closed")
        now = int(time.time())
        header = {"alg": "RS256", "kid": self._kid, "typ": "at+jwt"}
        claims = {
            "aud": _AUDIENCE,
            "exp": now + self._token_ttl_seconds,
            "iat": now,
            "iss": self.issuer,
            "jti": secrets.token_hex(16),
            "nbf": now - 1,
            "roles": ["kg:admin"],
            "scope": "kg:admin",
            "sub": self._subject,
            "tenant_id": self._tenant,
        }
        signing_input = (
            f"{_base64url(_json_bytes(header))}.{_base64url(_json_bytes(claims))}"
        ).encode("ascii")
        signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        with self._mint_lock:
            self._mint_count += 1
        return f"{signing_input.decode('ascii')}.{_base64url(signature)}"

    def token(self, request: _Request) -> tuple[HTTPStatus, dict[str, Any]]:
        content_type = request.headers.get("content-type", "").partition(";")[0]
        if content_type.strip().lower() != "application/x-www-form-urlencoded":
            return HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
        try:
            form = parse_qs(
                request.body.decode("ascii"),
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=_MAX_FORM_FIELDS,
            )
        except (UnicodeDecodeError, ValueError):
            return HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
        if any(len(values) != 1 for values in form.values()):
            return HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
        fields = {name: values[0] for name, values in form.items()}
        if set(fields) - {
            "audience",
            "client_id",
            "client_secret",
            "grant_type",
            "scope",
        }:
            return HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
        authorization = request.headers.get("authorization", "")
        body_credentials = "client_id" in fields or "client_secret" in fields
        if authorization and body_credentials:
            return HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}
        authenticated = (
            self._authenticate(authorization)
            if authorization
            else bool(
                set(fields) >= {"client_id", "client_secret"}
                and secrets.compare_digest(fields["client_id"], self._client_id)
                and secrets.compare_digest(fields["client_secret"], self._client_secret)
            )
        )
        if not authenticated:
            return HTTPStatus.UNAUTHORIZED, {"error": "invalid_client"}
        if fields.get("grant_type") != "client_credentials":
            return HTTPStatus.BAD_REQUEST, {"error": "unsupported_grant_type"}
        if fields.get("audience", _AUDIENCE) != _AUDIENCE:
            return HTTPStatus.BAD_REQUEST, {"error": "invalid_target"}
        if fields.get("scope", "kg:admin").split() != ["kg:admin"]:
            return HTTPStatus.BAD_REQUEST, {"error": "invalid_scope"}
        return HTTPStatus.OK, {
            "access_token": self.mint(),
            "expires_in": self._token_ttl_seconds,
            "scope": "kg:admin",
            "token_type": "Bearer",
        }


def _read_request(connection: socket.socket) -> _Request:
    deadline = time.monotonic() + _SOCKET_DEADLINE_SECONDS
    incoming = bytearray()
    header_end = -1
    while header_end < 0:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _RequestError(HTTPStatus.REQUEST_TIMEOUT)
        connection.settimeout(remaining)
        chunk = connection.recv(4_096)
        if not chunk:
            raise _RequestError(HTTPStatus.BAD_REQUEST)
        incoming.extend(chunk)
        header_end = incoming.find(b"\r\n\r\n")
        if header_end < 0 and len(incoming) > _MAX_HEADER_BYTES:
            raise _RequestError(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE)
    if header_end + 4 > _MAX_HEADER_BYTES:
        raise _RequestError(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE)
    header_block = bytes(incoming[:header_end])
    body = bytearray(incoming[header_end + 4 :])
    lines = header_block.split(b"\r\n")
    if not lines or len(lines[0]) > _MAX_REQUEST_LINE_BYTES:
        raise _RequestError(HTTPStatus.REQUEST_URI_TOO_LONG)
    try:
        method, target, version = lines[0].decode("ascii").split(" ")
    except (UnicodeDecodeError, ValueError):
        raise _RequestError(HTTPStatus.BAD_REQUEST) from None
    if method not in {"GET", "POST"} or version not in {"HTTP/1.0", "HTTP/1.1"}:
        raise _RequestError(HTTPStatus.METHOD_NOT_ALLOWED)
    parsed_target = urlsplit(target)
    if (
        not target.startswith("/")
        or parsed_target.scheme
        or parsed_target.netloc
        or parsed_target.query
        or parsed_target.fragment
    ):
        raise _RequestError(HTTPStatus.BAD_REQUEST)
    if len(lines) - 1 > _MAX_HEADER_COUNT:
        raise _RequestError(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or len(line) > _MAX_HEADER_LINE_BYTES or b":" not in line:
            raise _RequestError(HTTPStatus.BAD_REQUEST)
        raw_name, raw_value = line.split(b":", 1)
        try:
            name = raw_name.decode("ascii").lower()
            value = raw_value.strip().decode("ascii")
        except UnicodeDecodeError:
            raise _RequestError(HTTPStatus.BAD_REQUEST) from None
        if not _HEADER_NAME.fullmatch(name) or name in headers:
            raise _RequestError(HTTPStatus.BAD_REQUEST)
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise _RequestError(HTTPStatus.BAD_REQUEST)
        headers[name] = value
    if "transfer-encoding" in headers:
        raise _RequestError(HTTPStatus.BAD_REQUEST)
    raw_length = headers.get("content-length", "0")
    if not raw_length.isascii() or not raw_length.isdigit():
        raise _RequestError(HTTPStatus.BAD_REQUEST)
    content_length = int(raw_length)
    if content_length > _MAX_BODY_BYTES:
        raise _RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    if method == "GET" and content_length:
        raise _RequestError(HTTPStatus.BAD_REQUEST)
    if len(body) > content_length:
        raise _RequestError(HTTPStatus.BAD_REQUEST)
    while len(body) < content_length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _RequestError(HTTPStatus.REQUEST_TIMEOUT)
        connection.settimeout(remaining)
        chunk = connection.recv(min(4_096, content_length - len(body)))
        if not chunk:
            raise _RequestError(HTTPStatus.BAD_REQUEST)
        body.extend(chunk)
    return _Request(method, parsed_target.path, headers, bytes(body))


def _send_json(
    connection: socket.socket, status: HTTPStatus, payload: dict[str, Any]
) -> None:
    body = _json_bytes(payload)
    headers = [
        f"HTTP/1.1 {status.value} {status.phrase}",
        "Cache-Control: no-store",
        "Connection: close",
        f"Content-Length: {len(body)}",
        "Content-Type: application/json",
        "X-Content-Type-Options: nosniff",
    ]
    if status == HTTPStatus.UNAUTHORIZED:
        headers.append('WWW-Authenticate: Basic realm="graphos-certification"')
    connection.sendall("\r\n".join(headers).encode("ascii") + b"\r\n\r\n" + body)


class _LoopbackServer(socketserver.TCPServer):
    allow_reuse_address = False
    request_queue_size = 8

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        tls_context: ssl.SSLContext,
        stop_event: threading.Event,
    ) -> None:
        self.authority: _Authority | None = None
        self.tls_context = tls_context
        self.stop_event = stop_event
        self.request_count = 0
        super().__init__(server_address, _Handler, bind_and_activate=True)
        self.timeout = 0.1

    def get_request(self) -> tuple[socket.socket, Any]:
        connection, address = super().get_request()
        try:
            connection.settimeout(_SOCKET_DEADLINE_SECONDS)
            secured = self.tls_context.wrap_socket(connection, server_side=True)
            return secured, address
        except Exception:
            connection.close()
            raise

    def verify_request(self, request: socket.socket, client_address: Any) -> bool:
        return bool(client_address and client_address[0] == _BIND_HOST)

    def handle_error(self, request: socket.socket, client_address: Any) -> None:
        return

    def admit(self) -> bool:
        if self.request_count >= _MAX_REQUESTS:
            self.stop_event.set()
            return False
        self.request_count += 1
        return True


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        if not isinstance(server, _LoopbackServer) or server.authority is None:
            return
        try:
            if not server.admit():
                _send_json(
                    self.request, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "busy"}
                )
                return
            request = _read_request(self.request)
            expected_host = f"{_BIND_HOST}:{server.server_address[1]}"
            if request.headers.get("host") != expected_host:
                raise _RequestError(HTTPStatus.BAD_REQUEST)
            if request.method == "GET" and request.path == (
                "/.well-known/openid-configuration"
            ):
                status, payload = HTTPStatus.OK, server.authority.discovery()
            elif request.method == "GET" and request.path == "/jwks":
                status, payload = HTTPStatus.OK, server.authority.jwks
            elif request.method == "POST" and request.path == "/token":
                status, payload = server.authority.token(request)
            else:
                status, payload = HTTPStatus.NOT_FOUND, {"error": "not_found"}
            _send_json(self.request, status, payload)
        except _RequestError as exc:
            try:
                _send_json(self.request, exc.status, {"error": exc.oauth_error})
            except OSError:
                return
        except Exception:
            try:
                _send_json(
                    self.request,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "server_error"},
                )
            except OSError:
                return


def _private_file(root: Path, name: str, payload: bytes) -> Path:
    path = root / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise CertificationAuthorityError("private authority material is invalid")
    finally:
        os.close(descriptor)
    return path


def _tls_material(root: Path) -> tuple[bytes, Path, Path]:
    ca_key = rsa.generate_private_key(public_exponent=65_537, key_size=2_048)
    leaf_key = rsa.generate_private_key(public_exponent=65_537, key_size=2_048)
    now = datetime.now(UTC)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ephemeral-ca")])
    leaf_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "ephemeral-loopback")]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=8))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=8))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(_BIND_HOST))]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    cert_path = _private_file(
        root, "server-cert.pem", leaf_cert.public_bytes(serialization.Encoding.PEM)
    )
    key_path = _private_file(
        root,
        "server-key.pem",
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )
    return ca_pem, cert_path, key_path


class EphemeralLoopbackOidcAuthority:
    """Own one verified HTTPS authority for exactly one certification run."""

    def __init__(self, *, token_ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS) -> None:
        self.token_ttl_seconds = validated_token_ttl_seconds(token_ttl_seconds)
        self._work_root: Path | None = None
        self._server: _LoopbackServer | None = None
        self._authority: _Authority | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._client_id = ""
        self._client_secret = ""
        self._ca_pem = b""
        self._tls_verified = False

    @property
    def running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    @property
    def tls_verified(self) -> bool:
        return self._tls_verified

    @property
    def token_mint_count(self) -> int:
        return self._authority.mint_count if self._authority is not None else 0

    @property
    def issuer(self) -> str:
        server = self._server
        if server is None:
            raise CertificationAuthorityError("authority is not running")
        return f"https://{_BIND_HOST}:{server.server_address[1]}"

    def _client_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cadata=self._ca_pem.decode("ascii"))
        return context

    def _verified_json(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        server = self._server
        if server is None:
            raise CertificationAuthorityError("authority is not running")
        connection = http.client.HTTPSConnection(
            _BIND_HOST,
            server.server_address[1],
            timeout=_SOCKET_DEADLINE_SECONDS,
            context=self._client_context(),
        )
        try:
            connection.request(method, path, body=body, headers=dict(headers or {}))
            response = connection.getresponse()
            payload = response.read(_MAX_BODY_BYTES + 1)
            if response.status != HTTPStatus.OK or len(payload) > _MAX_BODY_BYTES:
                raise CertificationAuthorityError("authority verification failed")
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise CertificationAuthorityError("authority verification failed")
            return value
        except CertificationAuthorityError:
            raise
        except Exception as exc:
            raise CertificationAuthorityError("authority verification failed") from exc
        finally:
            connection.close()

    def start(self) -> EphemeralLoopbackOidcAuthority:
        if self._server is not None or self._thread is not None:
            raise CertificationAuthorityError("authority lifecycle is invalid")
        try:
            self._work_root = Path(tempfile.mkdtemp(prefix="graphos-skill-cert-"))
            self._work_root.chmod(0o700)
            self._ca_pem, cert_path, key_path = _tls_material(self._work_root)
            tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
            tls_context.load_cert_chain(str(cert_path), str(key_path))
            self._server = _LoopbackServer(
                (_BIND_HOST, 0),
                tls_context=tls_context,
                stop_event=self._stop_event,
            )
            self._client_id = secrets.token_urlsafe(24)
            self._client_secret = secrets.token_urlsafe(48)
            self._authority = _Authority(
                client_id=self._client_id,
                client_secret=self._client_secret,
                issuer=self.issuer,
                token_ttl_seconds=self.token_ttl_seconds,
            )
            self._server.authority = self._authority

            def serve() -> None:
                server = self._server
                if server is None:
                    return
                while not self._stop_event.is_set():
                    try:
                        server.handle_request()
                    except OSError:
                        if self._stop_event.is_set():
                            return
                        raise

            self._thread = threading.Thread(
                target=serve,
                name="graphos-skill-certification-oidc",
                daemon=False,
            )
            self._thread.start()
            discovery = self._verified_json("GET", "/.well-known/openid-configuration")
            self._tls_verified = bool(
                discovery.get("issuer") == self.issuer
                and discovery.get("jwks_uri") == f"{self.issuer}/jwks"
                and discovery.get("token_endpoint") == f"{self.issuer}/token"
            )
            if not self._tls_verified:
                raise CertificationAuthorityError("authority verification failed")
            return self
        except Exception:
            self.stop()
            raise

    def child_environment(
        self,
        base_environment: Mapping[str, str],
        *,
        model_private_hosts: list[str],
    ) -> dict[str, str]:
        """Return an isolated exact-child environment with reference-backed auth."""

        if not self.running or not self.tls_verified:
            raise CertificationAuthorityError("authority is not ready")
        hosts = {
            str(host).strip().casefold().rstrip(".")
            for host in model_private_hosts
            if str(host).strip()
        }
        hosts.add(_BIND_HOST)
        if len(hosts) > 256:
            raise CertificationAuthorityError("private host boundary exceeded")
        profile = {
            "ca_bundle_pem": self._ca_pem.decode("ascii"),
            "system_trust": False,
            "trust_env": False,
        }
        oauth2 = {
            "audience": _AUDIENCE,
            "client_id": self._client_id,
            "client_secret": _CLIENT_SECRET_REF,
            "scope": "kg:admin",
            "tls_profile_ref": _TLS_PROFILE_REF,
            "token_auth_style": "basic",
            "token_url": f"{self.issuer}/token",
        }
        environment = dict(base_environment)
        environment.update(
            {
                _CLIENT_SECRET_ENV: self._client_secret,
                _TLS_PROFILE_ENV: json.dumps(profile, sort_keys=True),
                "AUTH_TYPE": "jwt",
                "AUTH_JWT_ALGORITHMS": json.dumps(["RS256"]),
                "AUTH_JWT_AUDIENCE": _AUDIENCE,
                "AUTH_JWT_ISSUER": self.issuer,
                "AUTH_JWT_JWKS_URI": f"{self.issuer}/jwks",
                "FASTMCP_SERVER_AUTH_JWT_ALGORITHM": "RS256",
                "FASTMCP_SERVER_AUTH_JWT_AUDIENCE": _AUDIENCE,
                "FASTMCP_SERVER_AUTH_JWT_ISSUER": self.issuer,
                "FASTMCP_SERVER_AUTH_JWT_JWKS_URI": f"{self.issuer}/jwks",
                "FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY": "",
                "FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES": "kg:admin",
                "FASTMCP_SERVER_AUTH_JWT_SECRET_REF": "",
                "KG_AUTH_TOKEN_REF": "",
                "KG_IDENTITY_OAUTH2": json.dumps(oauth2, sort_keys=True),
                "KG_POLICY_VERSION": "skill-certification-v2",
                "MCP_CLIENT_AUTH": "oidc-client-credentials",
                "MODEL_HTTP_ALLOWED_PRIVATE_HOSTS": json.dumps(sorted(hosts)),
                "OIDC_AUDIENCE": _AUDIENCE,
                "OIDC_CLIENT_ID": self._client_id,
                "OIDC_CLIENT_SECRET_REF": _CLIENT_SECRET_REF,
                "OIDC_CONFIG_URL": "",
                "OIDC_HTTP_ALLOWED_PRIVATE_HOSTS": json.dumps([_BIND_HOST]),
                "OIDC_ISSUER": self.issuer,
                "OIDC_SCOPE": "kg:admin",
                "OIDC_TLS_PROFILE": "",
                "OIDC_TLS_PROFILE_REF": _TLS_PROFILE_REF,
                "OIDC_TOKEN_URL": f"{self.issuer}/token",
            }
        )
        return environment

    def prove_renewable(self) -> bool:
        """Mint and verify two distinct credentials through the HTTPS endpoint."""

        if not self.running or not self.tls_verified:
            return False
        credentials = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode("ascii")
        ).decode("ascii")
        previous_mint_count = self.token_mint_count
        responses = [
            self._verified_json(
                "POST",
                "/token",
                body=urlencode(
                    {
                        "audience": _AUDIENCE,
                        "grant_type": "client_credentials",
                        "scope": "kg:admin",
                    }
                ).encode("ascii"),
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            for _ in range(2)
        ]
        tokens = [response.get("access_token") for response in responses]
        return bool(
            all(isinstance(token, str) and token for token in tokens)
            and tokens[0] != tokens[1]
            and all(
                response.get("expires_in") == self.token_ttl_seconds
                for response in responses
            )
            and self.token_mint_count >= previous_mint_count + 2
        )

    def stop(self) -> None:
        self._stop_event.set()
        server = self._server
        if server is not None:
            server.server_close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=_SOCKET_DEADLINE_SECONDS + 1.0)
        if thread is not None and thread.is_alive():
            raise CertificationAuthorityError("authority did not stop")
        if self._authority is not None:
            self._authority.close()
        self._server = None
        self._authority = None
        self._thread = None
        self._client_id = ""
        self._client_secret = ""
        self._ca_pem = b""
        self._tls_verified = False
        if self._work_root is not None:
            shutil.rmtree(self._work_root, ignore_errors=True)
        self._work_root = None

    def __enter__(self) -> EphemeralLoopbackOidcAuthority:
        return self.start()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stop()


def self_check() -> bool:
    """Exercise HTTPS verification, renewal, and complete cleanup."""

    authority = EphemeralLoopbackOidcAuthority()
    try:
        authority.start()
        return bool(authority.prove_renewable() and authority.token_mint_count >= 2)
    finally:
        authority.stop()


def main(argv: list[str] | None = None) -> int:
    """Run the bounded source self-check; no external server mode is exposed."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments not in ([], ["--self-check"]):
        print(json.dumps({"ok": False}, sort_keys=True))
        return 1
    try:
        ok = self_check()
    except Exception:
        ok = False
    print(json.dumps({"ok": ok}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
