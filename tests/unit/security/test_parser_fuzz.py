"""Deterministic bounded fuzzing for security-sensitive public parsers."""

from __future__ import annotations

import random
import string

from agent_utilities.mcp.server_factory import (
    _FILTER_VALUE_RE,
    _bounded_filter_values,
)
from agent_utilities.security.cli_secrets import (
    RuntimeSecretReferenceError,
    _validated_reference,
)
from agent_utilities.security.http_boundary import parse_cidrs

_SEED = 0xA63E17
_ALPHABET = (
    string.ascii_letters
    + string.digits
    + "_./#-:, "
    + "".join(chr(value) for value in (0, 9, 10, 13))
)


def _random_text(randomizer: random.Random, maximum: int = 300) -> str:
    return "".join(
        randomizer.choice(_ALPHABET) for _ in range(randomizer.randrange(maximum + 1))
    )


def test_secret_reference_parser_fuzz_is_bounded_and_scheme_closed():
    randomizer = random.Random(_SEED)
    for _ in range(512):
        value = _random_text(randomizer)
        try:
            scheme, rendered = _validated_reference(value)
        except RuntimeSecretReferenceError as exc:
            assert "runtime secret reference" in str(exc)
            continue
        assert scheme in {"env", "vault", "secret"}
        assert rendered == value.strip()
        assert len(rendered.encode("utf-8")) <= 1_024
        assert not any(
            character.isspace() or ord(character) < 32 for character in rendered
        )

    for index in range(256):
        expected = f"FUZZ_SECRET_{index}"
        assert _validated_reference(f"env://{expected}") == (
            "env",
            f"env://{expected}",
        )


def test_visibility_filter_parser_fuzz_never_broadens_or_exceeds_limits():
    randomizer = random.Random(_SEED + 1)
    for _ in range(512):
        values = [_random_text(randomizer) for _ in range(randomizer.randrange(9))]
        try:
            parsed = _bounded_filter_values(values)
        except ValueError as exc:
            assert str(exc).startswith("MCP visibility filter")
            continue
        assert len(parsed) <= 256
        assert len(parsed) == len(set(parsed))
        assert all(value.lower() != "all" for value in parsed)
        assert all(_FILTER_VALUE_RE.fullmatch(value) for value in parsed)


def test_trusted_proxy_cidr_parser_fuzz_accepts_only_exact_networks():
    randomizer = random.Random(_SEED + 2)
    for _ in range(256):
        values = [_random_text(randomizer, 80) for _ in range(randomizer.randrange(5))]
        try:
            networks = parse_cidrs(values)
        except ValueError as exc:
            assert "trusted proxy CIDR" in str(exc)
            continue
        assert 1 <= len(networks) <= 64
        assert all(str(network.network_address) in str(network) for network in networks)

    exact = [f"10.{index}.0.0/16" for index in range(64)]
    assert len(parse_cidrs(exact)) == 64
