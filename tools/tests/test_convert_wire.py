# SPDX-License-Identifier: GPL-3.0-or-later
"""Original wire/error cases; no generated or vendored protobuf parser."""

from __future__ import annotations

import gzip
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from mouse_tools.convert.schema import normalize_format
from mouse_tools.convert.wire import ConvertError, Limits, decode_layer, parse_fields, read_gzip


def varint(value: int) -> bytes:
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result) + bytes([value])


def field(number: int, wire: int, value: bytes | int) -> bytes:
    prefix = varint(number * 8 + wire)
    if isinstance(value, int):
        return prefix + varint(value)
    return prefix + (varint(len(value)) if wire == 2 else b"") + value


def layer(low: float, high: float, values: tuple[int, ...]) -> bytes:
    return (
        field(1, 5, struct.pack("<f", low))
        + field(2, 5, struct.pack("<f", high))
        + field(3, 2, struct.pack("<" + "H" * len(values), *values))
    )


def test_unknown_fields_are_bounded_and_preserved() -> None:
    parsed = parse_fields(
        field(15, 0, 800) + field(16, 1, b"12345678") + field(17, 2, b"abc") + field(18, 5, b"1234")
    )
    assert [(item.number, item.wire) for item in parsed] == [(15, 0), (16, 1), (17, 2), (18, 5)]
    assert parsed[0].value == 800
    assert parsed[2].value == b"abc"


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00",
        b"\x0b",
        b"\x0c",
        b"\x0e",
        b"\x08\x80",
        b"\x08" + b"\xff" * 10,
        b"\x08\x80\x00",
        b"\x0a\x05a",
        b"\x0dabc",
        b"\x09abc",
        varint((1 << 29) * 8),
    ],
)
def test_rejects_malformed_wire(payload: bytes) -> None:
    with pytest.raises(ConvertError):
        parse_fields(payload)


def test_field_count_limit() -> None:
    with pytest.raises(ConvertError, match="field count"):
        parse_fields(field(1, 0, 1) * 4, max_fields=3)


def test_linear16_exact_float32_operation_order() -> None:
    actual = decode_layer(layer(-1.1698321104049683, 1.4447356462478638, (0, 1, 237, 32767, 65535)))
    code = np.array([0, 1, 237, 32767, 65535], dtype=np.float32)
    expected = code / np.float32(65535)
    expected *= np.float32(np.float32(1.4447356462478638) - np.float32(-1.1698321104049683))
    expected += np.float32(-1.1698321104049683)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


@pytest.mark.parametrize(
    ("low", "high"), [(float("nan"), 1), (0, float("inf")), (2, 1), (-3e38, 3e38)]
)
def test_rejects_invalid_quantization_ranges(low: float, high: float) -> None:
    with pytest.raises(ConvertError, match="range"):
        decode_layer(layer(low, high, (1,)))


@pytest.mark.parametrize(
    "payload",
    [
        field(1, 5, struct.pack("<f", 0)),
        layer(0, 1, (1,)) + field(1, 5, struct.pack("<f", 0)),
        field(1, 0, 0) + field(2, 5, struct.pack("<f", 1)) + field(3, 2, b"aa"),
        field(1, 5, struct.pack("<f", 0)) + field(2, 5, struct.pack("<f", 1)) + field(3, 2, b"a"),
    ],
)
def test_rejects_malformed_layer(payload: bytes) -> None:
    with pytest.raises(ConvertError):
        decode_layer(payload)


def test_gzip_hash_crc_and_size_limits(tmp_path: Path) -> None:
    path = tmp_path / "net.pb.gz"
    compressed = gzip.compress(b"x" * 100)
    path.write_bytes(compressed)
    data, digest = read_gzip(path, expected_sha256=hashlib.sha256(compressed).hexdigest())
    assert data == b"x" * 100
    assert digest == hashlib.sha256(compressed).hexdigest()
    with pytest.raises(ConvertError, match="SHA256"):
        read_gzip(path, expected_sha256="0" * 64)
    with pytest.raises(ConvertError, match="decompressed"):
        read_gzip(path, limits=Limits(max_decompressed_bytes=99))
    with pytest.raises(ConvertError, match="compressed"):
        read_gzip(path, limits=Limits(max_compressed_bytes=2))
    path.write_bytes(compressed[:-1] + bytes([compressed[-1] ^ 1]))
    with pytest.raises(ConvertError, match="gzip"):
        read_gzip(path)


def raw_format(network: int = 4) -> dict[int, int]:
    return {1: 1, 2: 2, 3: network, 4: 3, 5: 2, 6: 1, 7: 1}


def test_legacy_upgrade_is_explicit_and_preserves_raw_presence() -> None:
    raw = raw_format()
    parsed = normalize_format(raw, encoder_count=10, has_smolgen=True)
    assert parsed.network == 6 and parsed.smolgen_activation == 7
    assert parsed.ffn_activation == 8 and parsed.input_embedding == 1
    assert parsed.raw[3] == 4 and 10 not in parsed.raw
    assert parsed.upgrades
    assert raw == raw_format()


def test_legacy_activation_override_matches_loader_even_when_present() -> None:
    parsed = normalize_format(raw_format() | {8: 0, 9: 0}, encoder_count=1, has_smolgen=True)
    assert parsed.smolgen_activation == 7 and parsed.ffn_activation == 8


@pytest.mark.parametrize("change", [{1: 2}, {3: 2}, {4: 1}, {5: 1}, {6: 0}, {7: 0}, {10: 0}])
def test_unsupported_formats_fail_with_reason(change: dict[int, int]) -> None:
    with pytest.raises(ConvertError, match="unsupported"):
        normalize_format(raw_format() | change, encoder_count=10, has_smolgen=True)


def test_missing_presence_and_native_activation_defaults_are_not_guessed() -> None:
    with pytest.raises(ConvertError, match="missing"):
        normalize_format({3: 4}, encoder_count=10, has_smolgen=True)
    with pytest.raises(ConvertError, match="unsupported"):
        normalize_format(raw_format(6), encoder_count=10, has_smolgen=True)
    with pytest.raises(ConvertError, match="unsupported"):
        normalize_format(raw_format(), encoder_count=0, has_smolgen=True)


def test_bad_deflate_stream_and_empty_file_have_clear_gzip_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.gz"
    corrupted = bytearray(gzip.compress(b"test" * 100))
    corrupted[10:13] = b"\xff\xff\xff"
    for payload in (bytes(corrupted), b""):
        path.write_bytes(payload)
        with pytest.raises(ConvertError, match="gzip"):
            read_gzip(path)


def test_unknown_activation_enum_is_not_hidden_by_legacy_override() -> None:
    with pytest.raises(ConvertError, match="unsupported"):
        normalize_format(raw_format() | {8: 999}, encoder_count=10, has_smolgen=True)
