# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded protobuf wire primitives, authored without generated schema code.

Field facts: lczero-common b326b154/proto/net.proto; arithmetic behavior:
Lc0 fd71a2d9/src/utils/weights_adapter.cc. Neither source is vendored.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class ConvertError(ValueError):
    """Invalid or explicitly unsupported network input."""


@dataclass(frozen=True)
class Limits:
    max_compressed_bytes: int = 64 * 1024 * 1024
    max_decompressed_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_compressed_bytes <= 0 or self.max_decompressed_bytes <= 0:
            raise ConvertError("size limits must be positive")


@dataclass(frozen=True)
class Field:
    number: int
    wire: int
    value: int | memoryview


def _varint(data: memoryview, offset: int) -> tuple[int, int]:
    start = offset
    result = 0
    for shift in range(0, 70, 7):
        if offset == len(data):
            raise ConvertError("truncated varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise ConvertError("uint64 varint overflow")
        result |= (byte & 127) << shift
        if byte < 128:
            if offset - start > 1 and byte == 0:
                raise ConvertError("noncanonical varint")
            return result, offset
    raise ConvertError("unterminated varint")


def parse_fields(data: bytes | memoryview, *, max_fields: int = 4096) -> list[Field]:
    """Validate every field boundary before exposing zero-copy byte slices.

    Groups are deliberately unsupported. Nested known messages are interpreted
    by a fixed-depth schema, so untrusted nesting never drives recursion.
    """
    view = memoryview(data)
    offset = 0
    fields: list[Field] = []
    while offset < len(view):
        if len(fields) >= max_fields:
            raise ConvertError("protobuf field count limit exceeded")
        tag, offset = _varint(view, offset)
        number, wire = tag >> 3, tag & 7
        if not 1 <= number < (1 << 29):
            raise ConvertError("invalid protobuf field number")
        value: int | memoryview
        if wire == 0:
            value, offset = _varint(view, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                length, offset = _varint(view, offset)
            else:
                length = 8 if wire == 1 else 4
            if length > len(view) - offset:
                raise ConvertError("truncated protobuf field payload")
            value = view[offset : offset + length]
            offset += length
        else:
            raise ConvertError(f"unsupported protobuf wire type {wire}")
        fields.append(Field(number, wire, value))
    return fields


def message(
    data: bytes | memoryview, schema: dict[int, int], *, repeated: frozenset[int] = frozenset()
) -> dict[int, list[Field]]:
    """Unknown optional fields are skipped only after complete wire validation."""
    result: dict[int, list[Field]] = {}
    for item in parse_fields(data):
        if item.number not in schema:
            continue
        if item.wire != schema[item.number]:
            raise ConvertError(f"field {item.number}: wrong wire type {item.wire}")
        if item.number in result and item.number not in repeated:
            raise ConvertError(f"field {item.number}: duplicate singular field")
        result.setdefault(item.number, []).append(item)
    return result


def byte_field(items: dict[int, list[Field]], number: int) -> memoryview:
    if number not in items:
        raise ConvertError(f"missing required field {number}")
    value = items[number][0].value
    if not isinstance(value, memoryview):
        raise ConvertError(f"field {number}: expected bytes")
    return value


def int_field(items: dict[int, list[Field]], number: int) -> int:
    if number not in items:
        raise ConvertError(f"missing required field {number}")
    value = items[number][0].value
    if not isinstance(value, int):
        raise ConvertError(f"field {number}: expected integer")
    return value


def decode_layer(data: bytes | memoryview) -> NDArray[np.float32]:
    parts = message(data, {1: 5, 2: 5, 3: 2})
    low = float(struct.unpack("<f", byte_field(parts, 1))[0])
    high = float(struct.unpack("<f", byte_field(parts, 2))[0])
    if (
        not math.isfinite(low)
        or not math.isfinite(high)
        or low > high
        or high - low > float(np.finfo(np.float32).max)
    ):
        raise ConvertError("invalid finite float32 quantization range")
    payload = byte_field(parts, 3)
    if not payload or len(payload) % 2:
        raise ConvertError("LINEAR16 payload must contain nonempty uint16 values")
    result = np.frombuffer(payload, dtype="<u2").astype(np.float32)
    # Deliberately separate float32 operations: changing order changes weights.
    result /= np.float32(65535)
    result *= np.float32(np.float32(high) - np.float32(low))
    result += np.float32(low)
    return result


def read_gzip(
    path: Path, *, limits: Limits | None = None, expected_sha256: str | None = None
) -> tuple[bytes, str]:
    if expected_sha256 is not None and not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ConvertError("expected SHA256 must be 64 lowercase hexadecimal characters")
    bounds = Limits() if limits is None else limits
    try:
        if path.stat().st_size > bounds.max_compressed_bytes:
            raise ConvertError("compressed network size limit exceeded")
        with path.open("rb") as source:
            compressed = source.read(bounds.max_compressed_bytes + 1)
        if len(compressed) > bounds.max_compressed_bytes:
            raise ConvertError("compressed network size limit exceeded")
        if not compressed.startswith(b"\x1f\x8b"):
            raise ConvertError("invalid gzip header")
        digest = hashlib.sha256(compressed).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ConvertError(f"SHA256 mismatch: expected {expected_sha256}, got {digest}")
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            inflated = stream.read(bounds.max_decompressed_bytes + 1)
        if len(inflated) > bounds.max_decompressed_bytes:
            raise ConvertError("decompressed network size limit exceeded")
        return inflated, digest
    except (OSError, EOFError, zlib.error) as error:
        raise ConvertError(f"cannot read gzip network {path}: {error}") from error
