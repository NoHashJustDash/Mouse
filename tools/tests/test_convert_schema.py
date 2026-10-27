# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import numpy as np
import pytest
from test_convert_fixture import synthetic_net
from test_convert_wire import field

from mouse_tools.convert.schema import parse_net
from mouse_tools.convert.wire import ConvertError


def test_small_complete_net_has_exact_validated_shapes_and_count() -> None:
    net = parse_net(synthetic_net(), source_sha256="a" * 64)
    assert net.config.width == 4 and net.config.heads == 2 and net.config.encoders == 2
    assert net.tensors["ip_emb_w"].shape == (4, 176)
    assert net.tensors["encoder.1.mha.smolgen.dense1_w"].shape == (3, 64)
    assert net.parameter_count == sum(tensor.size for tensor in net.tensors.values())
    assert all(tensor.dtype == np.float32 for tensor in net.tensors.values())
    assert net.format.raw[3] == 4 and net.format.network == 6


@pytest.mark.parametrize(
    "name",
    [
        "ip_emb_w",
        "ip_mult_gate",
        "smolgen_w",
        "ip4_pol_w",
        "encoder.0.mha.k_w",
        "encoder.1.ffn.dense2_w",
        "encoder.0.mha.smolgen.dense1_w",
    ],
)
def test_every_tensor_length_is_checked(name: str) -> None:
    with pytest.raises(ConvertError, match="shape|dimension"):
        parse_net(synthetic_net(replacement=(name, 7)), source_sha256="a" * 64)


@pytest.mark.parametrize(
    "payload",
    [
        field(1, 2, b""),
        field(2, 2, b""),
        field(21, 2, b""),
        field(36, 2, b""),
        field(37, 2, b""),
        field(44, 2, b""),
    ],
)
def test_rejects_known_unsupported_architecture_weights(payload: bytes) -> None:
    with pytest.raises(ConvertError, match="unsupported"):
        parse_net(synthetic_net(extra_weights=payload), source_sha256="a" * 64)


def test_requires_global_linear16_and_valid_attention_dimensions() -> None:
    with pytest.raises(ConvertError, match="encoding"):
        parse_net(synthetic_net(encoding=2), source_sha256="a" * 64)
    for heads in (0, 3, 1000):
        with pytest.raises(ConvertError, match="head"):
            parse_net(synthetic_net(heads=heads), source_sha256="a" * 64)


def test_singular_top_level_and_weights_fields_cannot_duplicate() -> None:
    with pytest.raises(ConvertError, match="duplicate"):
        parse_net(synthetic_net() + field(10, 2, b""), source_sha256="a" * 64)
    with pytest.raises(ConvertError, match="duplicate"):
        parse_net(synthetic_net(extra_weights=field(28, 0, 2)), source_sha256="a" * 64)


def test_valid_unknown_optional_fields_can_be_skipped() -> None:
    expected = parse_net(synthetic_net(), source_sha256="a" * 64)
    actual = parse_net(
        synthetic_net(extra_weights=field(100, 2, b"extension")) + field(100, 0, 888),
        source_sha256="a" * 64,
    )
    assert actual.parameter_count == expected.parameter_count


def test_magic_source_hash_and_minimum_engine_version_validation() -> None:
    with pytest.raises(ConvertError, match="magic"):
        parse_net(b"\x0d\x00\x00\x00\x00", source_sha256="a" * 64)
    with pytest.raises(ConvertError, match="SHA256"):
        parse_net(synthetic_net(), source_sha256="invalid")
    payload = synthetic_net()
    marker = field(3, 2, field(1, 0, 0) + field(2, 0, 29) + field(3, 0, 0))
    future = field(3, 2, field(1, 0, 1) + field(2, 0, 0) + field(3, 0, 0))
    with pytest.raises(ConvertError, match="version"):
        parse_net(payload.replace(marker, future, 1), source_sha256="a" * 64)


def test_known_optional_nested_fields_still_require_valid_wire_and_utf8() -> None:
    for extra in (field(5, 2, b"\x08\x80"), field(2, 2, b"\xff")):
        with pytest.raises(ConvertError):
            parse_net(synthetic_net() + extra, source_sha256="a" * 64)
