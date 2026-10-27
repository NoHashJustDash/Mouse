# SPDX-License-Identifier: GPL-3.0-or-later
"""Presence-sensitive format normalization for the supported T1 architecture.

Facts: lczero-common b326b154/proto/net.proto; explicit compatibility migration
matches Lc0 fd71a2d9/src/neural/loader.cc:108-157, without copied source.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .wire import ConvertError, Field, byte_field, decode_layer, int_field, message


@dataclass(frozen=True)
class NetworkFormat:
    raw: Mapping[int, int]
    network: int
    smolgen_activation: int
    ffn_activation: int
    input_embedding: int
    upgrades: tuple[str, ...]

    @property
    def effective(self) -> dict[str, int]:
        names = (
            "input",
            "output",
            "network",
            "policy",
            "value",
            "moves_left",
            "default_activation",
        )
        result = {name: self.raw[index] for index, name in enumerate(names, 1)}
        return result | {
            "network": self.network,
            "smolgen_activation": self.smolgen_activation,
            "ffn_activation": self.ffn_activation,
            "input_embedding": self.input_embedding,
        }


def normalize_format(
    raw: Mapping[int, int], *, encoder_count: int, has_smolgen: bool
) -> NetworkFormat:
    for number in range(1, 8):
        if number not in raw:
            raise ConvertError(f"missing network_format field {number}")
    for number in (8, 9):
        if number in raw and raw[number] not in range(9):
            raise ConvertError(f"unsupported activation enum {number}={raw[number]}")
    effective = dict(raw)
    upgrades: list[str] = []
    if effective[3] == 4 and encoder_count > 0:
        effective[3] = 6
        upgrades.append("legacy network 4 with encoders upgraded to attention 6")
        if has_smolgen:
            effective[8], effective[9] = 7, 8
            upgrades.append(
                "legacy smolgen activations explicitly overridden to Swish/squared-ReLU"
            )
    if effective[3] == 6 and 10 not in effective:
        effective[10] = 1
        upgrades.append("absent attention input_embedding upgraded to PE_MAP")
    expected = {1: 1, 2: 2, 3: 6, 4: 3, 5: 2, 6: 1, 7: 1, 8: 7, 9: 8, 10: 1}
    for number, wanted in expected.items():
        if effective.get(number, 0) != wanted:
            raise ConvertError(
                f"unsupported network_format field {number}="
                f"{effective.get(number, 0)}; supported value is {wanted}"
            )
    if not 1 <= encoder_count <= 64 or not has_smolgen:
        raise ConvertError("unsupported attention body: requires 1..64 encoders with smolgen")
    return NetworkFormat(
        MappingProxyType(dict(raw)),
        effective[3],
        effective[8],
        effective[9],
        effective[10],
        tuple(upgrades),
    )


# These are independently expressed field facts, not generated protobuf code.
_ROOT_LAYERS = {
    4: "ip_pol_w",
    5: "ip_pol_b",
    7: "ip1_val_w",
    8: "ip1_val_b",
    9: "ip2_val_w",
    10: "ip2_val_b",
    13: "ip1_mov_w",
    14: "ip1_mov_b",
    15: "ip2_mov_w",
    16: "ip2_mov_b",
    17: "ip2_pol_w",
    18: "ip2_pol_b",
    19: "ip3_pol_w",
    20: "ip3_pol_b",
    22: "ip4_pol_w",
    25: "ip_emb_w",
    26: "ip_emb_b",
    29: "ip_val_w",
    30: "ip_val_b",
    31: "ip_mov_w",
    32: "ip_mov_b",
    33: "ip_mult_gate",
    34: "ip_add_gate",
    35: "smolgen_w",
}
_UNSUPPORTED_WEIGHTS = frozenset(
    {1, 2, 3, 6, 11, 12, 21, 24, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45}
)
_MHA_LAYERS = dict(enumerate(("q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "dense_w", "dense_b"), 1))
_SMOL_LAYERS = dict(
    enumerate(
        (
            "compress",
            "dense1_w",
            "dense1_b",
            "ln1_gammas",
            "ln1_betas",
            "dense2_w",
            "dense2_b",
            "ln2_gammas",
            "ln2_betas",
        ),
        1,
    )
)
_FFN_LAYERS = dict(enumerate(("dense1_w", "dense1_b", "dense2_w", "dense2_b"), 1))


@dataclass(frozen=True)
class Config:
    width: int
    heads: int
    encoders: int
    d_model: int
    ffn_hidden: int
    smol_compress: int
    smol_hidden: int
    smol_gen: int
    policy_embedding: int
    policy_model: int
    value_embedding: int
    value_hidden: int
    mlh_embedding: int
    mlh_hidden: int


@dataclass(frozen=True)
class Network:
    format: NetworkFormat
    config: Config
    tensors: Mapping[str, NDArray[np.float32]]
    source_sha256: str
    parameter_count: int
    minimum_version: tuple[int, int, int]


def _group(
    parts: dict[int, list[Field]],
    names: dict[int, str],
    prefix: str,
    target: dict[str, NDArray[np.float32]],
) -> None:
    for number, name in names.items():
        try:
            target[prefix + name] = decode_layer(byte_field(parts, number))
        except ConvertError as error:
            raise ConvertError(f"{prefix}{name}: {error}") from error


def _shape(tensors: dict[str, NDArray[np.float32]], name: str, shape: tuple[int, ...]) -> None:
    actual = tensors[name]
    if actual.size != prod(shape):
        raise ConvertError(
            f"{name}: shape {shape} requires {prod(shape)} values, got {actual.size}"
        )
    actual = actual.reshape(shape)
    actual.setflags(write=False)
    tensors[name] = actual


def _dimension(value: int, name: str, *, maximum: int = 8192) -> int:
    if not 1 <= value <= maximum:
        raise ConvertError(f"{name}: unsupported dimension {value}, expected 1..{maximum}")
    return value


def _quotient(size: int, divisor: int, name: str) -> int:
    if size % divisor:
        raise ConvertError(f"{name}: tensor shape does not divide into dimension {divisor}")
    return _dimension(size // divisor, name)


def _validate_shapes(tensors: dict[str, NDArray[np.float32]], heads: int, encoders: int) -> Config:
    width = _dimension(tensors["ip_emb_b"].size, "embedding width", maximum=2048)
    d_model = _dimension(tensors["encoder.0.mha.q_b"].size, "attention width", maximum=2048)
    if not 1 <= heads <= 64 or d_model % heads:
        raise ConvertError(f"invalid attention head count {heads} for dimension {d_model}")
    config = Config(
        width,
        heads,
        encoders,
        d_model,
        _dimension(tensors["encoder.0.ffn.dense1_b"].size, "FFN width"),
        _quotient(tensors["encoder.0.mha.smolgen.compress"].size, width, "smolgen compress"),
        _dimension(tensors["encoder.0.mha.smolgen.dense1_b"].size, "smolgen hidden"),
        _quotient(tensors["encoder.0.mha.smolgen.dense2_b"].size, heads, "smolgen generation"),
        _dimension(tensors["ip_pol_b"].size, "policy embedding"),
        _dimension(tensors["ip2_pol_b"].size, "policy model"),
        _dimension(tensors["ip_val_b"].size, "value embedding"),
        _dimension(tensors["ip1_val_b"].size, "value hidden"),
        _dimension(tensors["ip_mov_b"].size, "MLH embedding"),
        _dimension(tensors["ip1_mov_b"].size, "MLH hidden"),
    )
    shapes = {
        "ip_emb_w": (width, 176),
        "ip_emb_b": (width,),
        "ip_mult_gate": (width, 64),
        "ip_add_gate": (width, 64),
        "smolgen_w": (4096, config.smol_gen),
        "ip_pol_w": (config.policy_embedding, width),
        "ip_pol_b": (config.policy_embedding,),
        "ip2_pol_w": (config.policy_model, config.policy_embedding),
        "ip2_pol_b": (config.policy_model,),
        "ip3_pol_w": (config.policy_model, config.policy_embedding),
        "ip3_pol_b": (config.policy_model,),
        "ip4_pol_w": (4, config.policy_model),
    }
    for stem, embedded, hidden, outputs in (
        ("val", config.value_embedding, config.value_hidden, 3),
        ("mov", config.mlh_embedding, config.mlh_hidden, 1),
    ):
        shapes.update(
            {
                f"ip_{stem}_w": (embedded, width),
                f"ip_{stem}_b": (embedded,),
                f"ip1_{stem}_w": (hidden, 64 * embedded),
                f"ip1_{stem}_b": (hidden,),
                f"ip2_{stem}_w": (outputs, hidden),
                f"ip2_{stem}_b": (outputs,),
            }
        )
    for index in range(encoders):
        prefix = f"encoder.{index}."
        for stem in ("q", "k", "v"):
            shapes[prefix + f"mha.{stem}_w"] = (d_model, width)
            shapes[prefix + f"mha.{stem}_b"] = (d_model,)
        shapes.update(
            {
                prefix + "mha.dense_w": (width, d_model),
                prefix + "mha.dense_b": (width,),
                prefix + "ffn.dense1_w": (config.ffn_hidden, width),
                prefix + "ffn.dense1_b": (config.ffn_hidden,),
                prefix + "ffn.dense2_w": (width, config.ffn_hidden),
                prefix + "ffn.dense2_b": (width,),
            }
        )
        for norm in ("ln1", "ln2"):
            for suffix in ("gammas", "betas"):
                shapes[prefix + f"{norm}_{suffix}"] = (width,)
        smol = prefix + "mha.smolgen."
        shapes.update(
            {
                smol + "compress": (config.smol_compress, width),
                smol + "dense1_w": (config.smol_hidden, 64 * config.smol_compress),
                smol + "dense1_b": (config.smol_hidden,),
                smol + "dense2_w": (heads * config.smol_gen, config.smol_hidden),
                smol + "dense2_b": (heads * config.smol_gen,),
            }
        )
        for norm, size in (("ln1", config.smol_hidden), ("ln2", heads * config.smol_gen)):
            for suffix in ("gammas", "betas"):
                shapes[smol + f"{norm}_{suffix}"] = (size,)
    if shapes.keys() != tensors.keys():
        raise ConvertError("unsupported or missing learned tensors")
    for name, shape in shapes.items():
        _shape(tensors, name, shape)
    return config


def parse_net(data: bytes | memoryview, *, source_sha256: str) -> Network:
    if not re.fullmatch("[0-9a-f]{64}", source_sha256):
        raise ConvertError("source SHA256 must contain 64 lowercase hexadecimal characters")
    root = message(data, {1: 5, 2: 2, 3: 2, 4: 2, 5: 2, 10: 2, 11: 2})
    import struct

    if struct.unpack("<I", byte_field(root, 1))[0] != 0x1C0:
        raise ConvertError("invalid Lc0 weight magic")
    if 2 in root:
        try:
            bytes(byte_field(root, 2)).decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ConvertError("invalid UTF-8 network license") from error
    if 5 in root:
        training = message(byte_field(root, 5), {1: 0, 2: 5, 3: 5, 4: 5, 5: 5, 6: 2})
        if 6 in training:
            try:
                bytes(byte_field(training, 6)).decode("utf-8", errors="strict")
            except UnicodeError as error:
                raise ConvertError("invalid UTF-8 training parameters") from error
    if 11 in root:
        raise ConvertError("unsupported embedded ONNX network")
    version = message(byte_field(root, 3), {1: 0, 2: 0, 3: 0})
    minimum = (int_field(version, 1), int_field(version, 2), int_field(version, 3))
    if minimum > (0, 32, 1):
        raise ConvertError(f"unsupported minimum Lc0 version {minimum}")
    file_format = message(byte_field(root, 4), {1: 0, 2: 2})
    if int_field(file_format, 1) != 1:
        raise ConvertError("unsupported weights encoding: only global LINEAR16 is supported")
    raw = message(byte_field(file_format, 2), dict.fromkeys(range(1, 11), 0))
    raw_values = {number: int_field(raw, number) for number in raw}
    schema = dict.fromkeys(_ROOT_LAYERS, 2) | dict.fromkeys(_UNSUPPORTED_WEIGHTS, 2)
    schema[24] = 0
    schema.update({27: 2, 28: 0})
    weights = message(byte_field(root, 10), schema, repeated=frozenset({2, 21, 27}))
    unsupported = _UNSUPPORTED_WEIGHTS.intersection(weights)
    if unsupported:
        raise ConvertError(f"unsupported architecture weight fields: {sorted(unsupported)}")
    encoder_records = weights.get(27, [])
    format_info = normalize_format(
        raw_values, encoder_count=len(encoder_records), has_smolgen=35 in weights
    )
    tensors: dict[str, NDArray[np.float32]] = {}
    _group(weights, _ROOT_LAYERS, "", tensors)
    for index, record in enumerate(encoder_records):
        if not isinstance(record.value, memoryview):
            raise ConvertError("encoder must be a nested message")
        prefix = f"encoder.{index}."
        enc = message(record.value, dict.fromkeys(range(1, 7), 2))
        _group(
            enc, {2: "ln1_gammas", 3: "ln1_betas", 5: "ln2_gammas", 6: "ln2_betas"}, prefix, tensors
        )
        mha = message(byte_field(enc, 1), dict.fromkeys(range(1, 13), 2))
        if {10, 11, 12}.intersection(mha):
            raise ConvertError("unsupported relative-position attention weights")
        _group(mha, _MHA_LAYERS, prefix + "mha.", tensors)
        smol = message(byte_field(mha, 9), dict.fromkeys(range(1, 10), 2))
        _group(smol, _SMOL_LAYERS, prefix + "mha.smolgen.", tensors)
        ffn = message(byte_field(enc, 4), dict.fromkeys(range(1, 5), 2))
        _group(ffn, _FFN_LAYERS, prefix + "ffn.", tensors)
    config = _validate_shapes(tensors, int_field(weights, 28), len(encoder_records))
    return Network(
        format_info,
        config,
        MappingProxyType(tensors),
        source_sha256,
        sum(tensor.size for tensor in tensors.values()),
        minimum,
    )
