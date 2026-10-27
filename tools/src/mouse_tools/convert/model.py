# SPDX-License-Identifier: GPL-3.0-or-later
"""Original PyTorch equations for the validated gated T1 attention architecture.

Behavioral reference: Lc0 fd71a2d9/src/neural/onnx/converter.cc. Positional and
policy constants are derived from chess geometry; no upstream tables are copied.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as functional

from .schema import Network
from .wire import ConvertError


def _reachable(source: int, target: int) -> bool:
    dx, dy = abs(source % 8 - target % 8), abs(source // 8 - target // 8)
    return source != target and (dx == 0 or dy == 0 or dx == dy or sorted((dx, dy)) == [1, 2])


def positional_map() -> NDArray[np.float32]:
    return np.array(
        [[-1.0 if a == b else float(_reachable(a, b)) for b in range(64)] for a in range(64)],
        dtype=np.float32,
    )


def policy_gather() -> NDArray[np.int64]:
    ordinary = [
        source * 64 + target
        for source in range(64)
        for target in range(64)
        if _reachable(source, target)
    ]
    promotions = [
        4096 + (source * 8 + target) * 3 + piece
        for source in range(8)
        for target in range(8)
        if abs(source - target) <= 1
        for piece in range(3)
    ]
    return np.array(ordinary + promotions, dtype=np.int64)


def promotion_scores(scores: Tensor, keys: Tensor, projection: Tensor) -> Tensor:
    """Append all 8x8x3 promotion candidates in from/to/Q,R,B order."""
    offsets = keys[:, 56:64, :] @ projection.t()
    offsets = offsets[:, :, :3] + offsets[:, :, 3:4]
    promoted = scores[:, 48:56, 56:64].unsqueeze(-1) + offsets.unsqueeze(1)
    return torch.cat((scores.flatten(1), promoted.flatten(1)), dim=1)


def _tracing() -> bool:
    # Public PyTorch query lacks annotations in the pinned 2.14 type surface.
    return bool(torch.jit.is_tracing())  # type: ignore[attr-defined,no-untyped-call]


def _parameter(array: NDArray[np.float32]) -> nn.Parameter:
    return nn.Parameter(torch.from_numpy(np.array(array, copy=True)), requires_grad=False)


def _mish(value: Tensor) -> Tensor:
    return value * torch.tanh(functional.softplus(value))


class Dense(nn.Module):
    def __init__(self, net: Network, stem: str) -> None:
        super().__init__()
        self.weight = _parameter(net.tensors[stem + "_w"])
        self.bias = _parameter(net.tensors[stem + "_b"])

    def forward(self, value: Tensor) -> Tensor:
        # Explicit MatMul/Add keeps the independent export's arithmetic aligned
        # with the FP32 reference rather than selecting an implicit fused GEMM.
        return value @ self.weight.t() + self.bias


class Norm(nn.Module):
    def __init__(self, net: Network, stem: str, epsilon: float) -> None:
        super().__init__()
        self.gamma = _parameter(net.tensors[stem + "_gammas"])
        self.beta = _parameter(net.tensors[stem + "_betas"])
        self.width = self.gamma.numel()
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        return functional.layer_norm(value, (self.width,), self.gamma, self.beta, self.epsilon)


class Smolgen(nn.Module):
    def __init__(self, net: Network, prefix: str) -> None:
        super().__init__()
        self.compress = _parameter(net.tensors[prefix + "compress"])
        self.dense1 = Dense(net, prefix + "dense1")
        self.dense2 = Dense(net, prefix + "dense2")
        self.norm1 = Norm(net, prefix + "ln1", 1e-3)
        self.norm2 = Norm(net, prefix + "ln2", 1e-3)
        self.hidden_channels = net.config.smol_compress
        self.heads = net.config.heads
        self.generated = net.config.smol_gen

    def forward(self, value: Tensor, shared_weight: Tensor) -> Tensor:
        compressed = (value @ self.compress.t()).reshape(-1, 64 * self.hidden_channels)
        hidden = self.dense1.forward(compressed)
        hidden = self.norm1.forward(hidden * torch.sigmoid(hidden))
        generated = self.dense2.forward(hidden)
        generated = self.norm2.forward(generated * torch.sigmoid(generated))
        return (generated.reshape(-1, self.heads, self.generated) @ shared_weight.t()).reshape(
            -1, self.heads, 64, 64
        )


class Encoder(nn.Module):
    def __init__(self, net: Network, index: int) -> None:
        super().__init__()
        prefix = f"encoder.{index}."
        self.query = Dense(net, prefix + "mha.q")
        self.key = Dense(net, prefix + "mha.k")
        self.value = Dense(net, prefix + "mha.v")
        self.output = Dense(net, prefix + "mha.dense")
        self.smolgen = Smolgen(net, prefix + "mha.smolgen.")
        self.norm1 = Norm(net, prefix + "ln1", 1e-6)
        self.norm2 = Norm(net, prefix + "ln2", 1e-6)
        self.ffn1 = Dense(net, prefix + "ffn.dense1")
        self.ffn2 = Dense(net, prefix + "ffn.dense2")
        self.heads = net.config.heads
        self.depth = net.config.d_model // self.heads
        self.model_width = net.config.d_model
        self.scale = float(np.float32(1) / np.sqrt(np.float32(self.depth)))
        self.alpha = float(np.float32((2 * net.config.encoders) ** -0.25))

    def forward(self, tokens: Tensor, shared_weight: Tensor) -> Tensor:
        query = self.query.forward(tokens).reshape(-1, 64, self.heads, self.depth)
        key = self.key.forward(tokens).reshape(-1, 64, self.heads, self.depth)
        value = self.value.forward(tokens).reshape(-1, 64, self.heads, self.depth)
        scores = (query.permute(0, 2, 1, 3) @ key.permute(0, 2, 3, 1)) * self.scale
        scores = scores + self.smolgen.forward(tokens, shared_weight)
        attended = torch.softmax(scores, dim=3) @ value.permute(0, 2, 1, 3)
        attended = attended.permute(0, 2, 1, 3).reshape(-1, self.model_width)
        hidden = self.norm1.forward(tokens + self.alpha * self.output.forward(attended))
        feedforward = torch.relu(self.ffn1.forward(hidden))
        feedforward = self.ffn2.forward(feedforward * feedforward)
        return self.norm2.forward(hidden + self.alpha * feedforward)


class Policy(nn.Module):
    def __init__(self, net: Network) -> None:
        super().__init__()
        self.embedding = Dense(net, "ip_pol")
        self.query = Dense(net, "ip2_pol")
        self.key = Dense(net, "ip3_pol")
        self.promotion = _parameter(net.tensors["ip4_pol_w"])
        self.width = net.config.policy_model
        self.scale = float(np.float32(1) / np.sqrt(np.float32(self.width)))
        self.mapping: Tensor
        self.register_buffer("mapping", torch.from_numpy(policy_gather()))

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = _mish(self.embedding.forward(tokens))
        query = self.query.forward(hidden).reshape(-1, 64, self.width)
        key = self.key.forward(hidden).reshape(-1, 64, self.width)
        scores = (query @ key.transpose(1, 2)) * self.scale
        return promotion_scores(scores, key, self.promotion).index_select(1, self.mapping)


class ScalarHead(nn.Module):
    def __init__(self, net: Network, stem: str, width: int, *, wdl: bool) -> None:
        super().__init__()
        self.embedding = Dense(net, "ip_" + stem)
        self.dense1 = Dense(net, "ip1_" + stem)
        self.dense2 = Dense(net, "ip2_" + stem)
        self.width = width
        self.wdl = wdl

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = _mish(self.embedding.forward(tokens)).reshape(-1, 64 * self.width)
        result = self.dense2.forward(_mish(self.dense1.forward(hidden)))
        return torch.softmax(result, dim=1) if self.wdl else torch.relu(result)


class T1Model(nn.Module):
    """CPU conversion model; deployment providers are chosen by the engine."""

    def __init__(self, net: Network, *, precision: str = "fp32") -> None:
        super().__init__()
        if precision not in ("fp32", "fp16"):
            raise ConvertError(f"unsupported compute precision {precision!r}; use fp32 or fp16")
        self.precision = precision
        self.embedding = Dense(net, "ip_emb")
        self.multiply_gate = _parameter(net.tensors["ip_mult_gate"].T)
        self.add_gate = _parameter(net.tensors["ip_add_gate"].T)
        self.shared_smolgen = _parameter(net.tensors["smolgen_w"])
        self.encoders = nn.ModuleList([Encoder(net, index) for index in range(net.config.encoders)])
        self.policy = Policy(net)
        self.value = ScalarHead(net, "val", net.config.value_embedding, wdl=True)
        self.moves_left = ScalarHead(net, "mov", net.config.mlh_embedding, wdl=False)
        self.width = net.config.width
        self.position_map: Tensor
        self.register_buffer("position_map", torch.from_numpy(positional_map()))
        if precision == "fp16":
            self.half()
            if any(not bool(torch.isfinite(item).all()) for item in self.parameters()):
                raise ConvertError("network parameters must remain finite in fp16 precision")

    def forward(self, planes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not _tracing() and (
            planes.device.type != "cpu"
            or planes.dtype != torch.float32
            or planes.ndim != 4
            or planes.shape[0] < 1
            or tuple(planes.shape[1:]) != (112, 8, 8)
            or not bool(torch.isfinite(planes).all())
        ):
            raise ConvertError("input must be finite CPU float32 with shape [batch>=1,112,8,8]")
        if self.precision == "fp16":
            if not _tracing() and bool((planes.abs() > torch.finfo(torch.float16).max).any()):
                raise ConvertError("input cannot be represented as finite fp16")
            planes = planes.half()
        batch = planes.shape[0]
        tokens = planes.reshape(batch, 112, 64).transpose(1, 2)
        position = self.position_map.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat((tokens, position), dim=2).reshape(-1, 176)
        tokens = _mish(self.embedding.forward(tokens)).reshape(-1, 64, self.width)
        tokens = (tokens * self.multiply_gate + self.add_gate).reshape(-1, self.width)
        for encoder in self.encoders:
            tokens = cast(Encoder, encoder).forward(tokens, self.shared_smolgen)
        outputs = (
            self.policy.forward(tokens),
            self.value.forward(tokens),
            self.moves_left.forward(tokens),
        )
        if not _tracing() and any(not bool(torch.isfinite(item).all()) for item in outputs):
            raise ConvertError(f"network produced non-finite outputs in {self.precision} precision")
        if self.precision == "fp16":
            return outputs[0].float(), outputs[1].float(), outputs[2].float()
        return outputs
