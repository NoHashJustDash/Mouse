# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from test_convert_fixture import synthetic_net

from mouse_tools.convert.model import T1Model, policy_gather, positional_map, promotion_scores
from mouse_tools.convert.schema import Network, parse_net
from mouse_tools.convert.wire import ConvertError


def numpy_forward(net: Network, planes: NDArray[np.float32]) -> tuple[NDArray[np.float64], ...]:
    """Independent token-major NumPy equations, using float64 for the oracle."""
    weights = {name: value.astype(np.float64) for name, value in net.tensors.items()}
    config = net.config
    batch = planes.shape[0]

    def dense(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
        return np.asarray(value @ weights[name + "_w"].T + weights[name + "_b"], dtype=np.float64)

    def mish(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(value * np.tanh(np.logaddexp(0.0, value)), dtype=np.float64)

    def norm(value: NDArray[np.float64], name: str, eps: float) -> NDArray[np.float64]:
        centered = value - value.mean(axis=-1, keepdims=True)
        return np.asarray(
            centered
            / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + eps)
            * weights[name + "_gammas"]
            + weights[name + "_betas"],
            dtype=np.float64,
        )

    def softmax(value: NDArray[np.float64]) -> NDArray[np.float64]:
        exp = np.exp(value - value.max(axis=-1, keepdims=True))
        return np.asarray(exp / exp.sum(axis=-1, keepdims=True), dtype=np.float64)

    position = np.empty((64, 64), dtype=np.float64)
    for source in range(64):
        for target in range(64):
            dx, dy = abs(source % 8 - target % 8), abs(source // 8 - target // 8)
            position[source, target] = (
                -1
                if source == target
                else int(dx == 0 or dy == 0 or dx == dy or (dx * dy == 2 and dx + dy == 3))
            )
    tokens: NDArray[np.float64] = (
        planes.astype(np.float64).reshape(batch, 112, 64).transpose(0, 2, 1)
    )
    tokens = np.concatenate((tokens, np.broadcast_to(position, (batch, 64, 64))), axis=2)
    body = mish(dense(tokens, "ip_emb"))
    body = body * weights["ip_mult_gate"].T + weights["ip_add_gate"].T
    alpha = float(np.float32((2 * config.encoders) ** -0.25))
    for index in range(config.encoders):
        prefix = f"encoder.{index}."
        smol = prefix + "mha.smolgen."
        bias: NDArray[np.float64] = body @ weights[smol + "compress"].T
        bias = dense(bias.reshape(batch, -1), smol + "dense1")
        bias = norm(bias / (1 + np.exp(-bias)), smol + "ln1", 1e-3)
        bias = dense(bias, smol + "dense2")
        bias = norm(bias / (1 + np.exp(-bias)), smol + "ln2", 1e-3)
        bias = (
            bias.reshape(batch, config.heads, config.smol_gen) @ weights["smolgen_w"].T
        ).reshape(batch, config.heads, 64, 64)
        projected = [
            dense(body, prefix + "mha." + name)
            .reshape(batch, 64, config.heads, -1)
            .transpose(0, 2, 1, 3)
            for name in ("q", "k", "v")
        ]
        query, key, values = projected
        scale = float(np.float32(1) / np.sqrt(np.float32(config.d_model // config.heads)))
        probabilities = softmax((query @ key.transpose(0, 1, 3, 2)) * scale + bias)
        attended = (probabilities @ values).transpose(0, 2, 1, 3).reshape(batch, 64, -1)
        body = norm(body + alpha * dense(attended, prefix + "mha.dense"), prefix + "ln1", 1e-6)
        hidden = np.maximum(dense(body, prefix + "ffn.dense1"), 0) ** 2
        body = norm(body + alpha * dense(hidden, prefix + "ffn.dense2"), prefix + "ln2", 1e-6)
    policy_body = mish(dense(body, "ip_pol"))
    policy_query, policy_key = dense(policy_body, "ip2_pol"), dense(policy_body, "ip3_pol")
    scores = (policy_query @ policy_key.transpose(0, 2, 1)) * float(
        np.float32(1) / np.sqrt(np.float32(config.policy_model))
    )
    offsets = policy_key[:, 56:64] @ weights["ip4_pol_w"].T
    scores_list = []
    for source in range(64):
        for target in range(64):
            dx, dy = abs(source % 8 - target % 8), abs(source // 8 - target // 8)
            if source != target and (dx == 0 or dy == 0 or dx == dy or sorted((dx, dy)) == [1, 2]):
                scores_list.append(scores[:, source, target])
    for source in range(48, 56):
        for target in range(56, 64):
            if abs(source % 8 - target % 8) <= 1:
                for piece in range(3):
                    scores_list.append(
                        scores[:, source, target]
                        + offsets[:, target % 8, piece]
                        + offsets[:, target % 8, 3]
                    )
    policy = np.stack(scores_list, axis=1)
    value: NDArray[np.float64] = mish(dense(body, "ip_val")).reshape(batch, -1)
    value = softmax(dense(mish(dense(value, "ip1_val")), "ip2_val"))
    mlh: NDArray[np.float64] = mish(dense(body, "ip_mov")).reshape(batch, -1)
    mlh = np.maximum(dense(mish(dense(mlh, "ip1_mov")), "ip2_mov"), 0)
    return policy, value, mlh


def test_original_geometric_constants() -> None:
    mapping = policy_gather()
    assert mapping.shape == (1858,) and len(set(mapping.tolist())) == 1858
    assert mapping[103] == 263 and mapping[97] == 256
    np.testing.assert_array_equal(mapping[-66:-63], [4096, 4097, 4098])
    position = positional_map()
    assert position.shape == (64, 64) and position.dtype == np.float32
    assert position[0, 0] == -1 and position[0, 10] == 1 and position[0, 19] == 0


def test_promotion_offsets_have_destination_and_piece_order() -> None:
    scores = torch.arange(4096, dtype=torch.float32).reshape(1, 64, 64)
    keys = torch.zeros((1, 64, 2), dtype=torch.float32)
    keys[0, 56:, 0] = torch.arange(8)
    projection = torch.tensor([[1.0, 0], [2.0, 0], [3.0, 0], [4.0, 0]])
    actual = promotion_scores(scores, keys, projection).numpy()
    assert actual.shape == (1, 4288)
    for source in range(8):
        for target in range(8):
            for piece in range(3):
                assert actual[0, 4096 + (source * 8 + target) * 3 + piece] == (
                    scores[0, 48 + source, 56 + target].item() + target * (piece + 5)
                )


@pytest.mark.parametrize("batch", [1, 3])
def test_complete_small_model_matches_independent_numpy(batch: int) -> None:
    torch.set_num_threads(1)
    net = parse_net(synthetic_net(), source_sha256="a" * 64)
    model = T1Model(net).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == net.parameter_count
    inputs = np.random.default_rng(2718).normal(size=(batch, 112, 8, 8)).astype(np.float32)
    expected = numpy_forward(net, inputs)
    with torch.inference_mode():
        actual = model(torch.from_numpy(inputs))
    for wanted, observed in zip(expected, actual, strict=True):
        np.testing.assert_allclose(observed.numpy(), wanted, rtol=0, atol=1e-4)
    assert actual[0].shape == (batch, 1858) and actual[1].shape == (batch, 3)
    assert actual[2].shape == (batch, 1)
    np.testing.assert_allclose(actual[1].sum(dim=1).numpy(), np.ones(batch), atol=1e-6)
    assert bool(torch.all(actual[2] >= 0))


@pytest.mark.parametrize(
    "inputs",
    [
        torch.zeros(1, 111, 8, 8),
        torch.zeros(0, 112, 8, 8),
        torch.zeros(1, 112, 8, 8, dtype=torch.float64),
        torch.full((1, 112, 8, 8), float("nan")),
    ],
)
def test_model_rejects_invalid_input(inputs: torch.Tensor) -> None:
    model = T1Model(parse_net(synthetic_net(), source_sha256="a" * 64))
    with pytest.raises(ConvertError, match="input"):
        model(inputs)
