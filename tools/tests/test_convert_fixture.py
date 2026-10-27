# SPDX-License-Identifier: GPL-3.0-or-later
"""Small, independently authored serialized attention model fixtures."""

from __future__ import annotations

from math import prod

import numpy as np
from test_convert_wire import field, layer

ROOT_FIELDS = {
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


def synthetic_net(
    *,
    replacement: tuple[str, int] | None = None,
    extra_weights: bytes = b"",
    encoding: int = 1,
    heads: int = 2,
) -> bytes:
    rng = np.random.default_rng(618)

    def tensor(name: str, shape: tuple[int, ...]) -> bytes:
        count = prod(shape)
        if replacement is not None and replacement[0] == name:
            count = replacement[1]
        codes = tuple(int(value) for value in rng.integers(0, 65536, size=count))
        low, high = (0.5, 1.5) if "gammas" in name or name == "ip_mult_gate" else (-0.07, 0.08)
        return layer(low, high, codes)

    shapes = {
        "ip_pol_w": (4, 4),
        "ip_pol_b": (4,),
        "ip1_val_w": (3, 128),
        "ip1_val_b": (3,),
        "ip2_val_w": (3, 3),
        "ip2_val_b": (3,),
        "ip1_mov_w": (2, 64),
        "ip1_mov_b": (2,),
        "ip2_mov_w": (1, 2),
        "ip2_mov_b": (1,),
        "ip2_pol_w": (4, 4),
        "ip2_pol_b": (4,),
        "ip3_pol_w": (4, 4),
        "ip3_pol_b": (4,),
        "ip4_pol_w": (4, 4),
        "ip_emb_w": (4, 176),
        "ip_emb_b": (4,),
        "ip_val_w": (2, 4),
        "ip_val_b": (2,),
        "ip_mov_w": (1, 4),
        "ip_mov_b": (1,),
        "ip_mult_gate": (4, 64),
        "ip_add_gate": (4, 64),
        "smolgen_w": (4096, 2),
    }
    weights = b"".join(
        field(number, 2, tensor(name, shapes[name])) for number, name in ROOT_FIELDS.items()
    )
    for index in range(2):
        prefix = f"encoder.{index}."
        mha = b"".join(
            field(
                number, 2, tensor(prefix + "mha." + name, (4, 4) if name.endswith("_w") else (4,))
            )
            for number, name in enumerate(
                ("q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "dense_w", "dense_b"), 1
            )
        )
        smol_shapes = [
            ("compress", (1, 4)),
            ("dense1_w", (3, 64)),
            ("dense1_b", (3,)),
            ("ln1_gammas", (3,)),
            ("ln1_betas", (3,)),
            ("dense2_w", (4, 3)),
            ("dense2_b", (4,)),
            ("ln2_gammas", (4,)),
            ("ln2_betas", (4,)),
        ]
        smol = b"".join(
            field(i, 2, tensor(prefix + "mha.smolgen." + name, shape))
            for i, (name, shape) in enumerate(smol_shapes, 1)
        )
        mha += field(9, 2, smol)
        ffn = b"".join(
            field(i, 2, tensor(prefix + "ffn." + name, shape))
            for i, (name, shape) in enumerate(
                [
                    ("dense1_w", (6, 4)),
                    ("dense1_b", (6,)),
                    ("dense2_w", (4, 6)),
                    ("dense2_b", (4,)),
                ],
                1,
            )
        )
        encoder = field(1, 2, mha) + field(4, 2, ffn)
        for number, name in (
            (2, "ln1_gammas"),
            (3, "ln1_betas"),
            (5, "ln2_gammas"),
            (6, "ln2_betas"),
        ):
            encoder += field(number, 2, tensor(prefix + name, (4,)))
        weights += field(27, 2, encoder)
    weights += field(28, 0, heads) + extra_weights
    network = b"".join(field(i, 0, value) for i, value in enumerate((1, 2, 4, 3, 2, 1, 1), 1))
    version = field(1, 0, 0) + field(2, 0, 29) + field(3, 0, 0)
    import struct

    return (
        field(1, 5, struct.pack("<I", 0x1C0))
        + field(3, 2, version)
        + field(4, 2, field(1, 0, encoding) + field(2, 2, network))
        + field(10, 2, weights)
    )
