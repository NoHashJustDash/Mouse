# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest
from numpy.typing import ArrayLike
from onnx import TensorProto, helper, numpy_helper

from mouse_tools.convert.parity import run_parity
from mouse_tools.convert.wire import ConvertError


def constant_model(path: Path, *, reference: bool, policy_value: float = 0) -> str:
    """Directly authored ONNX oracle for harness tests; neither converter is used."""
    input_name = "/input/planes" if reference else "planes"
    names = (
        ("/output/policy", "/output/wdl", "/output/mlh") if reference else ("policy", "wdl", "mlh")
    )
    nodes = [
        helper.make_node("Shape", [input_name], ["shape"]),
        helper.make_node("Gather", ["shape", "zero_index"], ["batch"], axis=0),
    ]
    tensors = [numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_index")]
    for index, (name, width, value) in enumerate(
        zip(names, (1858, 3, 1), (policy_value, 1 / 3, 0), strict=True)
    ):
        nodes.extend(
            [
                helper.make_node("Concat", ["batch", f"width{index}"], [f"size{index}"], axis=0),
                helper.make_node("Expand", [f"value{index}", f"size{index}"], [name]),
            ]
        )
        tensors.extend(
            [
                numpy_helper.from_array(np.array([width], dtype=np.int64), f"width{index}"),
                numpy_helper.from_array(
                    np.full((1, width), value, dtype=np.float32), f"value{index}"
                ),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "independent_constant_oracle",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", 112, 8, 8])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", width])
            for name, width in zip(names, (1858, 3, 1), strict=True)
        ],
        tensors,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    if not reference:
        helper.set_model_props(
            model,
            {
                "mouse.schema_version": "1",
                "mouse.converter_version": "0.2.0-dev",
                "mouse.source_sha256": "a" * 64,
                "mouse.input_format": "1",
                "mouse.network_format": "6",
                "mouse.raw_network_format": "4",
                "mouse.policy_format": "3",
                "mouse.value_format": "2",
                "mouse.moves_left_format": "1",
                "mouse.policy_mapping": "lc0-1858-v1",
                "mouse.learned_parameter_count": "1",
            },
        )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_fixed_tolerance_gate_uses_all_positions(tmp_path: Path) -> None:
    own, reference, corpus = (
        tmp_path / "own.onnx",
        tmp_path / "reference.onnx",
        tmp_path / "planes.npy",
    )
    own_digest = constant_model(own, reference=False)
    digest = constant_model(reference, reference=True)
    distinct_inputs = np.zeros((1001, 112, 8, 8), dtype=np.float32)
    distinct_inputs[:, 109, :, :] = np.arange(1001, dtype=np.float32)[:, None, None]
    np.save(corpus, distinct_inputs, allow_pickle=False)
    input_digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    report = run_parity(own, reference, corpus, reference_sha256=digest, batch_size=17)
    assert report.passed and report.gate_satisfied and report.positions == 1001
    assert report.distinct_positions == 1001
    assert report.converted_sha256 == own_digest
    assert report.reference_sha256 == digest
    assert report.inputs_sha256 == input_digest
    assert report.max_abs == {"policy": 0.0, "wdl": 0.0, "mlh": 0.0}
    assert report.tolerance == 1e-4
    assert report.providers == ("CPUExecutionProvider",)


def test_large_difference_is_reported_and_cannot_pass(tmp_path: Path) -> None:
    own, reference, corpus = (
        tmp_path / "own.onnx",
        tmp_path / "reference.onnx",
        tmp_path / "planes.npy",
    )
    constant_model(own, reference=False)
    digest = constant_model(reference, reference=True, policy_value=0.001)
    np.save(corpus, np.zeros((2, 112, 8, 8), dtype=np.float32), allow_pickle=False)
    report = run_parity(own, reference, corpus, reference_sha256=digest, diagnostic=True)
    assert not report.passed and not report.gate_satisfied
    assert report.max_abs["policy"] > 1e-4


def test_short_corpus_cannot_claim_the_gate(tmp_path: Path) -> None:
    own, reference = tmp_path / "own.onnx", tmp_path / "reference.onnx"
    constant_model(own, reference=False)
    digest = constant_model(reference, reference=True)
    corpus = tmp_path / "planes.npy"
    np.save(corpus, np.zeros((999, 112, 8, 8), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ConvertError, match="1000"):
        run_parity(own, reference, corpus, reference_sha256=digest)


@pytest.mark.parametrize("kind", ["wrong_dtype", "wrong_shape", "nan", "pickle"])
def test_invalid_corpus_is_rejected_before_inference(tmp_path: Path, kind: str) -> None:
    own, reference = tmp_path / "own.onnx", tmp_path / "reference.onnx"
    constant_model(own, reference=False)
    digest = constant_model(reference, reference=True)
    corpus = tmp_path / "planes.npy"
    values: dict[str, ArrayLike] = {
        "wrong_dtype": np.zeros((1, 112, 8, 8), np.float64),
        "wrong_shape": np.zeros((1, 111, 8, 8), np.float32),
        "nan": np.full((1, 112, 8, 8), np.nan, np.float32),
        "pickle": np.array([{"untrusted": "object"}], dtype=object),
    }
    np.save(corpus, values[kind], allow_pickle=True)
    with pytest.raises(ConvertError, match="input|corpus"):
        run_parity(
            own,
            reference,
            corpus,
            reference_sha256=digest,
            diagnostic=True,
        )


def test_oracle_digest_is_required_and_checked(tmp_path: Path) -> None:
    own, reference, corpus = (
        tmp_path / "own.onnx",
        tmp_path / "reference.onnx",
        tmp_path / "planes.npy",
    )
    constant_model(own, reference=False)
    constant_model(reference, reference=True)
    np.save(corpus, np.zeros((1, 112, 8, 8), np.float32), allow_pickle=False)
    with pytest.raises(ConvertError, match="reference SHA256"):
        run_parity(own, reference, corpus, reference_sha256="0" * 64, diagnostic=True)


def test_repeated_tensor_rows_cannot_satisfy_the_position_gate(tmp_path: Path) -> None:
    own, reference = tmp_path / "own.onnx", tmp_path / "reference.onnx"
    constant_model(own, reference=False)
    digest = constant_model(reference, reference=True)
    corpus = tmp_path / "repeated.npy"
    np.save(corpus, np.zeros((1000, 112, 8, 8), np.float32), allow_pickle=False)
    with pytest.raises(ConvertError, match="distinct"):
        run_parity(own, reference, corpus, reference_sha256=digest)


@pytest.mark.parametrize("changed", ["converted", "reference", "inputs"])
def test_artifact_mutation_during_first_batch_invalidates_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    from typing import Protocol, cast

    import onnxruntime as ort  # type: ignore[import-untyped]  # ORT has no type marker.
    from numpy.typing import NDArray

    class Session(Protocol):
        def disable_fallback(self) -> None: ...
        def get_providers(self) -> list[str]: ...
        def run(
            self, names: list[str], feeds: dict[str, NDArray[np.float32]]
        ) -> list[NDArray[np.float32]]: ...

    own, reference, corpus = (
        tmp_path / "own.onnx",
        tmp_path / "reference.onnx",
        tmp_path / "planes.npy",
    )
    constant_model(own, reference=False)
    reference_digest = constant_model(reference, reference=True)
    values = np.zeros((1000, 112, 8, 8), dtype=np.float32)
    values[:, 109, :, :] = np.arange(1000, dtype=np.float32)[:, None, None]
    np.save(corpus, values, allow_pickle=False)
    paths = {"converted": own, "reference": reference, "inputs": corpus}
    target = paths[changed]
    original_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    real_factory = ort.InferenceSession
    mutated = False

    class MutatingSession:
        def __init__(self, delegate: Session) -> None:
            self.delegate = delegate

        def disable_fallback(self) -> None:
            self.delegate.disable_fallback()

        def get_providers(self) -> list[str]:
            return self.delegate.get_providers()

        def run(
            self, names: list[str], feeds: dict[str, NDArray[np.float32]]
        ) -> list[NDArray[np.float32]]:
            nonlocal mutated
            output = self.delegate.run(names, feeds)
            if not mutated:
                # Change bytes in place so this is also valid for a Windows
                # read-only NPY mapping; no truncate/rename permission is needed.
                with target.open("r+b") as stream:
                    stream.seek(-1, 2)
                    previous = stream.read(1)
                    stream.seek(-1, 2)
                    stream.write(bytes([previous[0] ^ 1]))
                mutated = True
            return output

    def factory(path: str, *, sess_options: object, providers: list[str]) -> MutatingSession:
        return MutatingSession(
            cast(Session, real_factory(path, sess_options=sess_options, providers=providers))
        )

    monkeypatch.setattr(ort, "InferenceSession", factory)
    with pytest.raises(ConvertError, match=changed + ".*changed"):
        run_parity(own, reference, corpus, reference_sha256=reference_digest, batch_size=17)
    assert mutated
    assert hashlib.sha256(target.read_bytes()).hexdigest() != original_digest


@pytest.mark.parametrize("runtime_unavailable", [False, True])
def test_zero_byte_corpus_cli_is_a_concise_error(tmp_path: Path, runtime_unavailable: bool) -> None:
    import subprocess
    import sys

    own, reference, corpus = (
        tmp_path / "own.onnx",
        tmp_path / "reference.onnx",
        tmp_path / "empty.npy",
    )
    constant_model(own, reference=False)
    digest = constant_model(reference, reference=True)
    corpus.write_bytes(b"")
    # A malformed corpus must be rejected before the optional runtime is loaded.
    # This also keeps an ORT platform warning from obscuring the actual CLI error.
    guarded = """import builtins, runpy, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'onnxruntime':
        raise RuntimeError('runtime imported before invalid corpus was rejected')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv[0] = 'mouse_tools.convert.parity'
runpy.run_module('mouse_tools.convert.parity', run_name='__main__')
"""
    invocation = ["-c", guarded] if runtime_unavailable else ["-m", "mouse_tools.convert.parity"]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            *invocation,
            str(own),
            str(reference),
            str(corpus),
            "--reference-sha256",
            digest,
            "--diagnostic",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("parity failed: invalid input corpus ")
    assert "Traceback" not in result.stderr


def test_all_hashes_are_frozen_before_corpus_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numpy.typing import NDArray

    import mouse_tools.convert.parity as parity_module

    own, reference, corpus = (
        tmp_path / "own.onnx",
        tmp_path / "reference.onnx",
        tmp_path / "planes.npy",
    )
    constant_model(own, reference=False)
    digest = constant_model(reference, reference=True)
    values = np.zeros((1000, 112, 8, 8), np.float32)
    values[:, 109, :, :] = np.arange(1000, dtype=np.float32)[:, None, None]
    np.save(corpus, values, allow_pickle=False)
    real_inputs = parity_module._inputs

    def mutate_at_first_load(path: Path, diagnostic: bool) -> NDArray[np.float32]:
        # This remains a valid, numerically identical graph. Only its artifact
        # identity changes before either model has been loaded by the runner.
        model = onnx.load(own)
        model.producer_name = "changed-before-first-artifact-load"
        onnx.save(model, own)
        return real_inputs(path, diagnostic)

    monkeypatch.setattr(parity_module, "_inputs", mutate_at_first_load)
    with pytest.raises(ConvertError, match="converted.*changed"):
        run_parity(own, reference, corpus, reference_sha256=digest)
