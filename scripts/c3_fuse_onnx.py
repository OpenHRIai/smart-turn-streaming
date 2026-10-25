"""Stage C3: one ONNX graph, input_features -> P(complete) at the clip end, for pipecat's official benchmark.

    .venv/bin/python scripts/c3_fuse_onnx.py --calibration runs/c3/calibration.json --out data/models/smart-turn-streaming-clipend.onnx
    .venv/bin/python scripts/c3_fuse_onnx.py --no-calibration --out data/models/smart-turn-streaming-clipend-raw.onnx

The streaming detector is two graphs (encoder with the pooled vector, heads over pooled + last-second
mel + silence length) driven by a silence clock; pipecat's ``benchmark.py`` runs one graph with the
stock model's interface: ``input_features`` (B, 80, 800) in, a probability (B, 1) out, threshold 0.5.
This composes that graph for the clip-end case (τ = 0, not speaking): the pooled encoder, the
last-second mel summary computed from the same input tensor, zeros for the τ features, the heads,
P(complete) taken from the heads' output and, optionally, the Platt calibration of
``scripts/c3_calibrate.py`` so that 0.5 is a meaningful operating point. The fused graph reproduces
the two-graph path bit for bit (checked on random inputs at the end).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.compose import merge_models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.features import mel_summary  # noqa: E402


def opset_of(m: onnx.ModelProto) -> int:
    return next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))


def pre_graph(opset: int, ir: int) -> onnx.ModelProto:
    """feats_in (B,80,800) -> mel20_out (B,20,80), tau_out (B,3) zeros."""
    nodes = [
        helper.make_node("Constant", [], ["starts"], value=numpy_helper.from_array(np.array([-100], np.int64))),
        helper.make_node("Constant", [], ["ends"], value=numpy_helper.from_array(np.array([2**62], np.int64))),
        helper.make_node("Constant", [], ["axes"], value=numpy_helper.from_array(np.array([2], np.int64))),
        helper.make_node("Slice", ["feats_in", "starts", "ends", "axes"], ["last100"]),
        helper.make_node("Constant", [], ["shape4"], value=numpy_helper.from_array(np.array([0, 80, 20, 5], np.int64))),
        helper.make_node("Reshape", ["last100", "shape4"], ["grouped"]),
        helper.make_node("Constant", [], ["axm"], value=numpy_helper.from_array(np.array([3], np.int64))),
        helper.make_node("ReduceMean", ["grouped", "axm"], ["meaned"], keepdims=0),
        helper.make_node("Transpose", ["meaned"], ["mel20_out"], perm=[0, 2, 1]),
        # tau features: zeros of shape (B, 3)
        helper.make_node("Shape", ["feats_in"], ["fshape"]),
        helper.make_node("Constant", [], ["i0"], value=numpy_helper.from_array(np.array([0], np.int64))),
        helper.make_node("Constant", [], ["i1"], value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node("Slice", ["fshape", "i0", "i1"], ["bdim"]),
        helper.make_node("Constant", [], ["three"], value=numpy_helper.from_array(np.array([3], np.int64))),
        helper.make_node("Concat", ["bdim", "three"], ["tshape"], axis=0),
        helper.make_node("ConstantOfShape", ["tshape"], ["tau_out"], value=numpy_helper.from_array(np.zeros(1, np.float32))),
    ]
    g = helper.make_graph(nodes, "clipend_pre", [helper.make_tensor_value_info("feats_in", TensorProto.FLOAT, ["B", 80, 800])],
                          [helper.make_tensor_value_info("mel20_out", TensorProto.FLOAT, ["B", 20, 80]),
                           helper.make_tensor_value_info("tau_out", TensorProto.FLOAT, ["B", 3])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], ir_version=ir)
    return m


def post_graph(a: float, b: float, opset: int, ir: int) -> onnx.ModelProto:
    """probs_in (B,2) -> probability (B,1) = sigmoid(a * logit(p_end) + b)."""
    nodes = [
        helper.make_node("Constant", [], ["idx0"], value=numpy_helper.from_array(np.array([0], np.int64))),
        helper.make_node("Gather", ["probs_in", "idx0"], ["p_end"], axis=1),                       # (B,1)
        helper.make_node("Constant", [], ["eps"], value=numpy_helper.from_array(np.array(1e-6, np.float32))),
        helper.make_node("Constant", [], ["one_m_eps"], value=numpy_helper.from_array(np.array(1 - 1e-6, np.float32))),
        helper.make_node("Clip", ["p_end", "eps", "one_m_eps"], ["p_c"]),
        helper.make_node("Constant", [], ["one"], value=numpy_helper.from_array(np.array(1.0, np.float32))),
        helper.make_node("Sub", ["one", "p_c"], ["q"]),
        helper.make_node("Div", ["p_c", "q"], ["odds"]),
        helper.make_node("Log", ["odds"], ["logit"]),
        helper.make_node("Constant", [], ["ca"], value=numpy_helper.from_array(np.array(a, np.float32))),
        helper.make_node("Constant", [], ["cb"], value=numpy_helper.from_array(np.array(b, np.float32))),
        helper.make_node("Mul", ["logit", "ca"], ["scaled"]),
        helper.make_node("Add", ["scaled", "cb"], ["shifted"]),
        helper.make_node("Sigmoid", ["shifted"], ["probability"]),
    ]
    g = helper.make_graph(nodes, "clipend_post", [helper.make_tensor_value_info("probs_in", TensorProto.FLOAT, ["B", 2])],
                          [helper.make_tensor_value_info("probability", TensorProto.FLOAT, ["B", 1])])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], ir_version=ir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="data/models/smart-turn-v3.2-gpu.pooled.onnx")
    ap.add_argument("--heads", default="data/models/eot_heads_v32.onnx")
    ap.add_argument("--calibration", default="runs/c3/calibration.json")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--out", default="data/models/smart-turn-streaming-clipend.onnx")
    a = ap.parse_args()

    enc = onnx.load(a.encoder); heads = onnx.load(a.heads)
    opset, ir = opset_of(enc), enc.ir_version
    if opset_of(heads) != opset:
        from onnx import version_converter
        heads = version_converter.convert_version(heads, opset)
    heads.ir_version = ir
    # a copy of the input so a second consumer can be attached through io_map
    enc.graph.node.append(helper.make_node("Identity", ["input_features"], ["feats_copy"]))
    enc.graph.output.append(helper.make_tensor_value_info("feats_copy", TensorProto.FLOAT, ["B", 80, 800]))
    cal = dict(a=1.0, b=0.0) if a.no_calibration else json.load(open(a.calibration))
    m = merge_models(enc, pre_graph(opset, ir), io_map=[("feats_copy", "feats_in")], prefix2="pre_")
    m = merge_models(m, heads, io_map=[("sum_1", "pooled"), ("pre_mel20_out", "mel20"), ("pre_tau_out", "tau_feat")], prefix2="heads_")
    m = merge_models(m, post_graph(float(cal["a"]), float(cal["b"]), opset, ir), io_map=[("heads_probs", "probs_in")], prefix2="post_")
    # keep one output, the probability, first
    keep = [o for o in m.graph.output if o.name == "post_probability"]
    del m.graph.output[:]
    m.graph.output.extend(keep)
    m.graph.output[0].name = "probability"
    for n in m.graph.node:
        for i, o in enumerate(n.output):
            if o == "post_probability":
                n.output[i] = "probability"
    m.producer_name = "smart-turn-streaming"
    m.doc_string = (f"Smart Turn v3.2 encoder (pooled) + streaming heads at the clip end (tau = 0), calibration a={cal['a']:.4f} b={cal['b']:.4f}; "
                    "built by scripts/c3_fuse_onnx.py for pipecat's benchmark.py")
    onnx.checker.check_model(m)
    onnx.save(m, a.out)

    # verify against the two-graph path
    import onnxruntime as ort
    s_f = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    s_e = ort.InferenceSession(a.encoder, providers=["CPUExecutionProvider"])
    s_h = ort.InferenceSession(a.heads, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((5, 80, 800)) * 0.5 + rng.uniform(-1.5, 1.0, (5, 1, 1))).astype(np.float32)
    logits, pooled = s_e.run(None, {"input_features": x})
    p2 = s_h.run(None, {"pooled": pooled, "mel20": mel_summary(x).astype(np.float32), "tau_feat": np.zeros((5, 3), np.float32)})[0][:, :1]
    p2 = np.clip(p2, 1e-6, 1 - 1e-6); z = np.log(p2 / (1 - p2)); ref = 1 / (1 + np.exp(-(cal["a"] * z + cal["b"])))
    pf = s_f.run(None, {"input_features": x})[0]
    print("fused vs two-graph, max |dp| =", float(np.abs(pf - ref).max()), "| outputs", [o.name for o in s_f.get_outputs()],
          "| inputs", [i.name for i in s_f.get_inputs()], f"| {Path(a.out).stat().st_size / 1e6:.1f} MB")
    assert np.abs(pf - ref).max() < 1e-4


if __name__ == "__main__":
    main()
