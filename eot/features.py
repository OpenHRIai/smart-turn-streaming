"""Frozen-encoder features for Stage B1.

Two frozen backbones are supported, both Whisper-tiny encoders with attention pooling:
pipecat's ``smart-turn-v3.2-gpu.onnx`` (the commercial baseline, trained on smart-turn-data) and
our audio-only export ``smart-turn-multimodal-audio.onnx``. ``expose_pooled`` rewrites a graph so
that, besides the model's own probability, it also returns the pooled 384-d vector (the output of
the top-level ReduceSum that follows the attention softmax, i.e. the input to the model's head).

Per window the cache stores:

* ``pooled``   (384,)  fp16  the frozen encoder's pooled vector
* ``mel20``    (20, 80) fp16 the last 1 s of the Whisper log-mel input averaged to 50 ms steps
                            (the "fresh" signal the per-hop GRU of the streaming design reads)
* ``p_base``   ()      fp32  the frozen model's own P(complete) for the window (free baseline)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16000
N_FRAMES = 800          # 8 s of 10 ms Whisper frames
MEL_STEPS = 20          # last 1 s at 50 ms
MEL_STEP_FRAMES = 5


def expose_pooled(src: str | Path, dst: str | Path) -> tuple[str, str]:
    """Write ``dst`` = ``src`` with the pooled vector added as a graph output. Returns the two
    output names (probability, pooled)."""
    import onnx
    from onnx import helper
    m = onnx.load(str(src))
    g = m.graph
    rs = [n for n in g.node if n.op_type == "ReduceSum" and not n.name.startswith("/encoder")]
    if len(rs) != 1:
        raise RuntimeError(f"expected one top-level ReduceSum, found {[n.name for n in rs]}")
    pooled = rs[0].output[0]
    if pooled not in [o.name for o in g.output]:
        g.output.append(helper.make_tensor_value_info(pooled, onnx.TensorProto.FLOAT, ["batch_size", 384]))
    onnx.save(m, str(dst))
    return g.output[0].name, pooled


def mel_summary(feats: np.ndarray) -> np.ndarray:
    """(B, 80, 800) Whisper log-mel -> (B, 20, 80): the last 100 frames averaged in groups of 5."""
    last = feats[:, :, -MEL_STEPS * MEL_STEP_FRAMES:]                      # (B, 80, 100)
    return last.reshape(feats.shape[0], feats.shape[1], MEL_STEPS, MEL_STEP_FRAMES).mean(-1).transpose(0, 2, 1)


class FeatureSession:
    """ONNX Runtime session over an ``expose_pooled`` graph."""

    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None, threads: int = 4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        outs = [o.name for o in self.sess.get_outputs()]
        self.out_prob, self.out_pooled = outs[0], outs[-1]

    def run(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, 80, 800) float32 -> (p_base (B,), pooled (B, 384))."""
        prob, pooled = self.sess.run([self.out_prob, self.out_pooled], {self.inp: feats.astype(np.float32)})
        return prob.reshape(-1), pooled
