#!/usr/bin/env python3
"""Assemble the Hugging Face release directory for R1 and print the upload command (nothing is uploaded).

    python release/stage.py [--repo-id OpenHRI/smart-turn-streaming] [--stage-dir build/hf_stage] [--encoder ...] [--heads ...]

Copies the two ONNX files from data/models under their release names, the model card, the rule tables,
the evaluation script and manifests, and the licence / attribution files from release/; checks that both
ONNX graphs load and have the documented inputs and outputs; writes SHA-256 sums; prints the `hf upload`
command and the equivalent `huggingface_hub` call.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "README.md": ROOT / "release/README.md",
    "LICENSE": ROOT / "release/LICENSE",
    "AMI_LICENCE.txt": ROOT / "release/AMI_LICENCE.txt",
    "ATTRIBUTION.md": ROOT / "release/ATTRIBUTION.md",
    "rules/theta_tables.csv": ROOT / "release/rules/theta_tables.csv",
    "rules/eager_tables.csv": ROOT / "release/rules/eager_tables.csv",
    "eval/ami_eval.py": ROOT / "release/eval/ami_eval.py",
    "eval/ami_dev.csv": ROOT / "release/eval/ami_dev.csv",
    "eval/ami_test.csv": ROOT / "release/eval/ami_test.csv",
    "eval/ami_eval_reference.md": ROOT / "release/eval/ami_eval_reference.md",
}


def check_onnx(enc: Path, heads: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    e = ort.InferenceSession(str(enc), providers=["CPUExecutionProvider"])
    assert [i.name for i in e.get_inputs()] == ["input_features"], e.get_inputs()
    assert [o.name for o in e.get_outputs()] == ["logits", "sum_1"], e.get_outputs()
    h = ort.InferenceSession(str(heads), providers=["CPUExecutionProvider"])
    assert [i.name for i in h.get_inputs()] == ["pooled", "mel20", "tau_feat"], h.get_inputs()
    assert [o.name for o in h.get_outputs()] == ["probs"], h.get_outputs()
    feats = np.zeros((1, 80, 800), np.float32)
    logits, pooled = e.run(None, {"input_features": feats})
    assert pooled.shape == (1, 384) and logits.shape == (1, 1)
    p = h.run(None, {"pooled": pooled, "mel20": np.zeros((1, 20, 80), np.float32), "tau_feat": np.zeros((1, 3), np.float32)})[0]
    assert p.shape == (1, 2) and (0 <= p).all() and (p <= 1).all(), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="OpenHRI/smart-turn-streaming")
    ap.add_argument("--stage-dir", default=str(ROOT / "build/hf_stage"))
    ap.add_argument("--encoder", default=str(ROOT / "data/models/smart-turn-v3.2-gpu.pooled.onnx"))
    ap.add_argument("--heads", default=str(ROOT / "data/models/eot_heads_v32.onnx"))
    a = ap.parse_args()
    FILES["smart-turn-v3.2-encoder-pooled.onnx"] = Path(a.encoder)
    FILES["eot-heads-v32.onnx"] = Path(a.heads)
    stage = Path(a.stage_dir) / a.repo_id.split("/")[-1]
    if stage.exists():
        shutil.rmtree(stage)
    sums = []
    for rel, src in FILES.items():
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sums.append(f"{hashlib.sha256(dst.read_bytes()).hexdigest()}  {rel}")
        print(f"{dst.stat().st_size / 1e6:8.2f} MB  {rel}")
    (stage / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    check_onnx(stage / "smart-turn-v3.2-encoder-pooled.onnx", stage / "eot-heads-v32.onnx")
    print(f"\nONNX graphs load and run; staged in {stage}\n")
    print("Upload (after `hf auth login`; creates the repo if needed; nothing is uploaded by this script):\n")
    print(f"  hf upload {a.repo_id} {stage} . --repo-type model --commit-message 'R1: streaming end-of-turn heads'\n")
    print("or from Python:\n")
    print(f"  from huggingface_hub import HfApi; api = HfApi(); api.create_repo('{a.repo_id}', repo_type='model', exist_ok=True)")
    print(f"  api.upload_folder(folder_path='{stage}', repo_id='{a.repo_id}', repo_type='model', commit_message='R1: streaming end-of-turn heads')")


if __name__ == "__main__":
    main()
