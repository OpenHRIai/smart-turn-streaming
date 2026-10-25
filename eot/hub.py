"""Model files from the Hugging Face Hub.

    from eot.hub import model_paths
    encoder_onnx, heads_onnx = model_paths()          # downloads once, then the local cache

The two files are `smart-turn-v3.2-encoder-pooled.onnx` (Pipecat's Smart Turn v3.2 encoder with the pooled
vector exposed, 31 MB) and `eot-heads-v32.onnx` (the trained heads, 0.7 MB); see UPSTREAM.md for their
provenance and hashes.
"""

from __future__ import annotations

REPO_ID = "OpenHRI/smart-turn-streaming"
ENCODER_FILE = "smart-turn-v3.2-encoder-pooled.onnx"
HEADS_FILE = "eot-heads-v32.onnx"


def model_paths(repo_id: str = REPO_ID, revision: str | None = None, cache_dir: str | None = None) -> tuple[str, str]:
    """Local paths of the encoder and heads ONNX files, downloaded from the Hub if not cached."""
    from huggingface_hub import hf_hub_download

    enc = hf_hub_download(repo_id, ENCODER_FILE, revision=revision, cache_dir=cache_dir)
    heads = hf_hub_download(repo_id, HEADS_FILE, revision=revision, cache_dir=cache_dir)
    return enc, heads
