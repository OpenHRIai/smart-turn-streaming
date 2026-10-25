# Upstream: Pipecat's Smart Turn

This repository is not a fork of [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn).
It depends on one artifact from it, the frozen Smart Turn v3.2 encoder, and on nothing in its code.

## Pinned encoder

| | |
|---|---|
| Hub repository | [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) |
| revision | `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` (2026-01-07) |
| file | `smart-turn-v3.2-gpu.onnx`, 32,411,198 bytes |
| sha256 | `ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be` |
| licence | BSD 2-Clause (Daily) |

What this repository ships is derived from that file:

| file on the Hub (`OpenHRI/smart-turn-streaming`) | derivation | sha256 |
|---|---|---|
| `smart-turn-v3.2-encoder-pooled.onnx` | the same graph with the pooled 384-d vector exposed as a second output, weights unchanged (`eot.features.expose_pooled`) | `37e2a4fe22e1c4efaeba4df40141e422a3b96519db9160ae9134962a936ba4ea` |
| `eot-heads-v32.onnx` | heads trained on the pooled vector, the last second of log-mel and the silence length (`scripts/b1_train.py`, run `knob_ami07_auc`) | `8d5b4039c9d7c27e22e84d372c343077489f2fe291ccd67caeeb5ce45b05ce1a` |

The code repository was at `pipecat-ai/smart-turn@4786657e242dfe77dd138699ac564ee074a2a543` (2026-01-29)
when this was last checked; the Pipecat integration targets `pipecat-ai` 1.14.

## Taking a new Smart Turn encoder

A new upstream model is a new file on the Hub, not a merge. To move to it:

1. Export the pooled vector: `python -c "from eot.features import expose_pooled as e; e('smart-turn-vX.onnx', 'smart-turn-vX.pooled.onnx')"`.
2. Rebuild the feature cache for the training items with the new encoder (`modal_features.py`, or
   `eot.features` locally) and retrain the heads (`scripts/b1_train.py`); the heads are specific to the
   encoder's pooled space.
3. Refit the commit and eager tables on AMI dev (`scripts/b1_train.py` writes the commit tables,
   `scripts/c1_eager_sweep.py` the eager tables), re-run the B2 fidelity check (`scripts/b2_validate.py`)
   and the Pipecat replay (`scripts/b3_replay.py`).
4. Update this file.

## If upstream code is ever needed

Vendor the directory rather than forking the repository:

```
git subtree add --prefix third_party/smart-turn https://github.com/pipecat-ai/smart-turn.git main --squash
git subtree pull --prefix third_party/smart-turn https://github.com/pipecat-ai/smart-turn.git main --squash
```
