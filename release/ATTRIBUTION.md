# Attribution and licences

## This release

The trained heads (`eot-heads-v32.onnx`), the decision-rule tables (`rules/`), the evaluation
script and manifests (`eval/`) and the model card are released by the OpenHRI project under the BSD 2-Clause
License (`LICENSE`), copyright Maxim Makatchev.

## Encoder: Smart Turn v3.2 (Daily / Pipecat)

`smart-turn-v3.2-encoder-pooled.onnx` is Pipecat's `smart-turn-v3.2-gpu.onnx`
(https://huggingface.co/pipecat-ai/smart-turn-v3, BSD 2-Clause, Copyright (c) 2024–2025 Daily)
with one change: the attention-pooled 384-d vector that feeds the model's own classifier is added
as a second graph output (`sum_1`). Weights are unchanged, so the first output (`logits`) is
bit-exact with the original model. The encoder architecture is OpenAI's Whisper-tiny encoder
(MIT), as fine-tuned by Pipecat on `pipecat-ai/smart-turn-data-v3.2-train`.

## Training data of the heads

- `pipecat-ai/smart-turn-data-v3.2-train` (Pipecat; the τ-padded and pre-offset variants described
  in the model card are derived from it on the fly and are not redistributed).
- **AMI Meeting Corpus** (University of Edinburgh and the AMI consortium),
  https://groups.inf.ed.ac.uk/ami/corpus/, released under the Creative Commons Attribution 4.0
  International licence (`AMI_LICENCE.txt`, the licence text shipped with the corpus annotations,
  release 1.7). The per-speaker close-talking headset recordings and the NXT word alignments were
  used; the heads were trained on the 137 Kaldi-recipe training meetings.

  Citation: J. Carletta et al., "The AMI Meeting Corpus: A Pre-announcement", in *Machine Learning
  for Multimodal Interaction (MLMI 2005)*, LNCS 3869, Springer, 2006, pp. 28–39.

## Evaluation manifests (`eval/ami_dev.csv`, `eval/ami_test.csv`)

Derived from the AMI Meeting Corpus (CC BY 4.0) and redistributed under the same licence with
attribution to the corpus authors above. Each row is one labelled silence of one speaker:
`meeting`, `speaker` (NXT agent letter), `channel` (headset number in the corpus audio), `t` (the
speaker's VAD stop, seconds into the meeting), `label` (`complete`: another participant took the
floor within 3 s; `incomplete`: the speaker resumed), `gap_s` (incomplete: length of the pause),
`b_onset_s` (complete: the next speaker's onset after `t`), `a_turn_s` (speech in the speaker's turn
so far). The VAD comes from Silero run on the headset audio and gated by the transcript; the
labelling rules are documented in the public repository (`docs/a2_ami.md`). The dev split is the
Kaldi AMI recipe's dev set (18 meetings), the test split its eval set (16 meetings). Audio is not
redistributed: `eval/ami_eval.py` downloads the headset files it needs from the corpus mirror.

## Not used

Nothing in this release was trained or fitted on Meta's Seamless Interaction dataset (CC BY-NC) or
on Meta's Casual Conversations dataset.
