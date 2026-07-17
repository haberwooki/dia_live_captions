"""Speaker diarization — "who is talking".

Offline (post-processing) diarization over saved audio. Two swappable backends:
  * pyannote.audio (best quality; needs a Hugging Face token — models are gated)
  * sherpa-onnx   (no account/token, ONNX/CPU, works out of the box)

The hard truth (see docs/diarization-research.md): our audio is a single
post-mix loopback stream, so speaker labels are inherently probabilistic —
good on 2-speaker low-overlap audio, mediocre on crowded/overlapping calls.
"""
