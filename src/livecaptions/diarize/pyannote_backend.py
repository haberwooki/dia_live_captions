"""pyannote.audio diarization backend — best local quality (~13% DER).

Needs PyTorch (hence the project's Python 3.12 runtime) and a Hugging Face
token: the pyannote pipelines are GATED, so you must accept the model terms
once and supply a token. community-1 is CC-BY-4.0 (commercial use OK).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Diarizer, SpeakerTurn

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"

_GATED_HELP = """
pyannote models are gated. One-time setup (~2 minutes):
  1. Create a free account:  https://huggingface.co/join
  2. Accept the terms:      https://huggingface.co/{model}
     (and, for 3.1, also:   https://huggingface.co/pyannote/segmentation-3.0)
  3. Create a READ token:   https://huggingface.co/settings/tokens
  4. Give it to the app, either:
       setx HF_TOKEN <your_token>        (then restart the shell)
     or put  hf_token = "<your_token>"  in your config.toml
Or use the no-account backend instead:  --diarizer sherpa
"""


class PyannoteDiarizer(Diarizer):
    name = "pyannote"

    def __init__(self, model: str = DEFAULT_MODEL, token: Optional[str] = None,
                 device: Optional[str] = None, num_speakers: int = -1):
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            import sys
            hint = ("this build does not include the pyannote backend; use the sherpa "
                    "backend instead" if getattr(sys, "frozen", False)
                    else "pyannote backend needs: pip install pyannote.audio")
            raise SystemExit(f"{hint}  ({e})")

        self._num_speakers = num_speakers
        try:
            # pyannote 4.x uses `token=`; older releases used `use_auth_token=`
            try:
                pipe = Pipeline.from_pretrained(model, token=token)
            except TypeError:
                pipe = Pipeline.from_pretrained(model, use_auth_token=token)
        except Exception as e:
            raise SystemExit(f"Could not load '{model}': {e}\n{_GATED_HELP.format(model=model)}")
        if pipe is None:
            # pyannote returns None (rather than raising) when the repo is gated/unauthorized
            raise SystemExit(f"Could not load '{model}' — it is gated or the token is invalid."
                             f"\n{_GATED_HELP.format(model=model)}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(torch.device(device))
        self._pipe = pipe
        self._torch = torch
        self.device = device

    def diarize(self, audio16k: np.ndarray) -> List[SpeakerTurn]:
        waveform = self._torch.from_numpy(audio16k.astype(np.float32)).unsqueeze(0)  # (1, N)
        kwargs = {}
        if self._num_speakers and self._num_speakers > 0:
            kwargs["num_speakers"] = self._num_speakers
        result = self._pipe({"waveform": waveform, "sample_rate": 16000}, **kwargs)

        # pyannote 4.x returns a DiarizeOutput; older versions return an Annotation.
        # Prefer `exclusive_speaker_diarization` — it strips overlapping turns, which is
        # exactly what we want when snapping each transcribed word to one speaker.
        annotation = getattr(result, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(result, "speaker_diarization", result)

        return [SpeakerTurn(float(seg.start), float(seg.end), str(label))
                for seg, _, label in annotation.itertracks(yield_label=True)]
