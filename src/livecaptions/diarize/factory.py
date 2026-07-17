"""Pick a diarization backend.

`auto` (the default) prefers **pyannote** — the best local quality — but it needs
a Hugging Face token because the models are gated. With no token available we
fall back to **sherpa-onnx**, which needs no account and runs on CPU/ONNX, and
say so rather than failing.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Optional

from .base import Diarizer


def _pyannote_available() -> bool:
    return (importlib.util.find_spec("pyannote.audio") is not None
            and importlib.util.find_spec("torch") is not None)


def resolve_token(settings) -> Optional[str]:
    """HF token from settings (config.toml / LC_HF_TOKEN) or the usual HF env vars,
    else whatever `huggingface-cli login` stored."""
    token = getattr(settings, "hf_token", None) or os.environ.get("HF_TOKEN") \
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def make_diarizer(settings, backend: str = "auto", num_speakers: int = -1) -> Diarizer:
    token = resolve_token(settings)

    if backend == "auto":
        # Prefer pyannote only if a token AND the library are both present, so a
        # tokenless user (or a build without torch) degrades to sherpa instead of
        # hard-failing on a missing/gated backend.
        if token and _pyannote_available():
            backend = "pyannote"
        else:
            backend = "sherpa"
            why = "no Hugging Face token found" if not token else "pyannote/torch not available"
            print(f"({why} -> using the sherpa-onnx backend. "
                  "For pyannote's better quality, see --diarizer pyannote for setup.)")

    if backend == "pyannote":
        from .pyannote_backend import PyannoteDiarizer
        return PyannoteDiarizer(
            model=getattr(settings, "diarize_model", "pyannote/speaker-diarization-community-1"),
            token=token, num_speakers=num_speakers)

    if backend == "sherpa":
        from .models import EMBEDDING_MODEL, SEGMENTATION_MODEL, download_sherpa_models, sherpa_models_present
        from .sherpa_backend import SherpaOnnxDiarizer
        if not sherpa_models_present():
            download_sherpa_models()
        return SherpaOnnxDiarizer(SEGMENTATION_MODEL, EMBEDDING_MODEL,
                                  num_speakers=num_speakers,
                                  threshold=getattr(settings, "diarize_threshold", 0.5))

    raise SystemExit(f"unknown diarizer backend: {backend!r} (use auto, pyannote, or sherpa)")
