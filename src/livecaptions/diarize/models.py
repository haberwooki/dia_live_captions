"""Diarization model files: where they live and how to fetch them.

Only the sherpa-onnx backend needs local files (pyannote pulls from the HF hub
into its own cache). These come from the k2-fsa sherpa-onnx GitHub releases —
ungated, no account required.
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import platformdirs

MODEL_DIR = Path(platformdirs.user_data_dir("live-captions", appauthor=False)) / "models"

_SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
_EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-recongition-models/nemo_en_titanet_small.onnx")   # (sic: upstream typo)

SEGMENTATION_MODEL = MODEL_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING_MODEL = MODEL_DIR / "nemo_en_titanet_small.onnx"


def sherpa_models_present() -> bool:
    return SEGMENTATION_MODEL.exists() and EMBEDDING_MODEL.exists()


def download_sherpa_models(force: bool = False) -> None:
    """Fetch the sherpa-onnx diarization models (~47 MB) into the user data dir."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if force or not EMBEDDING_MODEL.exists():
        print(f"Downloading speaker-embedding model -> {EMBEDDING_MODEL}")
        urllib.request.urlretrieve(_EMB_URL, EMBEDDING_MODEL)

    if force or not SEGMENTATION_MODEL.exists():
        archive = MODEL_DIR / "segmentation.tar.bz2"
        print(f"Downloading segmentation model -> {SEGMENTATION_MODEL.parent}")
        urllib.request.urlretrieve(_SEG_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(MODEL_DIR)
        archive.unlink(missing_ok=True)

    if not sherpa_models_present():
        raise SystemExit(f"model download failed; expected:\n  {SEGMENTATION_MODEL}\n  {EMBEDDING_MODEL}")
    print("Diarization models ready.")
