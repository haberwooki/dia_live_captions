"""sherpa-onnx diarization backend — ONNX/CPU, no Hugging Face account or token.

Pipeline: pyannote segmentation-3.0 (ONNX) + a speaker-embedding model
(NeMo TitaNet) + fast clustering. Runs on onnxruntime's CPU provider, so it uses
no VRAM and needs no torch. Models are fetched from the k2-fsa GitHub releases
into the user cache dir (see models.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .base import Diarizer, SpeakerTurn


class SherpaOnnxDiarizer(Diarizer):
    name = "sherpa-onnx"

    def __init__(self, segmentation_model: Path, embedding_model: Path, *,
                 num_speakers: int = -1, threshold: float = 0.5,
                 num_threads: int = 2, provider: str = "cpu"):
        try:
            import sherpa_onnx
        except ImportError:
            raise SystemExit("sherpa-onnx backend needs: pip install sherpa-onnx")

        for p in (segmentation_model, embedding_model):
            if not Path(p).exists():
                raise SystemExit(f"diarization model missing: {p}\n"
                                 f"Run with --download-models to fetch them.")

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation_model)),
                provider=provider, num_threads=num_threads),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding_model), provider=provider, num_threads=num_threads),
            # num_clusters=-1 -> infer the speaker count from `threshold`
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers, threshold=threshold),
            min_duration_on=0.3, min_duration_off=0.5,
        )
        if not config.validate():
            raise SystemExit("sherpa-onnx diarization config invalid (check model paths)")
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    @property
    def sample_rate(self) -> int:
        return self._sd.sample_rate

    def diarize(self, audio16k: np.ndarray) -> List[SpeakerTurn]:
        result = self._sd.process(audio16k.astype(np.float32))
        turns = []
        for seg in result.sort_by_start_time():
            spk = seg.speaker
            label = f"SPEAKER_{int(spk):02d}" if isinstance(spk, (int, float)) else str(spk)
            turns.append(SpeakerTurn(float(seg.start), float(seg.end), label))
        return turns
