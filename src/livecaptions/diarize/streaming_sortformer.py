"""Live speaker diarization via NVIDIA Streaming Sortformer (4-speaker cap).

Sortformer keeps speaker identity stable across a stream using an Arrival-Order
Speaker Cache, so we drive its true streaming API (`forward_streaming_step` with
a persistent state) rather than re-running a batch diarizer over a sliding
window — that would swap speaker labels between windows.

Model: nvidia/diar_streaming_sortformer_4spk-v2 (CC-BY-4.0, commercial use OK).
NOTE the offline diar_sortformer_4spk-v1 is CC-BY-NC — don't use that one.

Honest limits (docs/diarization-research.md): max 4 speakers, and our audio is a
single post-mix stream, so labels are best-effort — decent on 2-speaker,
low-overlap audio, weaker on crowded calls.

The streaming loop follows NVIDIA's own reference service (Apache-2.0); the
buffering helpers it needs are vendored in _nemo_vendor.py to avoid pulling in
the pipecat framework.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

WHISPER_SR = 16000
FRAME_SEC = 0.08          # Sortformer emits one prediction per 80 ms frame
MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2"


class StreamingSortformer:
    #: CPU by default on purpose: it runs at ~RTF 0.4 (comfortably real-time) and
    #: uses no VRAM, keeping the 8 GB free for the Whisper worker. (Also, the
    #: default PyPI torch wheel on Windows is CPU-only anyway.)
    def __init__(self, *, device: str = "cpu", max_speakers: int = 4,
                 threshold: float = 0.5, history_sec: float = 120.0,
                 left_offset: int = 8, right_offset: int = 8):
        try:
            import torch
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:
            import sys
            hint = ("this build does not include live diarization"
                    if getattr(sys, "frozen", False)
                    else "live diarization needs NeMo: pip install nemo_toolkit[asr]")
            raise SystemExit(f"{hint}  ({e})")
        from ._nemo_vendor import CacheFeatureBufferer

        self._torch = torch
        self.device = device
        self._threshold = threshold
        self._history = history_sec
        self._left_offset = left_offset
        self._right_offset = right_offset

        model = SortformerEncLabelModel.from_pretrained(MODEL_ID, map_location=device)
        # streaming hyper-parameters (NVIDIA's reference defaults for 4spk-v2)
        mods = model.sortformer_modules
        mods.chunk_len = 6
        mods.spkcache_len = 188
        mods.spkcache_refresh_rate = 144
        mods.fifo_len = 188
        mods.chunk_left_context = 1
        mods.chunk_right_context = 7
        mods.log = False
        model.eval()
        self._model = model
        self._max_speakers = mods.n_spk

        self._chunk_frames = mods.chunk_len
        chunk_sec = self._chunk_frames * FRAME_SEC
        buffer_sec = chunk_sec + (left_offset + right_offset) * 0.01
        self._bufferer = CacheFeatureBufferer(
            sample_rate=WHISPER_SR, buffer_size_in_secs=buffer_sec,
            chunk_size_in_secs=chunk_sec, preprocessor_cfg=model.cfg.preprocessor,
            device=device)

        self._state = mods.init_streaming_state(
            batch_size=1, async_streaming=model.async_streaming, device=model.device)
        self._total_preds = torch.zeros((1, 0, self._max_speakers), device=model.device)

        self._chunk_samples = int(chunk_sec * WHISPER_SR)
        self._pending = np.zeros(0, dtype=np.float32)
        self._t = 0.0                                   # global time of the next frame
        self._timeline: List[Tuple[float, float, Optional[str]]] = []

    @property
    def chunk_sec(self) -> float:
        return self._chunk_samples / WHISPER_SR

    def _step(self, chunk: np.ndarray) -> np.ndarray:
        """One streaming step -> [frames, speakers] probabilities for this chunk."""
        torch = self._torch
        self._bufferer.update(chunk)
        feats = self._bufferer.get_feature_buffer().unsqueeze(0).transpose(1, 2)
        lens = torch.tensor([feats.shape[1]], device=self.device)
        with torch.inference_mode():
            self._state, preds = self._model.forward_streaming_step(
                processed_signal=feats, processed_signal_length=lens,
                streaming_state=self._state, total_preds=self._total_preds,
                left_offset=self._left_offset, right_offset=self._right_offset)
        self._total_preds = preds
        return preds[:, -self._chunk_frames:, :].clone().cpu().numpy()[0]

    def feed(self, audio16k: np.ndarray) -> None:
        """Feed float32 mono 16 kHz audio; extends the speaker timeline."""
        self._pending = np.append(self._pending, audio16k.astype(np.float32))
        while len(self._pending) >= self._chunk_samples:
            chunk = self._pending[:self._chunk_samples]
            self._pending = self._pending[self._chunk_samples:]
            for row in np.atleast_2d(self._step(chunk)):
                idx = int(np.argmax(row))
                speaker = f"SPEAKER_{idx:02d}" if float(row[idx]) >= self._threshold else None
                self._timeline.append((self._t, self._t + FRAME_SEC, speaker))
                self._t += FRAME_SEC
        self._prune()

    def _prune(self) -> None:
        cutoff = self._t - self._history
        if cutoff > 0 and self._timeline and self._timeline[0][0] < cutoff:
            self._timeline = [f for f in self._timeline if f[1] >= cutoff]

    def speaker_at(self, start: float, end: float) -> Optional[str]:
        """The speaker holding most of [start, end) — None if nobody does."""
        totals: dict = {}
        for a, b, spk in self._timeline:
            if b <= start:
                continue
            if a >= end:
                break
            if spk:
                totals[spk] = totals.get(spk, 0.0) + (min(end, b) - max(start, a))
        return max(totals, key=totals.get) if totals else None

    def speakers_seen(self) -> List[str]:
        return sorted({s for _, _, s in self._timeline if s})
