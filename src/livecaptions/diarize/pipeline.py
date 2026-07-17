"""The offline post-processing pass: audio file -> speaker-labeled transcript.

WhisperX-style: transcribe with word timestamps, diarize the same audio, then
assign each word to the speaker whose turn overlaps it most, and group
consecutive same-speaker words into caption lines.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soxr

from ..asr.streaming import WHISPER_SR
from .assign import SpeakerSegment, Word, assign_speakers, group_into_segments

def load_wav_16k(path: str) -> np.ndarray:
    """Read a 16-bit PCM WAV as float32 mono at 16 kHz."""
    try:
        with wave.open(str(path), "rb") as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (OSError, wave.Error, EOFError) as e:
        raise SystemExit(f"{path}: cannot read as a WAV file ({e})")
    if width != 2:
        raise SystemExit(f"{path}: need a 16-bit PCM WAV (sample width {width} bytes)")
    data = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        usable = (len(data) // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    audio = data.astype(np.float32) / 32768.0
    if rate != WHISPER_SR:
        audio = soxr.resample(audio, rate, WHISPER_SR).astype(np.float32)
    return audio


def transcribe_words(model, audio16k: np.ndarray, settings) -> List[Word]:
    """Whisper word timestamps for the whole clip."""
    segments, _ = model.transcribe(
        audio16k, language=settings.language, beam_size=settings.beam_size,
        word_timestamps=True, vad_filter=True)
    words: List[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            if w.word.strip():
                words.append((float(w.start), float(w.end), w.word.strip()))
    return words


def diarize_file(path: str, model, settings, *, backend: str = "auto",
                 num_speakers: int = -1) -> Tuple[List[SpeakerSegment], int]:
    """Returns (speaker-labeled segments, speaker count)."""
    from .factory import make_diarizer

    audio = load_wav_16k(path)
    dur = len(audio) / WHISPER_SR
    print(f"Audio: {Path(path).name}  ({dur:.1f}s)")

    print("Transcribing (word timestamps)...")
    words = transcribe_words(model, audio, settings)
    print(f"  {len(words)} words")

    diarizer = make_diarizer(settings, backend=backend, num_speakers=num_speakers)
    print(f"Diarizing with '{diarizer.name}'...")
    turns = diarizer.diarize(audio)
    speakers = sorted({t.speaker for t in turns})
    print(f"  {len(turns)} turns, {len(speakers)} speaker(s): {', '.join(speakers) or '-'}")

    segments = group_into_segments(assign_speakers(words, turns))
    return segments, len(speakers)
