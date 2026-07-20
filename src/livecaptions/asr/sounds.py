"""Say what the audio IS when it isn't speech — music, or just sound.

Silence and "playing fine, but nobody is talking" look identical in a caption
overlay: both show nothing. That ambiguity has cost real debugging time on this
project, where a blank overlay could mean the wrong audio device, a stopped
pipeline, a muted output, or simply music. Naming the non-speech audio removes a
whole class of "is it broken?".

No new model. Silero already runs on every buffer and answers "is this speech";
what is left is separating tonal, sustained audio (music) from everything else
(notification sounds, keyboard, fans, applause). That distinction is cheap in the
frequency domain: music holds energy in a few strong harmonics, so its spectrum is
PEAKY, while noise spreads energy evenly and is FLAT. Spectral flatness measures
exactly that, and needs one FFT.

Deliberately coarse. This says "music" or "sound", never a genre or a source. A
confident wrong guess is worse than a vague right one.
"""
from __future__ import annotations

import numpy as np

#: Below this RMS there is nothing worth naming — no caption for a silent room.
SILENCE_RMS = 0.002

#: Geometric/arithmetic mean ratio of the power spectrum. Tonal audio concentrates
#: energy in few bins and scores LOW; broadband noise approaches 1.0. Music with
#: heavy percussion drifts upward, so the threshold sits well clear of pure tones.
MUSIC_FLATNESS = 0.28

#: Music sustains; a notification chirp does not. Requiring the audio to still be
#: there after a beat keeps single "ding"s from being announced as music.
MIN_MUSIC_SEC = 1.5


#: Windows sampled across the buffer. Measuring ONE window (the middle) reported
#: flatness 1.000 — "not music" — for real audio that was plainly music, because the
#: middle of a several-second buffer is often a pause between words: every bin zero,
#: no spectrum to measure. Several windows, silent ones skipped, fixes that.
_WINDOWS = 5
_WINDOW_N = 4096


def _window_flatness(window: np.ndarray) -> float | None:
    """Flatness of one window, or None if it holds no measurable signal."""
    window = window.astype(np.float64) * np.hanning(window.size)
    spectrum = np.abs(np.fft.rfft(window)) ** 2
    # Ignore the lowest bins: mains hum and DC offset are not musical content, but
    # they dominate the geometric mean and would make everything look tonal.
    spectrum = spectrum[3:]
    spectrum = spectrum[spectrum > 0]
    if spectrum.size < 8:
        return None
    geometric = np.exp(np.mean(np.log(spectrum)))
    arithmetic = np.mean(spectrum)
    if arithmetic <= 0:
        return None
    return float(np.clip(geometric / arithmetic, 0.0, 1.0))


def spectral_flatness(audio: np.ndarray, rate: int = 16000) -> float:
    """0 = a pure tone, 1 = white noise. NaN-safe, and pause-safe.

    Takes the MEDIAN across several windows: one loud transient (a door, a drum
    hit) should not decide what a whole passage is.
    """
    if audio.size < 512:
        return 1.0
    n = min(_WINDOW_N, audio.size)
    if audio.size <= n:
        return _window_flatness(audio[:n]) or 1.0
    starts = np.linspace(0, audio.size - n, _WINDOWS).astype(int)
    scores = [s for s in (_window_flatness(audio[i:i + n]) for i in starts) if s is not None]
    return float(np.median(scores)) if scores else 1.0


def classify(audio: np.ndarray, rate: int = 16000) -> str:
    """"silence" | "music" | "sound" for a buffer already known to lack speech."""
    if audio.size == 0:
        return "silence"
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < SILENCE_RMS:
        return "silence"
    return "music" if spectral_flatness(audio, rate) < MUSIC_FLATNESS else "sound"


class SoundLabeller:
    """Tracks non-speech audio and decides when to announce it.

    Keeps state because a label should appear once and persist, not flicker on
    every pass: the overlay is being read, not watched.
    """

    LABELS = {"music": "♪  music playing", "sound": "•  sound (no speech)"}

    def __init__(self, min_music_sec: float = MIN_MUSIC_SEC):
        self.min_music_sec = float(min_music_sec)
        self._kind = "silence"
        self._since = 0.0
        self._announced: str | None = None

    def update(self, audio: np.ndarray, now: float, rate: int = 16000) -> str | None:
        """Feed a non-speech buffer. Returns a label to show, or None.

        Returns the label only on a CHANGE, so the caller can leave it on screen
        without re-emitting it every pass.
        """
        kind = classify(audio, rate)
        if kind != self._kind:
            self._kind, self._since = kind, now
        if kind == "silence":
            self._announced = None
            return None
        # A chirp should not be announced as music; wait for it to sustain.
        if now - self._since < self.min_music_sec:
            return None
        if self._announced == kind:
            return None
        self._announced = kind
        return self.LABELS[kind]

    def speech_started(self) -> None:
        """Speech takes the overlay back; the next non-speech run re-announces."""
        self._kind, self._announced = "silence", None
