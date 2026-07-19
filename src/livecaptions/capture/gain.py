"""Make quiet system audio transcribable, so captions still work with the volume down.

WASAPI loopback hands us the endpoint mix AFTER Windows applies the volume, and
there is no pre-volume tap through this API. But the path is float32, so lowering
the volume just multiplies the waveform — the shape survives, only the scale is
lost, and it can be scaled back.

Measured on real captured audio before writing this, because the obvious
justification for it turned out to be false:
  - Silero's VAD still fires at 1% volume with no gain at all.
  - Whisper transcribes 5% and 1% volume byte-identically to full volume.
  - At 0.3% the ungained decode hallucinated a different sentence, while the
    gained one matched the full-volume reference exactly.
So this is insurance for extreme attenuation, NOT what makes quiet playback work
— that already worked. Do not let anyone (including a future me) justify a
bigger audio-processing chain with "otherwise quiet audio fails".

The one thing this cannot fix is mute (or 0%): those samples are exactly zero, and
no amount of gain recovers signal from zeros.
"""
from __future__ import annotations

import numpy as np

#: Below this RMS a block is treated as silence and left alone — amplifying it
#: would just turn the noise floor into something the VAD reacts to.
NOISE_FLOOR = 2e-5


class AutoGain:
    """Smoothly scale audio toward a target level, bounded and silence-aware.

    Deliberately slow: speech level varies constantly, and a fast-reacting gain
    would pump the level within a single utterance, which hurts recognition more
    than the quiet did. This is a level correction, not a compressor.
    """

    def __init__(self, target_rms: float = 0.05, max_gain: float = 30.0,
                 smoothing: float = 0.08):
        self.target_rms = float(target_rms)
        self.max_gain = float(max_gain)
        self.smoothing = float(smoothing)   # 0..1, per block; small = slow
        self.gain = 1.0
        self.last_input_rms = 0.0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        return self.process(block)

    def process(self, block: np.ndarray) -> np.ndarray:
        if block.size == 0:
            return block
        rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
        self.last_input_rms = rms
        if rms < NOISE_FLOOR:
            return block                      # silence (or mute): nothing to recover

        wanted = min(self.max_gain, max(1.0, self.target_rms / rms))
        # Rise slowly, but fall quickly: over-amplifying loud audio clips, whereas
        # under-amplifying quiet audio merely delays the correction by a block or two.
        alpha = self.smoothing if wanted > self.gain else min(1.0, self.smoothing * 4)
        self.gain += (wanted - self.gain) * alpha

        out = block.astype(np.float32) * self.gain
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:                        # never hand clipped audio to the model
            out /= peak
            self.gain /= peak
        return out
