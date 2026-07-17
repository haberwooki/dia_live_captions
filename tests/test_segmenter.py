"""Characterization tests for the pure segmenter.

These assert on deterministic segmenter behavior (utterance count, boundaries,
speech duration, timing) driven by a synthetic RMS envelope. They never touch
Whisper or decoded text, so they don't flake across model/precision/beam.
"""
import numpy as np
import pytest

from livecaptions.events import AudioBlock
from livecaptions.asr.segmenter import Segmenter

RATE = 16000
BLOCK = 0.1
NSAMP = int(RATE * BLOCK)

SPEECH = 0.05   # constant amplitude -> RMS ~1638 (int16), above the 350 threshold
SILENCE = 0.0


def blk(amp, t):
    return AudioBlock(samples=np.full(NSAMP, amp, dtype=np.float32), rate=RATE, t=t)


def feed(seg, amps):
    """Feed amplitudes as consecutive 0.1 s blocks; return all emitted utterances."""
    out = []
    t = 0.0
    for a in amps:
        out.extend(seg.push(blk(a, t)))
        t += BLOCK
    return out


def test_single_utterance_on_trailing_silence():
    seg = Segmenter()
    # 0.5 s speech, then 0.8 s silence (end_silence=0.6 fires mid-way)
    out = feed(seg, [SPEECH] * 5 + [SILENCE] * 8)
    assert len(out) == 1
    assert out[0].speech_sec == pytest.approx(0.5, abs=1e-6)
    assert out[0].samples.dtype == np.float32


def test_short_blip_is_filtered():
    # REGRESSION: a 0.1 s transient (keyboard click / notification ding) must NOT
    # produce an utterance. The old code gated on speech+trailing-silence, so the
    # 0.6 s silence tail pushed it over MIN_UTT_SEC and it always fired.
    seg = Segmenter()
    out = feed(seg, [SPEECH] * 1 + [SILENCE] * 8)
    assert out == []


def test_max_utterance_cap_forces_flush():
    seg = Segmenter(max_utt_sec=1.0)
    # 2.0 s of unbroken speech -> two 1.0 s utterances at the cap
    out = feed(seg, [SPEECH] * 20)
    assert len(out) == 2
    for u in out:
        assert u.speech_sec == pytest.approx(1.0, abs=1e-6)


def test_two_utterances_separated_by_pause():
    seg = Segmenter()
    out = feed(seg, [SPEECH] * 5 + [SILENCE] * 8 + [SPEECH] * 5 + [SILENCE] * 8)
    assert len(out) == 2


def test_flush_emits_trailing_speech():
    seg = Segmenter()
    out = feed(seg, [SPEECH] * 5)      # no trailing silence -> nothing emitted yet
    assert out == []
    u = seg.flush()
    assert u is not None
    assert u.speech_sec == pytest.approx(0.5, abs=1e-6)


def test_flush_drops_too_short_speech():
    seg = Segmenter()
    feed(seg, [SPEECH] * 2)            # 0.2 s speech < 0.4 s min
    assert seg.flush() is None


def test_utterance_timing():
    seg = Segmenter()
    out = feed(seg, [SPEECH] * 5 + [SILENCE] * 8)
    assert len(out) == 1
    # speech starts at t=0.0; flush fires on the 6th silence block (index 10),
    # whose end is (1.0 + 0.1) = 1.1 s
    assert out[0].t_start == pytest.approx(0.0, abs=1e-6)
    assert out[0].t_end == pytest.approx(1.1, abs=1e-6)


def test_leading_silence_is_ignored():
    seg = Segmenter()
    # silence before speech must not count toward the utterance or its start time
    out = feed(seg, [SILENCE] * 5 + [SPEECH] * 5 + [SILENCE] * 8)
    assert len(out) == 1
    assert out[0].t_start == pytest.approx(0.5, abs=1e-6)
    assert out[0].speech_sec == pytest.approx(0.5, abs=1e-6)
