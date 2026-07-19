"""Captions must survive the volume being turned down.

WASAPI loopback delivers the endpoint mix AFTER Windows applies the output volume,
so playing quietly shrinks the signal we receive until the VAD stops firing and
captions simply stop. The path is float32, so the attenuation is a multiply and the
waveform is recoverable — except at mute, where the samples are exactly zero.
"""
import numpy as np
import pytest

from livecaptions.capture.gain import NOISE_FLOOR, AutoGain


def speech_like(seconds=1.0, sr=16000, amp=1.0):
    """A rough speech-band signal: a few harmonics with an envelope."""
    t = np.arange(int(seconds * sr)) / sr
    sig = (np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 380 * t)
           + 0.25 * np.sin(2 * np.pi * 900 * t))
    env = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / seconds, 0, 1)))
    return (sig * env / 3.0 * amp).astype(np.float32)


def run(gain, audio, block=1600):
    return np.concatenate([gain.process(audio[i:i + block])
                           for i in range(0, len(audio), block)])


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2))) if x.size else 0.0


@pytest.mark.parametrize("volume", [1.0, 0.25, 0.05, 0.01])
def test_quiet_audio_is_brought_back_up(volume):
    """Down to 1% volume, the level must land near the target the VAD expects."""
    quiet = speech_like(3.0, amp=volume)
    out = run(AutoGain(target_rms=0.05), quiet)
    assert rms(out) > 0.02, f"still too quiet at volume {volume}: rms={rms(out):.4f}"


def test_mute_stays_muted():
    """Zeros carry no signal — amplifying them would only invent noise."""
    out = run(AutoGain(), np.zeros(16000, dtype=np.float32))
    assert rms(out) == 0.0


def test_near_silence_is_not_amplified():
    """A dead-quiet endpoint must not have its noise floor pumped up into
    something the speech detector reacts to."""
    hiss = (np.random.default_rng(0).standard_normal(16000) * NOISE_FLOOR * 0.1
            ).astype(np.float32)
    out = run(AutoGain(), hiss)
    assert rms(out) <= rms(hiss) * 1.01


def test_loud_audio_is_left_alone():
    """Already-loud audio must not be attenuated or clipped."""
    loud = speech_like(2.0, amp=1.0)
    out = run(AutoGain(target_rms=0.05), loud)
    assert np.max(np.abs(out)) <= 1.0 + 1e-6, "clipped"
    assert rms(out) >= rms(loud) * 0.9, "quietened audio that was already fine"


def test_never_clips():
    for volume in (0.02, 0.2, 1.0):
        out = run(AutoGain(), speech_like(2.0, amp=volume))
        assert np.max(np.abs(out)) <= 1.0 + 1e-6, f"clipped at volume {volume}"


def test_gain_is_bounded():
    g = AutoGain(max_gain=30.0)
    run(g, speech_like(3.0, amp=0.0005))
    assert g.gain <= 30.0 + 1e-6


def test_gain_does_not_pump_within_an_utterance():
    """A fast gain would swing the level mid-sentence, which hurts recognition
    more than the quiet did. Measure block-to-block level stability."""
    g = AutoGain()
    audio = speech_like(3.0, amp=0.05)
    levels = [rms(g.process(audio[i:i + 1600])) for i in range(0, len(audio), 1600)]
    levels = [x for x in levels if x > 0.001][3:]          # skip the initial ramp
    if len(levels) > 2:
        jumps = [abs(b - a) / max(a, 1e-6) for a, b in zip(levels, levels[1:])]
        assert max(jumps) < 1.5, f"level swings mid-utterance: {jumps}"
