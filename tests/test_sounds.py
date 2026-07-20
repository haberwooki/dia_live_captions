"""Naming non-speech audio, so a blank overlay stops being ambiguous.

Silence, the wrong audio device, a stopped pipeline and "music with nobody talking"
all render identically today: nothing. That ambiguity has repeatedly cost debugging
time on this project, so the audio gets named.

The classifier must be RIGHT about music vs sound often enough to be useful, and —
more importantly — must never announce anything over silence, because a caption
that appears in a quiet room is worse than no caption at all.
"""
import numpy as np
import pytest

from livecaptions.asr import sounds as S

RATE = 16000


def tone_music(seconds=2.0, rate=RATE):
    """A chord with harmonics and a beat — tonal, so spectrally peaky."""
    t = np.arange(int(seconds * rate)) / rate
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330, 440, 554))
    beat = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))          # slow amplitude pulse
    return (chord / 5 * beat * 0.3).astype(np.float32)


def white_noise(seconds=2.0, rate=RATE, amp=0.1):
    rng = np.random.default_rng(7)
    return (rng.standard_normal(int(seconds * rate)) * amp).astype(np.float32)


def silence(seconds=2.0, rate=RATE):
    return np.zeros(int(seconds * rate), dtype=np.float32)


class TestClassify:
    def test_silence_is_silence(self):
        assert S.classify(silence()) == "silence"

    def test_a_quiet_room_is_still_silence(self):
        """Room tone is not zero. Announcing "sound" for a quiet room would put a
        caption on screen when nothing is happening."""
        assert S.classify(white_noise(amp=0.0005)) == "silence"

    def test_tonal_audio_is_music(self):
        assert S.classify(tone_music()) == "music"

    def test_broadband_noise_is_sound_not_music(self):
        """A fan, a fingers-on-keyboard rattle: real, but not music."""
        assert S.classify(white_noise()) == "sound"

    def test_empty_buffer_does_not_crash(self):
        assert S.classify(np.zeros(0, dtype=np.float32)) == "silence"


class TestSpectralFlatness:
    def test_a_pure_tone_is_peaky(self):
        t = np.arange(RATE) / RATE
        assert S.spectral_flatness(np.sin(2 * np.pi * 440 * t).astype(np.float32)) < 0.05

    def test_white_noise_is_flat(self):
        assert S.spectral_flatness(white_noise()) > 0.3

    def test_ordering_holds(self):
        """The absolute thresholds may drift with tuning; the ORDER must not."""
        assert S.spectral_flatness(tone_music()) < S.spectral_flatness(white_noise())


class TestLabeller:
    def test_a_short_chirp_is_not_announced_as_music(self):
        """A notification "ding" must not put "music playing" on screen."""
        lab = S.SoundLabeller()
        assert lab.update(tone_music(0.3), now=0.0) is None
        assert lab.update(tone_music(0.3), now=0.4) is None

    def test_sustained_music_is_announced_once(self):
        lab = S.SoundLabeller()
        assert lab.update(tone_music(), now=0.0) is None      # just started
        first = lab.update(tone_music(), now=2.0)
        assert first is not None and "music" in first
        # Still music a moment later: must NOT re-emit, or the line flickers while
        # being read.
        assert lab.update(tone_music(), now=3.0) is None
        assert lab.update(tone_music(), now=9.0) is None

    def test_silence_clears_the_label(self):
        lab = S.SoundLabeller()
        lab.update(tone_music(), now=0.0)
        assert lab.update(tone_music(), now=2.0) is not None
        assert lab.update(silence(), now=3.0) is None
        # music returns -> announce again, because the screen was cleared
        lab.update(tone_music(), now=4.0)
        assert lab.update(tone_music(), now=6.0) is not None

    def test_speech_resets_it(self):
        """Once someone talks, the overlay belongs to the captions."""
        lab = S.SoundLabeller()
        lab.update(tone_music(), now=0.0)
        assert lab.update(tone_music(), now=2.0) is not None
        lab.speech_started()
        lab.update(tone_music(), now=10.0)
        assert lab.update(tone_music(), now=12.0) is not None, "never re-announced"

    def test_changing_kind_re_announces(self):
        lab = S.SoundLabeller()
        lab.update(white_noise(), now=0.0)
        first = lab.update(white_noise(), now=2.0)
        assert first is not None and "sound" in first
        lab.update(tone_music(), now=3.0)
        second = lab.update(tone_music(), now=5.0)
        assert second is not None and "music" in second

    def test_silence_never_produces_a_label(self):
        lab = S.SoundLabeller()
        for t in range(0, 20, 1):
            assert lab.update(silence(), now=float(t)) is None


def test_labels_are_short_enough_for_the_overlay():
    for text in S.SoundLabeller.LABELS.values():
        assert len(text) < 30, f"too long for a caption pill: {text!r}"


def test_labelling_is_on_by_default():
    from livecaptions.config import Settings
    assert Settings().label_sounds is True


def test_a_pause_in_the_middle_does_not_defeat_the_measurement():
    """Found on real audio: measuring ONE window from the middle of the buffer
    reported flatness 1.000 ("not music") for audio that was obviously music,
    because the middle of a several-second buffer is often a gap between phrases —
    every bin zero, nothing to measure. Several windows fix it."""
    music = tone_music(1.0)
    buf = np.concatenate([music, silence(1.0), music])
    assert S.spectral_flatness(buf) < S.MUSIC_FLATNESS
    assert S.classify(buf) == "music"


def test_one_transient_does_not_decide_a_whole_passage():
    """A door slam inside a musical passage is broadband; the median across windows
    keeps it from flipping the verdict."""
    music = tone_music(1.0)
    bang = white_noise(0.15, amp=0.6)
    buf = np.concatenate([music, bang, music])
    assert S.classify(buf) == "music"
