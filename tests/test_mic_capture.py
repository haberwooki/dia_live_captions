"""Your own voice is captioned, and attributed by MEASUREMENT rather than guesswork.

WASAPI loopback carries only what Windows plays, so in a call it has everyone except
you. Mixing the microphone in fixes that, and gives something diarization cannot:
we know which DEVICE the sound arrived on, so a line that was loud on the mic is
yours with certainty.

The risk worth testing is the opposite error. On speakers the microphone also hears
the far end, and labelling THEIR words as yours is much worse than missing a "You" —
a missed label just falls back to normal captioning.
"""
import numpy as np
import pytest

from livecaptions.capture import mixed as M
from livecaptions.events import AudioBlock


class FakeAudio:
    is_live, name, rate = True, "fake", 16000

    def start(self, on_block, on_end):
        pass

    def stop(self):
        pass


@pytest.fixture
def source():
    """A streaming source with no model, VAD or diarizer — only the attribution."""
    from livecaptions.config import Settings
    from livecaptions.sources.streaming_local import StreamingTranscriptionSource
    src = StreamingTranscriptionSource.__new__(StreamingTranscriptionSource)
    from collections import deque
    src._levels = deque(maxlen=4000)
    src._self_label = Settings().mic_label
    src._diarizer = None
    return src


def _levels(src, rows):
    for t, system, mic in rows:
        src._levels.append((t, system, mic))


class TestAttribution:
    def test_loud_on_the_mic_is_you(self, source):
        _levels(source, [(1.0, 0.001, 0.08), (1.5, 0.001, 0.09)])
        assert source._spoken_by_me(0.9, 1.6) is True

    def test_loud_on_the_system_is_not_you(self, source):
        """The far end, coming out of your headphones."""
        _levels(source, [(1.0, 0.09, 0.001), (1.5, 0.08, 0.0)])
        assert source._spoken_by_me(0.9, 1.6) is False

    def test_speaker_bleed_is_not_attributed_to_you(self, source):
        """On speakers the mic hears the far end too. Comparable levels must NOT be
        called yours — mislabelling their words is the worse failure."""
        _levels(source, [(1.0, 0.06, 0.05), (1.5, 0.07, 0.045)])
        assert source._spoken_by_me(0.9, 1.6) is False

    def test_room_noise_alone_is_not_speech(self, source):
        """A quiet room still has a nonzero mic level; with silence on both sides the
        ratio can look decisive while nothing was said."""
        _levels(source, [(1.0, 0.0000001, 0.0005)])
        assert source._spoken_by_me(0.9, 1.1) is False

    def test_only_the_words_time_range_is_considered(self, source):
        """Speaking earlier in the session must not make a later line yours."""
        _levels(source, [(1.0, 0.001, 0.09),      # you, at t=1
                         (9.0, 0.09, 0.001)])     # them, at t=9
        assert source._spoken_by_me(8.5, 9.5) is False
        assert source._spoken_by_me(0.5, 1.5) is True

    def test_no_microphone_means_never_you(self, source):
        """With system audio only there are no levels at all, and nothing should be
        attributed — this is the default configuration."""
        assert source._spoken_by_me(0.0, 5.0) is False


class TestMixing:
    def test_mixing_averages_instead_of_clipping(self):
        """Summing two loud sources clips, and clipping hurts recognition far more
        than being 6 dB quieter."""
        system = np.full(100, 0.9, dtype=np.float32)
        mic = np.full(100, 0.9, dtype=np.float32)
        mixed = (system + mic) * 0.5
        assert float(np.max(np.abs(mixed))) <= 1.0

    def test_short_mic_audio_is_padded_not_stalled(self):
        """The system stream is the clock. If the mic is behind, pad with silence —
        blocking would stall the capture callback and drop system audio."""
        src = M.MixedSource.__new__(M.MixedSource)
        import queue
        src._mic_q = queue.Queue()
        src._mic_buf = np.zeros(0, dtype=np.float32)
        src.rate = 16000
        out = src._take_mic(1600)
        assert out.shape == (1600,)
        assert float(np.max(np.abs(out))) == 0.0

    def test_a_mic_backlog_is_trimmed_to_the_recent_past(self):
        """Stale mic audio mixed against the wrong moment would mis-attribute a
        line, so surplus is dropped from the OLDEST end."""
        src = M.MixedSource.__new__(M.MixedSource)
        import queue
        src._mic_q = queue.Queue()
        src.rate = 16000
        # 10s of backlog, with a marker at the very end
        src._mic_buf = np.zeros(10 * 16000, dtype=np.float32)
        src._mic_buf[-1] = 0.5
        out = src._take_mic(1600)
        assert src._mic_buf.size + out.size <= M._MAX_MIC_BACKLOG_SEC * 16000 + 1600
        kept = np.concatenate([out, src._mic_buf])
        assert float(np.max(np.abs(kept))) == 0.5, "dropped the newest audio instead"


def test_mono_downmix_handles_stereo():
    raw = np.array([1000, 3000, -1000, -3000], dtype=np.int16).tobytes()
    mono = M._to_mono(raw, 2)
    assert mono.shape == (2,)
    assert mono[0] == pytest.approx(2000 / 32768.0, abs=1e-6)


def test_capture_mic_is_off_by_default():
    """Opening a microphone must be a deliberate choice, never a default."""
    from livecaptions.config import Settings
    assert Settings().capture_mic is False
