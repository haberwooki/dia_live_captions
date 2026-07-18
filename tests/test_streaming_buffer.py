"""The rolling buffer must stay bounded even when nothing commits.

`stream_max_buffer_sec` used to be decorative: trim_to_committed() sets
`offset = cut_time`, so the next call computes rel == 0, n == 0, and the
`0 < n` guard is structurally dead until a NEW word commits. When decoding falls
behind, nothing commits — so the buffer grew without bound (measured 46/71/100 s
against a 15 s cap) and every pass re-decoded more audio, feeding back into
multi-second frozen captions.

Two distinct stall shapes reach that dead state, and both are pinned here.
"""
import numpy as np
import pytest

from livecaptions.asr.streaming import WHISPER_SR, OnlineASR

CAP = 15.0


class StubModel:
    """Stands in for faster-whisper. `words_fn` decides what each pass 'hears',
    which is how we drive the never-commits and commit-then-stall shapes."""

    def __init__(self, words_fn=None):
        self.words_fn = words_fn or (lambda buf: [])
        self.passes = 0
        self.max_seen_sec = 0.0

    def transcribe(self, audio, **kw):
        self.passes += 1
        self.max_seen_sec = max(self.max_seen_sec, len(audio) / WHISPER_SR)
        return self.words_fn(audio), None


def _seg(words):
    class W:
        def __init__(self, s, e, t):
            self.start, self.end, self.word = s, e, t

    class S:
        def __init__(self, ws):
            self.words = ws

    return [S([W(*w) for w in words])]


def _feed(online, seconds, model, chunk=1.0):
    """Push `seconds` of audio through in chunks, running a pass after each."""
    for _ in range(int(seconds / chunk)):
        online.insert_audio(np.zeros(int(chunk * WHISPER_SR), dtype=np.float32))
        online.process()


def test_buffer_is_bounded_when_nothing_ever_commits():
    """Stall shape 1: the model never returns a stable word, so last_committed_time
    stays 0.0 and trim_to_committed can never fire."""
    model = StubModel(lambda buf: [])
    online = OnlineASR(model, language="en", beam_size=1, max_buffer_sec=CAP)

    _feed(online, 100.0, model)

    assert online.buffer_sec() <= CAP * online.HARD_CAP_FACTOR, (
        f"buffer ran away to {online.buffer_sec():.1f}s")
    # and the decoder was never handed a runaway buffer either
    assert model.max_seen_sec <= CAP * online.HARD_CAP_FACTOR + 1.0
    assert online.dropped_sec > 0


def test_buffer_is_bounded_after_committing_then_stalling():
    """Stall shape 2: words commit for a while (so a trim happens and sets
    offset == last_committed_time), then commits stop. This is the state where the
    `0 < n` guard is dead even though last_committed_time is non-zero."""
    state = {"stall": False}

    def words(buf):
        # One word per second of buffered audio, STABLE across passes: LocalAgreement-2
        # only commits what two consecutive hypotheses agree on, so a stub that returns
        # different words every pass never commits and would silently degenerate into
        # the never-commits case above.
        if state["stall"]:
            return []
        n = int(len(buf) / WHISPER_SR)
        return _seg([(i, i + 0.5, f"w{i}") for i in range(n)])

    model = StubModel(words)
    online = OnlineASR(model, language="en", beam_size=1, max_buffer_sec=CAP)

    _feed(online, 20.0, model)          # commit normally for a while
    committed_offset = online.offset
    assert committed_offset > 0, "precondition: a real commit+trim must have happened"
    assert online.dropped_sec == 0.0, "the healthy phase must not have hit the hard cap"

    state["stall"] = True
    _feed(online, 80.0, model)          # then stop committing entirely

    assert online.offset > committed_offset, "offset must keep advancing during a stall"
    assert online.buffer_sec() <= CAP * online.HARD_CAP_FACTOR, (
        f"buffer ran away to {online.buffer_sec():.1f}s after a commit-then-stall")


def test_healthy_run_is_not_cut():
    """The clamp must not fire on a healthy stream sitting just over the cap —
    cutting there would eat live speech mid-utterance."""
    model = StubModel(lambda buf: [])
    online = OnlineASR(model, language="en", beam_size=1, max_buffer_sec=CAP)
    online.insert_audio(np.zeros(int(15.5 * WHISPER_SR), dtype=np.float32))
    online.process()
    assert online.dropped_sec == 0.0
    assert online.buffer_sec() == pytest.approx(15.5, abs=0.01)


def test_clamp_advances_the_clock_exactly():
    """The live diarizer aligns speaker turns against these timestamps, so dropped
    audio must advance `offset` by exactly the dropped duration."""
    model = StubModel(lambda buf: [])
    online = OnlineASR(model, language="en", beam_size=1, max_buffer_sec=CAP)
    online.insert_audio(np.zeros(int(40.0 * WHISPER_SR), dtype=np.float32))

    before_end = online.offset + online.buffer_sec()
    online._clamp_buffer()
    after_end = online.offset + online.buffer_sec()

    assert after_end == pytest.approx(before_end, abs=1e-6), "clock drifted"
    assert online.offset == pytest.approx(online.dropped_sec, abs=1e-6)


def test_stale_unconfirmed_words_are_dropped():
    """Unconfirmed words older than the surviving audio can never be re-agreed, and
    would block the next LocalAgreement round on a guaranteed mismatch."""
    model = StubModel(lambda buf: [])
    online = OnlineASR(model, language="en", beam_size=1, max_buffer_sec=CAP)
    online.insert_audio(np.zeros(int(40.0 * WHISPER_SR), dtype=np.float32))
    online.hyp.buffer = [(0.5, 1.0, "stale"), (39.0, 39.5, "fresh")]

    online._clamp_buffer()

    kept = [w[2] for w in online.hyp.buffer]
    assert "stale" not in kept and "fresh" in kept


def test_streaming_decode_disables_temperature_fallback():
    """The fallback re-decodes a cut up to 6x on 'bad' audio, freezing the overlay
    for seconds. Streaming must pin temperature; the batch path keeps the default."""
    seen = {}

    class M(StubModel):
        def transcribe(self, audio, **kw):
            seen.update(kw)
            return [], None

    online = OnlineASR(M(), language="en", beam_size=1, max_buffer_sec=CAP)
    online.insert_audio(np.zeros(WHISPER_SR, dtype=np.float32))
    online.process()
    assert seen.get("temperature") == 0.0
