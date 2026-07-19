"""Saved audio is worth keeping only if it is still there and still openable.

These assert the promises the feature makes to the user: off unless you ask for
it, a real playable WAV of the right length, a cap that stops instead of filling
the disk, a file that survives the app being killed, and deletion that frees the
space it says it does.
"""
import wave

import numpy as np
import pytest

from livecaptions import config
from livecaptions.capture import recorder as R
from livecaptions.store import db as db_mod


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Never touch the real store, config, or the user's saved audio."""
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "store" / "transcripts.db")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    assert str(tmp_path) in str(R.audio_dir()), "audio would be written outside tmp_path"
    conn = db_mod.connect(db_mod.DB_PATH)
    assert str(tmp_path) in conn.execute("PRAGMA database_list").fetchone()["file"]
    conn.close()


def tone(seconds, freq=440.0, rate=R.SAMPLE_RATE):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return w.getparams(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def test_audio_is_off_unless_asked_for():
    """A privacy and disk decision: it must never happen by default."""
    assert getattr(config.Settings(), "save_audio", False) is False
    assert R.SessionRecorder.from_settings(config.Settings(), 1) is None


def test_from_settings_records_when_enabled_and_names_by_session():
    s = config.Settings()
    rec = R.SessionRecorder.from_settings(_with(s, save_audio=True, audio_max_mb=10), 7)
    assert rec is not None
    rec.write(tone(0.2))
    rec.stop()
    assert rec.path.name == "session_7.wav"
    assert R.find_session_audio(7) == rec.path      # a transcript row can find its audio
    assert R.find_session_audio(8) is None


def test_no_session_id_means_no_recording():
    """Without a session row there is nothing to attach the audio to."""
    assert R.SessionRecorder.from_settings(_with(config.Settings(), save_audio=True), None) is None


def test_incremental_chunks_produce_one_valid_wav(tmp_path):
    path = tmp_path / "a.wav"
    rec = R.SessionRecorder(path)
    for _ in range(30):                            # 30 x 0.1 s, as blocks arrive live
        assert rec.write(tone(0.1)) is True
    rec.stop()

    params, samples = read_wav(path)
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 2, 16000)
    assert params.nframes == 3 * R.SAMPLE_RATE
    assert params.nframes / params.framerate == pytest.approx(3.0)
    assert rec.duration_sec == pytest.approx(3.0)
    assert len(samples) == 3 * R.SAMPLE_RATE
    assert samples.max() > 16000, "silence written instead of the audio"


def test_audio_survives_the_round_trip(tmp_path):
    """Re-diarizing later is the whole point, so the samples must come back."""
    path = tmp_path / "b.wav"
    original = tone(0.5)
    with R.SessionRecorder(path) as rec:
        rec.write(original)
    _, samples = read_wav(path)
    back = samples.astype(np.float32) / 32767.0
    assert np.abs(back - original).max() < 1e-3


def test_reported_size_matches_the_file_on_disk(tmp_path):
    path = tmp_path / "c.wav"
    rec = R.SessionRecorder(path)
    rec.write(tone(1.0))
    assert rec.size_bytes == path.stat().st_size    # reported while still recording
    rec.stop()
    assert rec.size_bytes == path.stat().st_size
    assert rec.size_bytes == R.HEADER_BYTES + R.BYTES_PER_SEC


def test_a_killed_app_still_leaves_a_readable_wav(tmp_path):
    """Simulates a kill: take the bytes on disk mid-session, without any close."""
    rec = R.SessionRecorder(tmp_path / "d.wav")
    for _ in range(5):
        rec.write(tone(0.1))
    snapshot = tmp_path / "killed.wav"
    snapshot.write_bytes((tmp_path / "d.wav").read_bytes())   # nothing was flushed by us

    params, samples = read_wav(snapshot)
    assert params.framerate == 16000
    assert params.nframes == 0.5 * R.SAMPLE_RATE, "a completed write was lost"
    assert len(samples) >= params.nframes, "header claims more audio than the file holds"
    rec.stop()


def test_a_kill_before_any_audio_still_leaves_a_valid_wav(tmp_path):
    R.SessionRecorder(tmp_path / "e.wav")           # opened, never written to, never closed
    params, samples = read_wav(tmp_path / "e.wav")
    assert (params.nframes, len(samples)) == (0, 0)
    assert params.framerate == 16000


def test_the_cap_stops_recording_and_says_so(tmp_path):
    path = tmp_path / "f.wav"
    rec = R.SessionRecorder(path, max_mb=0.1)       # ~3.2 s of audio
    accepted = sum(1 for _ in range(20) if rec.write(tone(0.5)))

    assert 0 < accepted < 20, "cap never fired"
    assert rec.stopped and not rec.write(tone(0.5))
    assert rec.size_bytes <= 0.1 * 1024 * 1024, "cap overrun"
    assert path.stat().st_size <= 0.1 * 1024 * 1024
    assert "0.1 MB" in rec.stop_reason and "stopped" in rec.stop_reason.lower()
    assert "Captions continue" in rec.stop_reason


def test_capped_file_is_still_a_valid_recording(tmp_path):
    """Hitting the cap must leave usable audio, not a corpse."""
    path = tmp_path / "g.wav"
    rec = R.SessionRecorder(path, max_mb=0.05)
    for _ in range(10):
        rec.write(tone(0.5))
    rec.stop()
    params, samples = read_wav(path)
    assert params.nframes > 0 and len(samples) == params.nframes


def test_a_chunk_bigger_than_the_whole_cap_is_refused_not_truncated(tmp_path):
    rec = R.SessionRecorder(tmp_path / "h.wav", max_mb=0.001)
    assert rec.write(tone(5.0)) is False
    assert rec.stopped
    assert read_wav(tmp_path / "h.wav")[0].nframes == 0


def test_no_cap_means_no_cap(tmp_path):
    rec = R.SessionRecorder(tmp_path / "i.wav", max_mb=0)
    for _ in range(20):
        assert rec.write(tone(0.5)) is True
    rec.stop()


def test_stopping_twice_is_safe_and_keeps_the_first_reason(tmp_path):
    path = tmp_path / "j.wav"
    rec = R.SessionRecorder(path, max_mb=0.05)
    for _ in range(10):
        rec.write(tone(0.5))
    capped_reason = rec.stop_reason
    rec.stop()
    rec.stop()
    assert rec.stop_reason == capped_reason, "cap reason overwritten by a normal stop"
    assert read_wav(path)[0].nframes > 0


def test_writes_after_stop_are_ignored(tmp_path):
    path = tmp_path / "k.wav"
    rec = R.SessionRecorder(path)
    rec.write(tone(0.5))
    rec.stop()
    assert rec.write(tone(0.5)) is False
    assert read_wav(path)[0].nframes == 0.5 * R.SAMPLE_RATE


def test_a_disk_failure_stops_recording_without_raising(tmp_path, monkeypatch):
    """Audio is the optional half of this app; it must not take captions down."""
    rec = R.SessionRecorder(tmp_path / "l.wav")
    rec.write(tone(0.2))

    def boom(_data):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(rec._wav, "writeframes", boom)

    assert rec.write(tone(0.2)) is False            # no exception reaches the audio thread
    assert rec.stopped and "No space left" in rec.stop_reason
    assert read_wav(tmp_path / "l.wav")[0].nframes > 0, "audio recorded before the failure lost"


def test_an_unwritable_location_is_reported_not_raised(tmp_path):
    (tmp_path / "blocker").write_text("not a directory")
    rec = R.SessionRecorder(tmp_path / "blocker" / "deeper.wav")
    assert rec.stopped and rec.stop_reason
    assert rec.write(tone(0.1)) is False


def test_loud_audio_clips_instead_of_wrapping(tmp_path):
    """A wrap would turn over-driven speech into a full-scale click."""
    path = tmp_path / "m.wav"
    with R.SessionRecorder(path) as rec:
        rec.write(np.full(1000, 1.8, dtype=np.float32))
    _, samples = read_wav(path)
    assert samples.min() > 0 and samples.max() == 32767


def test_int16_input_is_accepted(tmp_path):
    path = tmp_path / "n.wav"
    with R.SessionRecorder(path) as rec:
        rec.write(np.arange(-100, 100, dtype=np.int16))
    assert read_wav(path)[0].nframes == 200


def test_totals_and_deletion(tmp_path):
    for sid, secs in ((1, 1.0), (2, 2.0), (3, 0.5)):
        with R.SessionRecorder(R.session_audio_path(sid), session_id=sid) as rec:
            rec.write(tone(secs))

    expected = sum(R.HEADER_BYTES + int(s * R.BYTES_PER_SEC) for s in (1.0, 2.0, 0.5))
    assert R.total_audio_bytes() == expected
    assert len(R.audio_files()) == 3

    freed = R.session_audio_path(2).stat().st_size
    assert R.delete_session_audio(2) is True
    assert R.find_session_audio(2) is None
    assert R.total_audio_bytes() == expected - freed
    assert R.delete_session_audio(2) is False, "reported deleting a file that was gone"
    assert R.delete_session_audio(99) is False


def test_delete_all_frees_everything(tmp_path):
    for sid in (1, 2):
        with R.SessionRecorder(R.session_audio_path(sid)) as rec:
            rec.write(tone(1.0))
    total = R.total_audio_bytes()
    assert R.delete_all_audio() == (2, total)
    assert R.total_audio_bytes() == 0
    assert R.audio_files() == []


def test_totals_are_zero_before_anything_is_recorded():
    assert R.total_audio_bytes() == 0 and R.audio_files() == []


def test_audio_lives_beside_the_transcripts(tmp_path):
    assert R.audio_dir() == db_mod.DB_PATH.parent / "audio"


def test_size_label_is_human_readable():
    assert R.format_bytes(500 * 1024) == "500 KB"
    assert R.format_bytes(3 * 1024 ** 2) == "3.0 MB"
    assert R.format_bytes(2 * 1024 ** 3) == "2.00 GB"
    assert R.BYTES_PER_HOUR == pytest.approx(115_200_000, rel=0.01)


def _with(settings, **kw):
    return settings.model_copy(update=kw)
