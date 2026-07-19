"""Saved audio is worth keeping only if it is still there and still openable.

These assert the promises the feature makes to the user: off unless you ask for
it, a real playable WAV of the right length, a cap that stops instead of filling
the disk, a file that survives the app being killed, and deletion that frees the
space it says it does.
"""
import struct
import threading
import time
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
    rec = R.SessionRecorder(tmp_path / "e.wav")     # opened, never written to
    try:
        params, samples = read_wav(tmp_path / "e.wav")   # read while still open
        assert (params.nframes, len(samples)) == (0, 0)
        assert params.framerate == 16000
    finally:
        rec.stop()


def _kill_mid_write(src, dest, frames_in_header=0):
    """A WAV as a process kill leaves it: every frame on disk, header sizes stale.

    `wave` patches the sizes after writing the frames, so the window where a kill
    lands is exactly this — bytes present, sizes describing an earlier moment.
    """
    raw = bytearray(src.read_bytes())
    stale = frames_in_header * R.SAMPLE_WIDTH
    struct.pack_into("<I", raw, 4, R.HEADER_BYTES - 8 + stale)
    struct.pack_into("<I", raw, 40, stale)
    dest.write_bytes(bytes(raw))


def test_audio_killed_mid_write_is_recovered_whole_not_just_readable(tmp_path):
    """The promise is a valid WAV OF THE RIGHT DURATION, not a valid empty one."""
    source = tmp_path / "source.wav"
    with R.SessionRecorder(source) as rec:
        for _ in range(10):
            rec.write(tone(0.2))                    # 2.0 s written and flushed

    killed = R.session_audio_path(31)
    killed.parent.mkdir(parents=True, exist_ok=True)
    _kill_mid_write(source, killed, frames_in_header=0)   # patch never landed at all
    assert read_wav(killed)[0].nframes == 0, "test setup did not actually stale the header"

    found = R.find_session_audio(31)
    params, samples = read_wav(found)
    assert params.nframes == pytest.approx(2.0 * R.SAMPLE_RATE)
    assert params.nframes / params.framerate == pytest.approx(2.0)
    assert len(samples) == params.nframes, "header claims more audio than the file holds"
    assert np.abs(samples).max() > 16000, "recovered silence instead of the audio"


def test_a_partially_stale_header_recovers_only_the_lost_tail(tmp_path):
    source = tmp_path / "source2.wav"
    with R.SessionRecorder(source) as rec:
        rec.write(tone(1.0))

    killed = R.session_audio_path(32)
    killed.parent.mkdir(parents=True, exist_ok=True)
    _kill_mid_write(source, killed, frames_in_header=R.SAMPLE_RATE // 2)   # half patched
    assert R.repair_wav_header(killed) == R.SAMPLE_RATE // 2      # frames recovered
    assert read_wav(killed)[0].nframes == R.SAMPLE_RATE
    assert R.repair_wav_header(killed) == 0, "repaired an already-correct header"


def test_repair_never_grows_a_header_beyond_the_bytes_present(tmp_path):
    """Over-counting is the one failure that makes a WAV unreadable, not just short."""
    path = tmp_path / "torn.wav"
    with R.SessionRecorder(path) as rec:
        rec.write(tone(0.3))
    with open(path, "r+b") as fh:                   # a torn final frame, as a kill leaves it
        fh.truncate(path.stat().st_size + 1 - 2)
        fh.seek(40)
        fh.write(struct.pack("<I", 0))

    R.repair_wav_header(path)
    params, samples = read_wav(path)
    assert len(samples) == params.nframes
    assert params.nframes * R.SAMPLE_WIDTH + R.HEADER_BYTES <= path.stat().st_size


def test_repair_leaves_a_healthy_file_untouched(tmp_path):
    path = tmp_path / "healthy.wav"
    with R.SessionRecorder(path) as rec:
        rec.write(tone(0.4))
    before = path.read_bytes()
    assert R.repair_wav_header(path) == 0
    assert path.read_bytes() == before


def test_repair_ignores_files_it_does_not_understand(tmp_path):
    junk = tmp_path / "notaudio.wav"
    junk.write_bytes(b"this is not a RIFF file at all, not even nearly, no")
    before = junk.read_bytes()
    assert R.repair_wav_header(junk) == 0
    assert junk.read_bytes() == before
    assert R.repair_wav_header(tmp_path / "missing.wav") == 0


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


def test_stereo_is_refused_instead_of_saved_at_double_speed(tmp_path):
    """Interleaved stereo written as mono halves the duration of everything after it."""
    path = tmp_path / "o.wav"
    rec = R.SessionRecorder(path)
    rec.write(tone(0.5))
    stereo = np.stack([tone(0.5), tone(0.5, freq=880.0)], axis=1)
    assert stereo.shape == (0.5 * R.SAMPLE_RATE, 2)

    assert rec.write(stereo) is False               # no exception into the audio thread
    assert rec.stopped and "mono" in rec.stop_reason
    rec.stop()
    params, _ = read_wav(path)
    assert params.nframes == 0.5 * R.SAMPLE_RATE, "stereo frames landed in the file"


def test_mono_shaped_as_a_column_is_accepted(tmp_path):
    """sounddevice hands single-channel blocks back as (N, 1)."""
    path = tmp_path / "p.wav"
    with R.SessionRecorder(path) as rec:
        assert rec.write(tone(0.25).reshape(-1, 1)) is True
    assert read_wav(path)[0].nframes == 0.25 * R.SAMPLE_RATE


def test_an_unconvertible_chunk_stops_recording_without_raising(tmp_path):
    """The audio thread has no handler of its own; conversion runs before any I/O."""
    path = tmp_path / "q.wav"
    rec = R.SessionRecorder(path)
    rec.write(tone(0.2))

    assert rec.write(np.array(["not", "audio"])) is False
    assert rec.stopped and rec.stop_reason
    assert read_wav(path)[0].nframes == 0.2 * R.SAMPLE_RATE, "audio before the bad chunk lost"


def test_stop_does_not_block_behind_a_stalled_write(tmp_path):
    """stop() runs on the GUI thread; a hung disk must not freeze the window."""
    rec = R.SessionRecorder(tmp_path / "r.wav")
    entered, release = threading.Event(), threading.Event()
    real = rec._wav.writeframes

    def stalls(data):
        entered.set()
        release.wait(10)
        return real(data)

    rec._wav.writeframes = stalls
    writer = threading.Thread(target=rec.write, args=(tone(0.3),), daemon=True)
    writer.start()
    assert entered.wait(5), "writer never reached the stalled write"

    began = time.perf_counter()
    rec.stop("user stopped recording")
    elapsed = time.perf_counter() - began

    release.set()
    writer.join(10)
    assert elapsed < 2.0, f"stop() blocked {elapsed:.1f}s behind the stalled write"
    assert rec.stopped and rec.stop_reason == "user stopped recording"
    assert read_wav(tmp_path / "r.wav")[0].nframes == 0.3 * R.SAMPLE_RATE


def test_orphan_audio_is_documented_until_deletion_is_wired_up(tmp_path):
    """Deleting a transcript leaves its voices on disk; that must not go unsaid."""
    doc = R.__doc__.lower()
    assert "orphan" in doc
    assert "delete_session_audio" in doc
    assert "no callers" in doc


def test_totals_and_deletion(tmp_path):
    for sid, secs in ((1, 1.0), (2, 2.0), (3, 0.5)):
        with R.SessionRecorder(R.session_audio_path(sid), session_id=sid) as rec:
            rec.write(tone(secs))

    expected = sum(R.HEADER_BYTES + int(s * R.BYTES_PER_SEC) for s in (1.0, 2.0, 0.5))
    assert R.total_audio_bytes() == expected
    assert len(R.audio_files()) == 3

    freed = R.session_audio_path(2).stat().st_size
    assert R.delete_session_audio(2) == (True, "")
    assert R.find_session_audio(2) is None
    assert R.total_audio_bytes() == expected - freed
    assert R.delete_session_audio(2) == (False, ""), "reported deleting a file that was gone"
    assert R.delete_session_audio(99) == (False, "")


def test_a_failed_deletion_is_returned_not_printed(tmp_path, monkeypatch, capsys):
    """print() goes nowhere in the windowed build, so a locked file would look deleted."""
    with R.SessionRecorder(R.session_audio_path(5)) as rec:
        rec.write(tone(0.2))

    def locked(self, *a, **kw):
        raise PermissionError(13, "The process cannot access the file")
    monkeypatch.setattr(R.Path, "unlink", locked)

    removed, error = R.delete_session_audio(5)
    assert removed is False
    assert "cannot access" in error and "session_5.wav" in error
    assert capsys.readouterr().out == "", "failure reported by print(), invisible in the GUI"


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
