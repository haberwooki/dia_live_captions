"""The Speakers tab has two promises worth pinning: it never offers a button that
cannot work, and re-diarizing one session cannot touch another.

The real offline pass needs Whisper, a diarization model and a WAV, so it is
stubbed here. What is exercised for real: the state machine around it, the
session-scoped writeback against a temporary SQLite store, and the rule that no
widget is touched from the worker thread (which would crash Qt, not just look
wrong).
"""
import os
import threading
import time
import wave

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pydantic_settings import SettingsConfigDict  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from livecaptions import config  # noqa: E402
from livecaptions.diarize.assign import SpeakerSegment  # noqa: E402
from livecaptions.store import db as db_mod  # noqa: E402
from livecaptions.ui import diarization as dmod  # noqa: E402


class Cfg(config.Settings):
    """`audio_save_dir` belongs to the recording feature and may not exist yet;
    allow it so these tests can point the tab at a temp folder."""

    model_config = SettingsConfigDict(env_prefix="LC_", extra="allow")


def _write_wav(path, seconds=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    dbp = tmp_path / "transcripts.db"
    monkeypatch.setattr(db_mod, "DB_PATH", dbp)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(dmod, "save_settings", config.save_settings)

    conn = db_mod.connect(dbp)
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (1,'2026-07-18 10:00','loopback')")
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (2,'2026-07-18 11:00','loopback')")
    rows = [
        (1, 0.0, 1.0, "SPEAKER_00", "who is even talking here"),
        (1, 1.2, 2.0, "SPEAKER_00", "still the same wrong label"),
        (1, 3.0, 4.0, "SPEAKER_00", "and this bit too"),
        (2, 0.0, 1.0, "Dana", "a different day entirely"),
    ]
    for sid, a, b, spk, text in rows:
        conn.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                     " VALUES (?,?,?,?,?,?)", (sid, a, b, "2026-07-18 10:00", spk, text))
    conn.commit()
    conn.close()

    audio = tmp_path / "audio"
    audio.mkdir()
    settings = Cfg(audio_save_dir=str(audio))
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return {"settings": settings, "db": dbp, "audio": audio, "tmp": tmp_path}


@pytest.fixture
def tab(env):
    w = dmod.DiarizationTab(env["settings"])
    got = w._ensure().execute("PRAGMA database_list").fetchone()["file"]
    assert str(env["db"]) in got, f"tab opened the WRONG database: {got}"
    w._load_sessions()
    return w


def _pump(predicate, timeout=5.0):
    """Spin the Qt loop until a queued signal has been delivered."""
    app = QtWidgets.QApplication.instance()
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---- live colours ----

def test_live_colours_persist_and_state_their_cost(tab, env):
    assert not tab._colors.isChecked(), "live diarization must be opt-in"
    tab._colors.setChecked(True)
    assert config.Settings().speaker_colors is True

    blurb = tab.findChildren(QtWidgets.QLabel)
    text = " ".join(lbl.text() for lbl in blurb)
    assert "CPU" in text, "the CPU cost of live diarization is not disclosed"
    assert "second model" in text, "the extra model load is not disclosed"


# ---- offline: no audio means no button ----

def test_no_saved_audio_disables_the_button_and_says_why(tab):
    assert not tab._run.isEnabled()
    msg = tab._audio.text()
    assert "no saved audio" in msg.lower()
    assert "needs the recording" in msg, "doesn't explain why it can't run"


def test_button_enables_once_the_wav_exists(tab, env):
    _write_wav(env["audio"] / "session-1.wav")
    tab._load_sessions()
    tab._sessions.setCurrentIndex(_index_of(tab, 1))
    assert tab._run.isEnabled()
    assert "session-1.wav" in tab._audio.text()


def _index_of(tab, session_id):
    for i in range(tab._sessions.count()):
        if tab._sessions.itemData(i)[0] == session_id:
            return i
    raise AssertionError(f"session {session_id} not offered")


def test_audio_lookup_refuses_a_lookalike_file(env):
    """Re-diarizing the wrong recording would silently corrupt a transcript."""
    _write_wav(env["audio"] / "meeting-2021.wav")
    assert dmod.session_audio_path(env["settings"], 1) is None
    assert dmod.session_audio_path(env["settings"], 2021) is None, \
        "matched a file that doesn't name a session"


def test_audio_lookup_finds_a_session_named_wav(env):
    p = _write_wav(env["audio"] / "session_7_2026-07-18.wav")
    assert dmod.session_audio_path(env["settings"], 7) == p


# ---- writeback ----

def test_writeback_only_touches_the_chosen_session(env):
    from livecaptions.diarize.base import SpeakerTurn
    conn = db_mod.connect(env["db"])
    changed = dmod.apply_speakers(conn, 1, [
        SpeakerTurn(0.0, 1.1, "SPEAKER_00"),
        SpeakerTurn(1.1, 5.0, "SPEAKER_01"),
    ])
    assert changed == 3
    got = {(r["session_id"], r["t_start"]): r["speaker"]
           for r in conn.execute("SELECT session_id,t_start,speaker FROM utterances")}
    assert got[(1, 0.0)] == "SPEAKER_00"
    assert got[(1, 1.2)] == "SPEAKER_01"
    assert got[(1, 3.0)] == "SPEAKER_01"
    assert got[(2, 0.0)] == "Dana", "re-diarizing session 1 relabelled session 2"


def test_writeback_leaves_lines_no_turn_covers(env):
    from livecaptions.diarize.base import SpeakerTurn
    conn = db_mod.connect(env["db"])
    dmod.apply_speakers(conn, 1, [SpeakerTurn(0.0, 1.0, "SPEAKER_00")])
    # best_speaker snaps to the nearest turn, so every line is attributed; what
    # must not happen is a line being blanked out.
    speakers = [r["speaker"] for r in
                conn.execute("SELECT speaker FROM utterances WHERE session_id=1")]
    assert all(speakers), "a line lost its speaker entirely"


# ---- the worker ----

def _stub_pass(recorder, segments):
    def fake(req, settings, progress, cancelled):
        recorder["thread"] = threading.current_thread()
        recorder["req"] = req
        progress("working…")
        return {"session": req.session_id, "segments": segments, "speakers": 2}
    return fake


def test_full_run_relabels_the_session_from_the_worker(tab, env, monkeypatch):
    _write_wav(env["audio"] / "session-1.wav")
    tab._load_sessions()
    tab._sessions.setCurrentIndex(_index_of(tab, 1))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    rec = {}
    monkeypatch.setattr(dmod, "run_offline_pass", _stub_pass(rec, [
        SpeakerSegment(0.0, 1.1, "SPEAKER_00", "a"),
        SpeakerSegment(1.1, 5.0, "SPEAKER_01", "b"),
    ]))

    tab._on_run()
    assert _pump(lambda: not tab._running), "the run never finished"

    assert rec["thread"] is not threading.main_thread(), "the slow pass blocked the GUI thread"
    assert rec["req"].session_id == 1 and rec["req"].wav_path.endswith("session-1.wav")
    conn = tab._ensure()
    assert [r["speaker"] for r in
            conn.execute("SELECT speaker FROM utterances WHERE session_id=1 ORDER BY t_start")] \
        == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    assert [r["speaker"] for r in
            conn.execute("SELECT speaker FROM utterances WHERE session_id=2")] == ["Dana"]
    assert "Relabelled 3 line(s)" in tab._status.text()


def test_no_widget_is_touched_from_the_worker_thread(tab, env, monkeypatch):
    """Reading or writing a widget off the GUI thread crashes Qt, so progress has
    to travel by signal."""
    _write_wav(env["audio"] / "session-1.wav")
    tab._load_sessions()
    tab._sessions.setCurrentIndex(_index_of(tab, 1))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))

    threads = []
    original = QtWidgets.QLabel.setText
    monkeypatch.setattr(QtWidgets.QLabel, "setText",
                        lambda self, text: (threads.append(threading.current_thread()),
                                            original(self, text))[1])
    rec = {}
    monkeypatch.setattr(dmod, "run_offline_pass", _stub_pass(rec, [
        SpeakerSegment(0.0, 5.0, "SPEAKER_00", "a")]))

    tab._on_run()
    assert _pump(lambda: not tab._running)

    assert threads, "the run never updated the UI at all"
    assert all(t is threading.main_thread() for t in threads), \
        "a label was written from the worker thread"


def test_cancelling_writes_nothing(tab, env, monkeypatch):
    _write_wav(env["audio"] / "session-1.wav")
    tab._load_sessions()
    tab._sessions.setCurrentIndex(_index_of(tab, 1))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))

    released = threading.Event()

    def slow(req, settings, progress, cancelled):
        released.wait(5)
        if cancelled():
            raise dmod.Cancelled()
        return {"session": req.session_id, "speakers": 1,
                "segments": [SpeakerSegment(0.0, 5.0, "SPEAKER_09", "a")]}
    monkeypatch.setattr(dmod, "run_offline_pass", slow)

    tab._on_run()
    assert tab._running and not tab._run.isEnabled(), "a second run could be started"
    tab._on_cancel()
    released.set()
    assert _pump(lambda: not tab._running)

    speakers = {r["speaker"] for r in
                tab._ensure().execute("SELECT speaker FROM utterances WHERE session_id=1")}
    assert speakers == {"SPEAKER_00"}, "a cancelled pass still rewrote the transcript"
    assert "unchanged" in tab._status.text()


def test_declining_the_confirmation_starts_nothing(tab, env, monkeypatch):
    _write_wav(env["audio"] / "session-1.wav")
    tab._load_sessions()
    tab._sessions.setCurrentIndex(_index_of(tab, 1))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No))
    started = []
    monkeypatch.setattr(dmod, "run_offline_pass",
                        lambda *a, **k: started.append(1) or {})
    tab._on_run()
    assert not started and not tab._running
    assert "Nothing was changed" in tab._status.text()


def test_a_failing_pass_reports_instead_of_hanging(tab, env, monkeypatch):
    _write_wav(env["audio"] / "session-1.wav")
    tab._load_sessions()
    tab._sessions.setCurrentIndex(_index_of(tab, 1))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))

    def boom(*a, **k):
        raise RuntimeError("no diarization model")
    monkeypatch.setattr(dmod, "run_offline_pass", boom)

    tab._on_run()
    assert _pump(lambda: not tab._running)
    assert "no diarization model" in tab._status.text()
    assert tab._run.isEnabled(), "the tab stayed locked after a failure"


def test_cancel_is_checked_before_any_model_loads(env):
    """Cancelling before the first checkpoint must not pay for a model load."""
    with pytest.raises(dmod.Cancelled):
        dmod.run_offline_pass(
            dmod.DiarizeRequest(1, "nope.wav", "sherpa", -1),
            env["settings"], lambda _m: None, lambda: True)


# ---- settings + legend ----

def test_backend_and_speaker_count_persist(tab):
    tab._backend.setCurrentIndex(2)
    assert config.Settings().diarizer == "sherpa"
    tab._num.setValue(3)
    assert config.Settings().diarize_num_speakers == 3


def test_legend_maps_a_colour_per_speaker_in_first_heard_order(env, tab):
    conn = db_mod.connect(env["db"])
    conn.execute("UPDATE utterances SET speaker='SPEAKER_01' WHERE session_id=1 AND t_start=0.0")
    conn.commit()
    conn.close()
    assert dmod.speaker_order(tab._ensure(), 1) == ["SPEAKER_01", "SPEAKER_00"]

    tab._refresh_legend(1)
    from livecaptions.ui.overlay import SPEAKER_COLORS
    swatches = [w for w in tab.findChildren(QtWidgets.QLabel)
                if SPEAKER_COLORS[0].name() in w.styleSheet()
                or SPEAKER_COLORS[1].name() in w.styleSheet()]
    assert len(swatches) == 2, "each speaker needs its own colour swatch"
    names = [w.text() for w in tab.findChildren(QtWidgets.QLabel)]
    assert "SPEAKER_01" in names and "SPEAKER_00" in names
