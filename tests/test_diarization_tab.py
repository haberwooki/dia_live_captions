"""The Speakers tab has two promises worth pinning: it never offers a button that
cannot work, and re-diarizing one session cannot touch another.

The real offline pass needs Whisper, a diarization model and a WAV, so it is
stubbed here. What is exercised for real: the state machine around it, the
session-scoped writeback against a temporary SQLite store, and the rule that no
widget is touched from the worker thread (which would crash Qt, not just look
wrong).
"""
import os
import sqlite3
import threading
import time
import wave
from types import SimpleNamespace

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


def test_finds_audio_where_the_recorder_actually_writes_it(env):
    """With `audio_save_dir` unset — the default — the tab must look in the
    recorder's DB-derived directory, not a directory of its own invention."""
    from livecaptions.capture import recorder
    settings = SimpleNamespace()            # a plain object: no audio_save_dir at all
    wav = _write_wav(recorder.session_audio_path(1))

    assert dmod.audio_dir(settings) == recorder.audio_dir(), \
        "the tab searches a different folder than the recorder writes to"
    assert dmod.session_audio_path(settings, 1) == wav


def test_configured_audio_dir_works_on_a_plain_settings_object(env):
    """Production passes the real Settings, not a test subclass; the configured
    branch has to work through a bare attribute, and has to prefer exactly the
    name the recorder writes."""
    from livecaptions.capture import recorder
    settings = SimpleNamespace(audio_save_dir=str(env["audio"]))
    theirs = _write_wav(env["audio"] / recorder.session_audio_path(5).name)
    _write_wav(env["audio"] / "session-5.wav")     # a lookalike from some other tool

    assert dmod.audio_dir(settings) == env["audio"]
    assert dmod.session_audio_path(settings, 5) == theirs, \
        "picked a lookalike over the file the recorder actually wrote"


# ---- writeback ----

def test_writeback_only_touches_the_chosen_session(env):
    from livecaptions.diarize.base import SpeakerTurn
    conn = db_mod.connect(env["db"])
    changed = dmod.apply_speakers(conn, 1, [
        SpeakerTurn(0.0, 1.1, "SPEAKER_00"),
        SpeakerTurn(1.1, 5.0, "SPEAKER_01"),
    ])
    # 3 lines are attributed, but the first was already SPEAKER_00 and is left as
    # it is: the count is what actually changed, not what was matched.
    assert changed == 2
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


def test_rerunning_the_same_labels_reports_nothing_changed(env):
    """The count is shown to the user as "Relabelled N line(s)". UPDATE's rowcount
    counts rows MATCHED, so re-diarizing to the labels already stored used to
    claim every line had been rewritten."""
    from livecaptions.diarize.base import SpeakerTurn
    turns = [SpeakerTurn(0.0, 1.1, "SPEAKER_00"), SpeakerTurn(1.1, 5.0, "SPEAKER_01")]
    conn = db_mod.connect(env["db"])
    assert dmod.apply_speakers(conn, 1, turns) == 2
    assert dmod.apply_speakers(conn, 1, turns) == 0, \
        "a pass that changed nothing reported relabelled lines"


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
    assert "Relabelled 2 line(s)" in tab._status.text()   # the third line was already right


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
        # The uninterruptible phase (Whisper, then the diarizer), followed by the
        # checkpoint run_offline_pass really does have at line ~159.
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


def test_cancelling_after_the_workers_last_checkpoint_writes_nothing(tab, env, monkeypatch):
    """_on_cancel promises "nothing will be written to the transcript". The worker
    has no checkpoint left after its final one, so a Cancel pressed in the window
    between that check and the writeback arrives with a *successful* payload —
    and the writeback happens on the GUI thread, well after the worker is done."""
    _write_wav(env["audio"] / "session-1.wav")
    tab._load_sessions()
    tab._sessions.setCurrentIndex(_index_of(tab, 1))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))

    past_last_checkpoint = threading.Event()
    released = threading.Event()

    def slow(req, settings, progress, cancelled):
        if cancelled():
            raise dmod.Cancelled()
        past_last_checkpoint.set()
        released.wait(5)          # nothing after this can observe a cancel
        return {"session": req.session_id, "speakers": 1,
                "segments": [SpeakerSegment(0.0, 5.0, "REDIARIZED", "a")]}
    monkeypatch.setattr(dmod, "run_offline_pass", slow)

    tab._on_run()
    assert past_last_checkpoint.wait(5), "the worker never started"
    tab._on_cancel()
    released.set()
    assert _pump(lambda: not tab._running)

    speakers = {r["speaker"] for r in
                tab._ensure().execute("SELECT speaker FROM utterances WHERE session_id=1")}
    assert speakers == {"SPEAKER_00"}, \
        f"transcript WAS rewritten after Cancel: {speakers}"
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


def test_cancel_is_checked_before_any_model_loads(env, monkeypatch):
    """Cancelling before the first checkpoint must not pay for a model load.

    pytest.raises(Cancelled) alone cannot tell "stopped before the load" from
    "stopped at the *next* checkpoint, seconds later, having loaded Whisper from
    cache" — so the load itself is what fails this test."""
    from livecaptions.asr import whisper as whisper_mod
    loads = []

    def must_not_load(*a, **k):
        loads.append(1)
        raise AssertionError("a pre-cancelled run still loaded the speech model")
    monkeypatch.setattr(whisper_mod, "load_model", must_not_load)

    with pytest.raises(dmod.Cancelled):
        dmod.run_offline_pass(
            dmod.DiarizeRequest(1, "nope.wav", "sherpa", -1),
            env["settings"], lambda _m: None, lambda: True)
    assert not loads


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


def test_a_legend_that_cannot_be_read_says_so_instead_of_saying_empty(tab, monkeypatch):
    """"That session has no speaker labels" would send the user off to re-diarize
    a session that is almost certainly labelled fine."""
    def boom(conn, session_id):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(dmod, "speaker_order", boom)

    tab._refresh_legend(1)
    text = tab._legend_empty.text()
    assert "database is locked" in text, f"the read failure was swallowed: {text!r}"
    assert "no speaker labels" not in text, "a DB error was reported as emptiness"


def test_closing_the_tab_closes_its_database_handle(tab):
    conn = tab._ensure()
    tab.close()
    assert tab._conn is None, "the tab kept its sqlite handle after closing"
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
