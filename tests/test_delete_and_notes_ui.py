"""Deleting a session must remove everything it left behind, and only that.

Until now nothing ever deleted a session's saved audio, so raw recordings of
private conversations accumulated with no way to remove them from inside the app.
The tests that matter here are the ones that prove deletion is COMPLETE (transcript,
search index, notes, audio) and BOUNDED (the neighbouring session is untouched).
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from livecaptions import config  # noqa: E402
from livecaptions.store import db as db_mod  # noqa: E402


@pytest.fixture
def tab(tmp_path, monkeypatch):
    dbp = tmp_path / "t.db"
    audio = tmp_path / "audio"
    audio.mkdir()
    monkeypatch.setattr(db_mod, "DB_PATH", dbp)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")

    conn = db_mod.connect(dbp)
    for sid in (1, 2):
        conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (?,?,?)",
                     (sid, f"2026-07-18 1{sid}:00", "loopback"))
        conn.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                     " VALUES (?,?,?,?,?,?)",
                     (sid, 0, 1, "2026-07-18 10:00", "SPEAKER_00",
                      f"unique marker for session {sid}"))
    conn.commit(); conn.close()

    # real WAVs for both sessions, so we can prove only one is removed
    import wave
    for sid in (1, 2):
        with wave.open(str(audio / f"session_{sid}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)

    from livecaptions.ui import transcripts as tmod
    monkeypatch.setattr(tmod, "save_settings", config.save_settings)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = config.Settings(audio_save_dir=str(audio))
    w = tmod.TranscriptsTab(settings)
    assert str(dbp) in w._ensure().execute("PRAGMA database_list").fetchone()["file"]
    w.refresh()
    w._audio_dir = audio
    return w


def _select(tab, session_id):
    for i in range(tab._list.count()):
        if (tab._list.item(i).data(0x0100) or {}).get("session") == session_id:
            tab._list.setCurrentRow(i)
            return
    raise AssertionError(f"session {session_id} not listed")


def _yes(monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))


def test_delete_removes_transcript_notes_index_and_audio(tab, monkeypatch):
    from livecaptions.notes import SessionNotes, StoredNotes, save_notes
    conn = tab._ensure()
    save_notes(conn, StoredNotes(session_id=1, summary="s", key_points=[], todos=[],
                                 decisions=[], generated_at="now", model_label="m",
                                 truncated=False))
    _select(tab, 1)
    _yes(monkeypatch)
    tab._on_delete()

    assert conn.execute("SELECT COUNT(*) FROM utterances WHERE session_id=1").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sessions WHERE id=1").fetchone()[0] == 0
    from livecaptions.notes import load_notes
    assert load_notes(conn, 1) is None
    assert not (tab._audio_dir / "session_1.wav").exists(), "audio left on disk"

    # the neighbour is entirely untouched
    assert conn.execute("SELECT COUNT(*) FROM utterances WHERE session_id=2").fetchone()[0] == 1
    assert (tab._audio_dir / "session_2.wav").exists()


def test_deleted_session_stops_showing_up_in_search(tab, monkeypatch):
    """A deleted transcript that still answers searches would be the worst kind of
    'deleted' — the full-text index is maintained by triggers, so this pins it."""
    from livecaptions.store.search import search
    conn = tab._ensure()
    assert search(conn, "marker", limit=10)
    _select(tab, 1)
    _yes(monkeypatch)
    tab._on_delete()
    hits = search(conn, "marker", limit=10)
    assert all(h.session_id != 1 for h in hits), "deleted session still searchable"


def test_the_dialog_lists_what_will_be_destroyed(tab, monkeypatch):
    shown = {}
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda p, t, text, *a, **k: shown.update(text=text)
                                     or QtWidgets.QMessageBox.StandardButton.No))
    _select(tab, 1)
    tab._on_delete()
    assert "transcript line" in shown["text"]
    assert "audio" in shown["text"], "did not warn that the recording goes too"
    assert "cannot be undone" in shown["text"]


def test_declining_deletes_nothing(tab, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No))
    _select(tab, 1)
    tab._on_delete()
    conn = tab._ensure()
    assert conn.execute("SELECT COUNT(*) FROM utterances WHERE session_id=1").fetchone()[0] == 1
    assert (tab._audio_dir / "session_1.wav").exists()
    assert "Nothing was deleted" in tab._status.text()


def test_a_session_without_audio_deletes_cleanly(tab, monkeypatch):
    (tab._audio_dir / "session_1.wav").unlink()
    _select(tab, 1)
    _yes(monkeypatch)
    tab._on_delete()
    assert tab._ensure().execute(
        "SELECT COUNT(*) FROM sessions WHERE id=1").fetchone()[0] == 0


def test_undeletable_audio_is_reported_not_hidden(tab, monkeypatch):
    """print() goes nowhere in the windowed build, so a failure must reach the UI."""
    from livecaptions.capture import recorder as rec
    monkeypatch.setattr(rec, "delete_session_audio",
                        lambda *a, **k: (False, "file is open in another program"))
    _select(tab, 1)
    _yes(monkeypatch)
    tab._on_delete()
    assert "NOT deleted" in tab._status.text()
    assert "another program" in tab._status.text()


# ---- Notes UI ----

@pytest.fixture
def ai_tab(tmp_path, monkeypatch):
    dbp = tmp_path / "t.db"
    monkeypatch.setattr(db_mod, "DB_PATH", dbp)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    conn = db_mod.connect(dbp)
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (1,'2026-07-18 10:00','x')")
    conn.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                 " VALUES (1,0,1,'2026-07-18 10:00','SPEAKER_00','I will send the report')")
    conn.commit(); conn.close()

    from livecaptions.ui import ai as ai_mod
    monkeypatch.setattr(ai_mod, "save_settings", config.save_settings)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = ai_mod.AITab(config.Settings(llm_provider="local", llm_model="llama3.1",
                                     llm_base_url="http://localhost:11434/v1"))
    w._load_sessions()
    return w


def test_notes_require_consent_and_send_nothing_when_declined(ai_tab, monkeypatch):
    from livecaptions.llm import providers as P
    seen, sent = {}, []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda p, t, text, *a, **k: seen.update(text=text)
                                     or QtWidgets.QMessageBox.StandardButton.No))
    monkeypatch.setattr(P, "from_settings", lambda *a, **k: sent.append(1))
    ai_tab._on_notes()
    assert "characters" in seen.get("text", ""), "did not say how much is sent"
    assert not sent, "contacted the model despite the user declining"
    assert "Nothing was sent" in ai_tab._notes_status.text()


def test_existing_notes_are_shown_without_asking_the_model_again(ai_tab):
    from livecaptions.notes import StoredNotes, ToDo, save_notes
    save_notes(ai_tab._ensure(), StoredNotes(
        session_id=1, summary="We agreed to ship on Friday.",
        key_points=["timeline"], decisions=["ship Friday"],
        todos=[ToDo(text="send the report", owner="SPEAKER_00",
                    evidence="I will send the report")],
        generated_at="2026-07-18", model_label="llama3.1", truncated=False))
    ai_tab._show_existing_notes()
    body = ai_tab._notes_view.toPlainText()
    assert "ship on Friday" in body and "send the report" in body
    assert ai_tab._notes_btn.text().startswith("Regenerate")
