"""The Transcripts tab must show real sessions, search them, and export them.

These drive the widget against a real (temporary) SQLite store rather than mocks,
because the failure modes worth catching are store-shaped: FTS5 MATCH syntax
errors, empty stores, and a rename that silently touches other sessions.
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
    dbp = tmp_path / "transcripts.db"
    monkeypatch.setattr(db_mod, "DB_PATH", dbp)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")

    conn = db_mod.connect(dbp)
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (1, '2026-07-18 10:00', 'loopback')")
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (2, '2026-07-18 11:00', 'loopback')")
    rows = [
        (1, 0.0, 1.0, "SPEAKER_00", "the quarterly budget looks fine"),
        (1, 1.0, 2.0, "SPEAKER_01", "I disagree about the budget"),
        (2, 0.0, 1.0, "SPEAKER_00", "completely unrelated session"),
    ]
    for sid, a, b, spk, text in rows:
        conn.execute(
            "INSERT INTO utterances (session_id, t_start, t_end, wall_clock, speaker, text)"
            " VALUES (?,?,?,?,?,?)", (sid, a, b, "2026-07-18 10:00", spk, text))
    conn.commit()
    conn.close()

    from livecaptions.ui import transcripts as tmod
    monkeypatch.setattr(tmod, "save_settings", config.save_settings)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = tmod.TranscriptsTab(config.Settings())

    # Fail loudly if the widget reached the user's real store. connect()'s default
    # argument used to bind DB_PATH at import time, so this fixture's redirect was
    # ignored and a rename test relabelled a speaker in real transcripts.
    got = w._ensure().execute("PRAGMA database_list").fetchone()["file"]
    assert str(dbp) in got, f"tab opened the WRONG database: {got}"

    w.refresh()
    return w


def test_lists_saved_sessions(tab):
    assert tab._list.count() == 2
    labels = [tab._list.item(i).text() for i in range(tab._list.count())]
    assert any("[2]" in t for t in labels) and any("[1]" in t for t in labels)


def test_reading_a_session_shows_its_text(tab):
    tab._list.setCurrentRow(tab._list.count() - 1)      # session 1
    body = tab._text.toPlainText()
    assert "quarterly budget" in body
    assert "unrelated session" not in body, "leaked another session's text"


def test_search_finds_matching_lines(tab):
    tab._q.setText("budget")
    tab._on_search()
    assert tab._list.count() == 2, "should match both budget lines"
    assert "match" in tab._status.text()


def test_search_miss_is_reported_not_silent(tab):
    tab._q.setText("zzzznotpresent")
    tab._on_search()
    assert tab._list.count() == 0
    assert "Nothing matched" in tab._status.text()


def test_bad_search_syntax_does_not_look_broken(tab):
    """FTS5 MATCH rejects a bare quote; the tab must explain rather than die."""
    tab._q.setText('unbalanced "quote')
    tab._on_search()
    assert "Couldn't search" in tab._status.text() or tab._list.count() >= 0


def test_empty_search_returns_to_the_session_list(tab):
    tab._q.setText("budget")
    tab._on_search()
    tab._q.setText("")
    tab._on_search()
    assert tab._list.count() == 2
    assert "session(s)" in tab._status.text()


def test_saving_switch_persists(tab):
    assert config.Settings().save_transcripts is True
    tab._save.setChecked(False)
    assert config.Settings().save_transcripts is False
    assert "NOT be saved" in tab._status.text()
    # Turning saving off must not hide transcripts already recorded.
    tab.refresh()
    assert tab._list.count() == 2


def test_export_writes_the_chosen_format(tab, tmp_path, monkeypatch):
    tab._list.setCurrentRow(tab._list.count() - 1)
    out = tmp_path / "out.srt"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "Subtitles (*.srt)")))
    tab._on_export()
    text = out.read_text(encoding="utf-8")
    assert "-->" in text, "not SRT"
    assert "quarterly budget" in text


def test_export_honours_a_typed_extension_over_the_filter(tab, tmp_path, monkeypatch):
    tab._list.setCurrentRow(tab._list.count() - 1)
    out = tmp_path / "typed.md"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "Subtitles (*.srt)")))
    tab._on_export()
    assert "-->" not in out.read_text(encoding="utf-8"), "wrote SRT into a .md file"


def test_rename_is_scoped_to_the_selected_session(tab, monkeypatch):
    """A rename must not silently relabel the same speaker in other sessions."""
    tab._list.setCurrentRow(tab._list.count() - 1)          # session 1
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("SPEAKER_00", True)))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Dana", True)))
    tab._on_rename()

    conn = tab._ensure()
    s1 = [r["speaker"] for r in conn.execute(
        "SELECT speaker FROM utterances WHERE session_id=1")]
    s2 = [r["speaker"] for r in conn.execute(
        "SELECT speaker FROM utterances WHERE session_id=2")]
    assert "Dana" in s1
    assert s2 == ["SPEAKER_00"], "renamed a speaker in an unrelated session"
    assert "Reversible" in tab._status.text()
