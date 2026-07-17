"""Tests for the transcript store: schema, FTS search, rename, exports."""
import pytest

from livecaptions.store.db import connect, has_fts
from livecaptions.store.export import export
from livecaptions.store.search import rename_speaker, search


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    c.execute("INSERT INTO sessions (started_at, source) VALUES ('2026-07-17T10:00:00','test')")
    rows = [
        (1, 0.0, 2.0, "2026-07-17T10:00:00", "SPEAKER_00", "the quick brown fox jumps"),
        (1, 2.0, 4.0, "2026-07-17T10:00:02", "SPEAKER_01", "diarization results looked good"),
        (1, 4.0, 6.0, "2026-07-17T10:00:04", "SPEAKER_00", "the benchmarks said it falls apart"),
    ]
    c.executemany("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                  " VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    yield c
    c.close()


def test_fts_is_available_and_indexed(conn):
    assert has_fts(conn)
    assert len(search(conn, "diarization")) == 1


def test_search_ranks_and_snippets(conn):
    hits = search(conn, "the")
    assert len(hits) == 2
    assert all("[" in h.snippet for h in hits)      # snippet() marks the match


def test_search_filter_by_speaker(conn):
    assert len(search(conn, "the", speaker="SPEAKER_00")) == 2
    assert search(conn, "the", speaker="SPEAKER_01") == []


def test_phrase_search(conn):
    assert len(search(conn, '"brown fox"')) == 1
    assert search(conn, '"fox brown"') == []


def test_rename_speaker_updates_rows_and_keeps_search(conn):
    n = rename_speaker(conn, "SPEAKER_00", "Sarah")
    assert n == 2
    assert len(search(conn, "the", speaker="Sarah")) == 2
    # external-content FTS: renaming never touched the index, text still matches
    assert len(search(conn, "quick")) == 1


def test_rename_is_reversible(conn):
    rename_speaker(conn, "SPEAKER_00", "Sarah")
    rename_speaker(conn, "Sarah", "SPEAKER_00")
    assert len(search(conn, "the", speaker="SPEAKER_00")) == 2


def test_delete_removes_from_index(conn):
    conn.execute("DELETE FROM utterances WHERE text LIKE '%diarization%'")
    conn.commit()
    assert search(conn, "diarization") == []


def test_exports(conn):
    srt = export(conn, 1, "srt")
    assert "00:00:00,000 --> 00:00:02,000" in srt and "SPEAKER_00: the quick" in srt
    assert export(conn, 1, "vtt").startswith("WEBVTT")
    assert len(export(conn, 1, "jsonl").strip().splitlines()) == 3
    assert "**SPEAKER_00**" in export(conn, 1, "md")


def test_unknown_export_format(conn):
    with pytest.raises(SystemExit):
        export(conn, 1, "docx")
