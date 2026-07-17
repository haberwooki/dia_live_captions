"""Tests for LLM speaker naming — the pure parts. No network calls."""
import sqlite3

import pytest

pytest.importorskip("pydantic")

from livecaptions.store.db import init_schema  # noqa: E402
from livecaptions.store.naming import (  # noqa: E402
    SpeakerName,
    SpeakerNames,
    build_transcript,
    propose_names,
    session_labels,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_schema(c)
    c.execute("INSERT INTO sessions (started_at, source) VALUES ('2026-07-15T10:00:00', 'test')")
    rows = [
        ("SPEAKER_00", "Hi everyone, this is Mike."),
        ("SPEAKER_01", "Thanks Mike. Sarah here, I'll take the update."),
        (None, "Some line with no speaker label."),
    ]
    for i, (spk, text) in enumerate(rows):
        c.execute(
            "INSERT INTO utterances (session_id, t_start, t_end, wall_clock, speaker, text)"
            " VALUES (1, ?, ?, '2026-07-15T10:00:00', ?, ?)",
            (float(i), float(i) + 1.0, spk, text),
        )
    c.commit()
    return c


def test_session_labels_lists_distinct_labels_and_skips_null(conn):
    assert session_labels(conn, 1) == ["SPEAKER_00", "SPEAKER_01"]


def test_session_labels_empty_for_unknown_session(conn):
    assert session_labels(conn, 999) == []


def test_build_transcript_renders_labelled_lines(conn):
    text, truncated = build_transcript(conn, 1)
    assert not truncated
    assert text.splitlines()[0] == "SPEAKER_00: Hi everyone, this is Mike."
    assert "SPEAKER_01: Thanks Mike." in text


def test_build_transcript_labels_unattributed_lines(conn):
    text, _ = build_transcript(conn, 1)
    assert "UNKNOWN: Some line with no speaker label." in text


def test_build_transcript_truncates_and_reports_it(conn):
    text, truncated = build_transcript(conn, 1, max_chars=40)
    assert truncated
    assert len(text) <= 40


def test_build_transcript_keeps_the_head_when_truncating(conn):
    # Names are established early, so truncation must drop the tail, not the head.
    text, _ = build_transcript(conn, 1, max_chars=40)
    assert text.startswith("SPEAKER_00: Hi everyone")


def test_name_is_optional(conn):
    s = SpeakerName(label="SPEAKER_00", name=None, confidence="low", evidence="no mention")
    assert s.name is None


# --- propose_names reconciliation (fake client; no network) -------------------

def _fake_anthropic(monkeypatch, returned, stop_reason="end_turn"):
    """Stub anthropic.Anthropic so propose_names runs without a network call."""
    anthropic = pytest.importorskip("anthropic")

    class FakeMessages:
        def parse(self, **kwargs):
            self.kwargs = kwargs
            return type("R", (), {
                "stop_reason": stop_reason,
                "parsed_output": SpeakerNames(speakers=returned),
            })()

    class FakeClient:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)


def test_fills_in_labels_the_model_omitted(monkeypatch):
    _fake_anthropic(monkeypatch, [
        SpeakerName(label="SPEAKER_00", name="Mike", confidence="high", evidence="this is Mike"),
    ])
    out = propose_names("SPEAKER_00: hi", ["SPEAKER_00", "SPEAKER_01"])
    assert [s.label for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[1].name is None  # omitted -> unnamed, not dropped


def test_drops_labels_the_model_invented(monkeypatch):
    _fake_anthropic(monkeypatch, [
        SpeakerName(label="SPEAKER_00", name="Mike", confidence="high", evidence="this is Mike"),
        SpeakerName(label="SPEAKER_99", name="Ghost", confidence="high", evidence="fabricated"),
    ])
    out = propose_names("SPEAKER_00: hi", ["SPEAKER_00"])
    assert [s.label for s in out] == ["SPEAKER_00"]


def test_refusal_raises_rather_than_returning_names(monkeypatch):
    _fake_anthropic(monkeypatch, [], stop_reason="refusal")
    with pytest.raises(RuntimeError):
        propose_names("SPEAKER_00: hi", ["SPEAKER_00"])
