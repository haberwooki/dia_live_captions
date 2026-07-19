"""Session notes — the promises, with a fake provider so nothing goes near a network.

What is pinned here: consent is required before any transcript is read, notes
survive a restart, regenerating replaces rather than duplicates, a to-do with no
clear owner stays unowned, and truncation is visible instead of silent.
"""
import json

import pytest

pytest.importorskip("pydantic")

from livecaptions import config  # noqa: E402
from livecaptions.llm.providers import LLMError  # noqa: E402
from livecaptions.notes.generate import (  # noqa: E402
    ConsentRequired,
    SessionNotes,
    StoredNotes,
    ToDo,
    generate_notes,
    load_notes,
    privacy_notice,
    to_markdown,
)
from livecaptions.store import db as db_mod  # noqa: E402

_REPLY = SessionNotes(
    summary="Dana agreed to send the Q3 draft; the launch date stayed open.",
    key_points=["The Q3 draft is with Dana", "Launch date not settled"],
    todos=[
        ToDo(text="Send the Q3 draft to legal", owner="Dana",
             evidence="I'll get the Q3 draft over to legal on Friday"),
        ToDo(text="Book the launch review", owner=None,
             evidence="somebody should book the review"),
    ],
    decisions=["Ship behind a flag first"],
)


class FakeProvider:
    """Stands in for llm.providers.Provider: records what it was asked, returns
    a canned SessionNotes (or raises)."""

    label = "fake-model @ localhost"

    def __init__(self, reply=_REPLY, error=None):
        self._reply, self._error = reply, error
        self.calls = []

    def complete(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self._error:
            raise self._error
        return self._reply


@pytest.fixture
def conn(tmp_path, monkeypatch):
    """A throwaway store. The real transcripts DB is never opened: DB_PATH and the
    config paths are redirected first, and the redirect is asserted below."""
    tmp_db = tmp_path / "notes-test.db"
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_db)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")

    c = db_mod.connect(tmp_db)
    assert str(tmp_db) in c.execute("PRAGMA database_list").fetchone()["file"]

    c.execute("INSERT INTO sessions (id, started_at, source) VALUES (1,'2026-07-18T10:00:00','x')")
    for i, (spk, text) in enumerate([
        ("SPEAKER_00", "Right, where are we on the Q3 draft?"),
        ("Dana", "I'll get the Q3 draft over to legal on Friday."),
        ("SPEAKER_00", "Good. And somebody should book the review."),
    ]):
        c.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                  " VALUES (1,?,?,'2026-07-18T10:00:00',?,?)",
                  (float(i), float(i) + 1.0, spk, text))
    c.commit()
    yield c
    c.close()


def test_refuses_without_consent(conn):
    prov = FakeProvider()
    with pytest.raises(ConsentRequired) as e:
        generate_notes(conn, 1, prov)
    assert not prov.calls, "transcript was sent despite no consent"
    assert "agreement" in str(e.value).lower()


def test_refuses_when_consent_is_merely_truthy(conn):
    """A GUI passing a Qt button code or a non-empty string must not read as yes."""
    prov = FakeProvider()
    for sneaky in ("yes", 1, [1]):
        with pytest.raises(ConsentRequired):
            generate_notes(conn, 1, prov, consented=sneaky)
    assert not prov.calls


def test_declined_run_stores_nothing(conn):
    with pytest.raises(ConsentRequired):
        generate_notes(conn, 1, FakeProvider())
    assert load_notes(conn, 1) is None


def test_generates_and_stores_notes(conn):
    notes = generate_notes(conn, 1, FakeProvider(), consented=True)
    assert notes.summary.startswith("Dana agreed")
    assert [t.text for t in notes.todos] == ["Send the Q3 draft to legal",
                                             "Book the launch review"]
    assert notes.model_label == "fake-model @ localhost"
    assert notes.generated_at


def test_notes_survive_a_reopen(conn, tmp_path):
    generate_notes(conn, 1, FakeProvider(), consented=True)
    conn.close()

    reopened = db_mod.connect(tmp_path / "notes-test.db")
    loaded = load_notes(reopened, 1)
    assert isinstance(loaded, StoredNotes)
    assert loaded.summary == _REPLY.summary
    assert loaded.decisions == ["Ship behind a flag first"]
    assert loaded.todos[0].owner == "Dana"
    assert loaded.todos[0].evidence == "I'll get the Q3 draft over to legal on Friday"
    reopened.close()


def test_unclear_owner_stays_unowned(conn):
    """The model is told null beats a guess — that must survive the round trip."""
    generate_notes(conn, 1, FakeProvider(), consented=True)
    assert load_notes(conn, 1).todos[1].owner is None


def test_regenerating_replaces_rather_than_accumulates(conn):
    generate_notes(conn, 1, FakeProvider(), consented=True)
    second = SessionNotes(summary="A different take.", key_points=[], todos=[], decisions=[])
    generate_notes(conn, 1, FakeProvider(reply=second), consented=True)

    assert load_notes(conn, 1).summary == "A different take."
    rows = conn.execute("SELECT COUNT(*) c FROM session_notes WHERE session_id=1").fetchone()
    assert rows["c"] == 1


def test_no_notes_for_a_session_without_any(conn):
    assert load_notes(conn, 999) is None


def test_missing_transcript_is_not_sent(conn):
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (2,'2026-07-18T11:00','x')")
    prov = FakeProvider()
    with pytest.raises(ValueError):
        generate_notes(conn, 2, prov, consented=True)
    assert not prov.calls


def test_provider_failure_surfaces_a_useful_message(conn):
    prov = FakeProvider(error=LLMError("Couldn't reach http://localhost:11434/v1 — "
                                       "is the server running?"))
    with pytest.raises(LLMError) as e:
        generate_notes(conn, 1, prov, consented=True)
    assert "is the server running?" in str(e.value)
    assert load_notes(conn, 1) is None, "stored notes for a run that failed"


def test_truncation_is_recorded_and_shown(conn):
    notes = generate_notes(conn, 1, FakeProvider(), consented=True, max_chars=60)
    assert notes.truncated
    assert load_notes(conn, 1).truncated, "truncation lost on the way to the DB"
    assert "earlier part" in to_markdown(notes)


def test_full_session_is_not_flagged_as_truncated(conn):
    notes = generate_notes(conn, 1, FakeProvider(), consented=True)
    assert not notes.truncated
    assert "earlier part" not in to_markdown(notes)


def test_the_model_cannot_claim_it_saw_everything(conn):
    """Truncation is ours to know, so it must not be in the schema we send."""
    assert "truncated" not in json.dumps(SessionNotes.model_json_schema())


def test_prompt_forbids_inventing_todos(conn):
    prov = FakeProvider()
    generate_notes(conn, 1, prov, consented=True)
    system = prov.calls[0]["system"]
    assert "empty list" in system, "model is not told that no to-dos is a valid answer"
    assert "verbatim" in system
    assert "null" in system, "model is not told to leave an unclear owner unset"


def test_speakers_are_offered_as_owners(conn):
    prov = FakeProvider()
    generate_notes(conn, 1, prov, consented=True)
    user = prov.calls[0]["user"]
    assert "Dana" in user and "SPEAKER_00" in user


def test_markdown_has_every_section(conn):
    md = to_markdown(generate_notes(conn, 1, FakeProvider(), consented=True))
    for heading in ("## Summary", "## Key points", "## To-dos", "## Decisions"):
        assert heading in md
    assert "Dana agreed to send the Q3 draft" in md
    assert "- [ ] Send the Q3 draft to legal — **Dana**" in md
    assert "- [ ] Book the launch review — **Unassigned**" in md
    assert "> I'll get the Q3 draft over to legal on Friday" in md
    assert "- Ship behind a flag first" in md


def test_markdown_says_so_when_there_are_no_todos(conn):
    empty = SessionNotes(summary="Chit-chat.", key_points=[], todos=[], decisions=[])
    md = to_markdown(generate_notes(conn, 1, FakeProvider(reply=empty), consented=True))
    assert "## To-dos" in md
    assert "No one committed to anything" in md
    assert "None recorded" in md


def test_privacy_notice_states_size_and_destination(conn):
    from livecaptions.llm.providers import ProviderConfig

    cfg = ProviderConfig(kind="local", model="llama3.1", base_url="http://localhost:11434/v1")
    notice = privacy_notice(cfg, conn, 1)
    assert "characters" in notice
    assert "localhost:11434" in notice
    assert "does not leave this machine" in notice


def test_privacy_notice_warns_when_the_session_will_be_cut(conn):
    from livecaptions.llm.providers import ProviderConfig

    notice = privacy_notice(ProviderConfig(kind="anthropic", model="claude-opus-4-8"),
                            conn, 1, max_chars=60)
    assert "leaves this machine" in notice
    assert "earlier part" in notice
