"""Session notes — the promises, with a fake provider so nothing goes near a network.

What is pinned here: consent is required before any transcript is read, notes
survive a restart, regenerating replaces rather than duplicates, a to-do with no
clear owner stays unowned, and truncation is visible instead of silent.
"""
import json
import re

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from livecaptions import config  # noqa: E402
from livecaptions.llm.providers import LLMError  # noqa: E402
from livecaptions.notes.generate import (  # noqa: E402
    _SYSTEM,
    ConsentRequired,
    NotesNotStored,
    NotesUnreadable,
    SessionNotes,
    StoredNotes,
    ToDo,
    delete_notes,
    generate_notes,
    load_notes,
    payload_chars,
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


def _characters_in(notice: str) -> int:
    m = re.search(r"([\d,]+) characters", notice)
    assert m, f"no character count in the notice: {notice!r}"
    return int(m.group(1).replace(",", ""))


def test_privacy_notice_counts_everything_that_is_sent(conn):
    """The number shown before consent must cover the prompt and headers too, not
    just the transcript — the system prompt alone is over a kilobyte."""
    from livecaptions.llm.providers import ProviderConfig

    prov = FakeProvider()
    generate_notes(conn, 1, prov, consented=True)
    really_sent = len(prov.calls[0]["system"]) + len(prov.calls[0]["user"])

    shown = _characters_in(privacy_notice(
        ProviderConfig(kind="anthropic", model="claude-opus-4-8"), conn, 1))
    assert shown >= really_sent, (
        f"consent dialog said {shown} characters but {really_sent} were sent")
    assert payload_chars(conn, 1) == really_sent


def test_privacy_notice_is_not_just_the_transcript_length(conn):
    from livecaptions.llm.providers import ProviderConfig
    from livecaptions.store.naming import build_transcript

    transcript, _ = build_transcript(conn, 1, max_chars=120_000)
    shown = _characters_in(privacy_notice(
        ProviderConfig(kind="local", model="llama3.1",
                       base_url="http://localhost:11434/v1"), conn, 1))
    assert shown > len(transcript) + len(_SYSTEM) - 1


# --- MAJOR 1: the store hands out ONE connection, shared with the live writer ---

def test_reading_notes_does_not_commit_the_callers_transaction(conn):
    """store.db opens a single check_same_thread=False connection that the live
    TranscriptWriter appends to. If anything on the read path issues an implicit
    COMMIT, opening the notes pane mid-session commits a half-written block."""
    conn.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                 " VALUES (1,9.0,10.0,'2026-07-18T10:00:09','Dana','half-written block')")
    assert conn.in_transaction, "this test needs an open transaction to be meaningful"

    assert load_notes(conn, 1) is None
    assert conn.in_transaction, "load_notes ended a transaction it did not start"

    conn.rollback()
    texts = [r["text"] for r in
             conn.execute("SELECT text FROM utterances WHERE session_id=1").fetchall()]
    assert "half-written block" not in texts, "the writer's uncommitted row was committed"


def test_ensure_schema_does_not_commit_the_callers_transaction(conn):
    """Same hazard reached through the other read path: the table already exists
    by now, so this must be a pure read."""
    from livecaptions.notes.generate import ensure_schema

    generate_notes(conn, 1, FakeProvider(), consented=True)
    conn.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                 " VALUES (1,11.0,12.0,'2026-07-18T10:00:11','Dana','also uncommitted')")
    ensure_schema(conn)
    assert conn.in_transaction, "ensure_schema ended a transaction it did not start"

    conn.rollback()
    texts = [r["text"] for r in
             conn.execute("SELECT text FROM utterances WHERE session_id=1").fetchall()]
    assert "also uncommitted" not in texts


# --- MAJOR 2: an incomplete reply must not read as "there was nothing" ---

def test_every_section_is_required_of_the_model():
    """With default_factory the lists drop out of `required`, and a model that
    answers with only a summary looks identical to a session with no to-dos."""
    assert set(SessionNotes.model_json_schema()["required"]) == {
        "summary", "key_points", "todos", "decisions"}


def test_a_summary_only_reply_is_rejected(conn):
    """llm.providers falls back to plain json_object mode whenever a server
    rejects strict json_schema — the common local case — and validates the raw
    reply against exactly this schema."""
    with pytest.raises(ValidationError):
        SessionNotes.model_validate_json('{"summary": "We talked about Q3."}')


def test_a_reply_missing_sections_is_refused_not_stored(conn):
    """Belt and braces for a provider that hands back an unvalidated object."""
    partial = SessionNotes.model_construct(summary="We talked about Q3.")
    with pytest.raises(LLMError) as e:
        generate_notes(conn, 1, FakeProvider(reply=partial), consented=True)
    assert "todos" in str(e.value)
    assert load_notes(conn, 1) is None, "an incomplete reply was stored"


def test_an_explicitly_empty_reply_is_still_accepted(conn):
    """Required does not mean non-empty: 'nobody committed to anything' is a
    real answer and must survive."""
    empty = SessionNotes(summary="Chit-chat.", key_points=[], todos=[], decisions=[])
    notes = generate_notes(conn, 1, FakeProvider(reply=empty), consented=True)
    assert notes.todos == []
    assert load_notes(conn, 1).todos == []


# --- a paid answer must not be thrown away by a database problem ---

def test_notes_that_cannot_be_saved_are_handed_back(conn):
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (3,'2026-07-18T12:00','x')")
    conn.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                 " VALUES (3,0.0,1.0,'2026-07-18T12:00:00','Dana','Something worth summarising.')")
    conn.commit()

    class SessionVanishes(FakeProvider):
        """The user deletes the session while the model is thinking."""

        def complete(self, system, user, schema):
            conn.execute("DELETE FROM sessions WHERE id=3")
            conn.commit()
            return super().complete(system, user, schema)

    with pytest.raises(NotesNotStored) as e:
        generate_notes(conn, 3, SessionVanishes(), consented=True)
    assert e.value.notes.summary == _REPLY.summary, "the model's answer was lost"
    assert "export" in str(e.value).lower(), "no advice on how to keep the result"
    assert "3" in str(e.value)


# --- store=False, and delete ---

def test_store_false_does_not_write_to_the_database(conn):
    notes = generate_notes(conn, 1, FakeProvider(), consented=True, store=False)
    assert notes.summary == _REPLY.summary
    assert load_notes(conn, 1) is None, "store=False wrote notes anyway"
    assert conn.execute(
        "SELECT COUNT(*) c FROM session_notes").fetchone()["c"] == 0


def test_delete_notes_removes_them(conn):
    generate_notes(conn, 1, FakeProvider(), consented=True)
    assert delete_notes(conn, 1) is True
    assert load_notes(conn, 1) is None
    assert conn.execute(
        "SELECT COUNT(*) c FROM session_notes WHERE session_id=1").fetchone()["c"] == 0


def test_delete_notes_reports_when_there_was_nothing_to_delete(conn):
    assert delete_notes(conn, 999) is False


def test_delete_notes_leaves_other_sessions_alone(conn):
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (4,'2026-07-18T13:00','x')")
    conn.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                 " VALUES (4,0.0,1.0,'2026-07-18T13:00:00','Dana','Another session entirely.')")
    conn.commit()
    generate_notes(conn, 1, FakeProvider(), consented=True)
    generate_notes(conn, 4, FakeProvider(), consented=True)

    assert delete_notes(conn, 4) is True
    assert load_notes(conn, 1) is not None, "deleting one session's notes took another's"


def test_deleting_a_session_takes_its_notes_with_it(conn):
    generate_notes(conn, 1, FakeProvider(), consented=True)
    conn.execute("DELETE FROM sessions WHERE id=1")
    conn.commit()
    assert load_notes(conn, 1) is None


# --- a damaged row must not surface as a raw decode error ---

def test_a_damaged_notes_row_is_reported_not_crashed(conn):
    generate_notes(conn, 1, FakeProvider(), consented=True)
    conn.execute("UPDATE session_notes SET notes_json='{\"summary\": ' WHERE session_id=1")
    conn.commit()

    with pytest.raises(NotesUnreadable) as e:
        load_notes(conn, 1)
    assert "again" in str(e.value), "no hint that regenerating fixes it"


def test_a_notes_row_from_another_shape_is_reported_not_crashed(conn):
    generate_notes(conn, 1, FakeProvider(), consented=True)
    conn.execute("UPDATE session_notes SET notes_json='{\"summary\": 5, \"todos\": \"lots\"}'"
                 " WHERE session_id=1")
    conn.commit()

    with pytest.raises(NotesUnreadable):
        load_notes(conn, 1)
