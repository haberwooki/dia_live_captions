"""The AI tab is the one place data can leave the machine — pin its guarantees.

These assert the promises made to the user in the UI, not implementation details:
off by default, keys never in config.toml, consent before sending with the size
and destination stated, and per-rename confirmation showing the evidence.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from livecaptions import config  # noqa: E402
from livecaptions.llm import providers as P  # noqa: E402
from livecaptions.store import db as db_mod  # noqa: E402


@pytest.fixture
def tab(tmp_path, monkeypatch):
    dbp = tmp_path / "t.db"
    monkeypatch.setattr(db_mod, "DB_PATH", dbp)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")

    conn = db_mod.connect(dbp)
    conn.execute("INSERT INTO sessions (id, started_at, source) VALUES (1,'2026-07-18 10:00','x')")
    for spk, text in (("SPEAKER_00", "hi everyone this is Mike"),
                      ("SPEAKER_01", "thanks Mike, I'm Dana")):
        conn.execute("INSERT INTO utterances (session_id,t_start,t_end,wall_clock,speaker,text)"
                     " VALUES (1,0,1,'2026-07-18 10:00',?,?)", (spk, text))
    conn.commit(); conn.close()

    from livecaptions.ui import ai as ai_mod
    monkeypatch.setattr(ai_mod, "save_settings", config.save_settings)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = ai_mod.AITab(config.Settings())
    assert str(dbp) in w._ensure().execute("PRAGMA database_list").fetchone()["file"]
    return w


def test_off_by_default(tab):
    assert config.Settings().llm_provider == "none"
    assert tab._kind.currentIndex() == 0
    assert "off" in tab._keynote.text().lower()
    assert not tab._test.isEnabled(), "nothing to test when unconfigured"


def test_choosing_a_provider_persists_it(tab):
    tab._kind.setCurrentIndex(3)                    # local
    assert config.Settings().llm_provider == "local"
    assert tab._test.isEnabled()


def test_local_provider_states_data_stays_put(tab):
    tab._kind.setCurrentIndex(3)
    assert "leaves this machine" in tab._keynote.text()


def test_key_never_reaches_the_config_file(tab, tmp_path, monkeypatch):
    """The whole point of using Credential Manager."""
    saved = {}
    from livecaptions.llm import credentials as creds
    monkeypatch.setattr(creds, "set_secret", lambda name, secret: saved.update({name: secret}))
    tab._kind.setCurrentIndex(2)                    # openai
    tab._key.setText("sk-super-secret-value")
    tab._on_save_key()

    assert saved == {"openai": "sk-super-secret-value"}
    text = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "sk-super-secret" not in text, "key leaked into config.toml"
    assert tab._key.text() == "", "key left visible on screen"


def test_naming_requires_consent_and_sends_nothing_when_declined(tab, monkeypatch):
    tab._kind.setCurrentIndex(3)
    tab._model.setText("llama3.1"); tab._model.editingFinished.emit()
    tab._url.setText("http://localhost:11434/v1"); tab._url.editingFinished.emit()
    tab._load_sessions()

    shown = {}

    def fake_question(parent, title, text, *a, **k):
        shown["title"], shown["text"] = title, text
        return QtWidgets.QMessageBox.StandardButton.No
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(fake_question))

    sent = []
    monkeypatch.setattr(P, "from_settings", lambda *a, **k: sent.append(1))

    tab._on_name()

    assert "Send this transcript" in shown["title"]
    assert "characters" in shown["text"], "must state how much is being sent"
    assert not sent, "asked the model despite the user declining"
    assert "Nothing was sent" in tab._name_status.text()


def test_consent_prompt_names_the_destination(tab, monkeypatch):
    tab._kind.setCurrentIndex(3)
    tab._url.setText("http://localhost:11434/v1"); tab._url.editingFinished.emit()
    tab._model.setText("llama3.1"); tab._model.editingFinished.emit()
    tab._load_sessions()
    seen = {}
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda p, t, text, *a, **k: seen.update(text=text)
                                     or QtWidgets.QMessageBox.StandardButton.No))
    tab._on_name()
    assert "localhost:11434" in seen["text"]
    assert "does not leave this machine" in seen["text"]


def test_each_rename_is_confirmed_with_its_evidence(tab, monkeypatch):
    from livecaptions.store.naming import SpeakerName
    prompts = []

    def fake_question(parent, title, text, *a, **k):
        prompts.append(text)
        return QtWidgets.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(fake_question))

    tab._on_names_done({"session": 1, "proposals": [
        SpeakerName(label="SPEAKER_00", name="Mike", confidence="high",
                    evidence='"hi everyone this is Mike"'),
        SpeakerName(label="SPEAKER_01", name=None, confidence="low",
                    evidence="no supporting quote"),
    ]})

    assert len(prompts) == 1, "confirmed a rename for a label with no name"
    assert "this is Mike" in prompts[0], "evidence not shown before renaming"
    assert "undone" in prompts[0], "reversibility not stated"

    conn = tab._ensure()
    speakers = [r["speaker"] for r in conn.execute("SELECT speaker FROM utterances")]
    assert "Mike" in speakers
    assert "SPEAKER_01" in speakers, "renamed a label the model declined to name"
    assert "No name found for SPEAKER_01" in tab._name_status.text()


def test_declining_a_rename_changes_nothing(tab, monkeypatch):
    from livecaptions.store.naming import SpeakerName
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No))
    tab._on_names_done({"session": 1, "proposals": [
        SpeakerName(label="SPEAKER_00", name="Mike", confidence="high", evidence="q")]})
    speakers = [r["speaker"] for r in tab._ensure().execute("SELECT speaker FROM utterances")]
    assert "Mike" not in speakers
    assert "Renamed 0 speaker(s)" in tab._name_status.text()


class TestOneClickNotes:
    """With nothing configured, the Notes button must still be the thing you click.

    The whole point: no provider dropdown, no port, no model name typed. If a local
    server is running, one click finds it, asks once, and proceeds.
    """

    def test_notes_button_is_clickable_with_nothing_configured(self, tab):
        assert tab._kind.currentIndex() == 0, "precondition: unconfigured"
        assert tab._notes_btn.isEnabled(), (
            "disabled the one button that needs no setup")

    def test_one_click_finds_a_local_model_and_configures_it(self, tab, monkeypatch):
        from livecaptions.llm import discover as D
        monkeypatch.setattr(D, "autoconfigure", lambda *a, **k: {
            "llm_provider": "local", "llm_base_url": "http://localhost:11434/v1",
            "llm_model": "llama3.1:8b", "_server": "Ollama"})
        asked = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                            staticmethod(lambda p, t, text, *a, **k: asked.append(text)
                                         or QtWidgets.QMessageBox.StandardButton.Yes))
        assert tab._autoconfigure_or_explain() == "local"

        saved = config.Settings()
        assert saved.llm_provider == "local"
        assert saved.llm_model == "llama3.1:8b"
        assert "Ollama" in asked[0] and "llama3.1:8b" in asked[0]
        assert "stays on this machine" in asked[0], "must say where the data goes"

    def test_declining_changes_nothing(self, tab, monkeypatch):
        from livecaptions.llm import discover as D
        monkeypatch.setattr(D, "autoconfigure", lambda *a, **k: {
            "llm_provider": "local", "llm_base_url": "http://localhost:11434/v1",
            "llm_model": "llama3.1:8b", "_server": "Ollama"})
        monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No))
        assert tab._autoconfigure_or_explain() is None
        assert config.Settings().llm_provider == "none", "configured without consent"

    def test_no_local_server_explains_what_to_do(self, tab, monkeypatch):
        """A dead end here is where someone gives up, so it must name the fix."""
        from livecaptions.llm import discover as D
        monkeypatch.setattr(D, "autoconfigure", lambda *a, **k: None)
        assert tab._autoconfigure_or_explain() is None
        msg = tab._notes_status.text()
        assert "Ollama" in msg and "ollama pull" in msg, f"unhelpful: {msg}"
        assert config.Settings().llm_provider == "none"

    def test_a_broken_discovery_does_not_look_like_a_crash(self, tab, monkeypatch):
        from livecaptions.llm import discover as D
        def boom(*a, **k):
            raise RuntimeError("network stack on fire")
        monkeypatch.setattr(D, "autoconfigure", boom)
        assert tab._autoconfigure_or_explain() is None
        assert "Couldn't look for a local model" in tab._notes_status.text()
