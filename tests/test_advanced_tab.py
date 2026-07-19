"""The Advanced tab hands people the knobs that can break transcription — pin the
promises that keep that safe: what you change is what gets saved, a combination
that cannot work is never stored, "reset" really means the shipped defaults, and
nothing here writes outside the config file it was pointed at.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from livecaptions import config  # noqa: E402
from livecaptions.ui import advanced as adv  # noqa: E402


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Point config at tmp_path and prove the real user config is untouchable."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(adv, "save_settings", config.save_settings)
    assert str(config.CONFIG_PATH).startswith(str(tmp_path))
    return cfg_path


@pytest.fixture
def make_tab(cfg, monkeypatch):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Never poke the real keyboard registry from a test run.
    monkeypatch.setattr(adv, "probe_hotkey", lambda spec: "free")

    def _make(**kw):
        tab = adv.AdvancedTab(config.Settings(), **kw)
        tab.RESTART_DELAY_MS = 0
        return tab
    return _make


@pytest.fixture
def tab(make_tab):
    return make_tab()


def _reloaded(cfg):
    """Settings as a fresh launch would read them back off disk."""
    assert cfg.exists(), "nothing was written to the config file"
    return config.Settings()


# ---- persistence round-trips ----------------------------------------------
def test_controls_start_on_the_saved_values(tab):
    s = config.Settings()
    assert tab._beam.value() == s.beam_size
    assert tab._block.value() == pytest.approx(s.block_sec)
    assert tab._vad.value() == pytest.approx(s.stream_vad_threshold)
    assert tab._hotkey_edits["hotkey_toggle"].text() == s.hotkey_toggle


@pytest.mark.parametrize("attr, field, value", [
    ("_beam", "beam_size", 5),
    ("_block", "block_sec", 0.24),
    ("_floor", "silence_rms_floor", 12.0),
    ("_interval", "stream_process_interval", 0.8),
    ("_endsil", "stream_end_silence_sec", 1.4),
    ("_maxline", "stream_max_line_sec", 6.0),
    ("_maxbuf", "stream_max_buffer_sec", 20.0),
    ("_vad", "stream_vad_threshold", 0.700),
    ("_nudge", "hotkey_nudge_px", 80),
])
def test_numeric_settings_round_trip(tab, cfg, attr, field, value):
    getattr(tab, attr).setValue(value)
    assert getattr(_reloaded(cfg), field) == pytest.approx(value)


def test_compute_and_language_round_trip(tab, cfg):
    tab._gpu.setCurrentText("int8_float16")
    tab._cpu.setCurrentText("float32")
    tab._lang.setCurrentText("de")
    tab._on_language()
    s = _reloaded(cfg)
    assert (s.gpu_compute, s.cpu_compute, s.language) == ("int8_float16", "float32", "de")


def test_unknown_language_is_refused(tab, cfg):
    tab._lang.setCurrentText("klingon")
    tab._on_language()
    assert config.Settings().language == "en", "stored a language Whisper cannot use"
    assert "not a language" in tab._lang_note.text()


def test_turning_hotkeys_off_persists(tab, cfg):
    tab._hk_enabled.setChecked(False)
    assert _reloaded(cfg).hotkeys_enabled is False


# ---- hotkey validation -----------------------------------------------------
@pytest.mark.parametrize("typed, expected", [
    ("ctrl+alt+c", "ctrl+alt+c"),
    ("  ALT + Shift+K ", "alt+shift+k"),
    ("alt+ctrl+f5", "ctrl+alt+f5"),          # canonical order, not typing order
    ("win+shift+left", "shift+win+left"),
    ("control+alt+space", "ctrl+alt+space"),
    ("ctrl+ctrl+alt+j", "ctrl+alt+j"),
])
def test_valid_combos_are_accepted_and_normalised(typed, expected):
    assert adv.normalize_hotkey(typed) == expected


@pytest.mark.parametrize("typed", [
    "",                 # nothing
    "   ",
    "ctrl+alt",         # modifiers only
    "banana",           # not a key
    "ctrl+alt+banana",
    "c",                # no modifier: would steal the key from every app
    "f5",
    "ctrl+alt+c+d",     # two keys
    "ctrl++c",          # empty part
])
def test_unusable_combos_are_rejected(typed):
    with pytest.raises(ValueError):
        adv.normalize_hotkey(typed)


def test_a_bad_combo_is_never_saved(tab, cfg):
    edit = tab._hotkey_edits["hotkey_toggle"]
    edit.setText("ctrl+alt+banana")
    edit.editingFinished.emit()

    assert config.Settings().hotkey_toggle == "ctrl+alt+c", "saved an unusable hotkey"
    assert edit.text() == "ctrl+alt+c", "left the broken text in the box"
    assert "Not a usable combination" in tab._hotkey_notes["hotkey_toggle"].text()


def test_a_good_combo_is_saved_normalised(tab, cfg):
    edit = tab._hotkey_edits["hotkey_toggle"]
    edit.setText(" CTRL+SHIFT+F9 ")
    edit.editingFinished.emit()

    assert _reloaded(cfg).hotkey_toggle == "ctrl+shift+f9"
    assert edit.text() == "ctrl+shift+f9"


def test_one_combo_cannot_do_two_things(tab, cfg):
    edit = tab._hotkey_edits["hotkey_pause"]
    edit.setText("ctrl+alt+c")               # already the show/hide shortcut
    edit.editingFinished.emit()

    assert config.Settings().hotkey_pause == "ctrl+alt+p", "two actions on one combo"
    assert "already used" in tab._hotkey_notes["hotkey_pause"].text()


def test_claimed_hotkeys_are_named_in_the_status(tab, monkeypatch):
    monkeypatch.setattr(adv, "probe_hotkey",
                        lambda spec: "in use" if spec == "ctrl+alt+c" else "free")
    tab.refresh_hotkey_status()
    assert "Show / hide captions (ctrl+alt+c)" in tab._hk_status.text()
    assert "Pause" not in tab._hk_status.text()
    assert "In use" in tab._hotkey_notes["hotkey_toggle"].text()
    assert "Available" in tab._hotkey_notes["hotkey_pause"].text()


def test_a_hotkey_the_app_failed_to_register_is_reported(make_tab):
    tab = make_tab(registered={"hotkey_toggle": False, "hotkey_pause": True})
    tab.refresh_hotkey_status()
    assert "Not working" in tab._hotkey_notes["hotkey_toggle"].text()
    assert "Working now" in tab._hotkey_notes["hotkey_pause"].text()
    assert "Show / hide captions" in tab._hk_status.text()


def test_all_free_says_so(tab):
    tab.refresh_hotkey_status()
    assert "Every shortcut is available" in tab._hk_status.text()


# ---- restart flagging ------------------------------------------------------
def test_a_pipeline_change_rebuilds_when_there_is_something_to_rebuild(make_tab):
    calls = []
    tab = make_tab(on_restart=lambda: calls.append(1))
    tab._beam.setValue(4)
    assert "reloading" in tab._note.text()
    tab._do_restart()
    assert calls == [1]


def test_a_pipeline_change_without_a_pipeline_says_next_launch(tab):
    tab._interval.setValue(1.0)
    assert "next time you start" in tab._note.text()
    tab._do_restart()          # nothing to call; must not raise


def test_a_hotkey_change_does_not_pretend_a_rebuild_applies_it(make_tab):
    """Hotkeys are claimed once at startup; rebuilding the audio pipeline won't
    re-register them, so the tab must not claim it did."""
    calls = []
    tab = make_tab(on_restart=lambda: calls.append(1))
    edit = tab._hotkey_edits["hotkey_quit"]
    edit.setText("ctrl+shift+f10")
    edit.editingFinished.emit()
    assert calls == []
    assert "next time you start" in tab._note.text()


# ---- reset -----------------------------------------------------------------
def test_reset_restores_every_owned_field(tab, cfg):
    tab._beam.setValue(6)
    tab._maxline.setValue(30.0)
    tab._gpu.setCurrentText("float32")
    tab._nudge.setValue(200)
    edit = tab._hotkey_edits["hotkey_toggle"]
    edit.setText("ctrl+shift+f9")
    edit.editingFinished.emit()

    tab.reset_to_defaults()

    fresh = _reloaded(cfg)
    for field in adv.OWNED_FIELDS:
        assert getattr(fresh, field) == config.Settings.model_fields[field].default, field
    assert tab._beam.value() == config.Settings().beam_size, "widgets still show the old values"
    assert tab._hotkey_edits["hotkey_toggle"].text() == "ctrl+alt+c"


def test_reset_reads_the_defaults_from_the_model(tab, cfg, monkeypatch):
    """Not a second hardcoded copy: move the model's default and reset follows."""
    monkeypatch.setattr(config.Settings.model_fields["beam_size"], "default", 7)
    tab.reset_to_defaults()
    assert "beam_size = 7" in cfg.read_text(encoding="utf-8")


def test_reset_is_confirmed_first(tab, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No))
    tab._beam.setValue(6)
    tab._on_reset()
    assert config.Settings().beam_size == 6, "reset happened without confirmation"


# ---- blast radius ----------------------------------------------------------
def test_nothing_is_written_outside_the_tmp_config(tab, cfg, tmp_path):
    real_dir = tmp_path.parent    # anything but the tmp config dir
    before = sorted(p.name for p in tmp_path.iterdir())

    tab._beam.setValue(3)
    tab._vad.setValue(0.6)
    edit = tab._hotkey_edits["hotkey_up"]
    edit.setText("ctrl+shift+f11")
    edit.editingFinished.emit()
    tab.reset_to_defaults()

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(set(before) | {"config.toml"})
    assert cfg.exists() and real_dir != cfg.parent


def test_the_users_real_config_is_never_opened(make_tab, monkeypatch):
    """The fixture repoints CONFIG_PATH; prove no code path ignores it."""
    import platformdirs
    real = platformdirs.user_config_dir("live-captions", appauthor=False)
    opened = []
    real_open = open

    def spy(path, *a, **k):
        opened.append(str(path))
        return real_open(path, *a, **k)
    monkeypatch.setattr("builtins.open", spy)

    tab = make_tab()
    tab._beam.setValue(2)
    tab.reset_to_defaults()

    assert not [p for p in opened if p.startswith(real)], f"touched the real config: {opened}"
