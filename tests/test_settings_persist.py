"""Settings that the GUI changes must survive a restart.

The 'Movable' checkbox was the one overlay control that only ever lived on the
running OverlayWindow, so it reset to unchecked on every launch. These tests pin
the round-trip: toggle -> config.toml -> next launch.
"""
import os

import pytest

from livecaptions import config


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """Point the config module at a throwaway file, never the user's real one."""
    path = tmp_path / "config.toml"
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    return path


def test_overlay_movable_round_trips(temp_config):
    assert config.Settings().overlay_movable is False       # default: click-through
    config.save_settings(overlay_movable=True)
    assert config.Settings().overlay_movable is True        # survives a "restart"


def test_saving_one_key_preserves_the_others(temp_config):
    config.save_settings(overlay_movable=True, overlay_font_pt=31)
    config.save_settings(speaker_colors=True)
    s = config.Settings()
    assert (s.overlay_movable, s.overlay_font_pt, s.speaker_colors) == (True, 31, True)


def test_settings_window_toggle_persists_movable(temp_config, monkeypatch):
    """The checkbox itself must write the setting, not just poke the overlay."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from livecaptions.ui import settings as settings_ui

    # the window imports save_settings by name, so patch it there too
    monkeypatch.setattr(settings_ui, "save_settings", config.save_settings)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # the box must SHOW the saved state when the window opens...
    config.save_settings(overlay_movable=True)
    shown = settings_ui.SettingsWindow(config.Settings())
    assert shown._movable.isChecked() is True
    shown.deleteLater()
    config.save_settings(overlay_movable=False)

    # ...and WRITE it when toggled.
    win = settings_ui.SettingsWindow(config.Settings())
    try:
        assert win._movable.isChecked() is False
        win._movable.setChecked(True)                    # emits toggled -> _on_movable
        assert config.Settings().overlay_movable is True
        win._movable.setChecked(False)
        assert config.Settings().overlay_movable is False
    finally:
        win.deleteLater()
        app.processEvents()
