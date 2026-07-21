"""Auto-update must inform, and only install when the user opted in.

The design the user asked for: on launch, say WHICH version is available; never
install behind their back; make automatic install a deliberate, separate choice.
These pin those promises against the settings window's update handlers.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from livecaptions import config  # noqa: E402


@pytest.fixture
def win(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    from livecaptions.ui import settings as sm
    monkeypatch.setattr(sm, "save_settings", config.save_settings)
    # No transport/registered needed; we only exercise the Updates group.
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = sm.SettingsWindow(config.Settings())
    return w


PLAN = {"tag": "v0.9.9", "kind": "patch", "url": "https://ex/patch.exe", "approx_mb": 200}


def test_update_settings_default_to_notify_not_auto_install():
    s = config.Settings()
    assert s.check_updates_on_launch is True
    assert s.auto_install_updates is False, "auto-install must be opt-in"


def test_an_available_update_names_the_version(win):
    win._check_interactive = False
    win._on_check_done({"plan": PLAN, "err": None})
    assert "v0.9.9" in win._ver.text()
    # isHidden(), not isVisible(): the latter is False in an unshown offscreen window.
    assert not win._install_btn.isHidden()
    assert "v0.9.9" in win._install_btn.text() and "200 MB" in win._install_btn.text()


def test_a_background_check_does_not_pop_a_dialog(win, monkeypatch):
    """The launch check must inform silently — no modal the user didn't ask for."""
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(1)))
    win._check_interactive = False
    win._on_check_done({"plan": PLAN, "err": None})
    assert not asked, "a background check must not open a dialog"


def test_clicking_check_offers_to_install(win, monkeypatch):
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(1)
                                     or QtWidgets.QMessageBox.StandardButton.No))
    win._check_interactive = True
    win._on_check_done({"plan": PLAN, "err": None})
    assert asked, "an explicit Check should offer to install"


def test_install_button_starts_the_download(win, monkeypatch):
    started = {}
    monkeypatch.setattr(win, "_start_download",
                        lambda url, tag: started.update(url=url, tag=tag))
    win._pending_plan = PLAN
    win._install_pending(confirm=False)
    assert started == {"url": PLAN["url"], "tag": "v0.9.9"}


def test_up_to_date_hides_the_install_button(win):
    win._pending_plan = PLAN
    win._install_btn.setVisible(True)
    win._check_interactive = False
    win._on_check_done({"plan": None, "err": None})
    assert win._install_btn.isHidden()
    assert win._pending_plan is None
    assert "Up to date" in win._ver.text()


def test_a_background_check_error_is_not_shown(win):
    """A flaky network on launch must not put an error where the version was."""
    win._ver.setText("Version 0.9.9")
    win._check_interactive = False
    win._on_check_done({"plan": None, "err": "URLError"})
    assert "Couldn't check" not in win._ver.text()


def test_an_interactive_check_error_is_shown(win):
    win._check_interactive = True
    win._on_check_done({"plan": None, "err": "URLError"})
    assert "Couldn't check" in win._ver.text()


def test_show_available_update_reflects_a_launch_find(win):
    """The overlay's launch check hands the plan to an open window via this."""
    win.show_available_update(PLAN)
    assert "v0.9.9" in win._ver.text()
    assert not win._install_btn.isHidden()
    assert win._pending_plan == PLAN


def test_toggling_the_checkboxes_persists(win):
    win._auto_check.setChecked(False)
    assert config.Settings().check_updates_on_launch is False
    win._auto_install.setChecked(True)
    assert config.Settings().auto_install_updates is True


def test_captions_tab_is_gone_and_model_moved_to_advanced(win):
    """The merge: no Captions tab, and the model picker now lives on Advanced."""
    names = [win._tabs.tabText(i) for i in range(win._tabs.count())]
    assert "Captions" not in names
    assert "Speakers" in names and "Advanced" in names
    from livecaptions.ui.advanced import AdvancedTab
    tab = AdvancedTab(config.Settings())
    assert hasattr(tab, "_model"), "model picker should be on the Advanced tab"
