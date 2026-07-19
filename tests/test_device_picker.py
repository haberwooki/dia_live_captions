"""Duplicate-named output devices must be distinguishable and selectable.

Real hardware case: two monitors on one GPU present two loopback endpoints with
byte-identical names (index 13 and 14). Windows' default output was the second one,
only index 14 carried audio, and index 13 delivered no data at all — which looks
exactly like the app being broken rather than the wrong device being selected.

The picker used to preselect by NAME alone, so it always landed on the first
duplicate: the panel showed the wrong device as chosen, with no way to tell them
apart.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from livecaptions import config  # noqa: E402
from livecaptions.capture.devices import name_ordinal  # noqa: E402
from livecaptions.ui import settings as settings_ui  # noqa: E402

NAME = "DELL S2721QS (NVIDIA High Definition Audio) [Loopback]"
DEVICES = [
    {"index": 13, "name": NAME, "maxInputChannels": 2, "defaultSampleRate": 48000},
    {"index": 14, "name": NAME, "maxInputChannels": 2, "defaultSampleRate": 48000},
    {"index": 15, "name": "Speakers (Realtek) [Loopback]", "maxInputChannels": 2,
     "defaultSampleRate": 48000},
]


@pytest.fixture
def picker(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(settings_ui, "save_settings", config.save_settings)
    monkeypatch.setattr(settings_ui, "_loopbacks", lambda: (DEVICES, 14))

    def build(**saved):
        config.save_settings(**saved) if saved else None
        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        return settings_ui.SettingsWindow(config.Settings())
    return build


def test_duplicate_names_are_numbered(picker):
    """Two identical names must not both render as the same unselectable string."""
    w = picker()
    labels = [w._dev.itemText(i) for i in range(1, w._dev.count())]
    dell = [t for t in labels if t.startswith("DELL")]
    assert len(dell) == 2
    assert dell[0] != dell[1], f"duplicates are indistinguishable: {dell}"
    assert "#1" in dell[0] and "#2" in dell[1]
    # A device with a unique name should NOT be cluttered with a number.
    assert not any("#" in t for t in labels if t.startswith("Speakers"))


def test_second_duplicate_can_be_preselected(picker):
    """The exact failure: saved ordinal 1 used to display ordinal 0 as selected."""
    w = picker(loopback_name=NAME, loopback_ordinal=1)
    chosen = w._dev.itemData(w._dev.currentIndex())
    assert chosen is not None, "fell back to auto instead of the saved device"
    assert chosen["index"] == 14, f"preselected index {chosen['index']}, wanted 14"


def test_first_duplicate_still_preselects(picker):
    w = picker(loopback_name=NAME, loopback_ordinal=0)
    assert w._dev.itemData(w._dev.currentIndex())["index"] == 13


def test_default_entry_is_auto_and_follows_windows(picker):
    """With nothing saved, the app must follow the Windows default rather than
    pinning an index that can change when devices come and go."""
    w = picker()
    assert w._dev.currentIndex() == 0
    assert w._dev.itemData(0) is None
    assert "auto" in w._dev.itemText(0).lower()


def test_selecting_a_duplicate_saves_its_ordinal(picker, monkeypatch):
    """Choosing '#2' must persist ordinal 1, not just the shared name."""
    w = picker()
    monkeypatch.setattr(settings_ui, "_loopbacks", lambda: (DEVICES, 14))
    import livecaptions.capture.devices as dev_mod
    monkeypatch.setattr(dev_mod, "enumerate_loopbacks", lambda p: DEVICES)

    class FakePA:
        def terminate(self):
            pass
    import sys
    import types
    fake = types.ModuleType("pyaudiowpatch")
    fake.PyAudio = lambda: FakePA()
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake)

    for i in range(1, w._dev.count()):
        if w._dev.itemData(i)["index"] == 14:
            w._dev.setCurrentIndex(i)
            break

    s = config.Settings()
    assert s.loopback_name == NAME
    assert s.loopback_ordinal == 1, "saved the wrong duplicate"
    assert name_ordinal(DEVICES, DEVICES[1]) == 1
