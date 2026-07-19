"""Smoke test for the v0.2.0 Start/Pause/Stop transport.

Drives the real run_overlay with a stub live source and walks the full state
machine, asserting that each control does what its label promises and that the
app survives all of it (the v0.1.6 bug class).
"""
import pathlib
import sys
import tempfile
import threading

sys.argv = ["smoke"]

from PySide6 import QtCore, QtWidgets  # noqa: E402

from livecaptions import config as _config  # noqa: E402

# Never read or write the real config: this test opens the Settings window and
# clicks things, and switching a tab alone persists a key. An earlier run wrote to
# the user's live settings, which is exactly the kind of surprise a test must not
# cause. Redirect to a throwaway file BEFORE anything reads it.
_tmp = pathlib.Path(tempfile.mkdtemp(prefix="lc-smoke-")) / "config.toml"
_config.CONFIG_DIR, _config.CONFIG_PATH = _tmp.parent, _tmp

from livecaptions.config import Settings  # noqa: E402
from livecaptions.ui import settings as _settings_ui  # noqa: E402
from livecaptions.ui.overlay import run_overlay  # noqa: E402
from livecaptions.ui.settings import SettingsWindow  # noqa: E402

_settings_ui.save_settings = _config.save_settings   # picks up the redirected path

built = []
released = []
states = []
steps = []
fail = []


class StubSource:
    def __init__(self):
        self.finished = threading.Event()
        self.dropped_blocks = 0
        self.stopped = False

    def start(self, on_event=None, monitor=None):
        pass

    def stop(self):
        self.stopped = True
        self.finished.set()


def factory():
    import time
    if built:
        time.sleep(1.0)          # a real rebuild is slow; that's when bugs bite
    s = StubSource()
    built.append(s)
    return s


factory.release_model = lambda: released.append(1)

settings = Settings()
settings.open_settings_on_launch = True
settings.startup_mode = "always"           # this test is about the running pipeline

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def win():
    for w in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(w, SettingsWindow):
            return w
    return None


def check(label, cond, detail=""):
    steps.append((label, bool(cond), detail))
    if not cond:
        fail.append(label)


def t_ready():
    w = win()
    if w is None or w._transport is None:
        fail.append("no transport wired")
        app.quit()
        return
    tr = w._transport
    tr.state_changed.connect(lambda s: states.append(s))
    check("starts running", tr.state == "running", tr.state)
    check("Start disabled while running", not w._btn_start.isEnabled())
    check("button says Pause", w._btn_pause.text() == "Pause")
    w._btn_pause.click()                      # -> paused


def t_after_pause():
    w = win(); tr = w._transport
    check("paused", tr.state == "paused", tr.state)
    check("source stopped on pause", built[0].stopped)
    check("model NOT released on pause", not released, f"released={len(released)}")
    check("button becomes Resume", w._btn_pause.text() == "Resume")
    check("Start enabled when paused", w._btn_start.isEnabled())
    w._btn_pause.click()                      # -> resume


def t_after_resume():
    w = win(); tr = w._transport
    check("resumed to running", tr.state == "running", tr.state)
    check("a second source was built", len(built) == 2, f"built={len(built)}")
    w._btn_stop.click()                       # -> stopped


def t_after_stop():
    w = win(); tr = w._transport
    check("stopped", tr.state == "stopped", tr.state)
    check("model released on stop", bool(released), f"released={len(released)}")
    check("Stop disabled when stopped", not w._btn_stop.isEnabled())
    w._btn_start.click()                      # -> start again


def t_after_restart():
    w = win(); tr = w._transport
    check("restarted after stop", tr.state == "running", tr.state)
    check("third source built", len(built) == 3, f"built={len(built)}")
    check("app still alive", True)
    # "how I leave it is how it re-opens" — the state must reach disk, not just the
    # in-memory settings object, or the next launch has nothing to resume from.
    saved = Settings().last_transport_state
    check("last state persisted for resume", saved == "running", saved)
    app.quit()


for ms, fn in ((1500, t_ready), (3000, t_after_pause), (5500, t_after_resume),
               (7000, t_after_stop), (9500, t_after_restart)):
    QtCore.QTimer.singleShot(ms, fn)
QtCore.QTimer.singleShot(20000, lambda: (fail.append("timed out"), app.quit()))

run_overlay(factory, settings, source_name="stub", is_live=True,
            on_release_model=factory.release_model)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("---- TRANSPORT ----")
for label, ok, detail in steps:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}{('  <- ' + str(detail)) if detail and not ok else ''}")
print("state sequence:", " -> ".join(states))
print("PASS" if not fail else f"FAIL: {fail}")
sys.exit(0 if not fail else 1)
