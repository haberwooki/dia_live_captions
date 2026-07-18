"""Regression smoke test for the 'enabling a setting quits the app' bug.

Drives the REAL run_overlay with a stub *live* source, then triggers the live
pipeline restart the Settings window uses. Before the fix, stopping the old
source set its `finished` event while the old done-timer was still running, and
that timer called app.quit() -> the whole app disappeared.

Calls the restart callback directly (not the checkbox) so the user's real
config.toml is never written to.
"""
import sys
import threading

sys.argv = ["smoke"]

from PySide6 import QtCore, QtWidgets  # noqa: E402

from livecaptions.config import Settings  # noqa: E402
from livecaptions.ui.overlay import run_overlay  # noqa: E402
from livecaptions.ui.settings import SettingsWindow  # noqa: E402

built = []
result = {"quit_early": True, "restarts": 0, "err": None}


class StubSource:
    def __init__(self, n):
        self.n = n
        self.finished = threading.Event()
        self.dropped_blocks = 0

    def start(self, on_event=None, monitor=None):
        pass

    def stop(self):
        self.finished.set()          # this is what used to kill the app


def factory():
    # The rebuild MUST be slow to reproduce the bug: it's the window between
    # "old source stopped (finished set)" and "new source started (timers replaced)"
    # in which the stale done-timer fires and quits. Loading the speaker model
    # takes seconds, which is why enabling colour captions was what killed the app.
    if built:
        import time
        time.sleep(2.0)
    s = StubSource(len(built))
    built.append(s)
    return s


settings = Settings()
settings.open_settings_on_launch = True   # in-memory only; never saved


def find_settings_window():
    for w in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(w, SettingsWindow):
            return w
    return None


def step_trigger_restart():
    w = find_settings_window()
    if w is None:
        result["err"] = "Settings window never appeared"
        QtWidgets.QApplication.instance().quit()
        return
    if w._on_restart is None:
        result["err"] = "no live-restart callback wired"
        QtWidgets.QApplication.instance().quit()
        return
    w._on_restart()               # <-- the exact path the checkboxes use
    result["restarts"] += 1


def step_verify():
    # Reaching here at all means the app did NOT quit during the restart.
    result["quit_early"] = False
    result["built"] = len(built)
    w = find_settings_window()
    result["note"] = w._restart_note.text() if w else "(no window)"
    QtWidgets.QApplication.instance().quit()


# The QApplication must exist before timers can be scheduled; run_overlay reuses it.
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
QtCore.QTimer.singleShot(2500, step_trigger_restart)
QtCore.QTimer.singleShot(9000, step_verify)
QtCore.QTimer.singleShot(20000, lambda: (result.__setitem__("err", "timed out"),
                                         app.quit()))

run_overlay(factory, settings, source_name="stub", is_live=True)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("---- RESULT ----")
print("error:            ", result["err"])
print("restarts issued:  ", result["restarts"])
print("sources built:    ", result.get("built"))
print("settings note:    ", result.get("note"))
print("app quit early:   ", result["quit_early"])
ok = (not result["err"]) and result["restarts"] == 1 and result.get("built", 0) >= 2 \
    and not result["quit_early"]
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
