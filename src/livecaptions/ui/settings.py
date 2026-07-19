"""Settings window (v0.1.2) — the GUI the CLI flags and config.toml used to be.

Opened from the tray's "Settings…" item. Appearance changes (font, colour,
opacity, lines, movable) apply to the running overlay immediately; changes that
touch the capture/model pipeline (audio device, model, speaker colours) persist
and take effect on the next launch — the window says so.
"""
from __future__ import annotations

import sys
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..config import save_settings

_MODELS = ["tiny.en", "base.en", "small.en", "medium", "large-v3"]


def _loopbacks() -> tuple[List[dict], Optional[int]]:
    """(loopback device infos, index of the default output's loopback)."""
    try:
        import pyaudiowpatch as pa
        from ..capture.devices import default_loopback, enumerate_loopbacks
        p = pa.PyAudio()
        try:
            lbs = enumerate_loopbacks(p)
            try:
                default_idx = default_loopback(p)["index"]
            except Exception:
                default_idx = None
            return lbs, default_idx
        finally:
            p.terminate()
    except Exception:
        return [], None


class SettingsWindow(QtWidgets.QWidget):
    # updater: marshal background download progress/results onto the GUI thread
    _check_done = QtCore.Signal(object)   # {"tag", "url", "err"}
    _dl_progress = QtCore.Signal(int)
    _dl_done = QtCore.Signal(str)         # "" on success, else an error string

    def __init__(self, settings, overlay=None, quit_on_close: bool = False, on_restart=None,
                 transport=None):
        super().__init__(None)
        self._settings = settings
        self._overlay = overlay
        self._quit_on_close = quit_on_close   # closing this window quits the whole app
        self._on_restart = on_restart         # rebuild the live pipeline (device/model/speaker), or None
        self._transport = transport           # start/pause/stop the pipeline, or None
        self._restart_dirty = False
        self.setWindowTitle("Live Captions — Settings")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(420)

        root = QtWidgets.QVBoxLayout(self)

        # Transport sits ABOVE the tabs: whatever you're configuring, the run/stop
        # state and the Start/Pause buttons stay visible.
        if transport is not None:
            root.addWidget(self._transport_group())

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._tab([self._features_group()]), "Captions")
        self._tabs.addTab(self._tab([self._audio_group()]), "Audio")
        self._tabs.addTab(self._tab([self._appearance_group(), self._overlay_group()]), "Overlay")
        self._tabs.addTab(self._tab([self._updates_group(), self._about_group()]), "Updates")
        # Reopen on the tab you left on — "how I leave it is how it re-opens".
        idx = int(getattr(self._settings, "settings_tab", 0) or 0)
        self._tabs.setCurrentIndex(idx if 0 <= idx < self._tabs.count() else 0)
        self._tabs.currentChanged.connect(lambda i: self._persist(settings_tab=int(i)))
        root.addWidget(self._tabs)

        self._check_done.connect(self._on_check_done)
        self._dl_progress.connect(self._on_dl_progress)
        self._dl_done.connect(self._on_dl_done)

        self._restart_note = QtWidgets.QLabel("")
        self._restart_note.setWordWrap(True)
        self._restart_note.setStyleSheet("color: #d08a30;")
        root.addWidget(self._restart_note)

        row = QtWidgets.QHBoxLayout()
        hint = QtWidgets.QLabel("Everything here saves as you change it.")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        row.addWidget(hint)
        row.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        root.addLayout(row)

    @staticmethod
    def _tab(widgets) -> QtWidgets.QWidget:
        """Wrap group boxes into a tab page, top-aligned."""
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        for w in widgets:
            v.addWidget(w)
        v.addStretch(1)
        return page

    def _about_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("About")
        v = QtWidgets.QVBoxLayout(g)
        lbl = QtWidgets.QLabel(
            "Live Captions runs entirely on this machine — audio, transcription and "
            "speaker detection never leave it.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(lbl)
        return g

    # ---- transport: run the captions without closing the app ----
    _STATE_TEXT = {
        "running":  ("● Captions running", "#3a8a4a"),
        "starting": ("◐ Starting…", "#d08a30"),
        "paused":   ("❚❚ Paused — model still loaded, resumes quickly", "#d08a30"),
        "stopped":  ("■ Stopped — model unloaded (frees video memory)", "#888888"),
        "error":    ("▲ Couldn't start — see the overlay", "#c04a3a"),
    }

    def _transport_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Captions")
        v = QtWidgets.QVBoxLayout(g)

        self._state_lbl = QtWidgets.QLabel("")
        v.addWidget(self._state_lbl)

        row = QtWidgets.QHBoxLayout()
        self._btn_start = QtWidgets.QPushButton("Start")
        self._btn_pause = QtWidgets.QPushButton("Pause")
        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_start.clicked.connect(lambda: self._transport.start())
        self._btn_pause.clicked.connect(self._on_pause_clicked)
        self._btn_stop.clicked.connect(lambda: self._transport.stop())
        for b in (self._btn_start, self._btn_pause, self._btn_stop):
            row.addWidget(b)
        v.addLayout(row)

        self._start_launch = QtWidgets.QCheckBox("Start captioning as soon as the app opens")
        self._start_launch.setChecked(bool(getattr(self._settings, "start_captions_on_launch", True)))
        self._start_launch.toggled.connect(lambda on: self._persist(start_captions_on_launch=on))
        v.addWidget(self._start_launch)

        self._transport.state_changed.connect(self._on_transport_state)
        self._on_transport_state(self._transport.state)
        return g

    def _on_pause_clicked(self) -> None:
        if self._transport.is_active:
            self._transport.pause()
        else:
            self._transport.start()          # the button doubles as Resume

    @QtCore.Slot(str)
    def _on_transport_state(self, state: str) -> None:
        text, colour = self._STATE_TEXT.get(state, (state, "#888888"))
        self._state_lbl.setText(text)
        self._state_lbl.setStyleSheet(f"color: {colour};")
        active = state in ("running", "starting")
        self._btn_start.setEnabled(not active)
        self._btn_stop.setEnabled(state != "stopped")
        self._btn_pause.setEnabled(state != "starting")
        self._btn_pause.setText("Pause" if active else "Resume")

    # ---- persistence + live apply helpers ----
    def _persist(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self._settings, k, v)
        save_settings(**kwargs)

    def _apply_appearance(self) -> None:
        if self._overlay is not None:
            self._overlay.apply_appearance()

    def pipeline_status(self, msg: str, *, warn: bool = False) -> None:
        """Called by the overlay when a live rebuild finishes (or fails), so the note
        reflects what actually happened instead of expiring on a guessed timer —
        loading the speaker model can take a while, and downloads it the first time."""
        self._restart_note.setStyleSheet("color: #c04a3a;" if warn else "color: #3a8a4a;")
        self._restart_note.setText(msg)
        if not warn:
            QtCore.QTimer.singleShot(4000, lambda: self._restart_note.setText(""))

    def _apply_pipeline(self, what: str) -> None:
        """A change that needs the capture/model pipeline rebuilt. Apply it live if
        there's a running pipeline to restart; otherwise say exactly what will apply
        on next launch (standalone `--settings` has nothing to restart)."""
        if self._on_restart is not None:
            self._restart_note.setStyleSheet("color: #d08a30;")
            self._restart_note.setText(
                f"Applying {what} — reloading… (the first time speaker colours are "
                f"enabled this downloads the speaker model, which can take a minute)"
                if what == "speaker colours" else f"Applying {what} change — reloading…")
            self._on_restart()
        else:
            self._restart_dirty = True
            self._restart_note.setText(
                f"↻ The {what} change will take effect the next time you start Live Captions.")

    def closeEvent(self, event) -> None:
        # This window is the app's main window: closing it quits everything.
        if self._quit_on_close:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.quit()
        event.accept()

    # ---- audio device ----
    def _audio_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Audio source")
        form = QtWidgets.QFormLayout(g)
        self._dev = QtWidgets.QComboBox()
        lbs, default_idx = _loopbacks()
        self._dev.addItem("Default output (auto)", userData=None)
        for lb in lbs:
            tag = "  ← default" if lb["index"] == default_idx else ""
            self._dev.addItem(f"{lb['name']} (index {lb['index']}){tag}", userData=lb)
        # preselect the saved device by name+ordinal
        saved = getattr(self._settings, "loopback_name", None)
        if saved:
            for i in range(1, self._dev.count()):
                if self._dev.itemData(i) and self._dev.itemData(i)["name"] == saved:
                    self._dev.setCurrentIndex(i)
                    break
        self._dev.currentIndexChanged.connect(self._on_device)
        form.addRow("Capture from:", self._dev)
        hint = QtWidgets.QLabel("Pick the output your audio actually plays through "
                                "(useful when two devices share a name).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(hint)
        return g

    def _on_device(self, i: int) -> None:
        lb = self._dev.itemData(i)
        if lb is None:
            self._persist(loopback_name=None, loopback_ordinal=0)
        else:
            from ..capture.devices import enumerate_loopbacks, name_ordinal
            import pyaudiowpatch as pa
            p = pa.PyAudio()
            try:
                ordinal = name_ordinal(enumerate_loopbacks(p), lb)
            finally:
                p.terminate()
            self._persist(loopback_name=lb["name"], loopback_ordinal=ordinal)
        self._apply_pipeline("audio device")

    # ---- appearance ----
    def _appearance_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Appearance")
        form = QtWidgets.QFormLayout(g)

        self._font = QtWidgets.QSpinBox()
        self._font.setRange(10, 60)
        self._font.setValue(int(self._settings.overlay_font_pt))
        self._font.valueChanged.connect(self._on_font)
        form.addRow("Text size:", self._font)

        self._color_btn = QtWidgets.QPushButton()
        self._color = QtGui.QColor(getattr(self._settings, "overlay_text_color", "#FFFFFF"))
        self._paint_color_btn()
        self._color_btn.clicked.connect(self._on_color)
        form.addRow("Text colour:", self._color_btn)

        self._opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._opacity.setRange(30, 100)
        self._opacity.setValue(int(self._settings.overlay_opacity * 100))
        self._opacity.valueChanged.connect(self._on_opacity)
        form.addRow("Opacity:", self._opacity)

        self._lines = QtWidgets.QSpinBox()
        self._lines.setRange(1, 8)
        self._lines.setValue(int(self._settings.overlay_max_lines))
        self._lines.valueChanged.connect(self._on_lines)
        form.addRow("Max lines:", self._lines)
        return g

    def _paint_color_btn(self) -> None:
        self._color_btn.setText(self._color.name())
        self._color_btn.setStyleSheet(
            f"background-color: {self._color.name()}; color: "
            f"{'#000' if self._color.lightness() > 128 else '#fff'};")

    def _on_font(self, v: int) -> None:
        self._persist(overlay_font_pt=v)
        self._apply_appearance()

    def _on_color(self) -> None:
        c = QtWidgets.QColorDialog.getColor(self._color, self, "Caption text colour")
        if c.isValid():
            self._color = c
            self._paint_color_btn()
            self._persist(overlay_text_color=c.name())
            self._apply_appearance()

    def _on_opacity(self, v: int) -> None:
        self._persist(overlay_opacity=v / 100.0)
        self._apply_appearance()

    def _on_lines(self, v: int) -> None:
        self._persist(overlay_max_lines=v)
        self._apply_appearance()

    # ---- overlay behaviour ----
    def _overlay_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Overlay")
        v = QtWidgets.QVBoxLayout(g)
        self._movable = QtWidgets.QCheckBox("Movable (drag to reposition; uncheck to click through)")
        self._movable.setChecked(bool(getattr(self._settings, "overlay_movable", False)))
        self._movable.toggled.connect(self._on_movable)
        v.addWidget(self._movable)
        reset = QtWidgets.QPushButton("Reset position to bottom-centre")
        reset.clicked.connect(self._on_reset_pos)
        v.addWidget(reset)

        self._open_launch = QtWidgets.QCheckBox("Open this Settings window when Live Captions starts")
        self._open_launch.setChecked(bool(getattr(self._settings, "open_settings_on_launch", True)))
        self._open_launch.toggled.connect(lambda on: self._persist(open_settings_on_launch=on))
        v.addWidget(self._open_launch)
        return g

    def _on_movable(self, on: bool) -> None:
        self._persist(overlay_movable=on)
        if self._overlay is not None:
            self._overlay.set_movable(on)

    def _on_reset_pos(self) -> None:
        if self._overlay is not None:
            self._overlay.reset_position()

    # ---- features (restart to apply) ----
    def _features_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Speech recognition")
        form = QtWidgets.QFormLayout(g)

        self._colors = QtWidgets.QCheckBox("Colour captions by speaker (live diarization)")
        self._colors.setChecked(bool(getattr(self._settings, "speaker_colors", False)))
        self._colors.toggled.connect(self._on_speaker_colors)
        form.addRow(self._colors)

        self._model = QtWidgets.QComboBox()
        self._model.addItems(_MODELS)
        cur = getattr(self._settings, "model_name", "medium")
        if cur in _MODELS:
            self._model.setCurrentText(cur)
        elif cur:
            self._model.addItem(cur)
            self._model.setCurrentText(cur)
        self._model.currentTextChanged.connect(self._on_model)
        form.addRow("Model:", self._model)
        hint = QtWidgets.QLabel("Smaller models start faster and use less; larger are more accurate.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(hint)
        return g

    def _on_speaker_colors(self, on: bool) -> None:
        self._persist(speaker_colors=on)
        self._apply_pipeline("speaker colours")

    def _on_model(self, name: str) -> None:
        self._persist(model_name=name)
        self._apply_pipeline("model")

    # ---- updates (upgrade the app; models/transcripts are kept) ----
    def _updates_group(self) -> QtWidgets.QGroupBox:
        from .. import updater
        g = QtWidgets.QGroupBox("Updates")
        v = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        self._ver = QtWidgets.QLabel(f"Version {updater.current_version()}")
        row.addWidget(self._ver)
        row.addStretch(1)
        self._check_btn = QtWidgets.QPushButton("Check for updates")
        self._check_btn.clicked.connect(self._check_updates)
        row.addWidget(self._check_btn)
        v.addLayout(row)
        note = QtWidgets.QLabel("Updating replaces the app and keeps your models and transcripts "
                                "(no big re-download).")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(note)
        return g

    def _check_updates(self) -> None:
        import threading
        from .. import updater
        self._check_btn.setEnabled(False)
        self._ver.setText("Checking for updates…")

        def _work():
            try:
                tag, url = updater.latest_release()
                self._check_done.emit({"tag": tag, "url": url, "err": None})
            except Exception as e:
                self._check_done.emit({"tag": None, "url": None, "err": type(e).__name__})
        threading.Thread(target=_work, daemon=True).start()

    def _on_check_done(self, res: dict) -> None:
        from .. import updater
        self._check_btn.setEnabled(True)
        if res["err"]:
            self._ver.setText(f"Couldn't check for updates ({res['err']})")
            return
        tag, url = res["tag"], res["url"]
        if not url or not updater.is_newer(tag):
            self._ver.setText(f"Up to date (version {updater.current_version()})")
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Update available",
            f"Version {tag} is available (you have {updater.current_version()}).\n\n"
            f"Download and install it now? Your models and transcripts are kept, and "
            f"Live Captions will close to finish installing.")
        if ans == QtWidgets.QMessageBox.StandardButton.Yes:
            self._start_download(url, tag)
        else:
            self._ver.setText(f"Update {tag} available")

    def _start_download(self, url: str, tag: str) -> None:
        import threading
        from .. import updater
        self._dlg = QtWidgets.QProgressDialog(f"Downloading {tag}…", None, 0, 100, self)
        self._dlg.setWindowTitle("Updating Live Captions")
        self._dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._dlg.setAutoClose(False)
        self._dlg.setValue(0)
        self._dlg.show()
        self._dl_path = None

        def _work():
            try:
                self._dl_path = updater.download(
                    url, on_progress=lambda f: self._dl_progress.emit(int(f * 100)))
                self._dl_done.emit("")
            except Exception as e:
                self._dl_done.emit(f"{type(e).__name__}: {e}")
        threading.Thread(target=_work, daemon=True).start()

    def _on_dl_progress(self, pct: int) -> None:
        if getattr(self, "_dlg", None) is not None:
            self._dlg.setValue(pct)

    def _on_dl_done(self, err: str) -> None:
        from .. import updater
        if getattr(self, "_dlg", None) is not None:
            self._dlg.close()
        if err:
            QtWidgets.QMessageBox.warning(self, "Update failed", f"The download failed:\n{err}")
            return
        updater.run_installer(self._dl_path)   # silent; upgrades in place, keeps models
        QtWidgets.QApplication.quit()           # close so the installer can replace our files


def run_settings(settings, screenshot_path: Optional[str] = None) -> None:
    """Open the settings window standalone (no capture). `livecaptions --settings`.
    Changes persist to config.toml and take effect on the next launch."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = SettingsWindow(settings, overlay=None)
    win.show()
    if screenshot_path:
        def _grab():
            win.grab().save(screenshot_path)
            app.quit()
        QtCore.QTimer.singleShot(400, _grab)
    app.exec()
