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
    def __init__(self, settings, overlay=None):
        super().__init__(None)
        self._settings = settings
        self._overlay = overlay
        self._restart_dirty = False
        self.setWindowTitle("Live Captions — Settings")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(420)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self._audio_group())
        root.addWidget(self._appearance_group())
        root.addWidget(self._overlay_group())
        root.addWidget(self._features_group())

        self._restart_note = QtWidgets.QLabel("")
        self._restart_note.setStyleSheet("color: #d08a30;")
        root.addWidget(self._restart_note)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        root.addLayout(row)

    # ---- persistence + live apply helpers ----
    def _persist(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self._settings, k, v)
        save_settings(**kwargs)

    def _apply_appearance(self) -> None:
        if self._overlay is not None:
            self._overlay.apply_appearance()

    def _mark_restart(self) -> None:
        self._restart_dirty = True
        self._restart_note.setText("↻ Some changes apply the next time you start Live Captions.")

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
        self._mark_restart()

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
        self._movable.setChecked(bool(getattr(self._overlay, "_movable", False)))
        self._movable.toggled.connect(self._on_movable)
        v.addWidget(self._movable)
        reset = QtWidgets.QPushButton("Reset position to bottom-centre")
        reset.clicked.connect(self._on_reset_pos)
        v.addWidget(reset)
        return g

    def _on_movable(self, on: bool) -> None:
        if self._overlay is not None:
            self._overlay.set_movable(on)

    def _on_reset_pos(self) -> None:
        if self._overlay is not None:
            self._overlay.reset_position()

    # ---- features (restart to apply) ----
    def _features_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Captions")
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
        self._mark_restart()

    def _on_model(self, name: str) -> None:
        self._persist(model_name=name)
        self._mark_restart()


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
