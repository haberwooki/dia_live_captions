"""Always-on-top, click-through, translucent caption overlay (PySide6).

Renders TranscriptEvents as a rounded caption "pill" at the bottom-centre of the
screen, over any application. Finals render solid; partials (is_final=False,
used by the M3 streaming path and the --demo generator) render dimmed and are
overwritten in place, then commit solid on the final — the render contract is
validated here so streaming "drops in" later.

Threading: sources call `bridge.emit_event` from a background thread; the Qt
signal marshals it to the GUI thread (auto QueuedConnection). All widget/paint
work stays on the GUI thread.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover - friendly message when the extra is missing
    raise SystemExit(
        "The overlay needs PySide6. Install it with:  pip install PySide6-Essentials\n"
        "(or `pip install -e .[gui]`). For a headless run, use the terminal sink.")

from ..events import TranscriptEvent

# ---- Win32 click-through (no-blink: toggles an ex-style bit, never recreates) ----
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020


def _set_click_through(hwnd: int, enabled: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    get = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    setw = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    get.restype = ctypes.c_longlong
    get.argtypes = [wintypes.HWND, ctypes.c_int]
    setw.restype = ctypes.c_longlong
    setw.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
    ex = get(hwnd, _GWL_EXSTYLE)
    if enabled:
        ex |= _WS_EX_LAYERED | _WS_EX_TRANSPARENT
    else:
        ex &= ~_WS_EX_TRANSPARENT
    setw(hwnd, _GWL_EXSTYLE, ex)


class CaptionBridge(QtCore.QObject):
    """Marshals TranscriptEvents from any thread onto the GUI thread."""

    event = QtCore.Signal(object)

    def emit_event(self, ev: TranscriptEvent) -> None:
        self.event.emit(ev)


#: per-speaker caption colours, assigned in first-seen order (Streaming Sortformer
#: caps at 4 speakers, so 4 distinct hues + a wrap-around is plenty).
SPEAKER_COLORS = [
    QtGui.QColor(130, 205, 255),   # blue
    QtGui.QColor(150, 245, 160),   # green
    QtGui.QColor(255, 205, 120),   # amber
    QtGui.QColor(255, 160, 205),   # pink
]


def _wrap(text: str, fm: QtGui.QFontMetrics, max_w: int) -> List[str]:
    """Greedy word-wrap `text` to `max_w` pixels using font metrics."""
    if not text:
        return []
    lines: List[str] = []
    cur = ""
    for word in text.split():
        trial = word if not cur else cur + " " + word
        if fm.horizontalAdvance(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class OverlayWindow(QtWidgets.QWidget):
    PADDING = 18
    RADIUS = 16
    MARGIN_BOTTOM = 70   # gap from the screen's bottom edge (logical px)

    def __init__(self, settings, *, source_name: str = "", movable: bool = False):
        super().__init__(None)
        self._settings = settings
        self._movable = movable
        self._source_name = source_name

        # each entry is (speaker_or_None, text) so turns can be coloured per speaker
        self._finals: "deque[Tuple[Optional[str], str]]" = deque(
            maxlen=max(6, settings.overlay_max_lines * 2))
        self._partial: Optional[Tuple[Optional[str], str]] = None
        self._speaker_slots: dict = {}      # speaker label -> colour index (first-seen order)
        self._paused = False
        self._status = "starting…"
        self._status_warn = False
        self._drag_offset: Optional[QtCore.QPoint] = None

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowOpacity(settings.overlay_opacity)

        self._font = QtGui.QFont("Segoe UI", settings.overlay_font_pt)
        self._font.setWeight(QtGui.QFont.Weight.DemiBold)

        self._qs = QtCore.QSettings("livecaptions", "overlay")
        self._offset = QtCore.QPoint(
            int(self._qs.value("offset_x", 0)), int(self._qs.value("offset_y", 0)))

        self.bridge = CaptionBridge()
        self.bridge.event.connect(self._on_event)   # auto QueuedConnection from worker threads
        self._last_block_t = time.monotonic()
        self._watch_screen()
        self._relayout()

    def _watch_screen(self) -> None:
        """Reposition when the screen changes. _relayout() reads the available
        geometry, and it used to be re-run ~3x/second by the health timer — so the
        overlay drifted back into place by accident after a resolution, DPI or
        monitor change. set_status now skips identical repaints, so that accident is
        gone and the screen has to be watched deliberately."""
        app = QtGui.QGuiApplication.instance()
        if app is None:
            return
        app.primaryScreenChanged.connect(lambda *_: self._relayout())
        for screen in app.screens():
            screen.availableGeometryChanged.connect(lambda *_: self._relayout())
            screen.logicalDotsPerInchChanged.connect(lambda *_: self._relayout())

    # ---- event/state (GUI thread) ----
    # ---- hotkey actions ----
    def toggle_visible(self) -> None:
        self.setVisible(not self.isVisible())

    def toggle_paused(self) -> None:
        """Pause = stop taking new captions (transcription keeps running)."""
        self._paused = not self._paused
        self.set_status("paused - press the toggle again to resume" if self._paused
                        else "listening…", warn=self._paused)

    def nudge(self, dx: int, dy: int) -> None:
        self._offset = QtCore.QPoint(self._offset.x() + dx, self._offset.y() + dy)
        self._qs.setValue("offset_x", self._offset.x())
        self._qs.setValue("offset_y", self._offset.y())
        self._relayout()

    @QtCore.Slot(object)
    def _on_event(self, ev: TranscriptEvent) -> None:
        if self._paused:
            return
        if ev.speaker and ev.speaker not in self._speaker_slots:
            self._speaker_slots[ev.speaker] = len(self._speaker_slots)
        entry = (ev.speaker, ev.text)
        if ev.is_final:
            self._finals.append(entry)
            self._partial = None
        else:
            self._partial = entry
        self._status_warn = False
        self._relayout()

    def _speaker_color(self, speaker: Optional[str]) -> QtGui.QColor:
        if not speaker:
            c = QtGui.QColor(getattr(self._settings, "overlay_text_color", "#FFFFFF"))
            return c if c.isValid() else QtGui.QColor(255, 255, 255)
        return QtGui.QColor(SPEAKER_COLORS[self._speaker_slots.get(speaker, 0) % len(SPEAKER_COLORS)])

    # ---- live settings apply (called from the Settings window) ----
    def apply_appearance(self) -> None:
        """Re-read appearance settings (size/colour/opacity/lines) and apply now."""
        self._font = QtGui.QFont("Segoe UI", int(self._settings.overlay_font_pt))
        self._font.setWeight(QtGui.QFont.Weight.DemiBold)
        self.setWindowOpacity(self._settings.overlay_opacity)
        self._relayout()

    def set_movable(self, movable: bool) -> None:
        self._movable = movable
        self.apply_click_through(not movable)   # movable == not click-through

    def reset_position(self) -> None:
        self._offset = QtCore.QPoint(0, 0)
        self._qs.setValue("offset_x", 0)
        self._qs.setValue("offset_y", 0)
        self._relayout()

    def set_status(self, text: str, warn: bool = False) -> None:
        # The health timer re-asserts the same status ~3x/second while no audio is
        # arriving, and each call used to relayout + repaint. That is the DEFAULT
        # state of a tray app nobody is speaking into, so it burned CPU for a frame
        # identical to the last one. Compare both fields: _on_event and the health
        # timer can differ only in `warn`.
        if text == self._status and warn == self._status_warn:
            return
        self._status = text
        self._status_warn = warn
        self._relayout()

    def note_block(self, rms: float) -> None:
        """Called from the segmenter thread (thread-safe float write)."""
        self._last_block_t = time.monotonic()

    def check_audio_health(self, is_live: bool) -> None:
        """Run on the GUI thread by a QTimer: surface a 'no audio' status."""
        if not is_live:
            return
        idle = time.monotonic() - self._last_block_t
        if idle > self._settings.no_blocks_warn_sec:
            self.set_status("no audio — is that the right output? try --device", warn=True)
        elif self._status_warn:
            self.set_status("listening…", warn=False)

    # ---- layout + paint ----
    def _target_screen(self) -> QtGui.QScreen:
        return QtWidgets.QApplication.primaryScreen()

    def _display_rows(self):
        """Rows to draw, newest-last: (kind, speaker, line). Each speaker turn is
        wrapped as its own block so turns stay visually separate and colourable."""
        fm = QtGui.QFontMetrics(self._font)
        scr = self._target_screen().availableGeometry()
        max_w = int(scr.width() * self._settings.overlay_width_frac) - 2 * self.PADDING

        if not self._finals and self._partial is None:
            return [("dim", None, ln) for ln in _wrap(self._status, fm, max_w)]

        rows = []
        for speaker, text in self._finals:
            label = f"{speaker}: " if speaker else ""
            for ln in _wrap(label + text, fm, max_w):
                rows.append(("solid", speaker, ln))
        if self._partial is not None:
            speaker, text = self._partial
            label = f"{speaker}: " if speaker else ""
            for ln in _wrap(label + text, fm, max_w):
                rows.append(("dim", speaker, ln))

        rows = rows[-self._settings.overlay_max_lines:]
        if self._status_warn:
            rows += [("dim", None, ln) for ln in _wrap(self._status, fm, max_w)]
        return rows

    def _relayout(self) -> None:
        fm = QtGui.QFontMetrics(self._font)
        rows = self._display_rows()
        if not rows:
            rows = [("dim", None, "…")]
        line_h = fm.height()
        text_w = max(fm.horizontalAdvance(ln) for _, _, ln in rows)
        w = text_w + 2 * self.PADDING + 6   # +6: allowance for the 3px text outline stroke
        h = line_h * len(rows) + 2 * self.PADDING

        scr = self._target_screen().availableGeometry()
        cx = scr.center().x() + self._offset.x()
        bottom = scr.bottom() - self.MARGIN_BOTTOM + self._offset.y()
        x = int(cx - w / 2)
        y = int(bottom - h)
        self.setGeometry(x, y, int(w), int(h))
        self._cache = (rows, line_h)
        self.update()

    def paintEvent(self, _event) -> None:
        rows, line_h = self._cache
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        # pill background
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 185))
        p.drawRoundedRect(self.rect(), self.RADIUS, self.RADIUS)

        p.setFont(self._font)
        fm = QtGui.QFontMetrics(self._font)
        y = self.PADDING + fm.ascent()
        for kind, speaker, line in rows:
            colour = self._speaker_color(speaker)
            if kind == "dim":                      # unconfirmed partial -> same hue, dimmed
                colour = QtGui.QColor(colour)
                colour.setAlpha(200)
                colour = colour.darker(115)
            self._draw_outlined(p, self.PADDING, y, line, fill=colour)
            y += line_h
        p.end()

    def _draw_outlined(self, p: QtGui.QPainter, x: int, baseline: int, text: str,
                       fill: QtGui.QColor) -> None:
        path = QtGui.QPainterPath()
        path.addText(float(x), float(baseline), self._font, text)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 230), 3))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(path)                       # outline for legibility over any content
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawPath(path)                       # fill

    # ---- click-through / dragging ----
    def apply_click_through(self, enabled: bool) -> None:
        _set_click_through(int(self.winId()), enabled)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._movable and e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._drag_offset is not None:
            self.move(e.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, _e: QtGui.QMouseEvent) -> None:
        if self._drag_offset is not None:
            self._drag_offset = None
            self._persist_position()

    def _persist_position(self) -> None:
        scr = self._target_screen().availableGeometry()
        # store the pill's bottom-centre offset from the default anchor
        self._offset = QtCore.QPoint(
            self.geometry().center().x() - scr.center().x(),
            self.geometry().bottom() - (scr.bottom() - self.MARGIN_BOTTOM))
        self._qs.setValue("offset_x", self._offset.x())
        self._qs.setValue("offset_y", self._offset.y())


class _SourceBuilder(QtCore.QObject):
    """Builds the source (which loads the Whisper model) OFF the GUI thread so the
    overlay can appear immediately with a 'loading' status. On first run that load
    includes a ~1.5 GB download that used to block for ~90 s with no window shown."""

    ready = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def build(self, factory: Callable[[], object]) -> None:
        def _work():
            try:
                source = factory()
            except BaseException as e:   # load_model raises SystemExit on total failure
                self.failed.emit(str(e) or e.__class__.__name__)
                return
            self.ready.emit(source)
        threading.Thread(target=_work, daemon=True).start()


class Transport(QtCore.QObject):
    """Start / pause / stop the capture+ASR pipeline without quitting the app.

    The states differ in what they release, which is what makes them worth having
    as separate controls:
      running  - capturing and transcribing
      starting - building the pipeline (model load / device open)
      paused   - capture and inference stopped, model still resident: resumes fast
      stopped  - as paused, and the model is released (frees ~2 GB of VRAM)
      error    - the last start failed; the message is on the overlay
    """
    state_changed = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._state = "starting"
        self.start = self.pause = self.stop = lambda: None   # wired by run_overlay

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state in ("running", "starting")

    def _set(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)


def run_overlay(source_factory: Callable[[], object], settings, *, source_name: str,
                is_live: bool, movable: bool = False,
                screenshot_path: Optional[str] = None, extra_sink=None,
                on_release_model=None) -> None:
    """Own the Qt event loop (main thread). Shows the overlay first, then builds
    the source (model load) off-thread so there's always a visible status."""
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)   # the tray keeps us alive even if the overlay is hidden

    # The saved preference wins unless --movable forced it on for this run.
    movable = bool(movable or getattr(settings, "overlay_movable", False))
    overlay = OverlayWindow(settings, source_name=source_name, movable=movable)
    overlay.set_status("starting…")
    overlay.show()
    if not movable:
        overlay.apply_click_through(True)   # after the native window exists

    # Created before the tray and the Settings window, which both drive it.
    transport = Transport()

    # --- system tray: the only visible way to quit, and the "it's running" signal ---
    tray = None
    settings_win = {"w": None}

    def _open_settings():
        from .settings import SettingsWindow
        if settings_win["w"] is None:
            # This is the app's main window: closing it quits everything. on_restart
            # applies device/model/speaker changes live (no restart).
            settings_win["w"] = SettingsWindow(settings, overlay, quit_on_close=True,
                                               on_restart=_restart_pipeline,
                                               transport=transport)
        settings_win["w"].show()
        settings_win["w"].raise_()
        settings_win["w"].activateWindow()

    if not screenshot_path:
        from .tray import install_tray
        tray = install_tray(app, overlay, on_quit=app.quit, on_settings=_open_settings,
                            transport=transport)

    # --- global hotkeys (the overlay is click-through, so Qt shortcuts can't work) ---
    hotkeys = None
    if settings.hotkeys_enabled and not screenshot_path and sys.platform == "win32":
        from .hotkeys import GlobalHotkeys
        px = settings.hotkey_nudge_px
        hotkeys = GlobalHotkeys(int(overlay.winId()))
        app.installNativeEventFilter(hotkeys)
        bound = [
            hotkeys.register(settings.hotkey_toggle, overlay.toggle_visible, "show/hide"),
            # Pause now really stops capture+inference (and frees the GPU), rather
            # than freezing a pipeline that keeps burning power behind a still overlay.
            hotkeys.register(settings.hotkey_pause,
                             lambda: transport.pause() if transport.is_active else transport.start(),
                             "pause/resume"),
            hotkeys.register(settings.hotkey_quit, app.quit, "quit"),
            hotkeys.register(settings.hotkey_left, lambda: overlay.nudge(-px, 0), "move left"),
            hotkeys.register(settings.hotkey_right, lambda: overlay.nudge(px, 0), "move right"),
            hotkeys.register(settings.hotkey_up, lambda: overlay.nudge(0, -px), "move up"),
            hotkeys.register(settings.hotkey_down, lambda: overlay.nudge(0, px), "move down"),
        ]
        if any(bound):
            print(f"Hotkeys: {settings.hotkey_toggle} show/hide, {settings.hotkey_pause} pause, "
                  f"{settings.hotkey_quit} quit, ctrl+alt+arrows move")

    # --- screenshot mode: render a canned partial+final sequence and save (no model) ---
    if screenshot_path:
        overlay._on_event(TranscriptEvent("did you get a chance to look at the results?",
                                          source="demo", t_start=0, t_end=1,
                                          is_final=True, speaker="SPEAKER_00"))
        overlay._on_event(TranscriptEvent("I did, the speaker separation looked good.",
                                          source="demo", t_start=1, t_end=2,
                                          is_final=True, speaker="SPEAKER_01"))
        overlay._on_event(TranscriptEvent("and this part is still unconfirmed",
                                          source="demo", t_start=2, t_end=3,
                                          is_final=False, speaker="SPEAKER_00"))

        def _grab():
            overlay.grab().save(screenshot_path)
            app.quit()
        QtCore.QTimer.singleShot(400, _grab)
        app.exec()
        return

    holder = {"src": None, "restarting": False}

    def _tell_settings(msg: str, warn: bool = False) -> None:
        """Report a live-rebuild outcome into the Settings window, if it's open."""
        w = settings_win.get("w")
        if w is not None and w.isVisible():
            w.pipeline_status(msg, warn=warn)

    # --- runs on the GUI thread once the model is loaded and the source is built ---
    def _start_source(source) -> None:
        for t in getattr(overlay, "_timers", ()):   # on a live rebuild, drop the old timers
            t.stop()
        holder["src"] = source
        overlay.set_status("listening — play some audio")
        transport._set("running")
        if holder["restarting"]:
            holder["restarting"] = False
            _tell_settings("✓ Applied — captions are running again.")
        on_event = overlay.bridge.emit_event
        if extra_sink is not None:                 # e.g. the transcript writer
            def on_event(ev, _bridge=overlay.bridge.emit_event, _extra=extra_sink):
                _bridge(ev)
                _extra(ev)
        source.start(on_event=on_event, monitor=(overlay.note_block if is_live else None))

        health = QtCore.QTimer(overlay)
        health.timeout.connect(lambda: overlay.check_audio_health(is_live))
        health.start(300)

        # Quit when a finite source (WAV/demo) finishes, holding the last captions
        # a few seconds so they're readable; a live source only ends via quit.
        hold = {"deadline": None}

        def _check_done():
            if holder["src"] is not source:
                return          # superseded by a live restart — its "finished" is not ours to act on
            if not source.finished.is_set():
                return
            if is_live:
                app.quit()
            elif hold["deadline"] is None:
                hold["deadline"] = time.monotonic() + 4.0
            elif time.monotonic() >= hold["deadline"]:
                app.quit()

        done = QtCore.QTimer(overlay)
        done.timeout.connect(_check_done)
        done.start(250)
        overlay._timers = (health, done)   # keep refs alive

    def _on_failed(msg: str) -> None:
        overlay.set_status(f"couldn't start: {msg[:110]} — press {settings.hotkey_pause} "
                           f"or use Settings to try again", warn=True)
        transport._set("error")
        if holder["restarting"]:
            holder["restarting"] = False
            _tell_settings(f"Couldn't apply that change: {msg[:200]}\n"
                           "Captions are stopped — undo it to start again.", warn=True)

    builder = _SourceBuilder()
    builder.ready.connect(_start_source)
    builder.failed.connect(_on_failed)
    # Whether captions run the moment the app opens is the user's call, not ours.
    if getattr(settings, "start_captions_on_launch", True) or not is_live:
        overlay.set_status("loading model… (first run downloads ~1.5 GB, please wait)")
        builder.build(source_factory)
    else:
        transport._set("stopped")
        overlay.set_status("stopped — click Start in Settings to begin", warn=True)

    def _detach_source():
        """Drop the running source and its timers, returning the old source to be
        stopped off-thread. Timers go FIRST: stopping a source sets its `finished`
        event, and a still-running done-timer would read that on a live source and
        quit the whole app."""
        for t in getattr(overlay, "_timers", ()):
            t.stop()
        overlay._timers = ()
        old = holder["src"]
        holder["src"] = None
        return old

    def _rebuild(status: str) -> None:
        """Stop the current source (if any) and build a fresh one from current
        settings, off the GUI thread. Used for both settings changes and Start."""
        old = _detach_source()
        holder["restarting"] = True
        transport._set("starting")
        overlay.set_status(status)

        def _work():
            if old is not None:
                try:
                    old.stop()
                except Exception:
                    pass
            try:
                source = source_factory()
            except BaseException as e:
                builder.failed.emit(str(e) or e.__class__.__name__)
                return
            builder.ready.emit(source)
        threading.Thread(target=_work, daemon=True).start()

    def _halt(state: str, status: str) -> None:
        """Stop capturing/transcribing. 'stopped' also releases the model (VRAM);
        'paused' keeps it resident so resuming is quick."""
        old = _detach_source()
        holder["restarting"] = False
        transport._set(state)
        overlay.set_status(status, warn=True)

        def _work():
            if old is not None:
                try:
                    old.stop()
                except Exception:
                    pass
            if state == "stopped" and on_release_model is not None:
                try:
                    on_release_model()
                except Exception:
                    pass
        threading.Thread(target=_work, daemon=True).start()

    def _restart_pipeline() -> None:
        _rebuild("applying settings — reloading…")

    transport.start = lambda: (None if transport.is_active
                               else _rebuild("starting…"))
    transport.pause = lambda: (_halt("paused", "paused — click Resume to start again")
                               if transport.is_active else None)
    transport.stop = lambda: (None if transport.state == "stopped"
                              else _halt("stopped", "stopped — click Start to begin"))

    # On launch, make the app discoverable: a tray notification (so you know it's
    # running and where Settings is) and, by default, open the Settings window so
    # the GUI is front-and-centre instead of hidden behind a tray right-click.
    def _on_launch():
        if tray is not None:
            tray.showMessage(
                "Live Captions",
                "Running in the tray — right-click the icon for Settings or Quit.",
                QtWidgets.QSystemTrayIcon.MessageIcon.Information, 6000)
        if getattr(settings, "open_settings_on_launch", True):
            _open_settings()
    QtCore.QTimer.singleShot(300, _on_launch)   # after the loop starts, so overlay/tray render first

    # let the interpreter process Ctrl+C while the Qt loop runs
    import signal
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    tick = QtCore.QTimer()
    tick.timeout.connect(lambda: None)
    tick.start(200)

    app.exec()
    if hotkeys is not None:
        hotkeys.unregister_all()
    if holder["src"] is not None:
        holder["src"].stop()
