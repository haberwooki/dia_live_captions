"""Advanced tab: the tuning knobs that used to be config.toml-only, plus hotkey remapping.

Three rules shape this tab:

  * every knob says what it *does* and what goes wrong if you move it. A number
    with no explanation is worse than no control at all — people change it, the
    captions get worse, and they can't tell which change did it;
  * recognition/streaming values are read when the capture pipeline is built, so
    changing one needs a rebuild. Saving is cheap and happens per keystroke;
    rebuilding is not, so the restart is debounced;
  * hotkeys are validated before they are stored. A combo that cannot be parsed
    is silently dead at runtime (registration just fails and prints), which is
    exactly the state this tab exists to get people out of.

Nothing here touches the transcript store or the network.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from ..config import Settings, save_settings
from .hotkeys import MOD_NOREPEAT, parse_hotkey

# Ordered so the normalised form of a combo is stable: "ctrl+alt+shift+win+key".
_MOD_ORDER = ("ctrl", "alt", "shift", "win")
_MOD_ALIASES = {"control": "ctrl", "super": "win", "windows": "win", "meta": "win"}

# ct2 compute types that actually exist. Anything else fails at model-load time
# with a message most people can't act on, so don't let it be typed.
_GPU_COMPUTE = ["float16", "int8_float16", "bfloat16", "float32"]
_CPU_COMPUTE = ["int8", "int8_float32", "float32", "bfloat16"]

_COMMON_LANGS = ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk",
                 "sv", "tr", "ar", "hi", "ja", "ko", "zh"]

#: field, label, description shown under the control
_HOTKEY_ACTIONS: Tuple[Tuple[str, str], ...] = (
    ("hotkey_toggle", "Show / hide captions"),
    ("hotkey_pause", "Pause / resume"),
    ("hotkey_quit", "Quit Live Captions"),
    ("hotkey_left", "Move overlay left"),
    ("hotkey_right", "Move overlay right"),
    ("hotkey_up", "Move overlay up"),
    ("hotkey_down", "Move overlay down"),
)

_RECOGNITION_FIELDS = ("language", "beam_size", "gpu_compute", "cpu_compute",
                       "block_sec", "silence_rms_floor")
_STREAMING_FIELDS = ("stream_process_interval", "stream_end_silence_sec",
                     "stream_max_line_sec", "stream_max_buffer_sec",
                     "stream_vad_threshold")

#: every field this tab owns — the set "Reset to defaults" restores
OWNED_FIELDS: Tuple[str, ...] = (
    _RECOGNITION_FIELDS + _STREAMING_FIELDS
    + tuple(f for f, _ in _HOTKEY_ACTIONS) + ("hotkey_nudge_px", "hotkeys_enabled"))


def model_default(field: str):
    """The pydantic default for a field. The Settings model is the only copy of
    the defaults; duplicating them here is how a 'reset' quietly starts lying."""
    return Settings.model_fields[field].default


def normalize_hotkey(spec: str) -> str:
    """Canonical form of a combo, e.g. ' ALT + Shift+K ' -> 'alt+shift+k'.

    Raises ValueError with a message meant for a human if the combo can't work:
    unknown key, no key, more than one key, or no modifier. Windows accepts a
    bare key, but a global hotkey on plain 'C' steals that key from every other
    app on the machine, so it is treated as a mistake.
    """
    if not spec or not spec.strip():
        raise ValueError("type a combination, e.g. ctrl+alt+c")
    mods: List[str] = []
    keys: List[str] = []
    for raw in spec.split("+"):
        part = raw.strip().lower()
        part = _MOD_ALIASES.get(part, part)
        if not part:
            raise ValueError(f"{spec.strip()!r} has an empty part — check the + signs")
        if part in _MOD_ORDER:
            if part not in mods:
                mods.append(part)
        else:
            keys.append(part)
    if not keys:
        raise ValueError("that is only modifier keys — add a key, e.g. ctrl+alt+c")
    if len(keys) > 1:
        raise ValueError(f"more than one key ({', '.join(keys)}) — a hotkey can only have one")
    if not mods:
        raise ValueError(f"{keys[0]!r} on its own would be taken from every other app — "
                         f"add ctrl, alt, shift or win")
    out = "+".join([m for m in _MOD_ORDER if m in mods] + keys)
    parse_hotkey(out)      # authoritative check: same parser the app registers with
    return out


def probe_hotkey(spec: str) -> str:
    """'free' | 'in use' | 'unknown' — ask Windows whether the combo is available.

    This registers the hotkey for an instant and releases it. Note the result is
    'in use' when *this* app is the one holding it, so it means "someone has it",
    not "it is broken" — the UI says as much.
    """
    try:
        mods, vk = parse_hotkey(spec)
    except ValueError:
        return "unknown"
    if sys.platform != "win32":
        return "unknown"
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hid = 0xBFF0        # top of the app-owned id range; ours are numbered from 1
        if not user32.RegisterHotKey(wintypes.HWND(0), hid, mods | MOD_NOREPEAT, vk):
            return "in use"
        user32.UnregisterHotKey(wintypes.HWND(0), hid)
        return "free"
    except Exception:
        return "unknown"


def _fmt(value: float) -> str:
    """Numbers as a person writes them: 40 not 40.0, 2.5 not 2.500000."""
    return f"{value:g}"


def _hint(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: gray; font-size: 11px;")
    return lbl


class AdvancedTab(QtWidgets.QWidget):
    """Tuning knobs + hotkey remapping.

    `on_restart` (same callable ui/settings.py passes around) rebuilds the live
    pipeline. When it is None there is nothing running to rebuild, and the tab
    says the change applies next launch instead of pretending it took effect.

    `registered` is optional: {field: bool} of what the running app actually
    managed to register, which is the only way to tell "another app has it" from
    "we have it, and it works".
    """

    #: how long to wait after the last edit before rebuilding the pipeline
    RESTART_DELAY_MS = 1200

    def __init__(self, settings, parent=None, on_restart=None,
                 registered: Optional[Dict[str, bool]] = None):
        super().__init__(parent)
        self._settings = settings
        self._on_restart = on_restart
        self._registered = dict(registered or {})
        # What each registration result was measured against. A remap makes the
        # result meaningless, and a stale "Working now." against a combo the app
        # never tried is worse than saying nothing.
        self._registered_specs: Dict[str, str] = {
            field: str(getattr(settings, field, model_default(field)))
            for field in self._registered}
        self._loading = True            # suppress persistence while we set widgets
        self._pending_restart: Optional[str] = None
        self._hotkey_edits: Dict[str, QtWidgets.QLineEdit] = {}
        self._hotkey_notes: Dict[str, QtWidgets.QLabel] = {}
        self._probed = False
        #: values found on disk that no widget can represent, reported not hidden
        self._out_of_range: List[str] = []

        self._restart_timer = QtCore.QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.timeout.connect(self._do_restart)

        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.addWidget(_hint(
            "These are the settings the app used to keep only in config.toml. The "
            "defaults are what was measured to work; change one thing at a time."))
        v.addWidget(self._recognition_group())
        v.addWidget(self._streaming_group())
        v.addWidget(self._hotkeys_group())
        v.addStretch(1)
        scroll.setWidget(page)
        outer.addWidget(scroll)

        self._range_note = QtWidgets.QLabel("")
        self._range_note.setWordWrap(True)
        self._range_note.setStyleSheet("color: #c04a3a; font-size: 11px;")
        self._range_note.setVisible(False)
        outer.addWidget(self._range_note)

        self._note = QtWidgets.QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: #d08a30;")
        outer.addWidget(self._note)

        row = QtWidgets.QHBoxLayout()
        # The window footer already says settings save as you change them; repeating
        # it directly above that line just looks like a rendering bug.
        row.addWidget(_hint("Changing recognition settings restarts captions."), 1)
        reset = QtWidgets.QPushButton("Reset to defaults")
        reset.clicked.connect(self._on_reset)
        row.addWidget(reset)
        outer.addLayout(row)

        self._loading = False
        self._show_out_of_range()
        self._check_buffer_vs_line()

    # ---- values the widgets were not built for -----------------------------
    def _fit_spin(self, widget, value: float, label: str, unit: str = ""):
        """Widen `widget` so it can show `value`, and remember that it had to.

        Clamping to the widget's range would put a number on screen that is not
        the number the pipeline is running with, and the user believes the screen.
        """
        lo, hi = widget.minimum(), widget.maximum()
        if lo <= value <= hi:
            return value
        self._out_of_range.append(
            f"{label} is {_fmt(value)}{unit}, outside the supported "
            f"{_fmt(lo)}{unit}–{_fmt(hi)}{unit}")
        widget.setRange(min(lo, value), max(hi, value))
        return value

    def _fit_choice(self, combo: QtWidgets.QComboBox, value: str, label: str,
                    allowed: List[str]) -> str:
        """Same for a fixed list: an unknown value is added so it can be shown."""
        if value in allowed:
            return value
        self._out_of_range.append(
            f"{label} is “{value}”, which is not one of {', '.join(allowed)}")
        combo.addItem(value)
        return value

    def _show_out_of_range(self) -> None:
        if not self._out_of_range:
            self._range_note.setVisible(False)
            self._range_note.setText("")
            return
        self._range_note.setVisible(True)
        self._range_note.setText(
            "Your config file has values this app was not tuned for, shown here "
            "exactly as the app is using them: " + "; ".join(self._out_of_range)
            + ". They keep working; pick a supported value if something misbehaves.")

    # ---- persistence -------------------------------------------------------
    def _persist(self, **kw) -> None:
        if self._loading:
            return
        for k, val in kw.items():
            setattr(self._settings, k, val)
        save_settings(**kw)

    def _persist_and_rebuild(self, what: str, **kw) -> None:
        """A value the capture/model pipeline reads at build time."""
        if self._loading:
            return
        self._persist(**kw)
        self._pending_restart = what
        if self._on_restart is not None:
            self._note.setText(f"Applying {what} — reloading…")
            self._restart_timer.start(self.RESTART_DELAY_MS)
        else:
            self._note.setText(f"↻ The {what} change takes effect the next time you "
                               f"start Live Captions.")

    def _do_restart(self) -> None:
        what, self._pending_restart = self._pending_restart, None
        if what is None or self._on_restart is None:
            return
        self._on_restart()

    def _needs_app_restart(self, what: str) -> None:
        """Hotkeys are claimed once, at startup — a rebuild of the audio pipeline
        does not re-register them, so never promise more than a relaunch gives."""
        self._note.setText(f"↻ {what} applies the next time you start Live Captions.")

    # ---- recognition -------------------------------------------------------
    def _recognition_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Recognition")
        form = QtWidgets.QFormLayout(g)

        self._lang = QtWidgets.QComboBox()
        self._lang.setEditable(True)
        self._lang.addItems(_COMMON_LANGS)
        cur = str(getattr(self._settings, "language", "en") or "en")
        self._lang.setCurrentText(cur)
        self._lang.activated.connect(lambda _i: self._on_language())
        self._lang.lineEdit().editingFinished.connect(self._on_language)
        form.addRow("Language:", self._lang)
        self._lang_note = _hint(
            "The language Whisper is told to expect. Set it wrong and you get "
            "confident nonsense or silent translation rather than an error. "
            "(There is no auto-detect: a wrong guess part-way through a session is "
            "worse than a fixed choice.)")
        form.addRow(self._lang_note)

        self._beam = QtWidgets.QSpinBox()
        self._beam.setRange(1, 10)
        self._beam.setValue(int(self._fit_spin(
            self._beam, int(getattr(self._settings, "beam_size", 1) or 1), "Beam size")))
        self._beam.valueChanged.connect(
            lambda v: self._persist_and_rebuild("beam size", beam_size=int(v)))
        form.addRow("Beam size:", self._beam)
        form.addRow(_hint(
            "How many wordings the decoder keeps in play. 1 is greedy and fastest; "
            "higher is slightly more accurate and much slower, and if decoding stops "
            "keeping up, captions fall behind the audio and stay behind."))

        self._gpu = QtWidgets.QComboBox()
        self._gpu.addItems(_GPU_COMPUTE)
        self._gpu.setCurrentText(self._fit_choice(
            self._gpu, str(getattr(self._settings, "gpu_compute", "float16")),
            "GPU precision", _GPU_COMPUTE))
        self._gpu.currentTextChanged.connect(
            lambda t: self._persist_and_rebuild("GPU precision", gpu_compute=t))
        form.addRow("GPU precision:", self._gpu)

        self._cpu = QtWidgets.QComboBox()
        self._cpu.addItems(_CPU_COMPUTE)
        self._cpu.setCurrentText(self._fit_choice(
            self._cpu, str(getattr(self._settings, "cpu_compute", "int8")),
            "CPU precision", _CPU_COMPUTE))
        self._cpu.currentTextChanged.connect(
            lambda t: self._persist_and_rebuild("CPU precision", cpu_compute=t))
        form.addRow("CPU precision:", self._cpu)
        form.addRow(_hint(
            "How model weights are stored while running. Smaller types (int8) use "
            "less memory and run faster, and lose a little accuracy. If your GPU "
            "refuses the type you pick, the app falls back to the CPU — which is "
            "much slower, so a bad choice here looks like 'captions got laggy'."))

        self._block = QtWidgets.QDoubleSpinBox()
        self._block.setRange(0.02, 1.0)
        self._block.setSingleStep(0.02)
        self._block.setDecimals(2)
        self._block.setSuffix(" s")
        self._block.setValue(self._fit_spin(
            self._block, float(getattr(self._settings, "block_sec", 0.1)),
            "Audio block", " s"))
        self._block.valueChanged.connect(
            lambda v: self._persist_and_rebuild("audio block size", block_sec=float(v)))
        form.addRow("Audio block:", self._block)
        form.addRow(_hint(
            "How much sound is taken from Windows at a time. Smaller reacts sooner "
            "but wakes the CPU more often; larger is steadier and adds its own length "
            "as delay to everything after it."))

        self._floor = QtWidgets.QDoubleSpinBox()
        self._floor.setRange(0.0, 1000.0)
        self._floor.setDecimals(1)
        self._floor.setValue(self._fit_spin(
            self._floor, float(getattr(self._settings, "silence_rms_floor", 5.0)),
            "Silence floor"))
        self._floor.valueChanged.connect(
            lambda v: self._persist_and_rebuild("silence floor", silence_rms_floor=float(v)))
        form.addRow("Silence floor:", self._floor)
        form.addRow(_hint(
            "Level under which a block counts as silence, used only for the "
            "'no audio' warning. Raise it if a hissy device never looks silent; "
            "lower it if you are told there is no audio while you can hear it."))
        return g

    def _on_language(self) -> None:
        code = self._lang.currentText().strip().lower()
        stored = str(getattr(self._settings, "language", "en") or "en")
        if not _valid_language(code):
            self._lang_note.setText(
                f"“{code}” is not a language Whisper knows — keeping "
                f"{stored}. Use a code like en, de, ja.")
            self._lang_note.setStyleSheet("color: #c04a3a; font-size: 11px;")
            # Leaving the rejected text on screen claims a language nobody stored.
            self._lang.setCurrentText(stored)
            return
        self._lang_note.setStyleSheet("color: gray; font-size: 11px;")
        if code != stored:
            self._persist_and_rebuild("language", language=code)

    # ---- streaming ---------------------------------------------------------
    def _streaming_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Streaming (live partial captions)")
        form = QtWidgets.QFormLayout(g)

        def spin(field: str, lo: float, hi: float, step: float, what: str):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(2)
            w.setSuffix(" s")
            w.setValue(self._fit_spin(
                w, float(getattr(self._settings, field, model_default(field))),
                what.capitalize(), " s"))
            w.valueChanged.connect(
                lambda v, f=field, n=what: self._on_stream_value(f, n, float(v)))
            return w

        self._interval = spin("stream_process_interval", 0.1, 3.0, 0.1, "update rate")
        form.addRow("Update every:", self._interval)
        form.addRow(_hint(
            "Seconds of fresh audio between decode passes. Lower makes the "
            "unconfirmed line move more often and costs GPU on every pass; too low "
            "and the decoder never finishes before the next pass is due."))

        self._endsil = spin("stream_end_silence_sec", 0.1, 5.0, 0.1, "end-of-line pause")
        form.addRow("Finish a line after:", self._endsil)
        form.addRow(_hint(
            "How long a pause has to last before the current line is locked in and "
            "saved. Shorter chops sentences at breaths; longer leaves text "
            "provisional (and unsaved) for longer."))

        self._maxline = spin("stream_max_line_sec", 1.0, 60.0, 1.0, "maximum line length")
        form.addRow("Force a new line after:", self._maxline)
        form.addRow(_hint(
            "A line is finished at this age even if nobody pauses, so someone who "
            "never stops talking still produces readable lines."))

        self._maxbuf = spin("stream_max_buffer_sec", 2.0, 60.0, 1.0, "buffer length")
        form.addRow("Audio kept in view:", self._maxbuf)
        self._buf_hint = (
            "How much recent audio the decoder may re-read each pass. More context "
            "means steadier wording and more work per pass. Keep it comfortably "
            "above the force-new-line time, or lines get cut off mid-thought.")
        self._buf_note = _hint(self._buf_hint)
        form.addRow(self._buf_note)

        self._vad = QtWidgets.QDoubleSpinBox()
        self._vad.setRange(0.05, 0.95)
        self._vad.setSingleStep(0.05)
        self._vad.setDecimals(2)
        self._vad.setValue(self._fit_spin(
            self._vad, float(getattr(self._settings, "stream_vad_threshold", 0.5)),
            "Speech threshold"))
        self._vad.valueChanged.connect(
            lambda v: self._on_stream_value("stream_vad_threshold", "speech threshold", float(v)))
        form.addRow("Speech threshold:", self._vad)
        form.addRow(_hint(
            "How sure the voice detector must be that a block contains speech. "
            "Raise it if background noise produces phantom captions; lower it if "
            "quiet or distant speech is being missed."))
        return g

    def _on_stream_value(self, field: str, what: str, value: float) -> None:
        self._persist_and_rebuild(what, **{field: value})
        self._check_buffer_vs_line()

    def _check_buffer_vs_line(self) -> None:
        """A buffer shorter than the max line means the start of a long line has
        already scrolled out of the decoder's view before the line is finalised."""
        if not hasattr(self, "_buf_note"):
            return
        bad = self._maxbuf.value() < self._maxline.value()
        self._buf_note.setStyleSheet(
            "color: #c04a3a; font-size: 11px;" if bad else "color: gray; font-size: 11px;")
        self._buf_note.setText(
            f"The buffer ({self._maxbuf.value():.0f}s) is shorter than the "
            f"force-new-line time ({self._maxline.value():.0f}s): long lines will "
            f"lose their beginning. Raise the buffer or lower the line length."
            if bad else self._buf_hint)

    # ---- hotkeys -----------------------------------------------------------
    def _hotkeys_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Hotkeys")
        v = QtWidgets.QVBoxLayout(g)
        v.addWidget(_hint(
            "System-wide shortcuts, asked of Windows by name. Only one app can own a "
            "combination: if another program already has it, ours simply never fires "
            "— that is what you are fixing here. Changes apply next launch."))

        self._hk_enabled = QtWidgets.QCheckBox("Use global hotkeys")
        self._hk_enabled.setChecked(bool(getattr(self._settings, "hotkeys_enabled", True)))
        self._hk_enabled.toggled.connect(self._on_hotkeys_enabled)
        v.addWidget(self._hk_enabled)

        form = QtWidgets.QFormLayout()
        for field, label in _HOTKEY_ACTIONS:
            edit = QtWidgets.QLineEdit(str(getattr(self._settings, field, model_default(field))))
            edit.setPlaceholderText("ctrl+alt+c")
            edit.editingFinished.connect(lambda f=field: self._on_hotkey(f))
            note = _hint("")
            self._hotkey_edits[field] = edit
            self._hotkey_notes[field] = note
            row = QtWidgets.QVBoxLayout()
            row.addWidget(edit)
            row.addWidget(note)
            form.addRow(f"{label}:", row)
        v.addLayout(form)
        # setChecked(False) on an already-unchecked box emits nothing, so launching
        # with hotkeys off would otherwise leave every field editable.
        self._apply_hotkeys_enabled(self._hk_enabled.isChecked())

        self._nudge = QtWidgets.QSpinBox()
        self._nudge.setRange(1, 500)
        self._nudge.setSuffix(" px")
        self._nudge.setValue(int(self._fit_spin(
            self._nudge, int(getattr(self._settings, "hotkey_nudge_px", 40)),
            "Move step", " px")))
        self._nudge.valueChanged.connect(self._on_nudge)
        nrow = QtWidgets.QFormLayout()
        nrow.addRow("Move step:", self._nudge)
        v.addLayout(nrow)
        v.addWidget(_hint("How far the arrow hotkeys shift the overlay each press."))

        btn = QtWidgets.QPushButton("Check which hotkeys are free")
        btn.clicked.connect(self.refresh_hotkey_status)
        v.addWidget(btn)
        self._hk_status = QtWidgets.QLabel("")
        self._hk_status.setWordWrap(True)
        v.addWidget(self._hk_status)
        return g

    def _apply_hotkeys_enabled(self, on: bool) -> None:
        for edit in self._hotkey_edits.values():
            edit.setEnabled(bool(on))

    def _on_hotkeys_enabled(self, on: bool) -> None:
        self._persist(hotkeys_enabled=bool(on))
        self._apply_hotkeys_enabled(on)
        if not self._loading:
            self._needs_app_restart("Turning hotkeys "
                                    + ("on" if on else "off"))

    def _on_nudge(self, px: int) -> None:
        self._persist(hotkey_nudge_px=int(px))

    def _on_hotkey(self, field: str) -> None:
        """Validate before storing. An unparseable combo is never written: at
        runtime it fails silently inside RegisterHotKey, which is indistinguishable
        from 'this app's hotkeys are broken'."""
        edit = self._hotkey_edits[field]
        typed = edit.text()
        current = str(getattr(self._settings, field, model_default(field)))
        if typed.strip().lower() == current:
            return
        try:
            spec = normalize_hotkey(typed)
        except ValueError as e:
            self._hotkey_notes[field].setText(f"Not a usable combination: {e}. Keeping {current}.")
            self._hotkey_notes[field].setStyleSheet("color: #c04a3a; font-size: 11px;")
            edit.setText(current)
            return
        clash = self._other_action_using(field, spec)
        if clash:
            self._hotkey_notes[field].setText(
                f"{spec} is already used for “{clash}” — one combination can only do "
                f"one thing. Keeping {current}.")
            self._hotkey_notes[field].setStyleSheet("color: #c04a3a; font-size: 11px;")
            edit.setText(current)
            return
        edit.setText(spec)
        self._hotkey_notes[field].setStyleSheet("color: gray; font-size: 11px;")
        self._hotkey_notes[field].setText("")
        self._persist(**{field: spec})
        self._needs_app_restart(f"The new {spec} shortcut")
        self.refresh_hotkey_status()

    def _other_action_using(self, field: str, spec: str) -> Optional[str]:
        """Compare what Windows sees, not what was typed: hotkeys.py maps both
        'esc' and 'escape' onto vk 0x1B, so string equality lets the same physical
        combination be assigned to two actions."""
        mine = _combo_id(spec)
        for other, label in _HOTKEY_ACTIONS:
            if other == field:
                continue
            theirs = _combo_id(str(getattr(self._settings, other, "")))
            if mine is not None and theirs == mine:
                return label
        return None

    def showEvent(self, event):
        super().showEvent(event)
        if not self._probed:
            self._probed = True
            self.refresh_hotkey_status()

    def _shown_spec(self, field: str) -> Tuple[str, bool]:
        """The combo the user is looking at, and whether it is usable at all.

        Status has to describe what is on screen. The stored value can lag the
        box (nothing is written until the combo validates), and reporting against
        the stored one is how a result for the old combo ends up under the new.
        """
        stored = str(getattr(self._settings, field, model_default(field)))
        edit = self._hotkey_edits.get(field)
        if edit is None:
            return stored, True
        try:
            return normalize_hotkey(edit.text()), True
        except ValueError:
            return edit.text().strip(), False

    def refresh_hotkey_status(self) -> None:
        """Say which combos are unavailable, because the failure is otherwise
        invisible: the app prints 'already claimed' to a console nobody sees."""
        taken: List[str] = []
        for field, label in _HOTKEY_ACTIONS:
            spec, usable = self._shown_spec(field)
            note = self._hotkey_notes[field]
            if not usable:
                # Don't probe, and above all don't leave the previous verdict
                # standing next to a combo it was never measured against.
                note.setText("Not checked — finish typing a usable combination.")
                note.setStyleSheet("color: #d08a30; font-size: 11px;")
                continue
            # A registration result is only about the combo it was tried with.
            live = (self._registered.get(field)
                    if _combo_id(self._registered_specs.get(field, "")) == _combo_id(spec)
                    else None)
            if live is True:
                note.setText("Working now.")
                note.setStyleSheet("color: #3a8a4a; font-size: 11px;")
                continue
            if live is False:
                note.setText("Not working — another app has this combination.")
                note.setStyleSheet("color: #c04a3a; font-size: 11px;")
                taken.append(f"{label} ({spec})")
                continue
            state = probe_hotkey(spec)
            if state == "in use":
                note.setText("In use — by another app, or by Live Captions itself if "
                             "captions are running.")
                note.setStyleSheet("color: #d08a30; font-size: 11px;")
                taken.append(f"{label} ({spec})")
            elif state == "free":
                note.setText("Available.")
                note.setStyleSheet("color: #3a8a4a; font-size: 11px;")
            else:
                note.setText("")
                note.setStyleSheet("color: gray; font-size: 11px;")
        if taken:
            self._hk_status.setStyleSheet("color: #d08a30; font-size: 11px;")
            self._hk_status.setText(
                "Unavailable: " + "; ".join(taken)
                + ". Type a different combination above — ctrl+shift+F8 style combos "
                  "are rarely taken.")
        else:
            self._hk_status.setStyleSheet("color: #3a8a4a; font-size: 11px;")
            self._hk_status.setText("Every shortcut is available.")

    # ---- reset -------------------------------------------------------------
    def _on_reset(self) -> None:
        ok = QtWidgets.QMessageBox.question(
            self, "Reset advanced settings?",
            "Put every setting on this tab back to the value it shipped with?\n\n"
            "Your overlay, audio device, model and transcripts are not touched.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if ok != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.reset_to_defaults()

    def reset_to_defaults(self) -> None:
        defaults = {f: model_default(f) for f in OWNED_FIELDS}
        for field, value in defaults.items():
            setattr(self._settings, field, value)
        save_settings(**defaults)
        self._reload_widgets()
        self._note.setText("Reset to the shipped defaults.")
        if self._on_restart is not None:
            self._pending_restart = "defaults"
            self._restart_timer.start(self.RESTART_DELAY_MS)
        self.refresh_hotkey_status()

    def _reload_widgets(self) -> None:
        """Push settings into the controls without re-triggering a save per widget."""
        self._loading = True
        try:
            s = self._settings
            self._lang.setCurrentText(str(getattr(s, "language", "en") or "en"))
            self._beam.setValue(int(s.beam_size))
            self._gpu.setCurrentText(str(s.gpu_compute))
            self._cpu.setCurrentText(str(s.cpu_compute))
            self._block.setValue(float(s.block_sec))
            self._floor.setValue(float(s.silence_rms_floor))
            self._interval.setValue(float(s.stream_process_interval))
            self._endsil.setValue(float(s.stream_end_silence_sec))
            self._maxline.setValue(float(s.stream_max_line_sec))
            self._maxbuf.setValue(float(s.stream_max_buffer_sec))
            self._vad.setValue(float(s.stream_vad_threshold))
            self._hk_enabled.setChecked(bool(s.hotkeys_enabled))
            self._nudge.setValue(int(s.hotkey_nudge_px))
            for field, _ in _HOTKEY_ACTIONS:
                self._hotkey_edits[field].setText(str(getattr(s, field)))
            self._apply_hotkeys_enabled(bool(s.hotkeys_enabled))
        finally:
            self._loading = False
        self._check_buffer_vs_line()


def _combo_id(spec: str) -> Optional[Tuple[int, int]]:
    """(modifiers, virtual-key) — the identity Windows actually registers on,
    or None when the combo cannot be parsed at all."""
    try:
        return parse_hotkey(spec)
    except ValueError:
        return None


def _valid_language(code: str) -> bool:
    if not code:
        return False
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        return code in _LANGUAGE_CODES
    except Exception:
        # No model library present (or it moved the table): fall back to shape, so
        # this tab still works rather than refusing every code.
        return code.isalpha() and 2 <= len(code) <= 3
