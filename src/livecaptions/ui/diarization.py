"""Speakers tab: live speaker colours, and a slow offline pass to do better.

Two different diarizers live behind this one tab, and the difference matters
enough to be spelled out in the UI rather than buried here:

  * LIVE (Streaming Sortformer) has to decide who is talking before the sentence
    is over, from a mixed stream, with at most 4 speakers. It costs a second
    model load and a CPU core for as long as captions run.
  * OFFLINE (pyannote / sherpa-onnx) sees the whole recording, so it can cluster
    voices properly — but it takes minutes and needs the audio to still exist.

So the offline pass is offered as a *repair*: re-diarize a saved session and
write the better labels back over the live guesses. It runs on a worker thread
that touches no Qt object; results come back through signals, the same seam the
AI tab uses.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from PySide6 import QtCore, QtWidgets

from ..config import save_settings

_BACKENDS = [
    ("auto", "Automatic (pyannote if you have a Hugging Face token, else sherpa)"),
    ("pyannote", "pyannote — best quality, needs a Hugging Face token"),
    ("sherpa", "sherpa-onnx — no account needed, runs on CPU"),
]


class Cancelled(Exception):
    """Raised inside the worker when the user asked to stop."""


@dataclass(frozen=True)
class DiarizeRequest:
    """Everything the worker needs — plain data only, so nothing Qt-owned can be
    read from the worker thread by accident."""

    session_id: int
    wav_path: str
    backend: str
    num_speakers: int


def audio_dir(settings) -> Path:
    """Where saved session audio lives.

    Unset `audio_save_dir` must resolve to the SAME directory the recorder writes
    to, or the tab reports "no saved audio" for sessions whose WAV is sitting on
    disk. capture.recorder derives it from DB_PATH at call time, so ask it rather
    than re-deriving here and drifting from it (it used to guess a platformdirs
    path the recorder never writes to).
    """
    # Delegate entirely, including the configured case: two implementations of
    # "where is the audio" is exactly how these drifted apart the first time.
    from ..capture.recorder import audio_dir as recorder_audio_dir
    return recorder_audio_dir(settings=settings)


def session_audio_path(settings, session_id: int,
                       source: Optional[str] = None) -> Optional[Path]:
    """The WAV for a session, or None if it wasn't kept.

    Re-diarizing the *wrong* audio would silently corrupt a transcript, so the
    match has to be conservative: exact names first, and the loose fallback only
    accepts a file that both says "session" and carries the id as a standalone
    number (so session 1 can't claim `meeting-2021.wav`).
    """
    if source and source.startswith("WAV:"):
        p = Path(source[4:])
        if p.is_file():
            return p

    d = audio_dir(settings)
    from ..capture.recorder import session_audio_path as recorder_name
    # The recorder's own name first, and taken from the recorder so the two can
    # never disagree; the rest are tolerated for hand-placed files.
    names = [recorder_name(session_id, d).name,
             f"session-{session_id}.wav", f"{session_id}.wav"]
    for name in names:
        p = d / name
        if p.is_file():
            return p

    token = re.compile(rf"(?<!\d){session_id}(?!\d)")
    try:
        candidates = sorted(d.glob("*.wav"))
    except OSError:
        return None
    for p in candidates:
        if "session" in p.stem.lower() and token.search(p.stem):
            return p
    return None


def segments_to_turns(segments: Sequence) -> List:
    """SpeakerSegment (caption lines) -> SpeakerTurn (speaker spans)."""
    from ..diarize.base import SpeakerTurn
    return [SpeakerTurn(s.start, s.end, s.speaker) for s in segments if s.speaker]


def speaker_order(conn, session_id: int) -> List[str]:
    """Labels in first-heard order — the same order the overlay hands out
    colours, so the legend matches what you actually saw on screen."""
    rows = conn.execute(
        "SELECT speaker, MIN(t_start) AS first_at FROM utterances "
        "WHERE session_id=? AND speaker IS NOT NULL "
        "GROUP BY speaker ORDER BY first_at", (session_id,)).fetchall()
    return [r["speaker"] for r in rows]


def apply_speakers(conn, session_id: int, turns: Sequence) -> int:
    """Write re-diarized labels back over one session. Returns rows changed.

    Every statement is filtered by session_id — a diarization pass on one
    recording must never be able to relabel a different day's transcript.
    """
    from ..diarize.assign import best_speaker
    rows = conn.execute(
        "SELECT id, t_start, t_end, speaker FROM utterances "
        "WHERE session_id=? ORDER BY t_start", (session_id,)).fetchall()
    changed = 0
    for r in rows:
        who = best_speaker(float(r["t_start"]), float(r["t_end"]), turns)
        if who is None:
            continue                    # no turn to attribute it to: leave it alone
        if who == r["speaker"]:
            # rowcount counts rows MATCHED, not rows altered, so counting it here
            # would report "Relabelled 200 lines" for a pass that agreed with the
            # labels already stored.
            continue
        conn.execute(
            "UPDATE utterances SET speaker=? WHERE id=? AND session_id=?",
            (who, r["id"], session_id))
        changed += 1
    conn.commit()
    return changed


def run_offline_pass(req: DiarizeRequest, settings, progress, cancelled) -> dict:
    """The worker body: WAV -> speaker segments. No Qt, no DB, no widget state.

    `progress` is a callable taking a string and `cancelled` a zero-arg predicate;
    the tab passes thread-safe things (a signal emit and an Event). Whisper and
    the diarizer are single blocking calls we cannot interrupt, so cancellation is
    checked between phases — the UI says as much rather than promising instant.
    """
    if cancelled():
        raise Cancelled()
    progress("Loading the speech model…")
    from ..asr.whisper import load_model
    try:
        from ..capture.cuda import bootstrap_cuda_dlls
        bootstrap_cuda_dlls()
    except Exception:
        pass                            # CPU-only box: the model load handles it
    model = load_model(settings)

    if cancelled():
        raise Cancelled()
    progress("Transcribing and diarizing the recording… (minutes, not seconds)")
    from ..diarize.pipeline import diarize_file
    segments, n_speakers = diarize_file(
        req.wav_path, model, settings,
        backend=req.backend, num_speakers=req.num_speakers)

    if cancelled():
        raise Cancelled()
    return {"session": req.session_id, "segments": segments, "speakers": n_speakers}


class DiarizationTab(QtWidgets.QWidget):
    _progress = QtCore.Signal(str)
    _finished = QtCore.Signal(object)    # the run_offline_pass dict, or {"err": ...}

    def __init__(self, settings, parent=None, apply_pipeline=None):
        super().__init__(parent)
        self._settings = settings
        # Set by the integrator so toggling live colours rebuilds the running
        # pipeline immediately; without it the change waits for the next launch.
        self._apply_pipeline = apply_pipeline
        self._conn = None
        self._cancel = threading.Event()
        self._running = False

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self._live_group())
        v.addWidget(self._audio_group())
        v.addWidget(self._offline_group())
        v.addWidget(self._legend_group())
        v.addStretch(1)

        self._progress.connect(self._on_progress)
        self._finished.connect(self._on_finished)

    # ---- live colours ----
    def _live_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Live speaker colours")
        v = QtWidgets.QVBoxLayout(g)
        self._colors = QtWidgets.QCheckBox("Colour captions by who is speaking")
        self._colors.setChecked(bool(getattr(self._settings, "speaker_colors", False)))
        self._colors.toggled.connect(self._on_colors)
        v.addWidget(self._colors)
        cost = QtWidgets.QLabel(
            "The honest cost: this loads a second model (Streaming Sortformer) "
            "alongside speech recognition and runs it on the CPU for as long as "
            "captions are on, so expect one more busy core and a slower start. It "
            "leaves the GPU free for transcription. It handles up to 4 speakers and "
            "has to guess before a sentence finishes, so it does get turns wrong — "
            "the offline pass below fixes those afterwards.")
        cost.setWordWrap(True)
        cost.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(cost)
        return g

    def _on_colors(self, on: bool) -> None:
        self._persist(speaker_colors=on)
        if self._apply_pipeline is not None:
            self._apply_pipeline("speaker colours")

    def _persist(self, **kw) -> None:
        for k, val in kw.items():
            setattr(self._settings, k, val)
        save_settings(**kw)

    # ---- offline re-diarization ----
    def _audio_group(self) -> QtWidgets.QGroupBox:
        """Turn on keeping the audio.

        Without this there is nothing to re-diarize, ever: the setting existed and
        the recorder was wired up, but no control switched it on, so the section
        below was permanently greyed out with "no saved audio". It lives here rather
        than in the Transcripts tab because keeping audio is only worth its disk
        space for re-diarization, and this is where that happens.
        """
        g = QtWidgets.QGroupBox("Keep the audio")
        v = QtWidgets.QVBoxLayout(g)

        self._save_audio = QtWidgets.QCheckBox(
            "Save each session's audio so it can be re-diarized later")
        self._save_audio.setChecked(bool(getattr(self._settings, "save_audio", False)))
        self._save_audio.toggled.connect(self._on_save_audio)
        v.addWidget(self._save_audio)

        cost = QtWidgets.QLabel(
            "About 110 MB per hour, kept on this PC and never uploaded. It applies to "
            "NEW sessions — anything already recorded has no audio to save. Deleting a "
            "session in Transcripts deletes its audio too.")
        cost.setWordWrap(True)
        cost.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(cost)

        row = QtWidgets.QHBoxLayout()
        self._audio_usage = QtWidgets.QLabel("")
        self._audio_usage.setStyleSheet("color: gray; font-size: 11px;")
        row.addWidget(self._audio_usage, 1)
        self._audio_clear = QtWidgets.QPushButton("Delete all saved audio")
        self._audio_clear.clicked.connect(self._on_clear_audio)
        row.addWidget(self._audio_clear)
        v.addLayout(row)
        self._refresh_audio_usage()
        return g

    def _on_save_audio(self, on: bool) -> None:
        self._persist(save_audio=on)
        self._refresh_audio_usage()
        if on:
            self._status.setText(
                "Audio will be kept from the next session you record — restart "
                "captions if one is running now.")

    def _refresh_audio_usage(self) -> None:
        try:
            from ..capture.recorder import audio_files, format_bytes, total_audio_bytes
            n = len(audio_files(settings=self._settings))
            total = total_audio_bytes(settings=self._settings)
        except Exception as e:
            self._audio_usage.setText(f"(couldn't read the audio folder: {e})")
            return
        self._audio_usage.setText(
            f"{n} recording(s), {format_bytes(total)}" if n else "No audio saved yet.")
        self._audio_clear.setEnabled(bool(n))

    def _on_clear_audio(self) -> None:
        from ..capture.recorder import delete_all_audio, format_bytes, total_audio_bytes
        total = total_audio_bytes(settings=self._settings)
        ok = QtWidgets.QMessageBox.question(
            self, "Delete all saved audio?",
            f"This permanently deletes {format_bytes(total)} of recordings.\n\n"
            f"Transcripts are kept, but those sessions can no longer be re-diarized.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if ok != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        removed, failed = delete_all_audio(settings=self._settings)
        self._refresh_audio_usage()
        self._status.setText(f"Deleted {removed} recording(s)."
                             + (f" {failed} could not be deleted." if failed else ""))
        self._refresh_sessions()

    def _offline_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Re-diarize a saved session")
        form = QtWidgets.QFormLayout(g)

        blurb = QtWidgets.QLabel(
            "Runs the whole recording through a proper diarizer and replaces the "
            "live guesses with better labels. Only works if the session's audio "
            "was saved.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(blurb)

        self._backend = QtWidgets.QComboBox()
        for value, label in _BACKENDS:
            self._backend.addItem(label, userData=value)
        cur = str(getattr(self._settings, "diarizer", "auto") or "auto")
        idx = next((i for i, (v, _) in enumerate(_BACKENDS) if v == cur), 0)
        self._backend.setCurrentIndex(idx)
        self._backend.currentIndexChanged.connect(self._on_backend)
        form.addRow("Diarizer:", self._backend)

        self._num = QtWidgets.QSpinBox()
        self._num.setRange(-1, 20)
        self._num.setSpecialValueText("Work it out")       # shown at the -1 minimum
        self._num.setValue(int(getattr(self._settings, "diarize_num_speakers", -1)))
        self._num.valueChanged.connect(
            lambda n: self._persist(diarize_num_speakers=int(n)))
        form.addRow("Speakers:", self._num)
        hint = QtWidgets.QLabel(
            "Telling it the real number is the single biggest accuracy win when you "
            "know it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(hint)

        row = QtWidgets.QHBoxLayout()
        self._sessions = QtWidgets.QComboBox()
        self._sessions.currentIndexChanged.connect(self._on_session_changed)
        row.addWidget(self._sessions, 1)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self._load_sessions)
        row.addWidget(refresh)
        form.addRow("Session:", row)

        self._audio = QtWidgets.QLabel("")
        self._audio.setWordWrap(True)
        self._audio.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(self._audio)

        btns = QtWidgets.QHBoxLayout()
        self._run = QtWidgets.QPushButton("Re-diarize…")
        self._run.clicked.connect(self._on_run)
        btns.addWidget(self._run)
        self._stop = QtWidgets.QPushButton("Cancel")
        self._stop.clicked.connect(self._on_cancel)
        self._stop.setEnabled(False)
        btns.addWidget(self._stop)
        btns.addStretch(1)
        form.addRow(btns)

        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(self._status)
        return g

    def _on_backend(self, i: int) -> None:
        self._persist(diarizer=self._backend.itemData(i) or "auto")

    # ---- colour legend ----
    def _legend_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Which colour is who")
        self._legend = QtWidgets.QVBoxLayout(g)
        self._legend_empty = QtWidgets.QLabel("Pick a session to see its speakers.")
        self._legend_empty.setStyleSheet("color: gray; font-size: 11px;")
        self._legend.addWidget(self._legend_empty)
        return g

    def _refresh_legend(self, session_id: Optional[int]) -> None:
        while self._legend.count() > 1:
            item = self._legend.takeAt(1)
            w = item.widget()
            if w is not None:
                # unparent now, not at deleteLater time: a second refresh before
                # the event loop runs would otherwise stack duplicate swatches
                w.setParent(None)
                w.deleteLater()

        labels = []
        if session_id is not None:
            try:
                labels = speaker_order(self._ensure(), session_id)
            except Exception as e:
                # "no speaker labels" would be a lie about a store we failed to
                # read, and would send the user off to re-diarize a session that
                # is probably labelled fine.
                self._legend_empty.setText(f"Couldn't read the speakers: {e}")
                return
        if not labels:
            self._legend_empty.setText(
                "That session has no speaker labels — turn on live colours before "
                "recording, or re-diarize it above." if session_id is not None
                else "Pick a session to see its speakers.")
            return
        self._legend_empty.setText(
            "Colours are handed out in the order people first speak, so they can "
            "differ between sessions.")
        from .overlay import SPEAKER_COLORS
        for i, label in enumerate(labels):
            colour = SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(28, 14)
            swatch.setStyleSheet(
                f"background: {colour.name()}; border: 1px solid #666; border-radius: 3px;")
            h.addWidget(swatch)
            h.addWidget(QtWidgets.QLabel(label), 1)
            self._legend.addWidget(row)

    # ---- sessions ----
    def _ensure(self):
        if self._conn is None:
            from ..store.db import DB_PATH, connect
            self._conn = connect(DB_PATH)
        return self._conn

    def closeEvent(self, event):
        # _ensure() opens a long-lived sqlite handle; without this it outlives the
        # widget and keeps the WAL files open until the process exits.
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self._colors.setChecked(bool(getattr(self._settings, "speaker_colors", False)))
        self._load_sessions()

    def _load_sessions(self) -> None:
        try:
            from ..store.search import recent_sessions
            rows = recent_sessions(self._ensure(), limit=50)
        except Exception as e:
            self._status.setText(f"Couldn't read sessions: {e}")
            return
        self._sessions.blockSignals(True)
        self._sessions.clear()
        for r in rows:
            if not r["utterances"]:
                continue                # nothing to relabel
            self._sessions.addItem(
                f"[{r['id']}] {r['started_at']} · {r['utterances']} lines",
                userData=(r["id"], r["source"]))
        self._sessions.blockSignals(False)
        if not self._sessions.count():
            self._audio.setText("No saved sessions yet.")
            self._run.setEnabled(False)
            self._refresh_legend(None)
            return
        self._on_session_changed(self._sessions.currentIndex())

    def _selected(self):
        data = self._sessions.currentData()
        return data if data else (None, None)

    def _on_session_changed(self, _i: int) -> None:
        session_id, source = self._selected()
        self._refresh_legend(session_id)
        if session_id is None:
            return
        wav = session_audio_path(self._settings, session_id, source)
        if wav is None:
            # Offering a button that cannot work is worse than saying why.
            self._audio.setText(
                f"No saved audio for session {session_id}, so it can't be "
                f"re-diarized — the offline pass needs the recording, not just the "
                f"text. Turn on saving audio before recording to make this possible. "
                f"Looked in {audio_dir(self._settings)}.")
            self._run.setEnabled(False)
        else:
            try:
                mb = wav.stat().st_size / (1024 * 1024)
                size = f" ({mb:.1f} MB)"
            except OSError:
                size = ""
            self._audio.setText(f"Audio: {wav}{size}")
            self._run.setEnabled(not self._running)

    # ---- the slow pass ----
    def _on_run(self) -> None:
        if self._running:
            return
        session_id, source = self._selected()
        if session_id is None:
            self._status.setText("Pick a session first.")
            return
        wav = session_audio_path(self._settings, session_id, source)
        if wav is None:
            self._status.setText("That session has no saved audio.")
            return

        existing = speaker_order(self._ensure(), session_id)
        warn = (f"\n\nThis overwrites the current labels ({', '.join(existing)}), "
                f"including any names you typed." if existing else "")
        ok = QtWidgets.QMessageBox.question(
            self, "Re-diarize this session?",
            f"Session {session_id} will be transcribed and diarized again from "
            f"{wav.name}. That takes minutes and works this machine hard; captions "
            f"will be slower while it runs.{warn}\n\nGo ahead?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if ok != QtWidgets.QMessageBox.StandardButton.Yes:
            self._status.setText("Cancelled. Nothing was changed.")
            return

        req = DiarizeRequest(
            session_id=int(session_id), wav_path=str(wav),
            backend=self._backend.currentData() or "auto",
            num_speakers=int(self._num.value()))
        self._start(req)

    def _start(self, req: DiarizeRequest) -> None:
        self._cancel.clear()
        self._set_running(True)
        self._status.setText("Starting…")
        settings = self._settings
        progress, finished = self._progress.emit, self._finished.emit
        cancelled = self._cancel.is_set

        def work():
            # Only plain data crosses into this thread; results go back through
            # signals, which Qt queues onto the GUI thread for us.
            try:
                finished(run_offline_pass(req, settings, progress, cancelled))
            except Cancelled:
                finished({"cancelled": True})
            except Exception as e:
                finished({"err": str(e)})

        threading.Thread(target=work, daemon=True,
                         name="rediarize").start()

    def _set_running(self, on: bool) -> None:
        self._running = on
        self._run.setEnabled(not on and self._sessions.count() > 0)
        self._stop.setEnabled(on)
        self._sessions.setEnabled(not on)
        self._backend.setEnabled(not on)
        self._num.setEnabled(not on)

    def _on_cancel(self) -> None:
        self._cancel.set()
        # Whisper and the diarizer are uninterruptible C calls; be honest that
        # this stops at the next checkpoint rather than immediately.
        self._status.setText("Stopping after the current step… nothing will be "
                             "written to the transcript.")

    @QtCore.Slot(str)
    def _on_progress(self, message: str) -> None:
        self._status.setText(message)

    @QtCore.Slot(object)
    def _on_finished(self, payload: dict) -> None:
        self._set_running(False)
        # The worker's last cancellation checkpoint is before it returns, so a
        # Cancel pressed after it — including during this signal's own hop onto
        # the GUI thread — arrives with a perfectly successful payload. _on_cancel
        # promised nothing would be written; honour that here, the last place that
        # still can.
        if payload.get("cancelled") or self._cancel.is_set():
            self._status.setText("Stopped. The transcript is unchanged.")
            return
        if payload.get("err"):
            self._status.setText(f"Re-diarization failed: {payload['err']}")
            return

        session_id = int(payload["session"])
        turns = segments_to_turns(payload.get("segments") or [])
        if not turns:
            self._status.setText("The pass found no speech, so nothing was changed.")
            return
        try:
            changed = apply_speakers(self._ensure(), session_id, turns)
        except Exception as e:
            self._status.setText(f"Couldn't save the new labels: {e}")
            return
        self._status.setText(
            f"Relabelled {changed} line(s) in session {session_id} as "
            f"{payload.get('speakers', 0)} speaker(s). Only this session was "
            f"touched; rename the labels from the Transcripts tab.")
        self._refresh_legend(session_id)
