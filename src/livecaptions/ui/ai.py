"""AI tab: choose a model provider, and use it to name speakers.

This is the only feature that can send your data anywhere, so the rules are
strict and visible in the UI, not just in a docstring:
  * off by default — "Not configured" until you choose otherwise;
  * a local provider (Ollama / LM Studio) keeps everything on the machine, and is
    offered first-class rather than as an afterthought;
  * before anything is sent you are told exactly how much and to where, and you
    have to agree;
  * every proposed rename is shown with the verbatim quote that justifies it and
    confirmed individually;
  * renames are reversible, and the UI says so.

API keys go to Windows Credential Manager, never to config.toml.
"""
from __future__ import annotations

import threading
from typing import List, Optional

from PySide6 import QtCore, QtWidgets

from ..config import save_settings
from ..llm import providers as P

_KIND_BY_INDEX = ["none", "anthropic", "openai", "local"]


class AITab(QtWidgets.QWidget):
    _test_done = QtCore.Signal(object)      # {"ok": str} | {"err": str}
    _names_done = QtCore.Signal(object)     # {"proposals": [...]} | {"err": str}

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._conn = None
        self._proposals: List = []

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self._provider_group())
        v.addWidget(self._naming_group())
        v.addStretch(1)

        self._test_done.connect(self._on_test_done)
        self._names_done.connect(self._on_names_done)
        self._sync_fields()

    # ---- provider setup ----
    def _provider_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Model provider")
        form = QtWidgets.QFormLayout(g)

        self._kind = QtWidgets.QComboBox()
        self._kind.addItems(P.available_labels())
        cur = str(getattr(self._settings, "llm_provider", "none") or "none")
        self._kind.setCurrentIndex(_KIND_BY_INDEX.index(cur) if cur in _KIND_BY_INDEX else 0)
        self._kind.currentIndexChanged.connect(self._on_kind)
        form.addRow("Provider:", self._kind)

        self._preset = QtWidgets.QComboBox()
        self._preset.addItem("Choose a local server…", userData="")
        for name, url in P.LOCAL_PRESETS.items():
            self._preset.addItem(name, userData=url)
        self._preset.currentIndexChanged.connect(self._on_preset)
        form.addRow("", self._preset)

        self._url = QtWidgets.QLineEdit(str(getattr(self._settings, "llm_base_url", "") or ""))
        self._url.setPlaceholderText("http://localhost:11434/v1")
        self._url.editingFinished.connect(
            lambda: self._persist(llm_base_url=self._url.text().strip()))
        form.addRow("Server URL:", self._url)

        self._model = QtWidgets.QLineEdit(str(getattr(self._settings, "llm_model", "") or ""))
        self._model.setPlaceholderText("e.g. claude-opus-4-8, gpt-4o, llama3.1")
        self._model.editingFinished.connect(
            lambda: self._persist(llm_model=self._model.text().strip()))
        form.addRow("Model:", self._model)

        self._key = QtWidgets.QLineEdit()
        self._key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._key.setPlaceholderText("(stored in Windows Credential Manager)")
        keyrow = QtWidgets.QHBoxLayout()
        keyrow.addWidget(self._key, 1)
        savebtn = QtWidgets.QPushButton("Save key")
        savebtn.clicked.connect(self._on_save_key)
        keyrow.addWidget(savebtn)
        clearbtn = QtWidgets.QPushButton("Forget")
        clearbtn.clicked.connect(self._on_clear_key)
        keyrow.addWidget(clearbtn)
        form.addRow("API key:", keyrow)

        self._keynote = QtWidgets.QLabel("")
        self._keynote.setWordWrap(True)
        self._keynote.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(self._keynote)

        self._test = QtWidgets.QPushButton("Test connection")
        self._test.clicked.connect(self._on_test)
        form.addRow(self._test)

        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(self._status)
        return g

    def _persist(self, **kw) -> None:
        for k, val in kw.items():
            setattr(self._settings, k, val)
        save_settings(**kw)

    def _on_kind(self, i: int) -> None:
        kind = _KIND_BY_INDEX[i] if 0 <= i < len(_KIND_BY_INDEX) else "none"
        self._persist(llm_provider=kind)
        self._sync_fields()

    def _on_preset(self, i: int) -> None:
        url = self._preset.itemData(i)
        if url:
            self._url.setText(url)
            self._persist(llm_base_url=url)

    def _sync_fields(self) -> None:
        kind = str(getattr(self._settings, "llm_provider", "none") or "none")
        local, openai, anth = kind == "local", kind == "openai", kind == "anthropic"
        configured = kind != "none"
        self._preset.setVisible(local)
        self._url.setEnabled(local or openai)
        self._model.setEnabled(configured)
        self._key.setEnabled(openai or anth)
        self._test.setEnabled(configured)
        self._name_btn.setEnabled(configured) if hasattr(self, "_name_btn") else None

        stored = self._stored_key_hint(kind)
        if local:
            self._keynote.setText("Local models usually need no key. Nothing you "
                                  "send to a local server leaves this machine.")
        elif anth:
            self._keynote.setText(
                "Uses ANTHROPIC_API_KEY or an `ant auth login` profile if present. "
                + stored)
        elif openai:
            self._keynote.setText("Works with OpenAI, OpenRouter, Groq, Together and "
                                  "anything else speaking the same API. " + stored)
        else:
            self._keynote.setText("AI features are off. Nothing is sent anywhere.")

    def _stored_key_hint(self, kind: str) -> str:
        try:
            from ..llm.credentials import get_secret
            return "A key is saved for this provider." if get_secret(kind) else \
                "No key saved yet."
        except Exception:
            return ""

    def _on_save_key(self) -> None:
        kind = str(getattr(self._settings, "llm_provider", "none") or "none")
        key = self._key.text().strip()
        if not key:
            self._status.setText("Type a key first.")
            return
        try:
            from ..llm.credentials import set_secret
            set_secret(kind, key)
        except OSError as e:
            self._status.setText(f"Couldn't save the key: {e}")
            return
        self._key.clear()      # don't leave it on screen once it's stored
        self._status.setText("Key saved to Windows Credential Manager (not to config.toml).")
        self._sync_fields()

    def _on_clear_key(self) -> None:
        kind = str(getattr(self._settings, "llm_provider", "none") or "none")
        try:
            from ..llm.credentials import delete_secret
            gone = delete_secret(kind)
        except Exception:
            gone = False
        self._status.setText("Key forgotten." if gone else "There was no saved key.")
        self._sync_fields()

    # ---- connection test ----
    def _on_test(self) -> None:
        self._test.setEnabled(False)
        self._status.setText("Testing…")

        def work():
            try:
                kind = str(getattr(self._settings, "llm_provider", "none") or "none")
                prov = P.from_settings(self._settings, P.resolve_api_key(kind))
                self._test_done.emit({"ok": prov.test()})
            except Exception as e:
                self._test_done.emit({"err": str(e)})
        threading.Thread(target=work, daemon=True).start()

    @QtCore.Slot(object)
    def _on_test_done(self, payload: dict) -> None:
        self._test.setEnabled(True)
        self._status.setText(payload.get("ok") or f"Failed: {payload.get('err')}")

    # ---- speaker naming ----
    def _naming_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Name the speakers in a session")
        v = QtWidgets.QVBoxLayout(g)
        blurb = QtWidgets.QLabel(
            "Reads a saved transcript and suggests real names for SPEAKER_00 and "
            "friends. You'll see how much text is sent before anything happens, and "
            "each rename is confirmed with the quote that justifies it.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(blurb)

        row = QtWidgets.QHBoxLayout()
        self._sessions = QtWidgets.QComboBox()
        row.addWidget(self._sessions, 1)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self._load_sessions)
        row.addWidget(refresh)
        v.addLayout(row)

        self._name_btn = QtWidgets.QPushButton("Suggest names…")
        self._name_btn.clicked.connect(self._on_name)
        v.addWidget(self._name_btn)

        self._name_status = QtWidgets.QLabel("")
        self._name_status.setWordWrap(True)
        self._name_status.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(self._name_status)
        return g

    def showEvent(self, event):
        super().showEvent(event)
        self._load_sessions()
        self._sync_fields()

    def _ensure(self):
        if self._conn is None:
            from ..store.db import DB_PATH, connect
            self._conn = connect(DB_PATH)
        return self._conn

    def _load_sessions(self) -> None:
        try:
            from ..store.search import recent_sessions
            rows = recent_sessions(self._ensure(), limit=50)
        except Exception as e:
            self._name_status.setText(f"Couldn't read sessions: {e}")
            return
        self._sessions.clear()
        for r in rows:
            if not r["speakers"]:
                continue           # nothing to name without diarization labels
            self._sessions.addItem(
                f"[{r['id']}] {r['started_at']} · {r['utterances']} lines, "
                f"{r['speakers']} speaker(s)", userData=r["id"])
        if not self._sessions.count():
            self._name_status.setText(
                "No sessions with speaker labels yet — turn on speaker colours to "
                "record who spoke.")

    def _on_name(self) -> None:
        sid = self._sessions.currentData()
        if sid is None:
            self._name_status.setText("Pick a session first.")
            return
        from ..store.naming import build_transcript, session_labels
        conn = self._ensure()
        labels = session_labels(conn, int(sid))
        transcript, truncated = build_transcript(conn, int(sid))
        if not labels or not transcript:
            self._name_status.setText("That session has nothing to name.")
            return

        kind = str(getattr(self._settings, "llm_provider", "none") or "none")
        cfg = P.config_from_settings(self._settings)
        # GATE 1 — consent, with the size and destination stated plainly.
        detail = P.describe_privacy(cfg, len(transcript))
        if truncated:
            detail += "\n\n(The session is long, so only the earlier part is sent.)"
        ok = QtWidgets.QMessageBox.question(
            self, "Send this transcript?",
            f"{detail}\n\nSpeakers to name: {', '.join(labels)}\n\nGo ahead?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if ok != QtWidgets.QMessageBox.StandardButton.Yes:
            self._name_status.setText("Cancelled. Nothing was sent.")
            return

        self._name_btn.setEnabled(False)
        self._name_status.setText("Asking the model…")

        def work():
            try:
                from ..store.naming import propose_names
                prov = P.from_settings(self._settings, P.resolve_api_key(kind))
                out = propose_names(transcript, labels, provider=prov)
                self._names_done.emit({"proposals": out, "session": int(sid)})
            except Exception as e:
                self._names_done.emit({"err": str(e)})
        threading.Thread(target=work, daemon=True).start()

    @QtCore.Slot(object)
    def _on_names_done(self, payload: dict) -> None:
        self._name_btn.setEnabled(True)
        if payload.get("err"):
            self._name_status.setText(f"Naming failed: {payload['err']}")
            return
        sid = payload["session"]
        applied = 0
        from ..store.search import rename_speaker
        conn = self._ensure()
        for p in payload["proposals"]:
            if not p.name:
                continue
            # GATE 2 — confirm each rename, showing the evidence it rests on.
            ok = QtWidgets.QMessageBox.question(
                self, "Rename this speaker?",
                f"{p.label}  →  {p.name}\n\nConfidence: {p.confidence}\n\n"
                f"Evidence: {p.evidence}\n\n"
                f"This can be undone by renaming {p.name} back to {p.label}.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if ok == QtWidgets.QMessageBox.StandardButton.Yes:
                rename_speaker(conn, p.label, p.name, session_id=sid)
                applied += 1
        skipped = [p.label for p in payload["proposals"] if not p.name]
        msg = f"Renamed {applied} speaker(s)."
        if skipped:
            msg += (f" No name found for {', '.join(skipped)} — the model is told to "
                    f"return nothing rather than guess.")
        self._name_status.setText(msg)
