"""Transcripts tab: browse, search, export and re-label saved sessions.

Everything here reads the local SQLite store (WAL, so these queries never block the
writer appending the session you are currently recording). Nothing leaves the
machine.

The store layer already provides search/export/rename; this is the GUI over it, so
the transcript features stop being CLI-only.
"""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from ..config import save_settings


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


class TranscriptsTab(QtWidgets.QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._conn = None
        self._session_id: Optional[int] = None

        v = QtWidgets.QVBoxLayout(self)

        # --- saving on/off: the privacy switch, so put it first and state where it goes
        box = QtWidgets.QGroupBox("Saving")
        bv = QtWidgets.QVBoxLayout(box)
        self._save = QtWidgets.QCheckBox("Save transcripts on this PC")
        self._save.setChecked(bool(getattr(settings, "save_transcripts", True)))
        self._save.toggled.connect(self._on_save_toggled)
        bv.addWidget(self._save)
        row = QtWidgets.QHBoxLayout()
        self._where = QtWidgets.QLabel("")
        self._where.setWordWrap(True)
        self._where.setStyleSheet("color: gray; font-size: 11px;")
        row.addWidget(self._where, 1)
        openbtn = QtWidgets.QPushButton("Open folder")
        openbtn.clicked.connect(self._open_folder)
        row.addWidget(openbtn)
        bv.addLayout(row)
        v.addWidget(box)

        # --- search
        srow = QtWidgets.QHBoxLayout()
        self._q = QtWidgets.QLineEdit()
        self._q.setPlaceholderText("Search everything you've captioned…")
        self._q.returnPressed.connect(self._on_search)
        srow.addWidget(self._q, 1)
        btn = QtWidgets.QPushButton("Search")
        btn.clicked.connect(self._on_search)
        srow.addWidget(btn)
        v.addLayout(srow)

        # --- sessions / results
        self._list = QtWidgets.QListWidget()
        self._list.setMaximumHeight(130)
        self._list.currentItemChanged.connect(self._on_pick)
        v.addWidget(self._list)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlaceholderText("Select a session to read it.")
        v.addWidget(self._text, 1)

        brow = QtWidgets.QHBoxLayout()
        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        brow.addWidget(self._status, 1)
        self._rename = QtWidgets.QPushButton("Rename speaker…")
        self._rename.clicked.connect(self._on_rename)
        brow.addWidget(self._rename)
        self._delete = QtWidgets.QPushButton("Delete…")
        self._delete.clicked.connect(self._on_delete)
        brow.addWidget(self._delete)
        self._export = QtWidgets.QPushButton("Export…")
        self._export.clicked.connect(self._on_export)
        brow.addWidget(self._export)
        v.addLayout(brow)

        self._sync_enabled()

    # ---- lazy DB access: don't touch the store until this tab is actually used ----
    def _ensure(self):
        if self._conn is None:
            from ..store.db import DB_PATH, connect
            self._conn = connect(DB_PATH)
            try:
                size = DB_PATH.stat().st_size
                self._where.setText(f"{DB_PATH}  ({_fmt_bytes(size)})")
            except OSError:
                self._where.setText(str(DB_PATH))
        return self._conn

    def showEvent(self, event):
        super().showEvent(event)
        try:
            self.refresh()
        except Exception as e:
            self._status.setText(f"Couldn't read the transcript store: {e}")

    def refresh(self) -> None:
        from ..store.search import recent_sessions
        conn = self._ensure()
        self._list.clear()
        rows = recent_sessions(conn, limit=50)
        if not rows:
            self._status.setText("No saved sessions yet.")
            return
        for r in rows:
            title = f" — {r['title']}" if r["title"] else ""
            spk = f", {r['speakers']} speaker(s)" if r["speakers"] else ""
            item = QtWidgets.QListWidgetItem(
                f"[{r['id']}] {r['started_at']}  ·  {r['utterances']} lines{spk}{title}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, {"session": r["id"]})
            self._list.addItem(item)
        self._status.setText(f"{len(rows)} session(s).")

    # ---- saving ----
    def _on_save_toggled(self, on: bool) -> None:
        self._settings.save_transcripts = on
        save_settings(save_transcripts=on)
        self._status.setText(
            "New sessions will be saved." if on else
            "New sessions will NOT be saved. Existing transcripts are kept.")
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        # Browsing still works with saving off — turning it off must not hide the
        # transcripts you already have.
        pass

    def _open_folder(self) -> None:
        from ..store.db import DB_PATH
        QtWidgets.QApplication.clipboard().setText(str(DB_PATH.parent))
        try:
            from PySide6 import QtGui
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(DB_PATH.parent)))
        except Exception:
            self._status.setText(f"Path copied to clipboard: {DB_PATH.parent}")

    # ---- search ----
    def _on_search(self) -> None:
        q = self._q.text().strip()
        if not q:
            self.refresh()
            return
        from ..store.search import search
        conn = self._ensure()
        try:
            hits = search(conn, q, limit=100)
        except Exception as e:
            # FTS5 MATCH syntax is unforgiving (a bare quote is a syntax error), so
            # report it rather than letting the tab look broken.
            self._status.setText(f"Couldn't search for that: {e}")
            return
        self._list.clear()
        for h in hits:
            who = f"{h.speaker}: " if h.speaker else ""
            item = QtWidgets.QListWidgetItem(
                f"[s{h.session_id}] {h.wall_clock}  {who}{h.snippet}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, {"session": h.session_id})
            self._list.addItem(item)
        self._status.setText(f"{len(hits)} match(es) for “{q}”."
                             if hits else f"Nothing matched “{q}”.")

    # ---- reading ----
    def _on_pick(self, cur, _prev) -> None:
        if cur is None:
            return
        data = cur.data(QtCore.Qt.ItemDataRole.UserRole) or {}
        sid = data.get("session")
        if sid is None:
            return
        self._session_id = int(sid)
        from ..store.export import to_markdown
        try:
            self._text.setPlainText(to_markdown(self._ensure(), self._session_id))
        except Exception as e:
            self._text.setPlainText(f"(couldn't read session {self._session_id}: {e})")

    # ---- export ----
    def _on_export(self) -> None:
        if self._session_id is None:
            self._status.setText("Pick a session to export first.")
            return
        from ..store.export import FORMATS, export
        filters = {"Markdown (*.md)": "md", "Subtitles (*.srt)": "srt",
                   "WebVTT (*.vtt)": "vtt", "JSON Lines (*.jsonl)": "jsonl"}
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export transcript", f"session-{self._session_id}.md",
            ";;".join(filters))
        if not path:
            return
        fmt = filters.get(chosen, "md")
        # Trust the extension the user actually typed over the filter dropdown.
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in FORMATS:
            fmt = ext
        try:
            text = export(self._ensure(), self._session_id, fmt)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._status.setText(f"Exported to {path}")
        except Exception as e:
            self._status.setText(f"Export failed: {e}")

    # ---- delete a session, and everything attached to it ----
    def _on_delete(self) -> None:
        """Remove a session's transcript, its notes, and its saved audio.

        Audio is the reason this exists: until now nothing ever deleted a session's
        WAV, so raw recordings of private conversations accumulated with no way to
        remove them from inside the app. The dialog lists every artefact by name and
        size, because "delete this session" should not quietly leave 90 MB of audio
        behind — nor delete it as a surprise.
        """
        if self._session_id is None:
            self._status.setText("Pick a session to delete first.")
            return
        sid = self._session_id
        conn = self._ensure()

        lines = conn.execute("SELECT COUNT(*) FROM utterances WHERE session_id=?",
                             (sid,)).fetchone()[0]
        items = [f"• {lines} transcript line(s)"]

        try:
            from ..notes import load_notes
            has_notes = load_notes(conn, sid) is not None
        except Exception:
            has_notes = False
        if has_notes:
            items.append("• the generated notes for this session")

        audio = None
        try:
            from ..capture.recorder import find_session_audio, format_bytes
            audio = find_session_audio(sid, settings=self._settings)
            if audio is not None:
                items.append(f"• the saved audio ({format_bytes(audio.stat().st_size)}) "
                             f"— re-diarizing this session will no longer be possible")
        except Exception:
            audio = None

        ok = QtWidgets.QMessageBox.question(
            self, "Delete this session?",
            f"Session {sid} will be permanently removed:\n\n" + "\n".join(items) +
            "\n\nThis cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if ok != QtWidgets.QMessageBox.StandardButton.Yes:
            self._status.setText("Nothing was deleted.")
            return

        removed = []
        from ..store.search import delete_session
        removed.append(f"{delete_session(conn, sid)} line(s)")
        if has_notes:
            try:
                from ..notes import delete_notes
                if delete_notes(conn, sid):
                    removed.append("notes")
            except Exception:
                pass
        if audio is not None:
            try:
                from ..capture.recorder import delete_session_audio
                gone, why = delete_session_audio(sid, settings=self._settings)
                # delete_session_audio returns a reason rather than printing, because
                # print() is a no-op in the windowed build — surface it.
                removed.append("audio" if gone else f"audio NOT deleted ({why})")
            except Exception as e:
                removed.append(f"audio NOT deleted ({e})")

        self._session_id = None
        self._text.clear()
        self.refresh()
        self._status.setText("Deleted: " + ", ".join(removed) + ".")

    # ---- rename a speaker ----
    def _on_rename(self) -> None:
        if self._session_id is None:
            self._status.setText("Pick a session first.")
            return
        from ..store.naming import session_labels
        from ..store.search import rename_speaker
        conn = self._ensure()
        labels = session_labels(conn, self._session_id)
        if not labels:
            self._status.setText("That session has no speaker labels to rename.")
            return
        old, ok = QtWidgets.QInputDialog.getItem(
            self, "Rename speaker", "Which speaker?", labels, 0, False)
        if not ok or not old:
            return
        new, ok = QtWidgets.QInputDialog.getText(
            self, "Rename speaker", f"New name for {old}:")
        new = (new or "").strip()
        if not ok or not new:
            return
        try:
            n = rename_speaker(conn, old, new, session_id=self._session_id)
        except Exception as e:
            self._status.setText(f"Rename failed: {e}")
            return
        # Renames are plain relabels, so they are reversible by swapping the two —
        # say so, rather than leaving the user wondering if it is permanent.
        self._status.setText(f"Renamed {n} line(s): {old} → {new}. "
                             f"Reversible: rename {new} back to {old}.")
        self._on_pick(self._list.currentItem(), None)
