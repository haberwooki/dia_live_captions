"""System tray icon + menu for the overlay.

The overlay is a frameless, click-through, always-on-top window with no taskbar
button — so without this there is no visible way to quit, pause, or even tell it's
running. The tray icon is that anchor: its presence is the "it's running" signal,
and its menu gives Quit / Pause / Show-Hide (and, later, Settings).

The icon is drawn in code so there's no asset file to ship (and nothing for
PyInstaller to miss).
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


def make_icon() -> QtGui.QIcon:
    """A small speech-bubble 'CC' badge, drawn at a few sizes for crisp scaling."""
    icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64):
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        m = size * 0.08
        rect = QtCore.QRectF(m, m, size - 2 * m, size - 2 * m * 1.4)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(20, 20, 24, 235))
        p.drawRoundedRect(rect, size * 0.22, size * 0.22)
        # little tail
        tail = QtGui.QPolygonF([
            QtCore.QPointF(size * 0.30, rect.bottom() - 1),
            QtCore.QPointF(size * 0.30, rect.bottom() + size * 0.20),
            QtCore.QPointF(size * 0.52, rect.bottom() - 1),
        ])
        p.drawPolygon(tail)
        f = QtGui.QFont("Segoe UI", int(size * 0.34))
        f.setWeight(QtGui.QFont.Weight.Black)
        p.setFont(f)
        p.setPen(QtGui.QColor(130, 205, 255))
        p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "CC")
        p.end()
        icon.addPixmap(pm)
    return icon


def install_tray(app: QtWidgets.QApplication, overlay, *, on_quit,
                 on_settings=None) -> QtWidgets.QSystemTrayIcon | None:
    """Add a tray icon whose menu controls the overlay. Returns it (keep a
    reference alive) or None if the platform has no system tray."""
    if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
        return None

    tray = QtWidgets.QSystemTrayIcon(make_icon(), app)
    tray.setToolTip("Live Captions")
    menu = QtWidgets.QMenu()

    act_show = menu.addAction("Show / hide overlay")
    act_show.triggered.connect(overlay.toggle_visible)

    act_pause = menu.addAction("Pause captions")
    act_pause.triggered.connect(overlay.toggle_paused)

    menu.addSeparator()
    act_settings = menu.addAction("Settings…")
    if on_settings is not None:
        act_settings.triggered.connect(on_settings)
    else:
        act_settings.setEnabled(False)

    menu.addSeparator()
    act_quit = menu.addAction("Quit Live Captions")
    act_quit.triggered.connect(on_quit)

    # Keep the Pause label in sync with the actual state each time the menu opens.
    def _sync():
        act_pause.setText("Resume captions" if getattr(overlay, "_paused", False)
                          else "Pause captions")
    menu.aboutToShow.connect(_sync)

    tray.setContextMenu(menu)
    # Left-click toggles the overlay (right-click opens the menu).
    def _activated(reason):
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.Trigger:
            overlay.toggle_visible()
    tray.activated.connect(_activated)

    tray.show()
    tray._menu = menu   # prevent GC of the menu
    return tray
