"""Global hotkeys via Win32 RegisterHotKey.

Why not QShortcut: the overlay is a click-through, never-focused Tool window, so
Qt shortcuts never fire. Why not the `keyboard` package: it installs a low-level
keyboard hook (i.e. a keylogger) — RegisterHotKey asks the OS for exactly the
combos we want and nothing else.

Windows delivers WM_HOTKEY to the registered window; we catch it with a
QAbstractNativeEventFilter and dispatch to a callback. If a combo is already
claimed by another app, RegisterHotKey fails — we say so and carry on so the
user can remap it in config.toml.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Callable, Dict, Optional, Tuple

from PySide6 import QtCore

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN = 0x0001, 0x0002, 0x0004, 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

_MODS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT,
         "shift": MOD_SHIFT, "win": MOD_WIN, "super": MOD_WIN}
_KEYS = {"left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
         "space": 0x20, "esc": 0x1B, "escape": 0x1B,
         **{f"f{i}": 0x6F + i for i in range(1, 13)}}


def parse_hotkey(spec: str) -> Tuple[int, int]:
    """'ctrl+alt+c' -> (modifiers, virtual-key). Raises ValueError on nonsense."""
    mods, vk = 0, None
    for part in (p.strip().lower() for p in spec.split("+")):
        if part in _MODS:
            mods |= _MODS[part]
        elif part in _KEYS:
            vk = _KEYS[part]
        elif len(part) == 1:
            vk = ord(part.upper())
        else:
            raise ValueError(f"unknown key {part!r} in hotkey {spec!r}")
    if vk is None:
        raise ValueError(f"hotkey {spec!r} has no key, only modifiers")
    return mods, vk


class GlobalHotkeys(QtCore.QAbstractNativeEventFilter):
    def __init__(self, hwnd: int):
        super().__init__()
        self._hwnd = hwnd
        self._handlers: Dict[int, Callable[[], None]] = {}
        self._next_id = 1
        self._user32 = ctypes.windll.user32

    def register(self, spec: str, callback: Callable[[], None],
                 label: Optional[str] = None) -> bool:
        try:
            mods, vk = parse_hotkey(spec)
        except ValueError as e:
            print(f"(bad hotkey config: {e})")
            return False
        hid = self._next_id
        self._next_id += 1
        # MOD_NOREPEAT: one event per press, not a stream while held
        if not self._user32.RegisterHotKey(wintypes.HWND(self._hwnd), hid,
                                           mods | MOD_NOREPEAT, vk):
            print(f"(hotkey {spec} for {label or 'action'} is already claimed by another app "
                  f"- remap it in config.toml)")
            return False
        self._handlers[hid] = callback
        return True

    def nativeEventFilter(self, event_type, message):
        if event_type in (b"windows_generic_MSG", "windows_generic_MSG"):
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY:
                cb = self._handlers.get(int(msg.wParam))
                if cb is not None:
                    cb()
                    return True, 0
        return False, 0

    def unregister_all(self) -> None:
        for hid in list(self._handlers):
            self._user32.UnregisterHotKey(wintypes.HWND(self._hwnd), hid)
        self._handlers.clear()
