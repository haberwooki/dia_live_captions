"""Tests for hotkey spec parsing (pure; no Win32 calls)."""
import pytest

pytest.importorskip("PySide6")

from livecaptions.ui.hotkeys import (  # noqa: E402
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    parse_hotkey,
)


def test_parses_modifiers_and_letter():
    mods, vk = parse_hotkey("ctrl+alt+c")
    assert mods == MOD_CONTROL | MOD_ALT
    assert vk == ord("C")


def test_case_and_spacing_insensitive():
    assert parse_hotkey(" Ctrl + Alt + C ") == parse_hotkey("ctrl+alt+c")


def test_named_keys():
    assert parse_hotkey("ctrl+alt+left")[1] == 0x25
    assert parse_hotkey("ctrl+alt+down")[1] == 0x28
    assert parse_hotkey("shift+f5") == (MOD_SHIFT, 0x74)


def test_control_alias():
    assert parse_hotkey("control+p") == parse_hotkey("ctrl+p")


def test_rejects_modifiers_only():
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+alt")


def test_rejects_unknown_key():
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+alt+notakey")
