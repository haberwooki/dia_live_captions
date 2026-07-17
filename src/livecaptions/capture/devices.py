"""WASAPI loopback device enumeration and selection.

Ports M0.1's device logic: robust to machines with multiple identically-named
outputs (e.g. two identical monitors), an explicit index, a name substring, an
interactive pick, or a remembered choice (by loopback name + ordinal, since
PortAudio indices are volatile).
"""
from __future__ import annotations

from typing import List, Optional

import pyaudiowpatch as pyaudio


def enumerate_loopbacks(p) -> List[dict]:
    return list(p.get_loopback_device_info_generator())


def name_ordinal(loopbacks: List[dict], dev: dict) -> int:
    """Position of `dev` among loopbacks sharing its exact name (0-based)."""
    same = [lb for lb in loopbacks if lb["name"] == dev["name"]]
    for i, lb in enumerate(same):
        if lb["index"] == dev["index"]:
            return i
    return 0


def default_loopback(p) -> dict:
    """Loopback for the default output, disambiguating duplicate names by
    mapping the default output's position among same-named render devices to the
    same position in the loopback list."""
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if default_speakers.get("isLoopbackDevice", False):
        return default_speakers

    name = default_speakers["name"]
    matches = [lb for lb in p.get_loopback_device_info_generator() if name in lb["name"]]
    if not matches:
        raise RuntimeError("No loopback device found for the default output. "
                           "Make sure you're on Windows with WASAPI.")
    if len(matches) == 1:
        return matches[0]

    same_named = []
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if (d["name"] == name and d.get("hostApi") == wasapi["index"]
                and d.get("maxOutputChannels", 0) > 0
                and not d.get("isLoopbackDevice", False)):
            same_named.append(d["index"])
    try:
        pos = same_named.index(default_speakers["index"])
    except ValueError:
        pos = 0
    return matches[min(pos, len(matches) - 1)]


def load_saved(loopbacks: List[dict], name: Optional[str], ordinal: int) -> Optional[dict]:
    if not name:
        return None
    same = [lb for lb in loopbacks if lb["name"] == name]
    if not same:
        return None
    return same[min(ordinal, len(same) - 1)]


def interactive_pick(loopbacks: List[dict]) -> dict:
    print("Loopback capture devices:")
    for i, lb in enumerate(loopbacks):
        print(f"  [{i}] index {lb['index']}: {lb['name']} "
              f"({int(lb['defaultSampleRate'])} Hz, {lb['maxInputChannels']} ch)")
    while True:
        try:
            choice = int(input("Pick a number: "))
            if not 0 <= choice < len(loopbacks):
                raise IndexError
            return loopbacks[choice]
        except (ValueError, IndexError):
            print("  invalid choice, try again")
        except EOFError:
            raise SystemExit("--pick needs an interactive terminal")


def print_device_list(p) -> None:
    loopbacks = enumerate_loopbacks(p)
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    try:
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        default_lb = default_loopback(p)
    except Exception:
        default_out = None
        default_lb = None
    if default_out is not None:
        print(f"Default output: {default_out['name']} (index {default_out['index']})\n")
    else:
        print("Default output: (none detected)\n")
    print("Loopback capture devices:")
    for lb in loopbacks:
        mark = "   <- default (auto)" if (default_lb and lb["index"] == default_lb["index"]) else ""
        print(f"  index {lb['index']}: {lb['name']} "
              f"({int(lb['defaultSampleRate'])} Hz, {lb['maxInputChannels']} ch){mark}")


def resolve_loopback(p, *, index: Optional[int] = None, device_substr: Optional[str] = None,
                     pick: bool = False, saved_name: Optional[str] = None,
                     saved_ordinal: int = 0) -> dict:
    """Pick the loopback to capture: index > substr > pick > saved > default."""
    loopbacks = enumerate_loopbacks(p)

    if index is not None:
        try:
            dev = p.get_device_info_by_index(index)
        except (OSError, ValueError):
            raise SystemExit(f"No device at index {index}. Try --list-devices.")
        if not dev.get("isLoopbackDevice", False):
            raise SystemExit(f"Index {index} ('{dev['name']}') is not a loopback capture "
                             f"device. Try --list-devices.")
        return dev

    if device_substr:
        matches = [lb for lb in loopbacks if device_substr.lower() in lb["name"].lower()]
        if not matches:
            raise SystemExit(f"No loopback device name contains '{device_substr}'. "
                             f"Try --list-devices.")
        if len(matches) > 1:
            try:
                dfl = default_loopback(p)
            except Exception:
                dfl = None
            chosen = dfl if (dfl and any(m["index"] == dfl["index"] for m in matches)) else matches[0]
            print(f"(--device '{device_substr}' matched {len(matches)} devices; using "
                  f"index {chosen['index']} - use --loopback-index to force another)")
            return chosen
        return matches[0]

    if pick:
        return interactive_pick(loopbacks)

    saved = load_saved(loopbacks, saved_name, saved_ordinal)
    if saved is not None:
        return saved

    try:
        return default_loopback(p)
    except (RuntimeError, OSError):
        raise SystemExit(
            "No usable audio output/loopback device found. Make sure something is set as your "
            "Windows default playback device (a sleeping monitor or unplugged output can remove "
            "it). Run --list-devices to see what's available.")
