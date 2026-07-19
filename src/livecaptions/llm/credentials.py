"""API keys, stored in Windows Credential Manager — never in config.toml.

config.toml is plain text in a folder people copy around, paste into issues and
back up to cloud drives. A key does not belong there. The Credential Manager is
the OS-provided store, per-user and DPAPI-protected, so that is where keys go.

Implemented with ctypes against advapi32 rather than a library: no new dependency
in a 947 MB installer, and nothing extra for PyInstaller to collect.
"""
from __future__ import annotations

import sys
from typing import Optional

TARGET_PREFIX = "live-captions"

CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2


def _target(name: str) -> str:
    return f"{TARGET_PREFIX}:{name}"


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD)]

    class _CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", _FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    _advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    _advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.POINTER(ctypes.POINTER(_CREDENTIAL))]
    _advapi.CredReadW.restype = wintypes.BOOL
    _advapi.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIAL), wintypes.DWORD]
    _advapi.CredWriteW.restype = wintypes.BOOL
    _advapi.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    _advapi.CredDeleteW.restype = wintypes.BOOL
    _advapi.CredFree.argtypes = [ctypes.c_void_p]
    _advapi.CredFree.restype = None

    def set_secret(name: str, secret: str) -> None:
        blob = secret.encode("utf-16-le")          # what the API expects
        buf = (ctypes.c_byte * len(blob)).from_buffer_copy(blob)
        cred = _CREDENTIAL(
            Flags=0, Type=CRED_TYPE_GENERIC, TargetName=_target(name),
            Comment="Live Captions API key", CredentialBlobSize=len(blob),
            CredentialBlob=ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)),
            Persist=CRED_PERSIST_LOCAL_MACHINE, AttributeCount=0, Attributes=None,
            TargetAlias=None, UserName=name)
        if not _advapi.CredWriteW(ctypes.byref(cred), 0):
            raise OSError(f"couldn't save the key (error {ctypes.get_last_error()})")

    def get_secret(name: str) -> Optional[str]:
        ptr = ctypes.POINTER(_CREDENTIAL)()
        if not _advapi.CredReadW(_target(name), CRED_TYPE_GENERIC, 0, ctypes.byref(ptr)):
            return None
        try:
            cred = ptr.contents
            if not cred.CredentialBlobSize:
                return None
            raw = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
            return raw.decode("utf-16-le")
        finally:
            _advapi.CredFree(ptr)

    def delete_secret(name: str) -> bool:
        return bool(_advapi.CredDeleteW(_target(name), CRED_TYPE_GENERIC, 0))

else:                                   # non-Windows: no store, and no silent file fallback
    def set_secret(name: str, secret: str) -> None:
        raise OSError("Credential storage is only implemented on Windows. "
                      "Set the key in the environment instead.")

    def get_secret(name: str) -> Optional[str]:
        return None

    def delete_secret(name: str) -> bool:
        return False


def resolve_key(provider: str, env_var: Optional[str] = None) -> Optional[str]:
    """The key for a provider: environment first, then Credential Manager.

    Environment wins so a key exported for a session (or by a parent process, or
    an `ant auth login` profile for Anthropic) is never shadowed by a stale saved
    one. Returns None when there is no key — callers must treat that as
    "unconfigured", not as an error to paper over.
    """
    import os
    if env_var:
        value = os.environ.get(env_var)
        if value:
            return value.strip()
    try:
        return get_secret(provider)
    except OSError:
        return None
