"""Make CUDA libraries from the nvidia-*-cu12 pip wheels loadable.

CTranslate2 (faster-whisper's backend) loads cuBLAS dynamically by *bare name*
via LoadLibrary at first inference — a call PyInstaller's dependency analyzer
can't see and the OS won't resolve on its own. So we add the DLLs' directory to
the search path and preload them by name; the later bare-name LoadLibrary then
returns the already-resident module. Best-effort and silent: on CPU-only
machines the wheels aren't present and we run on CPU.

Two layouts to handle:
- **Source run:** the DLLs are in ``site-packages/nvidia/*/bin`` (pip wheels).
- **Frozen build (PyInstaller --onedir):** the build hook copies cuBLAS next to
  the app; ``sys._MEIPASS`` is the search root, and ``sysconfig`` no longer
  points anywhere useful. We look there instead.

Preload order matters: ``cublas64_12.dll`` statically imports
``cublasLt64_12.dll``, so cuBLASLt must be resident first.
"""
from __future__ import annotations

import glob
import os
import sys

# cuBLASLt before cuBLAS (cublas depends on it). cudart is statically linked
# into ctranslate2.dll and never actually dlopened, but preloading it is cheap
# and harmless, so we keep it for older/edge ct2 builds.
_PRELOAD = ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll")


def _search_dirs() -> list[str]:
    """Directories that may hold the nvidia CUDA DLLs, most-specific first."""
    if getattr(sys, "frozen", False):
        # --onedir: _MEIPASS is <app>/_internal. The build hook may place the
        # DLLs at the root or preserve the nvidia/cublas/bin subtree; cover both.
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return [base,
                os.path.join(base, "nvidia", "cublas", "bin"),
                os.path.join(base, "nvidia", "cuda_runtime", "bin")]
    import sysconfig
    site = sysconfig.get_paths()["purelib"]
    return glob.glob(os.path.join(site, "nvidia", "*", "bin"))


def bootstrap_cuda_dlls() -> None:
    if os.name != "nt":
        return
    import ctypes

    for d in _search_dirs():
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    for name in _PRELOAD:
        try:
            ctypes.WinDLL(name)
        except OSError:
            pass
