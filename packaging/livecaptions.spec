# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller --onedir spec for live-captions (Tier A: everything bundled).

Build (from the repo root, in the project venv):
    pyinstaller packaging/livecaptions.spec --noconfirm

Produces  dist/LiveCaptions/  with two EXEs sharing one _internal/ payload:
    livecaptions.exe          console  — the CLI (all verbs)
    livecaptions-overlay.exe  windowed — double-click to launch the overlay

WHY --onedir, never --onefile: --onefile re-unpacks the whole payload (incl. the
~700 MB cuBLAS DLL) to a temp dir on every launch. --onedir unpacks once at
install.

The non-obvious collection decisions below are the output of the M7
investigation; each cites what it fixes. Freezing the NeMo/torch tree (Tier A)
is the hard part — after the FIRST build, read build/livecaptions/warn-*.txt and
add any still-missing dynamic imports to `hiddenimports` / `_safe_collect_all`.
"""
import glob
import importlib.util
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

REPO = os.path.dirname(os.path.abspath(SPECPATH))          # noqa: F821 (PyInstaller global)
SRC = os.path.join(REPO, "src")
PKG = os.path.join(REPO, "packaging")
ICON = os.path.join(PKG, "livecaptions.ico")
icon_arg = ICON if os.path.exists(ICON) else None

binaries = []
datas = []
hiddenimports = [
    "_portaudiowpatch",     # PyAudioWPatch's C ext is imported by bare name
    "hf_xet",               # huggingface_hub imports this dynamically for fast downloads
]

# --- faster-whisper's bundled Silero VAD model -------------------------------
# whisper.py hardcodes vad_filter=True, so silero_vad_v6.onnx MUST ship or every
# transcribe raises FileNotFoundError. Not collected by any default hook.
datas += collect_data_files("faster_whisper")

# --- CTranslate2's own DLLs --------------------------------------------------
# Keep ctranslate2.dll / cudnn64_9.dll / libiomp5md.dll INSIDE ctranslate2/ so
# the package's __init__ preload loop still fires. cudnn64_9.dll is bare-name
# loaded (default analysis misses it); cuDNN ships in this wheel, so DO NOT add
# an nvidia-cudnn-cu12 dependency.
binaries += collect_dynamic_libs("ctranslate2")


# --- NVIDIA cuBLAS (GPU Whisper) ---------------------------------------------
# ctranslate2 loads cublas by bare-name LoadLibrary at first inference, invisible
# to PyInstaller's analyzer, so we collect it explicitly. Only cuBLAS is needed
# for float16 medium inference (verified by a live residency probe): cudart is
# statically linked into ctranslate2.dll, and nvrtc (179 MB) never loads. Ship
# preserving nvidia/cublas/bin so capture/cuda.py's frozen search path finds it.
def _nvidia_bin(sub: str):
    spec = importlib.util.find_spec(f"nvidia.{sub}")
    if not spec or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], "bin")


_cublas_bin = _nvidia_bin("cublas")
if _cublas_bin:
    for dll in ("cublas64_12.dll", "cublasLt64_12.dll"):
        p = os.path.join(_cublas_bin, dll)
        if os.path.exists(p):
            binaries.append((p, os.path.join("nvidia", "cublas", "bin")))
else:
    print("WARNING: nvidia-cublas-cu12 not found — the frozen app will be CPU-only. "
          "Install it (pip install nvidia-cublas-cu12) before building the GPU bundle.")

# --- The dynamic-import ML tree (Tier A) -------------------------------------
# NeMo and pyannote resolve models/configs/submodules dynamically, so import-
# following alone misses data files and lazily-imported submodules. collect_all
# each present package; skip absent ones so the spec is portable. torch and
# transformers have hooks-contrib hooks, but collect_all is a safe superset.
def _safe_collect_all(name: str):
    if importlib.util.find_spec(name) is None:
        return
    ds, bs, hs = collect_all(name)
    datas.extend(ds)
    binaries.extend(bs)
    hiddenimports.extend(hs)


for _pkg in (
    "nemo", "pyannote", "lightning", "pytorch_lightning", "lightning_fabric",
    "torchmetrics", "asteroid_filterbanks", "omegaconf", "hydra",
    "sentencepiece", "huggingface_hub", "sherpa_onnx", "onnxruntime",
    "speechbrain", "julius",
):
    _safe_collect_all(_pkg)

# cuda-bindings (`cuda`) — a NAMESPACE package whose compiled .pyd extensions are
# invisible to collect_all AND collect_dynamic_libs (both return 0 binaries).
# NeMo's ASR decoding imports `cuda.bindings.cydriver` dynamically, so without
# this the frozen live-diarization path dies with "No module named
# cuda.bindings.cydriver". Verified: reconstructing the package tree on disk
# (.py via include_py_files + the .pyd globbed as binaries) imports cleanly in a
# frozen bundle. (This was the one real fix the first Tier A build surfaced.)
if importlib.util.find_spec("cuda") is not None:
    datas += collect_data_files("cuda", include_py_files=True)
    _cuda_root = list(importlib.util.find_spec("cuda").submodule_search_locations)[0]
    for _pyd in glob.glob(os.path.join(_cuda_root, "**", "*.pyd"), recursive=True):
        _rel = os.path.relpath(os.path.dirname(_pyd), _cuda_root)
        binaries.append((_pyd, os.path.join("cuda", _rel) if _rel != "." else "cuda"))
    hiddenimports += ["cuda", "cuda.bindings", "cuda.bindings.cydriver",
                      "cuda.bindings.cyruntime"]

# --- Conservative excludes ----------------------------------------------------
# Tier A bundles the ML stack, so we do NOT exclude torch/nemo/transformers.
# Only drop things that never run at inference. Keep this list small: an over-
# eager exclude that NeMo imports dynamically fails at runtime, not build time.
excludes = ["pytest", "_pytest", "pyinstaller", "PyInstaller"]

_common = dict(
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(PKG, "hooks")],
    excludes=excludes,
    noarchive=False,
)

a_cli = Analysis([os.path.join(PKG, "cli_launch.py")], **_common)      # noqa: F821
a_gui = Analysis([os.path.join(PKG, "overlay_launch.py")], **_common)  # noqa: F821

pyz_cli = PYZ(a_cli.pure)   # noqa: F821
pyz_gui = PYZ(a_gui.pure)   # noqa: F821

exe_cli = EXE(   # noqa: F821
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True,
    name="livecaptions",
    console=True,
    icon=icon_arg,
)
exe_gui = EXE(   # noqa: F821
    pyz_gui, a_gui.scripts, [],
    exclude_binaries=True,
    name="livecaptions-overlay",
    console=False,       # windowed: no console flash; stdio -> log file
    icon=icon_arg,
)

# One COLLECT for both EXEs — shared _internal/, files deduped by name.
coll = COLLECT(   # noqa: F821
    exe_cli, a_cli.binaries, a_cli.datas,
    exe_gui, a_gui.binaries, a_gui.datas,
    strip=False,
    upx=False,            # UPX + these native DLLs is a known crash source; leave off
    name="LiveCaptions",
)
