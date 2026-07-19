"""The installer trim: prove it drops the dead Qt payload and nothing else.

A real build takes ~30 minutes in CI, so these exercise the spec's filter against
a realistic fake TOC instead. The failure mode being guarded is not "the filter
crashes" but "the filter matches nothing, saves 0 MB, and looks like it worked" —
so the no-op case is asserted as loudly as the happy path.
"""
import ast
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1] / "packaging" / "livecaptions.spec"

# The pure, testable surface of the spec. Everything else in the file calls
# Analysis()/COLLECT() and needs PyInstaller's injected globals to even parse-run.
WANTED = {"_is_software_opengl", "_is_qt_translation", "TRIM_RULES",
          "trim_toc", "trim_analyses"}


def _load_spec_helpers():
    src = SPEC.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(SPEC))

    def wanted(node):
        if isinstance(node, ast.Import):                      # the spec's `import os`
            return True
        if isinstance(node, ast.FunctionDef):
            return node.name in WANTED
        if isinstance(node, ast.Assign):
            return any(getattr(t, "id", None) in WANTED for t in node.targets)
        return False

    body = [n for n in tree.body if wanted(n)]
    # SourceFileLoader by hand: .spec is not a recognized source suffix, so
    # importlib cannot infer a loader for it.
    loader = importlib.machinery.SourceFileLoader("livecaptions_spec", str(SPEC))
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("livecaptions_spec", SPEC, loader=loader))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SPEC), "exec"), mod.__dict__)

    missing = WANTED - set(vars(mod))
    assert not missing, f"{SPEC.name} no longer defines {sorted(missing)}"
    return mod


S = _load_spec_helpers()

# Shaped like a real PyInstaller TOC: (dest name, source path, typecode), with
# Windows separators as the hooks emit them.
BINARIES = [
    ("PySide6\\opengl32sw.dll", "C:\\v\\Lib\\site-packages\\PySide6\\opengl32sw.dll", "BINARY"),
    ("PySide6\\Qt6Core.dll", "C:\\v\\Lib\\site-packages\\PySide6\\Qt6Core.dll", "BINARY"),
    ("PySide6\\Qt6Gui.dll", "C:\\v\\Lib\\site-packages\\PySide6\\Qt6Gui.dll", "BINARY"),
    ("PySide6\\Qt6Widgets.dll", "C:\\v\\Lib\\site-packages\\PySide6\\Qt6Widgets.dll", "BINARY"),
    ("PySide6\\Qt6OpenGL.dll", "C:\\v\\Lib\\site-packages\\PySide6\\Qt6OpenGL.dll", "BINARY"),
    ("PySide6\\Qt6OpenGLWidgets.dll",
     "C:\\v\\Lib\\site-packages\\PySide6\\Qt6OpenGLWidgets.dll", "BINARY"),
    ("PySide6\\QtOpenGL.pyd", "C:\\v\\Lib\\site-packages\\PySide6\\QtOpenGL.pyd", "EXTENSION"),
    ("PySide6\\plugins\\platforms\\qwindows.dll",
     "C:\\v\\Lib\\site-packages\\PySide6\\plugins\\platforms\\qwindows.dll", "BINARY"),
    ("python3.dll", "C:\\Python312\\python3.dll", "BINARY"),
    ("python312.dll", "C:\\Python312\\python312.dll", "BINARY"),
    ("ctranslate2\\ctranslate2.dll",
     "C:\\v\\Lib\\site-packages\\ctranslate2\\ctranslate2.dll", "BINARY"),
    ("nvidia\\cublas\\bin\\cublas64_12.dll",
     "C:\\v\\Lib\\site-packages\\nvidia\\cublas\\bin\\cublas64_12.dll", "BINARY"),
    ("hf_xet\\hf_xet.pyd", "C:\\v\\Lib\\site-packages\\hf_xet\\hf_xet.pyd", "EXTENSION"),
]

DATAS = [
    ("PySide6\\translations\\qtbase_de.qm",
     "C:\\v\\Lib\\site-packages\\PySide6\\translations\\qtbase_de.qm", "DATA"),
    ("PySide6\\translations\\assistant_ar.qm",
     "C:\\v\\Lib\\site-packages\\PySide6\\translations\\assistant_ar.qm", "DATA"),
    ("PySide6\\translations\\qt_help_ja.qm",
     "C:\\v\\Lib\\site-packages\\PySide6\\translations\\qt_help_ja.qm", "DATA"),
    ("PySide6\\qt.conf", "C:\\v\\Lib\\site-packages\\PySide6\\qt.conf", "DATA"),
    ("faster_whisper\\assets\\silero_vad_v6.onnx",
     "C:\\v\\Lib\\site-packages\\faster_whisper\\assets\\silero_vad_v6.onnx", "DATA"),
    ("nemo\\collections\\asr\\conf\\config.yaml",
     "C:\\v\\Lib\\site-packages\\nemo\\collections\\asr\\conf\\config.yaml", "DATA"),
    ("packaging\\licenses\\LICENSE.txt", "C:\\repo\\packaging\\licenses\\LICENSE.txt", "DATA"),
]


class FakeAnalysis:
    def __init__(self, binaries, datas):
        self.binaries = list(binaries)
        self.datas = list(datas)


def _names(toc):
    return {entry[0] for entry in toc}


def test_the_software_rasterizer_is_dropped():
    kept, dropped = S.trim_toc(BINARIES)
    assert _names(dropped["Qt software OpenGL rasterizer"]) == {"PySide6\\opengl32sw.dll"}
    assert "PySide6\\opengl32sw.dll" not in _names(kept)


def test_qt_translations_are_dropped():
    kept, dropped = S.trim_toc(DATAS)
    assert len(dropped["Qt UI translations"]) == 3
    assert not [n for n in _names(kept) if n.endswith(".qm")]


def test_the_real_qt_opengl_libraries_survive():
    """Qt6Gui links these whether or not the app draws with GL; only the ~20 MB
    software fallback is dead weight."""
    kept, _ = S.trim_toc(BINARIES)
    for name in ("PySide6\\Qt6OpenGL.dll", "PySide6\\Qt6OpenGLWidgets.dll",
                 "PySide6\\QtOpenGL.pyd"):
        assert name in _names(kept)


def test_nothing_else_is_touched():
    kept_b, _ = S.trim_toc(BINARIES)
    kept_d, _ = S.trim_toc(DATAS)
    assert _names(kept_b) == _names(BINARIES) - {"PySide6\\opengl32sw.dll"}
    assert _names(kept_d) == _names(DATAS) - {n for n in _names(DATAS) if n.endswith(".qm")}
    # Order matters: COLLECT writes the payload in TOC order.
    assert kept_b == [e for e in BINARIES if e[0] != "PySide6\\opengl32sw.dll"]


@pytest.mark.parametrize("name", [
    "PySide6/opengl32sw.dll",          # forward slashes
    "PySide6\\OPENGL32SW.DLL",         # Windows is case-insensitive
    "opengl32sw.dll",                  # collected to the bundle root
    "PySide6\\Qt\\translations\\qtbase_fr.qm",   # the older hook layout
    "PySide6/translations/qtbase_fr.qm",
])
def test_layout_variants_are_still_matched(name):
    kept, _ = S.trim_toc([(name, "C:\\src", "BINARY")])
    assert kept == []


@pytest.mark.parametrize("name", [
    "PySide6\\Qt6Core.dll",
    "translations.dll",                       # substring, not a path component
    "livecaptions\\ui\\translations.py",
    "docs\\opengl32sw.dll.txt",
])
def test_lookalikes_are_kept(name):
    kept, _ = S.trim_toc([(name, "C:\\src", "BINARY")])
    assert len(kept) == 1


def test_all_four_tocs_of_both_analyses_are_trimmed():
    """The CLI and overlay Analyses each carry their own binaries+datas; missing
    one leaves the payload in the shared _internal/."""
    cli, gui = FakeAnalysis(BINARIES, DATAS), FakeAnalysis(BINARIES, DATAS)
    counts = S.trim_analyses([cli, gui])

    for analysis in (cli, gui):
        assert "PySide6\\opengl32sw.dll" not in _names(analysis.binaries)
        assert not [n for n in _names(analysis.datas) if n.endswith(".qm")]
        assert "PySide6\\Qt6Core.dll" in _names(analysis.binaries)
        assert "faster_whisper\\assets\\silero_vad_v6.onnx" in _names(analysis.datas)

    assert counts == {"Qt software OpenGL rasterizer": 2, "Qt UI translations": 6}


def test_a_rule_that_matches_nothing_fails_the_build():
    """The whole point: a silently-inert filter saves 0 MB and looks like it worked."""
    clean = FakeAnalysis([e for e in BINARIES if e[0] != "PySide6\\opengl32sw.dll"], DATAS)
    with pytest.raises(SystemExit) as err:
        S.trim_analyses([clean])
    assert "Qt software OpenGL rasterizer" in str(err.value)
    assert "Qt UI translations" not in str(err.value), "blamed a rule that did fire"


def test_the_spec_actually_applies_the_trim():
    """A helper nobody calls is worth 0 MB too. Parsed, not grepped, so that
    commenting the call out counts as removing it."""
    tree = ast.parse(SPEC.read_text(encoding="utf-8"), filename=str(SPEC))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node)

    trims = calls.get("trim_analyses", [])
    assert trims, "the spec defines the trim but never runs it"
    fed = {n.id for n in ast.walk(trims[0]) if isinstance(n, ast.Name)}
    assert {"a_cli", "a_gui"} <= fed, "both Analyses must be trimmed, not just one"
    assert trims[0].lineno < min(n.lineno for n in calls["COLLECT"]), \
        "trim must run before COLLECT reads the TOCs"
