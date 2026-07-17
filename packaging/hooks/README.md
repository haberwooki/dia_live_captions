# Custom PyInstaller hooks

The spec adds this directory to `hookspath`. It's empty by design — the Tier A
collection logic lives inline in `livecaptions.spec` (via `collect_all` /
`collect_dynamic_libs`) so it's all visible in one place.

Add a `hook-<package>.py` here only when the first build's
`build/LiveCaptions/warn-LiveCaptions.txt` reveals a dynamic import that
`collect_all` in the spec didn't catch — NeMo and pyannote are the usual
culprits. Example:

```python
# hook-some.nemo.submodule.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
hiddenimports = collect_submodules("some.nemo.submodule")
datas = collect_data_files("some.nemo.submodule")
```
