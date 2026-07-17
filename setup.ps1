# One-time setup for live-captions.
# Run from this folder in PowerShell:  .\setup.ps1
#
# If you get "running scripts is disabled on this system", allow local scripts
# for your user once with:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Python 3.12 specifically: the diarization stack (NeMo / pyannote) is PyTorch-based
# and torch has no wheels for 3.13+. Everything else supports 3.12 too.
# Install it once with:  winget install -e --id Python.Python.3.12 --scope user
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# The livecaptions package + the always-on-top overlay (PySide6).
pip install -e ".[gui,dev]"

# NVIDIA GPU acceleration (CUDA 12). Harmless on non-NVIDIA machines, but if you
# have no NVIDIA GPU you can comment this out to skip a ~630 MB download - the app
# falls back to CPU automatically.
pip install -r requirements-gpu.txt

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run it with:  python -m livecaptions --streaming --overlay"
