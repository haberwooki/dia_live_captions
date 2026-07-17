# Live Captions

A cross-app live-captions overlay for Windows: it captures your system audio and
shows live transcription in an always-on-top overlay, transcribed locally with
`faster-whisper` on your GPU (automatic CPU fallback). It can also colour captions
by speaker (live diarization), save and search every session, and optionally name
speakers. Developed on Windows 11 + RTX 3070, but it detects your hardware at
runtime and adapts (any Windows PC, GPU or CPU).

Milestones **M0–M7** are done — streaming ASR, the overlay, offline and live
speaker diarization, saved/searchable transcripts, global hotkeys, and a Windows
installer. See [ROADMAP.md](ROADMAP.md) and [docs/architecture.md](docs/architecture.md).

## Install

**Just want to use it (Windows):** download the latest `LiveCaptions-Setup-*.exe`
from the [Releases page](https://github.com/haberwooki/dia_live_captions/releases)
and run it. It installs per-user (no admin); it's unsigned, so Windows SmartScreen
will warn — click **More info → Run anyway**. First launch downloads the model
weights (~1–2 GB) into `%LOCALAPPDATA%`. No Python or CUDA toolkit required.

> No release published yet? You (or anyone) can build the installer from source —
> see [Build the Windows installer](#build-the-windows-installer) below.

**Run from source (developers):** clone and follow the Quickstart.

## Quickstart

```powershell
git clone https://github.com/haberwooki/dia_live_captions
cd dia_live_captions
# Python 3.12 specifically (the diarization stack is torch-based; torch has no 3.13+ wheels):
#   winget install -e --id Python.Python.3.12 --scope user
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[gui]"                    # the livecaptions package + overlay (PySide6)
pip install -r requirements-gpu.txt        # NVIDIA GPU only; skip for CPU-only
python -m livecaptions --streaming --overlay   # continuous captions in an overlay
python -m livecaptions                         # or utterance-at-a-time in the terminal
```

> Or just run `.\setup.ps1`.

Options: `--streaming` (continuous partial+final captions, LocalAgreement-2 +
Silero VAD), `--overlay` (always-on-top caption bar), `--demo` (overlay demo, no
audio/GPU), `--movable` (drag the overlay — remembers where), `--opacity F`;
`--list-devices`, `--device <substr>`, `--pick`, `--loopback-index N`,
`--wav clip.wav` (deterministic, no device), `--fake`, `--model <name>`, `--cpu`.
Your device choice and overlay position are remembered
(`%LOCALAPPDATA%\live-captions\`; `LC_*` env vars also work).

### Live speaker colours

```powershell
python -m livecaptions --diarize-live --overlay
```

Runs **NVIDIA Streaming Sortformer** alongside the streaming ASR and colours each
caption line by speaker. Runs on **CPU** (~0.4× real time, no VRAM — the GPU stays
free for Whisper). Max **4 speakers**.

> Best-effort: on 2-speaker, low-overlap audio it's accurate; on crowded, overlapping
> calls expect mistakes. It's a single mixed stream — see the caveat below.

### Who said what (offline diarization)

Post-process a recording into a speaker-labeled transcript:

```powershell
python -m livecaptions --diarize call.wav                 # auto backend
python -m livecaptions --diarize call.wav --num-speakers 2 --out transcript.txt
```
```
[00:00.00] SPEAKER_00: Hey Sarah, did you get a chance to look at the results?
[00:05.68] SPEAKER_01: I did, and the speaker separation looked better than I expected.
```

Two backends (`--diarizer`):
- **`sherpa`** — no account, ONNX/CPU, models auto-downloaded (`--download-models`). Works out of the box.
- **`pyannote`** — best quality, needs PyTorch + a **free Hugging Face token** and accepting the
  model terms at [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1);
  then `huggingface-cli login` (or set `HF_TOKEN`).
- **`auto`** (default) — pyannote if a token is available, else sherpa.

> Diarization runs on a *single mixed* loopback stream, so labels are best-effort: reliable on
> 2-speaker/low-overlap audio, weaker on crowded calls. See [docs/diarization-research.md](docs/diarization-research.md).

### Saved transcripts & search

Every session is saved automatically to a local SQLite database
(`%LOCALAPPDATA%\live-captions\transcripts.db`). Use `--no-save` to opt out.

```powershell
python -m livecaptions --sessions                        # list recent sessions
python -m livecaptions --search "quarterly results"      # full-text search (BM25-ranked)
python -m livecaptions --search "\"exact phrase\"" --speaker Sarah --since 2026-07-01
python -m livecaptions --export srt --session 3 --out call.srt   # srt | vtt | jsonl | md
```

Search uses SQLite FTS5 where the build has it, and falls back to a `LIKE` scan where it doesn't.

### Naming speakers

Rename a diarization label by hand — always reversible, just swap the two:

```powershell
python -m livecaptions --rename-speaker SPEAKER_01=Sarah --session 3
```

Or have **Claude** read the transcript and suggest names from what people call each other:

```powershell
python -m livecaptions --name-speakers --session 3
```

> ⚠️ **This is the only feature that leaves your machine.** It sends the session's transcript
> text to the Anthropic API. It's off by default, tells you exactly how much text it will send
> and asks before sending, then shows the quote behind each suggested name and asks before
> applying it. Needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile) — no key is ever read
> from `config.toml`. Model defaults to `claude-opus-4-8` (`--name-model` to change).

### Hotkeys

Global, and work while the overlay is click-through and unfocused:

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+C` | show / hide the overlay |
| `Ctrl+Alt+P` | pause / resume captions |
| `Ctrl+Alt+` arrows | nudge the overlay |

If another app already owns a combo, registration fails with a note — remap it in `config.toml`
(`hotkey_toggle`, `hotkey_pause`, `hotkey_left`/`right`/`up`/`down`, `hotkey_nudge_px`), or set
`hotkeys_enabled = false`.

## Project structure

```
live-captions/
├── src/livecaptions/       # the M1 package (python -m livecaptions)
│   ├── events.py           #   AudioBlock, TranscriptEvent (the narrow-waist seam)
│   ├── config.py           #   pydantic-settings + TOML
│   ├── capture/            #   WASAPI loopback + WAV audio sources, CUDA bootstrap
│   ├── asr/                #   segmenter, off-thread worker, streaming (LocalAgreement-2), VAD
│   ├── sources/            #   TranscriptionSource: local (batch), streaming, fake, demo
│   ├── diarize/            #   "who is talking": pyannote/sherpa offline, Sortformer live
│   ├── store/              #   SQLite+FTS5 transcripts, writer thread, search, export, LLM naming
│   ├── ui/                 #   terminal sink, always-on-top overlay (PySide6), global hotkeys
│   └── app.py, __main__.py #   wiring + CLI
├── tests/                  # 45 tests: segmenter, hypothesis, assign, store, hotkeys, naming
├── docs/architecture.md    # design + the M1 decisions
├── m0_live_captions.py     # the original M0 single-file spike (reference)
├── pyproject.toml · requirements-gpu.txt · setup.ps1 · ROADMAP.md
└── .vscode/
```

## Open in VS Code

1. Open this folder in VS Code (`File → Open Folder…`, pick `live-captions`).
2. When prompted, install the recommended extensions (Python, Pylance, debugpy).
3. Create the environment and install dependencies — either run `setup.ps1`:
   ```powershell
   .\setup.ps1
   ```
   or do it manually:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install -r requirements-gpu.txt   # NVIDIA GPU only; skip for CPU-only
   ```
4. VS Code should auto-select `.venv` as the interpreter (bottom-right status bar).
   If not, `Ctrl+Shift+P → Python: Select Interpreter → .venv`.

> If PowerShell blocks the activate script ("running scripts is disabled"), run
> once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Run

- Press **F5** (uses the "Run M0 live captions" debug config), or
- In the integrated terminal: `python m0_live_captions.py`

Then play any audio — a Discord/Teams/Meet call, or a YouTube video — and
timestamped lines will print, each tagged with the clip length and how long your
GPU took to transcribe it. A live level meter shows the incoming audio; if nothing
is captured after a few seconds it tells you (wrong/idle output device) instead of
sitting silent. `Ctrl+C` to stop. The first run downloads the Whisper weights (a
few hundred MB for `medium`).

### Options

```powershell
python m0_live_captions.py --list-devices     # show loopback capture devices and exit
python m0_live_captions.py --device DELL       # capture the output whose name contains "DELL" (remembered)
python m0_live_captions.py --pick              # choose a device interactively (remembered)
python m0_live_captions.py --loopback-index 14 # capture an exact device index (see --list-devices)
python m0_live_captions.py --wav clip.wav      # transcribe a WAV file — deterministic, no capture device
```

Your device choice is remembered in `%APPDATA%\live-captions\config.json`. `--wav` is the
reproducible test path (it drives the same segmenter/transcriber with no audio hardware).

> **No sound card / testing the real capture path?** Route audio through a virtual output like
> [VB-CABLE](https://vb-audio.com/Cable/) and select it — loopback needs an output device that is
> actually carrying an audio stream.

## Tuning (top of `m0_live_captions.py`)

- `MODEL_NAME` — `medium` is a good 8 GB-GPU balance; `large-v3` for best accuracy
  (still fits 8 GB in float16), `small` for lowest latency.
- `DEVICE` — `auto` uses the GPU if it works and falls back to CPU otherwise;
  force `cuda` or `cpu` if you want. `GPU_COMPUTE`/`CPU_COMPUTE` set the precision.
- `LOOPBACK_INDEX` — leave `None` to auto-pick the default output's loopback. If
  you have several identical output devices and it grabs a silent one, set this to
  the right index (the run prints which indices exist).
- `SILENCE_RMS` — raise if utterances never end (nothing prints); lower if lines
  get cut off mid-sentence. Depends on your system volume.
- `LANGUAGE` — `None` for auto multilingual (slower). `BEAM_SIZE` — `1` fast, `5`
  more accurate.

## GPU troubleshooting

`faster-whisper` (CTranslate2) needs CUDA 12 libraries. CTranslate2 bundles cuDNN 9
but **not** cuBLAS or the CUDA runtime, so `requirements-gpu.txt` supplies them
(`nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`); the app adds their DLL dirs to
the search path and preloads them at startup.

- If you skipped `requirements-gpu.txt`, install it: `pip install -r requirements-gpu.txt`.
- The app validates the GPU with a tiny self-test at startup and **auto-falls back
  to CPU** (`int8`) if it fails — you'll see `Model ready on cpu ...`. That's slower
  but confirms the pipeline while you sort out CUDA. Keep your NVIDIA driver current.
- Force behavior with `DEVICE = "cuda"` or `DEVICE = "cpu"` at the top of the script.

## Build the Windows installer

Everything for packaging lives in [packaging/](packaging/). It produces a per-user Windows
installer that bundles the whole app — GPU Whisper, the overlay, offline **and** live diarization,
and LLM naming (weights download on first run). Nothing is set up globally: it installs under
`%LOCALAPPDATA%`, and uninstalling **keeps your saved transcripts** by default.

```powershell
pip install -e ".[all,dev]"                       # PyInstaller + the full feature set
# put vc_redist.x64.exe in packaging\  (aka.ms/vs/17/release/vc_redist.x64.exe)
.\packaging\build.ps1                              # -> Output\LiveCaptions-Setup-0.1.0.exe
.\packaging\build.ps1 -SkipInstaller               # just the dist\LiveCaptions bundle
```

Needs **Python 3.12** (torch has no 3.13/3.14 wheels) and **Inno Setup 6.3+** on PATH for the
installer step. The bundle is ~2.2 GB unpacked (it includes the torch/NeMo stack for live speaker
colours); the compressed installer is smaller. The build script checks that the load-bearing pieces
(cuBLAS, CTranslate2, the Silero VAD model) actually landed.

The PyInstaller bundle has been built and run on an RTX 3070 — GPU Whisper, live diarization, and the
overlay all work frozen. The Inno installer compile and fresh-/non-NVIDIA-box testing are the parts
still to do (see [ROADMAP.md](ROADMAP.md) M7).

> The freeze of the NeMo/pyannote stack is the fragile part — if a NeMo/torch bump makes the frozen
> app hit a `ModuleNotFoundError`, read `build\LiveCaptions\warn-LiveCaptions.txt`, find the missing
> dynamic import, and add it to the spec (or a hook in `packaging\hooks\`). The one known case
> (`cuda.bindings.cydriver`, a namespace-package `.pyd`) is already handled in the spec.

## Roadmap

- **M0** (here) — system-audio capture → live transcription in the terminal.
- **M1** — always-on-top overlay window instead of terminal text.
- **M2** — streaming speaker diarization (who's talking) for the universal path.
- **M3** — overlay polish, hotkeys, saved/searchable transcripts, optional LLM
  name assignment.
- **M4** — cloud STT/diarization option (Deepgram/AssemblyAI) selectable in settings.
- **M5** — Discord bot source (fork Scripty) for perfect per-speaker labels.
