# Architecture (M1 foundation)

M1 turns the M0 single-file spike into a small layered package built around the
smallest seam M2 (overlay) and M3 (streaming) actually need. It does **not** add
features — same behavior as M0, reorganized, with inference moved off the audio
thread.

## Pipeline

```
AudioSource ──AudioBlock──▶ Segmenter ──Utterance──▶ WhisperWorker ──TranscriptEvent──▶ Sink
 (capture/)                  (asr/)      (thread hop)   (asr/)          (thread hop)      (ui/)
```

- **AudioSource** (`capture/`): produces `AudioBlock`s (mono float32 PCM at a
  device-native rate). Backends: WASAPI loopback (callback mode), WAV file.
- **Segmenter** (`asr/segmenter.py`): a *pure* RMS state machine — no I/O, fully
  unit-testable. Turns a stream of `AudioBlock`s into `Utterance`s at silence
  boundaries / a max-length cap. This is the keystone the characterization tests
  target (never on decoded Whisper text, which is non-deterministic).
- **WhisperWorker** (`asr/whisper.py`): a single background thread with a
  bounded, drop-oldest queue. Runs `faster-whisper` (one GPU worker only —
  `model.transcribe` is not safely concurrent, and 8 GB VRAM holds one `medium`
  model). Emits `TranscriptEvent`s.
- **TranscriptionSource** (`sources/`): the higher-level "produces caption
  events" seam. `LocalTranscriptionSource` wires capture → segmenter → worker.
  `FakeTranscriptionSource` emits canned events (UI/wiring tests, no audio/GPU).
  Cloud (M4) and Discord (M5) sources will implement the same interface.
- **Sink** (`ui/`): consumes `TranscriptEvent`s. M1 ships a terminal sink; M2
  adds the overlay. Sinks receive events from a background thread and must be
  thread-safe (or marshal to their own thread — the Qt overlay will).

## The two decisions the roadmap said to make before freezing the seam

### 1. Multi-source merge — DECIDED: independent sources, tagged events
Loopback and a future local **mic** (and later cloud/Discord) are **separate**
`TranscriptionSource`s. Every `TranscriptEvent` carries a `source` id
(`"loopback"`, `"mic"`, `"discord:alice"`). A sink merges multiple sources into
one transcript by `t_start`. Adding `MicCaptureSource` later needs **no schema
change** — this is why `source`, `speaker`, and `is_final` all live on the event
now, even though M1 only ever emits `source="loopback"`, `speaker=None`,
`is_final=True`. (The local mic is never present in the loopback stream, so for a
1:1 call the user's own side requires this second source — reserved here.)

### 2. Python runtime — REVERSED (2026-07-17): now CPython **3.12**
Originally 3.14: the whole M0–M3 stack (faster-whisper, ctranslate2,
PyAudioWPatch, soxr, numpy, PySide6 6.11.1, onnxruntime 1.27.0) has cp314 wheels,
so 3.14 was correct *for everything through M3*.

**What changed:** the diarization research (see
[diarization-research.md](diarization-research.md)) concluded local speaker
diarization is worth doing, and the good tools — **NVIDIA NeMo / Streaming
Sortformer** (live) and **pyannote.audio** (offline) — are **PyTorch-based, and
torch has no cp314 wheels**. Rather than split the app across two environments or
settle for the narrower torch-free (ONNX-only) toolset, we migrated the whole
project to **Python 3.12**, which every dependency supports.

Migrated 2026-07-17 and re-verified end-to-end on 3.12: 13/13 tests, GPU model
load (CUDA wheels + DLL preload), batch + streaming transcription, device
enumeration, and the overlay all pass. `requires-python` stays `>=3.11` (the
package code is version-agnostic); **3.12 is a runtime/venv choice driven by torch**,
so create the venv with `py -3.12 -m venv .venv`.

## Reserved / deferred (stated, not silently dropped)
Per the roadmap, these are **not** built in M1 and their dirs are not
pre-created: the `SourceCapabilities` enum, a provider factory, keyring, and the
`cloud/` `discord/` `diarize/` `store/` packages. They arrive in M4/M5 when a
real async or cross-process source forces their shape.

## Layout

```
src/livecaptions/
  events.py         AudioBlock, TranscriptEvent  (the narrow waist)
  config.py         Settings (pydantic-settings + TOML), device persistence
  capture/          cuda.py (DLL bootstrap), devices.py (WASAPI resolve),
                    wasapi.py (loopback AudioSource), wavfile.py (WAV AudioSource)
  asr/              segmenter.py (pure), whisper.py (off-thread worker + model load)
  sources/          base.py (TranscriptionSource), local.py, fake.py
  ui/               terminal.py (sink: captions + VU meter + watchdog)
  app.py            wires config -> source -> sink
  __main__.py       CLI (argparse) -> app;  python -m livecaptions
tests/
  test_segmenter.py characterization tests on the pure segmenter
```
