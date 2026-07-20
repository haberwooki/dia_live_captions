# Session Handoff

> Maintained by Claude during work sessions. Any fresh session should be able to
> read this file and continue with zero re-explanation. Keep it under ~150 lines —
> move finished history to the archive, don't let it grow forever.

**Last updated:** 2026-07-19 — shipped v0.3.4; roadmap complete, awaiting real-world testing

## Goal

A Windows live-captions overlay that runs entirely on the local machine: WASAPI
loopback → faster-whisper (streaming, LocalAgreement-2) → always-on-top overlay,
with optional local speaker diarization, a transcript store, and an optional AI
layer for speaker naming and session notes. Public repo:
`haberwooki/dia_live_captions`; CI builds a PyInstaller+Inno installer per tag.

## Current state

### Done
- v0.1.x: startup 18–20s → ~2s; caption freeze fixes; in-app updater with relaunch.
- v0.2.x: tabbed control panel; real Start/Pause/Stop transport; Transcripts tab;
  AI tab (Claude / any OpenAI-compatible / local Ollama+LM Studio); duplicate-device
  disambiguation; "resume how I left it" startup.
- v0.3.x: Speakers tab (live colours, offline re-diarization, audio-saving switch);
  Advanced tab (all tuning + hotkey remapping); optional audio recording; notes
  engine + UI; full session deletion (transcript + FTS + notes + audio).
- v0.3.3–v0.3.4: one-click notes (local-server discovery), real-server response-format
  fallback, and MICROPHONE capture — your own voice mixed into the same stream and
  labelled "You" by measuring which device the sound arrived on.
- 337 tests, plus manual scripts in `tests/manual/` (GUI smoke, transport, offline
  diarization ground-truth check).

### In progress / not verified by a human
- **Offline re-diarization has never run on real recorded audio.** Verified only via
  `tests/manual/offline_diarization_check.py` (synthetic 2-voice TTS conversation):
  6/6 segments, 2 speakers, 4.7s for 34.9s. Real conference audio is much harder.
- Notes UI never exercised against a real LLM; every test uses a fake provider.
- Hotkeys confirmed working by the user, but remap-and-reprobe is untested by a human.

## Key decisions & why

- **Local-first.** Only the AI tab can send data off-machine, and only after a consent
  dialog stating the exact character count and destination. A local provider is
  first-class, not a fallback.
- **API keys live in Windows Credential Manager** (`llm/credentials.py`, ctypes/
  advapi32), never `config.toml` — that file gets copied, pasted into issues, and
  cloud-backed. A test asserts no key-shaped field exists in `Settings`.
- **Transport states differ by what they RELEASE:** pause keeps the model resident
  (fast resume), stop releases it (~2 GB VRAM). Before v0.2.0 "pause" only froze the
  overlay while the GPU kept working at full cost.
- **Startup is `startup_mode` (resume|always|never), not a checkbox.** A checkbox
  cannot express "I pressed Start last time"; `last_transport_state` is written to
  disk as it changes.
- **`session_audio_path()` is the only place the audio filename is spelled.** The
  recorder and the Speakers tab each had their own idea once, and re-diarization
  reported "no saved audio" forever.
- **`store.db.connect()` resolves `DB_PATH` at CALL time.** As a default argument it
  bound at import, so a test's redirect was ignored and a rename test relabelled a
  speaker in the user's real transcripts.
- **Streaming decode pins `temperature=0.0`** (batch/offline keep the default). The
  fallback re-decoded up to 6× on ~10.7% of cuts, freezing the overlay 7–9.5s, and
  produced identical text. `is_degenerate()` replaces the compression-ratio check that
  removed.
- **`OnlineASR._clamp_buffer()` enforces a HARD cap before the decode**, with 1.5×
  hysteresis. `trim_to_committed()` sets `offset = cut_time`, so its `0 < n` guard is
  structurally dead until a new word commits — buffers reached 100s against a 15s cap.
- **All settings tabs are wrapped in a QScrollArea.** Qt squeezes group boxes when a
  tab is taller than the window, and wrapped text then overlaps the controls below.

## Gotchas discovered

- **This machine has TWO identically-named loopback devices** (DELL S2721QS, index 13
  and 14). Only **14** carries audio. `pyaudiowpatch.get_default_wasapi_loopback()`
  returns 13 — it matches by name. Our `capture/devices.default_loopback()` is
  ordinal-aware and correct.
- **An idle WASAPI loopback delivers NO data at all** (a blocking read never returns),
  so the wrong device is indistinguishable from a broken app. Hence `capture/probe.py`
  and the "Find the device that's playing audio" button.
- **Two `PyAudio()` instances across threads segfaults PortAudio.** Use one instance
  with callback streams (see `capture/probe.py`).
- **Every `ctrl+alt+*` hotkey is already claimed by another app here** — all seven
  defaults fail to register. That is why Advanced has remapping.
- **The HF snapshot symlinks break intermittently** (blobs intact, links rebuilt at
  20:24 and 22:01 on 2026-07-18). Cause still unknown. `_new_model` repairs the cache,
  then `_open_with_retry` retries — a freshly written 75 MB blob is briefly unopenable
  (virus scanner). `_describe_cache()` dumps the directory on final failure.
- **Volume does NOT break transcription.** Measured on real audio: 5% and 1% volume
  transcribe byte-identically to full volume. Only mute (literal zeros) defeats it.
  Auto-gain helps only below ~0.3%.
- **PyInstaller:** lazily-imported GUI/LLM modules must be in `hiddenimports`, or the
  app launches fine and dies when the tab opens. `opengl32sw.dll` CANNOT be dropped via
  `excludes` (Python modules only) — filter all four TOCs, and assert the filter
  matched something or it silently saves 0 MB.
- **Tests MUST monkeypatch `config.CONFIG_DIR`, `config.CONFIG_PATH` and
  `store.db.DB_PATH`, and ASSERT it** (`PRAGMA database_list`). This has bitten twice:
  a rename test hit the real transcripts, and a startup test read the real config.
- **Workflow scripts:** no `require`, and the script file must be LF-only with no
  control characters or the permission layer rejects it.
- Windows TTS (`Microsoft David Desktop` / `Zira Desktop`) is available, and is how the
  diarization ground-truth fixture is built.

## Next steps (in order)

1. **User testing of v0.3.2**: Speakers → "Save each session's audio" → record →
   Re-diarize. Confirm the offline pass beats the live labels on real audio.
2. **Notes against a real provider** (local Ollama, or a key). Watch the consent
   dialog's character count, and whether to-dos are grounded in real quotes.
3. Minor review findings, none blocking — see the v0.3.0/v0.3.1 commit bodies:
   `_on_hotkey` alias comparison on the same field; `probe_hotkey` never exercised
   live; `repair_wav_header` only handles the canonical 44-byte layout; notes'
   `_reject_partial_reply` skips duck-typed providers.
4. Consider showing a session's notes in the Transcripts tab (currently AI tab only).

## Dead ends — do NOT retry

Measured and rejected during the speed audit (details in the v0.1.7 commit body):
- **Streaming model medium → base.en**: 0.17s of median latency, and it silently
  breaks non-English. The "4.9× faster" claim came from a harness that omitted three
  buffer-shortening mechanisms.
- **`beam_size` 1 → 2**: +48% GPU on tiny.en for 0.00pp measured accuracy gain.
- **Moving `diarizer.feed()` off the caption thread**: the diarizer is on the
  line-segmentation control path, not just the colour path — async merges two speakers'
  turns into one mislabelled line.
- **Caching Sortformer across pipeline rebuilds**: worth 1.5s, not 43s, and `stop()`
  never joins, so a shared instance races the old thread.
- **Warming Sortformer at startup**: races v0.1.5's `_HideModules` blocker →
  `SystemExit` swallowed in a daemon thread → speaker colours vanish nondeterministically.
- **Splitting the GPU pack out of the installer**: the download lands inside
  `build_source()`, freezing the overlay for minutes; `device="cuda"` users get
  `SystemExit(1)` with no CPU fallback; 735 MB is orphaned on uninstall.
- **A second ASR-only build tier**: `updater.py` picks the FIRST `.exe` asset in a
  release, so publishing two installers silently downgrades full installs.
- **Sharing one `PYZ` object between the two EXEs**: saves exactly 0 bytes
  (`EXE.__init__` appends the PYZ into each exe's own archive).
- **`OMP_WAIT_POLICY=PASSIVE`**: −23% CPU but +17% diarizer wall RTF — spending
  real-time headroom to free cores nothing wants (Whisper is on the GPU).
- **`print()` for user-facing errors**: a no-op in the windowed PyInstaller build.
