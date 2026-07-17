# Live-Captions — Engineering Roadmap

*Produced from a multi-agent design review (8 parallel area designs → synthesis → adversarial
critique → finalize), then revised against the actual M0 code. Supersedes the informal M0–M5 list in
the README; milestones are renumbered (see the map at the bottom).*

Design principle: **hardware-independent — any Windows PC, GPU with CPU fallback.** Detect
capabilities at runtime; never hardcode one machine's shape.

---

## Status (2026-07-16)

**M0.1, M1, M2, and M3 are complete.** The single-file spike is now a layered
package (`src/livecaptions/`, `pip install -e ".[gui]"`, `python -m livecaptions`)
with inference off the audio thread, an always-on-top caption overlay, and a
streaming ASR core. See [docs/architecture.md](docs/architecture.md).

**M3 delivered & verified:** continuous partial+final captions via a
re-implemented **LocalAgreement-2** (`asr/hypothesis.py`) fed by faster-whisper
word timestamps (`asr/streaming.py` OnlineASR: rolling ≤15 s buffer, trim at last
committed word). **Silero VAD v6** runs on onnxruntime's **CPU** execution
provider (verified `['CPUExecutionProvider']` — zero VRAM, torch-free, bundled in
faster-whisper) for speech-gating + endpointing. `StreamingTranscriptionSource`
streams words into a "pending line" that finalises on sentence punctuation / a
pause / a max length — *without* losing the tail audio; latest-wins backpressure;
junk-phrase hallucination blocklist. Run `python -m livecaptions --streaming
--overlay`. Verified: 5/5 LocalAgreement tests; a **54 s no-pause monologue
streams continuously** (finals appear throughout, sentence-by-sentence, never
rewriting) and **keeps up with real time** (RTF < 1); partials refine in place;
streaming drives both the terminal and the overlay.

**M2 delivered & verified:** a frameless, always-on-top, translucent, click-through
caption bar (PySide6) at the bottom-centre of the screen — rounded pill, outlined
text for legibility over any content, HiDPI + multi-monitor aware, position/opacity
persisted (QSettings). Finals render solid; partials (is_final=False) render dimmed
and commit solid — the render contract M3 streaming needs, validated now via a
`DemoTranscriptionSource` (growing partials → final) and `--demo`. A `CaptionBridge`
(QObject signal) marshals events from the source's worker thread to the GUI thread.
Verified: on-screen render confirmed via PrintWindow; **live system audio →
overlay caption confirmed end-to-end** (WASAPI loopback → Whisper → overlay);
demo partial/final render; click-through + `--movable` drag-to-reposition. Run
`python -m livecaptions --overlay`.

M1 delivered & verified: the `TranscriptEvent`/`AudioBlock` seam (source-tagged for
multi-source merge); `TranscriptionSource` with local + fake implementations (both
drive the terminal sink via `on_event`); off-thread `WhisperWorker` (bounded,
drop-oldest) fed by a segmenter thread; pydantic-settings + TOML config with device
persistence; 8 segmenter characterization tests (green). Verified: `--fake`, `--wav`,
and live capture all transcribe; env/TOML config precedence works. The original
`m0_live_captions.py` remains as the M0 reference.

**Local diarization (2026-07-17)** — deep research ([docs/diarization-research.md](docs/diarization-research.md))
revisited the "mixed-stream diarization is not a milestone" call: it's now worth doing *locally*, in
two layers. **Layer 1 shipped:** an offline post-processing pass —
`python -m livecaptions --diarize call.wav` → speaker-labeled transcript, via a pluggable backend
(`sherpa-onnx`, no account, ONNX/CPU; or `pyannote`, best quality, needs an HF token). Both verified
100% correct on a 2-speaker test (6/6 turns). **Layer 2 shipped:** live speaker colours —
`python -m livecaptions --diarize-live --overlay` runs **NVIDIA Streaming Sortformer**
(CC-BY-4.0; the offline v1 is CC-BY-NC — avoid) alongside the streaming ASR, tags committed words
with the active speaker, and colours each line in the overlay. Runs on **CPU at ~RTF 0.4 with zero
VRAM**, keeping the GPU for Whisper. Verified 9/9 lines correctly attributed with stable speaker
identity across a 38 s 2-speaker conversation. Caps at **4 speakers**.

This does NOT change the core strategic call: that test is the *easy* case (clean audio, no overlap,
2 speakers). On crowded, overlapping, codec-degraded calls expect materially worse (~20–35 %+ DER),
and reliable *ground-truth* "who" still only comes from per-speaker audio (M5/Discord).

> **Python runtime: now 3.12** (migrated 2026-07-17, superseding the earlier "stay on
> 3.14" call). M0–M3 all had cp314 wheels, but the diarization stack (NeMo/Streaming
> Sortformer, pyannote.audio) is **torch-based and torch has no cp314 wheels**. Migrated
> the whole project rather than split environments; re-verified end-to-end on 3.12
> (13/13 tests, GPU, streaming, overlay) and confirmed torch 2.13 / pyannote.audio 4.0.7 /
> nemo_toolkit 2.7.3 all install. Use `py -3.12 -m venv .venv`. See
> [docs/architecture.md](docs/architecture.md) and [docs/diarization-research.md](docs/diarization-research.md).

<details><summary>Earlier — M0.1 (done)</summary>

**M0.1** — the spike is a diagnosable, hardware-independent capture app:

- ✅ Transcribes on GPU (cuda/float16), ~0.3 s for a 3 s utterance; auto-falls back to CPU (int8),
  validated by a startup self-test. CUDA cuBLAS/cudart via bundled wheels + DLL preload.
- ✅ **Callback-mode capture** (non-blocking) — the primary path; `--blocking` kept as a documented fallback.
- ✅ **Device picker**: `--list-devices`, `--device <substr>`, `--pick`, `--loopback-index`; choice
  persisted by loopback name + ordinal (handles two identically-named monitors) in `%APPDATA%`.
- ✅ **Two-signal audio-health watchdog** (no-blocks-in-4 s idle warning; blocks-but-silent-8 s
  wrong/muted warning) + live dBFS VU meter (tty only, clean logs when redirected).
- ✅ **`WavFileAudioSource`** — deterministic `--wav` test path through the same segmenter/transcriber.

Verified: `--wav` produces captions with no device; selecting the dead endpoint warns in ~4 s instead of
hanging; live callback capture transcribes accurately; persistence round-trips.

Inference is now off the consumer thread as of M1 (was the last M0.1 nice-to-have); VB-CABLE smoke test
of the real WASAPI path still not run on this box.

</details>

> **Hardware truth:** the dev box's DELL S2721QS monitors have no built-in speakers (audio only via
> their 3.5 mm jack over DisplayPort). It's a dev/test machine — it cannot demonstrate the real
> "hear-it-and-caption-it" end-user experience without real output hardware.

---

## Top-line strategic calls

1. **Mixed-stream diarization is *not* a milestone.** A summed WASAPI loopback stream cannot yield
   reliable "who." Reliable speaker labels have exactly one ground-truth tier — **Discord per-user
   audio (M5)**. Cloud diarization (M4) is *materially better but still probabilistic*. Local
   mixed-stream diarization is the pathological case (~25–45% DER) and is built **only if a spike
   proves it beats cloud** — expected default: render loopback as single-speaker.
2. **Overlay ships finals-first (M2) before the streaming rewrite (M3)** — but the M2 demo harness
   emits partials too, so the overwrite/reflow behavior is validated early, not discovered as rework.
3. **In-process `on_event` callbacks everywhere**, except the one justified cross-process boundary:
   the Rust→Python WebSocket for the Discord bot.
4. **Callback-mode capture is the primary path** (not blocking `stream.read`) — the verified cause of
   the silent hang, and the precondition for the watchdog *and* runtime device switching.
5. **Don't fork Scripty** (EUPL copyleft + irrelevant bloat); **don't transcribe in Rust** (duplicate
   STT + VRAM contention). The bot ships speaker-tagged audio to the one shared STT service.
6. **The local mic is never in the loopback stream.** Multi-source merge (loopback + mic) is a
   first-class question that must be answered **before M1 freezes the event contract**.

**Timeline:** ~**16–22 weeks** realistic single-dev for the full M0.1–M7 path. ~**9–12 weeks** is a
*stretch* target for the demoable core only (M0.1 + M1 + M2 + M3).

**Verify now (load-bearing risk):** cp314 Windows wheels for PySide6, onnxruntime, soxr,
PyAudioWPatch, and the Discord libs. If any core GUI/ML wheel is missing, **pin to Python 3.12** before
starting M1. (torch already has no cp314 wheels — the design routes around it with ONNX everywhere.)

---

## Milestones

### M0.1 — Prove capture end-to-end (immediate unblock) · ~3–4 days
**Objective:** never a silent black box again; demonstrably transcribes a deterministic source today.
- Switch capture to **PyAudio callback (non-blocking) mode** as the primary path (stuck/idle endpoint
  is timer-detectable and the stream tears down cleanly).
- **Device picker**: `--list-devices`, `--device <substr>`, `--pick`; persist the *loopback* device's
  own resolved name (with `[Loopback]` suffix), never the volatile PortAudio index; handle
  duplicate/truncated endpoint names.
- **Two-signal audio-health watchdog** + live dBFS VU meter: "no blocks in ~4 s" → idle/phantom
  warning; "blocks but ~silence in ~8 s" → wrong/muted-output warning. Priming probe warns-and-continues.
- **`WavFileAudioSource`** — replay a WAV through the same segmenter/inference path (the deterministic
  CI/dev signal and latency rig).
- Install **VB-CABLE** as the only way to exercise the real WASAPI path with non-zero frames on this box.
- Stays single-file (still a spike).

**Exit:** WAV → transcribed text with no capture device (CI); VB-CABLE moves the VU meter and produces
text (manual smoke); selecting the sink-less endpoint warns within ~8 s instead of hanging;
`--list-devices` works and a saved choice survives replug.

---

### M1 — Minimal foundation & source seam (the keystone, scoped down) · ~1.5–2 wk · needs M0.1
**Objective:** turn the spike into a layered package around the *smallest* event/source seam M2/M3
need, and move inference off the audio thread.
- Package `livecaptions/` (pyproject/hatchling, editable install, `python -m livecaptions`). Create
  only the dirs earned now: `events`, `capture/`, `asr/`, `sources/`, `ui/` (stub). **Do not**
  pre-create cloud/discord/diarize/store.
- **`events.py` minimal seam:** `TranscriptEvent{text, is_final, t_start, t_end, speaker?, confidence,
  source}` + `AudioBlock`. `is_final` reserved (finals-only for now); `speaker` optional. Don't freeze
  the capabilities enum yet.
- **Decide multi-source merge before freezing** the contract (loopback + future mic → one transcript?
  reserve source identity + per-source timelines).
- **`TranscriptionSource`** interface: `start(on_event)`, `stop()`; thread-safe callback / bounded
  queue. No asyncio forced into a GUI/consumer thread.
- **Off-thread inference:** segmenter thread + single bounded-queue drop-oldest GPU worker (surface
  "[dropped N — GPU behind]"). One GPU worker only (8 GB, medium/float16).
- **Config:** pydantic-settings + TOML in `%APPDATA%\live-captions\`, platformdirs for cache/weights.
- **Tests first:** characterization tests assert on **segmenter** outputs (boundaries, `utt_sec`, flush
  timing from a synthetic RMS envelope) — never on decoded Whisper text (flaky).
- Deferred to M4/M5 explicitly: capabilities enum, provider factory, keyring, cloud/discord/store dirs.

**Exit:** `pip install -e .` reproduces M0 behavior through the new layers; a Fake source + WAV source
both drive the same sink via `on_event`; segmenter tests pass unchanged; cp314-vs-3.12 decision recorded.

---

### M2 — Live overlay, finals-first · ~4–6 days · needs M1
**Objective:** always-on-top, click-through, per-pixel-transparent caption bar over any app.
- **PySide6** (LGPL) frameless `Qt.Tool` window; `WA_TranslucentBackground`,
  `WA_ShowWithoutActivating`; custom `paintEvent` rounded caption pill with outlined text.
- Thread-safe `CaptionBridge(QObject)` `Signal(str, bool, int)` → auto QueuedConnection to the GUI thread.
- Multi-monitor/HiDPI (logical px, `availableGeometry()`, bottom-center); persist screen/offset/opacity.
- No-flicker click-through via native `SetWindowLongPtrW(GWL_EXSTYLE, WS_EX_TRANSPARENT)`.
- Status states in the pill: Loading / Listening / "No audio — pick a device" / Error.
- **Critical:** the `--demo` generator emits partials *and* finals so word-overwrite / reflow / scroll
  are validated here — don't assume streaming "drops in with zero overlay rework."

**Exit:** floats over fullscreen video on the 4K DELL (150% scaling) + a second monitor; click-through
verified; demo drives dim-partials-into-solid-finals with correct reflow; position/opacity persist.

---

### M3 — Streaming ASR core (real-time feel) · ~2–3 wk · needs M1, M2
**Objective:** replace utterance-batching with continuous partial+final captions.
- Vendor **ufal/whisper_streaming** (MIT LocalAgreement-2: HypothesisBuffer, OnlineASRProcessor); write
  our own front end. Avoid WhisperLive/WhisperLiveKit (websocket-server; AlignAtt is non-commercial).
- **Silero VAD v5 (ONNX, not torch)**, pinned to the onnxruntime **CPU** execution provider (zero VRAM);
  turn faster-whisper's internal `vad_filter` off in streaming mode.
- Stateful `soxr.ResampleStream`, rolling ≤15 s buffer with forced trim at last committed word;
  `word_timestamps=True`, `condition_on_previous_text=False`.
- Latest-wins backpressure (maxsize-1 coalescing slot sheds stale ticks, never audio).
- Hallucination handling: VAD gating + `avg_logprob`/`no_speech_prob` thresholds + junk-phrase blocklist
  ("Thank you.", "Subtitles by…") for loopback music/ads/silence.
- Presets: medium/float16 default, small.en lowest latency, large-v3-turbo quality (benchmark).

**Exit:** interim caption within ~0.5–1.0 s; each word finalized within ~1.5–2.5 s; **finals never
rewrite**; continuous captions through a 60 s no-pause monologue; sustained RTF < 1.0 with margin;
Silero confirmed on CPU-EP with zero added VRAM.

---

### M4 — Cloud STT/diarization backend + spike-gated local labeler · ~3 wk · needs M1, M2
**Objective:** first materially-better "who" on the universal loopback path, opt-in.
- Introduce the deferred abstractions now that a real async source exists: capabilities enum,
  `make_source(settings)` factory (local | assemblyai | deepgram), **keyring** (Windows Credential Mgr).
- **AssemblyAI Universal-3.5 Pro Realtime + streaming diarization** = default cloud backend
  (turn-based A/B labels). **Deepgram Nova-3** = secondary. Characterize both honestly: materially
  better than local, still probabilistic — *not* ground-truth.
- **Verify billing** before building the cost story (is it socket-open-time, not per-audio-second?).
- VAD-gated connection (open on speech, close after 30–60 s idle) + reconnect/backlog replay.
- **Gate the local labeler behind a 1–2 day diart spike** — diart must beat AssemblyAI on your own
  audio to justify *any* local diarizer. Default (spike fails): ship no local "who", single-speaker.
- **`MicCaptureSource`** so the user's own side of a 1:1 call is captured/labeled (loopback lacks it).

**Exit:** 2-speaker WAV → correct A/B lines with per-speaker color; billing verified; idle tears down
socket and recovers on reconnect; diart verdict recorded as ship-or-shelve; mic merged into one transcript.

---

### M5 — Discord per-user bot: the only ground-truth labels · ~3–4 wk · needs M1, M4
**Objective:** perfect non-ML per-speaker labels where they're achievable — a Discord call.
- 1-day Python de-risk spike (discord-ext-voice-recv/py-cord) → almost certainly commit to **Rust**.
- Lean Rust bot on **serenity 0.12+ / songbird 0.5+** (receive + DAVE). *Reference* Scripty, don't fork.
- `DecodeMode::Decode`; `VoiceTick` (per-20 ms SSRC→PCM); `SpeakingStateUpdate` for SSRC↔user_id →
  display name. Per-user endpointing off packet gaps (~700 ms), not RMS.
- **The bot does not transcribe** — it ships 48k-mono i16 speaker-tagged segments over **localhost WS**
  (length-prefixed binary + JSON header) to the shared faster-whisper service. One model, one overlay.
- Per-SSRC packets/sec heartbeat that fails loudly on "joined but 0 packets" (the DAVE-downgrade analogue
  of the silent hang).

**Exit:** 3-way conversation renders with correct display-name labels per utterance, colored per
speaker; overlapping speakers → two correct sequential final lines; zero-packet condition warns visibly.

---

### M6 — Polish: transcripts, search, hotkeys, LLM naming · **shipped** (2026-07-17)
- ✅ **Shared persistence bus** — `store/writer.py`: dedicated writer thread, batched inserts (1 s /
  25 rows), never on the audio thread. Fan-out sink so the overlay and the writer both see events.
  Finals only. `--no-save` opts out.
- ✅ **Saved & searchable transcripts** — SQLite (WAL) + external-content **FTS5** (BM25 ranking,
  `snippet()` highlighting), runtime-probed with a `LIKE` fallback for builds without FTS5.
  `--sessions`, `--search` (+ `--speaker`/`--since` filters). External-content means a speaker rename
  costs **zero reindex**. Exports (`--export srt|vtt|jsonl|md`) are exports, not storage.
- ✅ **Global hotkeys** — Win32 `RegisterHotKey` + `QAbstractNativeEventFilter`, so they fire while the
  overlay is click-through and unfocused (QShortcut can't). Deliberately *not* the `keyboard` package:
  that installs a low-level hook (a keylogger); RegisterHotKey asks the OS for exactly our combos.
  Show/hide, pause, nudge; a claimed combo warns and stays remappable in `config.toml`.
  Verified: 6/6 combos registered on the live overlay.
- ✅ **Optional LLM speaker-naming** (`--name-speakers`, off by default) — Anthropic SDK structured
  outputs (`messages.parse` + pydantic) mapping SPEAKER_N → names; default `claude-opus-4-8`,
  `--name-model` to switch. Privacy *is* the design: the only feature that leaves the box, so it states
  the byte count and gets consent **before sending**, shows the cited verbatim quote and confirms
  **before each rename**, and every rename is reversible. Prompt forbids guessing from role/style and
  calls out the off-by-one trap ("Thanks, Sarah" is usually *not* Sarah). Model output is reconciled
  against our own label list — it cannot invent or drop a speaker. Key comes from `ANTHROPIC_API_KEY`
  or an `ant auth login` profile, never from `config.toml`.
  Verified up to the network boundary (consent gate, reconciliation, refusal handling — fake client);
  **the live API call is unverified** — no key on this box.

**M6 vs. plan:** dropped the Credential Manager (the SDK's own env/profile resolution is the
better-trodden path and keeps secrets out of the repo) and `FileTranscriptSource` (the existing
`--demo` and `--wav` paths already made M6 developable without live audio). Offline HQ diarization
moved out — it shipped early, in M4, and torch is already a dependency since the 3.12 migration.

---

### M7 — Packaging & distribution · **built & core-verified on the RTX 3070 box 2026-07-17**
Preceded by a 7-area investigation (fan-out + adversarial verification of every load-bearing
claim against the real venv). Tier decision: **A — bundle everything** (per owner). Artifacts in
`packaging/`. The **PyInstaller bundle was actually built and run** on this machine; the Inno
installer compile and fresh-/non-NVIDIA-box testing remain (see "Owner must verify").

**Verified by real frozen builds (RTX 3070, this box):**
- The **Tier A `--onedir` bundle builds** — 2.2 GB, both EXEs, all of cuBLAS/torch/NeMo/PySide6/
  Silero collected.
- **GPU Whisper runs inside the bundle** (`Model ready on cuda`), confirmed in both a core-only and
  the full Tier A bundle — the #1 technical risk, closed.
- **Live diarization runs frozen** — NeMo/Sortformer loads and initializes (`Live diarizer ready`).
- **No torch↔CTranslate2 OpenMP conflict** even under strict `KMP_DUPLICATE_LIB_OK=FALSE` (the Tier A
  hazard the investigation flagged did not materialize).
- **The windowed overlay EXE renders** (PySide6 frozen) and the SQLite store writes.
- **The frozen HF-cache redirect works** — models (Whisper + Sortformer, 914 MB) downloaded to the
  app-owned `%LOCALAPPDATA%\live-captions\hf`, *not* the shared cache, so uninstall can clean them.
- **One real bug found & fixed by the build:** the first Tier A bundle's live diarization died with
  `No module named 'cuda.bindings.cydriver'` — `cuda` (cuda-bindings) is a **namespace package**
  whose compiled `.pyd` are invisible to both `collect_all` and `collect_dynamic_libs`, and NeMo
  imports it dynamically. Fix (validated via a probe bundle, now in the spec): reconstruct the tree
  with `collect_data_files("cuda", include_py_files=True)` + glob the `.pyd` in as binaries.

- **PyInstaller `--onedir`** (never `--onefile` — re-unpacks the ~772 MB cuBLAS DLL per launch).
  `packaging/livecaptions.spec` builds **two EXEs sharing one `_internal/`**: `livecaptions.exe`
  (console, all CLI verbs) and `livecaptions-overlay.exe` (windowed, double-click to launch the
  overlay). Both route through `main()` so `configure_runtime()` always runs.
- **CUDA:** the spec collects **only** `nvidia/cublas/bin/{cublas64_12,cublasLt64_12}.dll` (772 MB) —
  a residency probe proved cudart is statically linked into `ctranslate2.dll` and nvrtc (179 MB)
  never loads. `capture/cuda.py` gained a `sys._MEIPASS` branch so the bare-name preload finds the
  bundled DLLs (preload order cuBLASLt→cuBLAS; the frozen search path matches the spec dest exactly).
- **Weights download on first run** into an **app-owned** dir. New `runtime.py` redirects
  `HF_HOME`/`HF_HUB_CACHE`/`HF_ASSETS_CACHE` to `%LOCALAPPDATA%\live-captions\hf` **when frozen**
  (source runs untouched), so an uninstaller can clean them — the shared `~/.cache/huggingface`
  couldn't be. It also captures the windowed EXE's stdout/stderr to a log file.
- **Non-NVIDIA / CPU fallback:** `whisper.py` now gates the GPU attempt on
  `ctranslate2.get_cuda_device_count() > 0` — a machine with no GPU starts straight on CPU, cleanly;
  a GPU present but cuBLAS failing is reported as a packaging bug, not "pip install" advice (useless
  in a bundle). CPU + a large model prints a "streaming won't keep up, try --model small.en" note.
- **Installer:** `packaging/livecaptions.iss` (Inno Setup **6.3+**) — **per-user** (no admin;
  `%LOCALAPPDATA%\Programs\LiveCaptions`), because models + the transcripts DB are per-user anyway.
  `x64os` blocks Arm64. **Uninstall preserves `transcripts.db` + `config.toml` by default** and
  *offers* to delete the multi-GB models; never touches anything outside our own dir. Bundles
  `vc_redist` (ctranslate2/onnxruntime statically import msvcp140). `build.ps1` drives both stages
  and asserts the load-bearing assets (cuBLAS, ctranslate2.dll, Silero VAD) actually landed.
- **Licensing:** `packaging/licenses/NOTICE.txt` — no ship-blocker; obligations are LGPL relink for
  **PySide6 and soxr** (both satisfied by `--onedir`'s loose DLLs + shipped license texts), the
  cuDNN/Intel-OpenMP texts that ride inside the ctranslate2 wheel, the NVIDIA CUDA EULA, and
  attribution for sherpa/TitaNet(Apache-2.0)/NeMo. Live colours use **Sortformer v2 (CC-BY-4.0,
  commercial OK)** — never swap to v1 (CC-BY-NC).

**Where the old M7 plan was wrong (corrected by the investigation):**
- "collect `nvidia/*/bin`" → collect **only** cuBLAS; `nvidia/*/bin` ships 179 MB of dead nvrtc.
- "pin ctranslate2/cuDNN together" / "add cuDNN" → cuDNN ships **inside** the ctranslate2 wheel;
  **do not** add `nvidia-cudnn-cu12`.
- "`os.add_dll_directory` in `__main__`" → the bootstrap is in `capture/cuda.py`; the `__main__`
  change that actually mattered was the HF-cache redirect, which the old plan omitted.
- "download into the user cache dir" hid a blocker: the default cache is the **shared** HF cache,
  uncleanable on uninstall — must be redirected to an app-owned dir.
- "~1 wk": realistic given the four code fixes, the two-EXE spec, the Inno `[Code]` (PATH surgery +
  silent-aware uninstall), and iterating a multi-minute NeMo freeze against a fresh box.

**Owner must verify (needs an installer toolchain / fresh / non-NVIDIA box — not available here):**
1. **The Inno installer compile** — Inno Setup isn't on the build box; `build.ps1`'s PyInstaller stage
   ran, the `iscc` stage did not. Install Inno 6.3+, drop `vc_redist.x64.exe` into `packaging/`, run
   the full `.\packaging\build.ps1`.
2. Clean **install + uninstall on a fresh Windows box** with no global CUDA toolkit *and no VC++
   redist* — confirm first-run weight download works and uninstall keeps the transcripts DB.
3. **int8/CPU fallback on a truly non-NVIDIA machine** (this box has an RTX 3070, so the *no-device*
   path was only simulated). Also worth confirming: nvrtc really is droppable, and `opengl32sw.dll` for the
   translucent overlay on a GL-less/RDP session.
4. **The pyannote offline gated-model path** — its torch/pyannote import is exercised by the (working)
   Sortformer freeze, but the gated `speaker-diarization-community-1` download wasn't run. Low risk.

**Deferred:** live diarization was kept in the bundle (Tier A), so no split installer. A CUDA-optional
Inno component (saves non-NVIDIA users ~772 MB) and a PySide6 trim (~100 MB) are easy later size wins.
The Rust bot (M5) doesn't exist, so nothing to co-package.

---

## Immediate next steps (this week)

1. Rewrite M0's capture to **PyAudio callback mode** (direct fix for the blocking-read hang; unlocks
   watchdog + runtime device switching).
2. Add the **two-signal audio-health watchdog** + live dBFS VU meter; priming probe warns-and-continues.
3. Build **`WavFileAudioSource`**; make "WAV → text, no capture device" the deterministic acceptance
   test; demote VB-CABLE to a documented manual smoke test.
4. Fix device persistence to store the **loopback device's own name** (handle duplicate/truncated names).
5. **Verify cp314 wheels now** for PySide6, onnxruntime, soxr, PyAudioWPatch; record the 3.12 fallback
   decision before M1.
6. Write the **segmenter characterization tests** (assert on boundaries/`utt_sec`/flush timing, never
   decoded text) so the M1 off-thread extraction is safe.

## Open questions (decide before they block a milestone)

- **Multi-source merge** (before M1 freezes the contract): loopback + mic as two sources or one? How to
  handle echo/double-capture and cross-source timeline alignment for the 1:1 call?
- **Cloud billing model** (before M4 cost story): socket-open-time vs per-audio-second vs minimums?
- **cp314 wheels** for the full stack, or pin 3.12 now?
- **diart spike** — does it beat AssemblyAI on your audio? Gates whether any local diarizer is built.
- **Async Rust** — known quantity for whoever builds M5? If not, it's 3–4+ weeks, not 1–2.
- **pyannote 4.0 license / HF gating** and whether torch offline diarization isolates cleanly on 3.14.

---

## Milestone renumbering (vs the old README list)

| Old README | New roadmap |
|---|---|
| M0 capture → terminal | M0.1 (hardened) — *done as of today* |
| M1 overlay | **M2** (overlay) — after M1 foundation + M3-precursor demo |
| M2 diarization | **demoted** — not a milestone; see M4 (cloud) / M5 (Discord) |
| M3 polish (hotkeys, transcripts, LLM naming) | **M6** |
| M4 cloud STT | **M4** |
| M5 Discord bot | **M5** |
| *(new)* streaming ASR | **M3** |
| *(new)* foundation/source seam | **M1** |
| *(new)* packaging | **M7** |
