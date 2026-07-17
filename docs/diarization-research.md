# Local Speaker Diarization — Research Report (2026-07)

*Deep-research synthesis for the live-captions project. Question: is LOCAL (on-device)
speaker diarization feasible and good enough — as a real-time path and/or an offline
pass — given a single post-mix WASAPI-loopback stream, faster-whisper word timestamps,
and an 8 GB RTX 3070, with the project willing to migrate to Python 3.12 (so PyTorch is
available)? 23 sources fetched, 25 claims adversarially verified (18 confirmed, 0 refuted,
7 unverified due to a mid-run session limit — those are low-risk repo facts, flagged below).*

## Verdict

**Yes, local diarization is feasible — and 2025 changed the streaming story.** Do it in two
layers:

1. **Offline pass on saved transcripts — do this first.** Mature, high quality (~13 % DER
   open-source), low effort. This is the reliable "who said what" for the searchable
   transcript feature (M6). Strongly recommended.
2. **Live best-effort speaker colors — worth a spike, manage expectations.** The new
   **NVIDIA Streaming Sortformer** (mid-2025) makes real-time local diarization genuinely
   viable for the first time, is commercially licensed (CC-BY-4.0), and fits easily in VRAM.
   BUT it's capped at **4 speakers**, and on a **single post-mix stream** with overlap and
   unknown speaker count the quality is inherently limited. Spike it; ship it only if it's
   good enough on your real audio, otherwise render single-speaker live and rely on the
   offline pass for labels.

The honest ceiling: **no local tool fixes the fundamental problem** — the conferencing app
already summed everyone into one stream, your own mic isn't in it, speakers overlap, and the
count is unknown. Tools reduce the error; they don't remove the limitation.

## Tool landscape

| Tool | Streaming? | Approx footprint | License | Notes |
|---|---|---|---|---|
| **NVIDIA Streaming Sortformer 4spk-v2** (NeMo) | ✅ real-time (Arrival-Order Speaker Cache) | small, RTF 0.005–0.18 on GPU | **CC-BY-4.0** (commercial OK) | **The live option.** ≤4 speakers hard cap. Streaming DER ≈ offline. Torch/NeMo. |
| **pyannote.audio community-1** (v4.0, 2025) | ❌ offline | ~1 GB VRAM, RTF ~0.025 on GPU | **CC-BY-4.0**, HF-gated | Newest open pyannote; best offline quality tier. Torch. |
| **pyannote.audio 3.1** | ❌ offline | similar | MIT, HF-gated | Previous gen; very widely used. Torch. |
| **WhisperX** | ❌ offline (batch files) | Whisper + wav2vec2 + pyannote | BSD-2 (code); pyannote gated | **Canonical offline word-level pipeline** (`assign_word_speakers`). |
| **whisper-diarization** (MahmoudAshraf97) | ❌ offline | NeMo MarbleNet(VAD)+TitaNet(embed)+CTC aligner | **BSD-2** *(unverified this run)* | Offline, **no pyannote / no HF gating**. Torch/NeMo. |
| **diart** (juanmc2005) | ✅ online (rolling 500 ms buffer, latency 0.5–5 s) | pyannote seg+embed | **MIT** | Mature streaming lib on pyannote models; the classic "live diart+Whisper" demos. |
| **sherpa-onnx** (k2-fsa) | ❌ offline | pyannote-seg + 3D-Speaker/NeMo embed, **ONNX** | Apache-2 | **Torch-FREE** (onnxruntime) — the option if you *don't* migrate to 3.12. |
| **NVIDIA Sortformer 4spk-v1** (offline) | ❌ offline | small | ⚠️ **CC-BY-NC-4.0 (NON-commercial)** | Avoid for a shippable app. (The *streaming* v2 is fine.) |
| **WhisperLiveKit** (QuentinFuxa) | ✅ (integration) | faster-whisper + Sortformer default | open source | **The reference implementation** — streaming Whisper (LocalAgreement/AlignAtt) + streaming diarization, exactly this project's shape. |

## Realistic accuracy (DER)

- **Open-source floor ≈ 13 % DER** on hard multi-speaker conversational audio: a 2025
  benchmark over 196.6 h of multilingual audio (all standardized to **mono 16 kHz** — i.e.
  the single-channel case) found the best open-source system (DiariZen) at **13.3 % DER** vs
  the commercial cloud PyannoteAI at **11.2 %**. So local ≈ 2 points behind cloud, not a
  chasm. [arxiv 2509.26177]
- **Streaming ≈ offline now.** Streaming Sortformer scores **6.05 % DER on CALLHOME
  2-speaker** and **13.75 % on DIHARD III (1–4 spk)** at 10 s latency, *including overlapping
  speech*; its own paper notes streaming can even **outperform** offline. This is the key
  2025 result — the old "online is much worse" gap has largely closed. [HF card; arxiv 2507.18446]
- **Reality check for your case:** those numbers are on clean-ish research corpora. On a
  *conferencing/VoIP-codec, post-mix, overlapping, unknown-count* stream, expect **worse** —
  plausibly 20–35 %+ DER for live, better for the offline pass. Two speakers, low overlap →
  quite usable. 4+ speakers, heavy overlap → mediocre.

## Fundamental limitations (unchanged by any tool)

- **Single post-mix stream** — remote participants are already summed before loopback; there
  are no per-speaker channels to exploit.
- **Your own mic is absent** — in a 1:1 call your side is never labeled from loopback at all.
- **Overlap + unknown/variable count + online, simultaneously,** remains an open research
  problem; the 4-speaker cap on Sortformer is a concrete symptom.
- **No enrollment** — labels are "Speaker 1/2/3", not names (name assignment is a separate,
  later step, e.g. LLM or manual).

## VRAM / compute — not a blocker

Whisper `medium` float16 ≈ 2–3 GB, leaving ~5 GB. Diarizer models are small (Sortformer /
pyannote ~ few-hundred-MB–1 GB) and run at tiny RTF (Sortformer 0.005–0.18; pyannote ~0.025
offline). Both fit alongside Whisper on the 3070. Embedding-based diarizers (pyannote/NeMo
TitaNet) can also run on **CPU** if you want to keep the GPU purely for Whisper.

## Integrating with our word timestamps

We already emit committed words with `(start, end, text)`. Standard pattern (WhisperX
`assign_word_speakers`): the diarizer yields speaker turns `(start, end, speaker)`; assign
each word to the speaker whose turn maximally overlaps the word's midpoint. Offline: run over
the whole saved audio, one clean pass. Live: run the streaming diarizer in parallel with our
streaming ASR; when LocalAgreement commits a word, look up the active speaker at its
timestamp and color it in the overlay (`TranscriptEvent.speaker` already reserved).

## The article you were thinking of

Most likely one of:
- **"Color Your Captions: Streamlining Live Transcriptions with diart and OpenAI's Whisper"**
  (Medium / Better Programming) — live captions + diart speaker colors, matches your recall.
- **WhisperLiveKit** (github.com/QuentinFuxa/WhisperLiveKit) — the closest full implementation.
- **WhisperX** (github.com/m-bain/whisperX) — the canonical offline whisper+diarization repo.
- **whisper-diarization** (github.com/MahmoudAshraf97/whisper-diarization) — offline, no HF gating.

## Licensing gotchas (important for shipping)

- ✅ **Streaming Sortformer 4spk-v2**: CC-BY-4.0 — commercial OK.
- ⚠️ **Offline Sortformer 4spk-v1**: **CC-BY-NC-4.0 — non-commercial. Avoid.**
- ✅ **pyannote community-1 & 3.1**: CC-BY-4.0 / MIT respectively — commercial OK, but **HF-gated**
  (must accept terms + supply an HF token; unauthorized access returns 403).
- ✅ **diart** MIT, **sherpa-onnx** Apache-2, **whisper-diarization** BSD-2 *(unverified this run)* —
  permissive, and the latter two avoid pyannote's HF gating.

## Recommendation for this project

1. **Migrate to Python 3.12** (you're already willing) — unlocks torch, hence NeMo + pyannote.
   Keep it in its own env; it's mechanical (our package is `requires-python >=3.11`, version-agnostic).
2. **Ship the offline pass first** (fold into M6 "saved transcripts"): after a session, run a
   WhisperX-style diarization over the saved audio and attach speakers to our word
   timestamps. Use **pyannote community-1** (best quality, accept the HF terms) or
   **whisper-diarization / NeMo** (BSD, no gating) if you want to avoid HF tokens. ~13 % DER,
   genuinely useful, ~3–5 days.
3. **Spike Streaming Sortformer for the live path** (~3–5 days): run it in parallel with the
   streaming ASR, color committed words by active speaker in the overlay. Gate shipping on a
   real-audio test: if 2-speaker calls look good, ship it as "best-effort live speaker
   colors" with a clear caveat; if 4+ speaker / heavy-overlap is bad, render single-speaker
   live and let the offline pass carry the labels. **Study WhisperLiveKit** as the reference.
4. **Names come later** — diarization gives "Speaker 1/2"; mapping to real names is a separate
   opt-in step (manual, or the LLM naming from the roadmap's M6).

Bottom line: **local is viable and the right call.** The offline pass is a clear win now; the
live path is finally worth attempting in 2025 thanks to Streaming Sortformer, but temper
expectations on a single mixed stream.

## Sources (verified)

- NVIDIA Streaming Sortformer 4spk-v2 — https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
- Streaming Sortformer paper — https://arxiv.org/html/2507.18446v1 ; NVIDIA blog — https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- Diarization benchmark 2025 — https://arxiv.org/html/2509.26177v1
- diart — https://github.com/juanmc2005/diart
- sherpa-onnx — https://github.com/k2-fsa/sherpa-onnx
- WhisperX — https://github.com/m-bain/whisperX
- whisper-diarization — https://github.com/MahmoudAshraf97/whisper-diarization
- WhisperLiveKit — https://github.com/QuentinFuxa/WhisperLiveKit
- pyannote community-1 — https://huggingface.co/pyannote/speaker-diarization-community-1 ; pyannote 3.1 — https://huggingface.co/pyannote/speaker-diarization-3.1
- diart+Whisper live captions (Medium) — https://medium.com/better-programming/color-your-captions-streamlining-live-transcriptions-with-diart-and-openais-whisper-6203350234ef
- Online EEND (overlap + flexible speakers) — https://arxiv.org/pdf/2101.08473
