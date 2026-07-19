"""End-to-end check of the offline diarization pass, scored against ground truth.

Everything else about re-diarization is tested with stubs, because a real pass needs
models on disk and minutes of CPU. This runs the actual thing: load_model ->
diarize_file (sherpa) -> word/speaker assignment, on a conversation built from two
different Windows TTS voices so we know exactly who spoke when.

Run:  python tests/manual/offline_diarization_check.py

First measured result (2026-07-19, sherpa-onnx, tiny.en, CPU):
    6/6 segments correct, 2 speakers found, 4.7s for 34.9s of audio.

Read that as a floor, not a headline: synthetic voices, no overlapping speech,
clean gaps between turns, and the speaker count given in advance. Real conference
audio — overlapping talk, VoIP codecs, unknown speaker count — is materially harder,
which is the whole reason the live pass exists as "best effort".
"""
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

TURNS = [
    ("Microsoft David Desktop", "A",
     "Right, let's go through the quarterly numbers before anyone leaves for lunch today."),
    ("Microsoft Zira Desktop", "B",
     "I pulled the report this morning and the margins look better than we forecast in April."),
    ("Microsoft David Desktop", "A",
     "That's good news, but I want to understand what changed in the supply costs."),
    ("Microsoft Zira Desktop", "B",
     "Mostly shipping. The new carrier contract took about eleven percent off every shipment."),
    ("Microsoft David Desktop", "A",
     "Then we should lock that in for another year before the rate goes back up again."),
    ("Microsoft Zira Desktop", "B",
     "Agreed. I will draft the renewal and send it round for review by Thursday afternoon."),
]

_PS = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('{voice}')
$s.SetOutputToWaveFile('{path}')
$s.Speak('{text}')
$s.Dispose()
"""


def build_conversation(out_dir: Path):
    """Two voices, alternating, with a gap between turns. Returns (wav, truth)."""
    chunks, truth, t = [], [], 0.0
    for i, (voice, speaker, text) in enumerate(TURNS):
        part = out_dir / f"turn{i}.wav"
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        _PS.format(voice=voice, path=part, text=text.replace("'", "''"))],
                       check=True, capture_output=True)
        with wave.open(str(part), "rb") as w:
            rate, ch = w.getframerate(), w.getnchannels()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        mono = data.astype(np.float32) / 32768.0
        if ch == 2:
            mono = mono.reshape(-1, 2).mean(axis=1)
        mono = soxr.resample(mono, rate, 16000).astype(np.float32)
        truth.append({"speaker": speaker, "start": t, "end": t + mono.size / 16000})
        chunks += [mono, np.zeros(int(0.35 * 16000), dtype=np.float32)]
        t = truth[-1]["end"] + 0.35

    audio = np.concatenate(chunks)
    wav = out_dir / "conversation.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return wav, truth


def main() -> int:
    from livecaptions.asr.whisper import load_model
    from livecaptions.config import Settings
    from livecaptions.diarize.pipeline import diarize_file

    out = Path(tempfile.mkdtemp(prefix="lc-diar-"))
    wav, truth = build_conversation(out)
    print(f"conversation: {wav}  ({truth[-1]['end']:.1f}s, {len(truth)} turns, 2 speakers)")

    settings = Settings(device="cpu", model_name="tiny.en")
    model = load_model(settings)
    segments, n_speakers = diarize_file(str(wav), model, settings,
                                        backend="sherpa", num_speakers=2)

    def truth_at(mid):
        return next((r["speaker"] for r in truth if r["start"] <= mid <= r["end"]), None)

    rows, counts = [], {}
    for seg in segments:
        actual = truth_at((seg.start + seg.end) / 2)
        rows.append((seg.speaker, actual, seg.start, seg.end, seg.text.strip()[:56]))
        if actual:
            counts.setdefault(seg.speaker, {}).setdefault(actual, 0)
            counts[seg.speaker][actual] += 1

    # The diarizer's labels are arbitrary, so map each to whichever true speaker it
    # most often covers before scoring — the question is consistency, not naming.
    best = {label: max(c, key=c.get) for label, c in counts.items()}
    scored = [(p, a) for p, a, *_ in rows if a]
    ok = sum(1 for p, a in scored if best.get(p) == a)

    print(f"speakers found: {n_speakers} (truth 2)   segments: {len(segments)}")
    for label, actual, start, end, text in rows:
        mark = "OK   " if (actual and best.get(label) == actual) else (
            "WRONG" if actual else "n/a  ")
        print(f"  {mark} {label} (truth {actual})  {start:6.2f}-{end:6.2f}  \"{text}\"")
    pct = 100 * ok / max(1, len(scored))
    print(f"\nCORRECT: {ok}/{len(scored)} segments ({pct:.0f}%)")

    if n_speakers != 2 or pct < 80:
        print("REGRESSION: the offline pass should nail this easy case.")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
